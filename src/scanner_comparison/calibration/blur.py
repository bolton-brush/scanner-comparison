"""Blur calibration: separate resampling blur from the scanner-pair device gap.

Two blur variables combine per comparison direction (the model validated in
the diag13/14 investigations and re-validated under the signed
common-support convention in diag16):

- ``r`` = resampling blur incurred by warping the *moving* scan. It depends
  on the interpolation kernel (``align._WARP_INTERPOLATION``) and the
  sub-pixel sampling phase; a per-pair value ``r_i`` is computed on the fly
  by integrating a measured phase response table over the warp's phase
  field. (Real warps drift through all phases across a 3000 px frame via
  the cross-axis terms, so ``r_i`` sits close to the phase average ``r_bar``.)
- ``sigma_dev`` = signed device blur constant of the scanner pair
  (positive = the second directory's scanner is intrinsically blurrier),
  solved from the data by the bidirectional signed-gap measurement
  (``pipeline.solve_blur_constant``); the edge-energy instrument cannot see
  the resampling component when the device gap dominates (diag16's
  quadrature compression), so ``r`` always comes from the phase table.

Per side, ``sigma_eff^2 = sigma_intrinsic^2 + r^2`` (``r`` only when warped),
so the sharper side is blurred by ``sqrt(|sigma_eff_a^2 - sigma_eff_b^2|)``:
with the signed gap ``d2 = sigma_dev^2 + r^2`` (moving minus reference),
blur the REFERENCE by ``sqrt(d2)`` when ``d2 > 0``, else the MOVING side by
``sqrt(-d2)``. The correction applies a mask-normalized Gaussian to the rank
images only (metric path); registration features are never blurred. The
per-pair ``blur_sigma`` metric stays a *reported* residual — solving blur
per pair at comparison time would erase real per-film focus variation.

Measurement convention: the signed common-support edge-energy instrument
(``metrics.signed_blur_gap``). The previous own-support convention
(constants 0.685/0.482 for the SC108/SC113 pair) is superseded; the
common-support constants for that pair are sigma_dev = 0.533, r_bar =
0.283 (diag16, defect-masked).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, SupportsFloat, cast

import cv2
import numpy as np

from scanner_comparison.core.imtypes import (
    BoolMask,
    F32Image,
    F64Image,
    U16Image,
    as_f32,
    erode_mask,
    shape2,
)
from scanner_comparison.imaging.align import film_roi_mask, warp_image
from scanner_comparison.imaging.metrics import signed_blur_gap
from scanner_comparison.imaging.normalize import robust_rescale, surround_mask

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

_BLUR_VERSION = 2
# Sub-pixel phase grid for the resampling-penalty table.
PHASE_GRID_N = 6
# Content-mask erosion for the phase sweep: past the blur metric's own
# erosion plus the 1 px shift and border-constant fill.
_PROBE_ERODE_PX = 24
# The phase field is evaluated on this stride: it is smooth, so dense
# sampling adds nothing.
_PHASE_SAMPLE_STRIDE = 2
# Denominator floor for the mask-normalized Gaussian.
_MASK_NORMALIZE_EPS = 1e-6


class BlurCalibrationError(ValueError):
    """Raised when a blur calibration cannot be computed from the data."""


@dataclass(frozen=True)
class PairBlur:
    """Per-pair bidirectional evidence of the device-blur solve.

    ``m_forward`` / ``m_reverse`` are the signed common-support blur gaps
    measured with the second directory's scan moving (forward) and with
    the first directory's scan moving (reverse, scale correction inverted).
    ``sigma_dev_sq_signed`` is the per-pair signed device variance
    ``(m_f^2 - m_r^2 - r_f^2 + r_r^2) / 2`` (signed squares; the table
    resampling penalties ``r_forward_table`` / ``r_reverse_table`` correct
    for asymmetric phase sampling). ``r_data`` is the data-solved
    resampling component ``sqrt((m_f^2 + m_r^2) / 2)`` — a cross-check
    only: the instrument compresses it toward 0 when the device gap
    dominates, so the table values are the operative ``r``.
    """

    name: str
    m_forward: float
    m_reverse: float
    sigma_dev_sq_signed: float
    r_data: float
    r_forward_table: float
    r_reverse_table: float


@dataclass(frozen=True)
class BlurCalibration:
    """A solved scanner-pair device blur constant with its evidence.

    ``sigma_dev`` is the signed constant for ``--blur-correction``
    (positive = the second directory's scanner is intrinsically blurrier).
    ``sigma_dev_forward`` / ``sigma_dev_reverse`` are the one-sided arms
    (forward gaps alone / reverse gaps alone, each table-corrected) — they
    should bracket ``sigma_dev``; a wide split flags direction asymmetry.
    ``r_bar`` is the phase-averaged resampling penalty of the warp kernel
    (both sides' content tables combined); ``r_data`` is the median
    per-pair data-solved resampling component, expected to under-read
    ``r_bar`` (quadrature compression) and recorded only as a cross-check.
    """

    sigma_dev: float
    r_bar: float
    scale_correction: float
    sigma_dev_forward: float
    sigma_dev_reverse: float
    r_data: float
    pairs: tuple[PairBlur, ...]


def masked_gaussian(img: F32Image, sigma: float, mask: BoolMask) -> F32Image:
    """Gaussian blur that never mixes mask-outside zeros into the mask rim.

    ``G(img * m) / G(m)`` is the ordinary blur in the mask interior and a
    renormalized local average near the boundary, so applying a blur
    correction does not darken the rim and perturb value metrics.

    Returns:
        The blurred image, zero outside ``mask``; ``img`` itself when
        ``sigma`` is 0.

    """
    if sigma <= 0.0:
        return img
    m = mask.astype(np.float32)
    num = as_f32(cv2.GaussianBlur(img * m, (0, 0), sigma))
    den = as_f32(cv2.GaussianBlur(m, (0, 0), sigma))
    out = as_f32(num / np.maximum(den, _MASK_NORMALIZE_EPS))
    out[den <= _MASK_NORMALIZE_EPS] = 0.0
    out[~mask] = 0.0
    return out


def correction_sigmas(sigma_dev: float, r: float) -> tuple[float, float]:
    """Blur sigmas ``(on_reference, on_moving)`` for the two-variable rule.

    ``sigma_dev`` is signed: the moving scanner's intrinsic blur minus the
    reference scanner's. The moving side additionally incurs the resampling
    penalty ``r``. The net gap ``d2 = sign(sigma_dev)*sigma_dev^2 + r^2`` is
    positive when the (warped) moving side is blurrier — then the REFERENCE
    is blurred by ``sqrt(d2)``; otherwise the moving side is blurred by
    ``sqrt(-d2)``.

    Returns:
        The Gaussian sigmas to apply to the reference and moving rank
        images.

    """
    d2 = math.copysign(sigma_dev * sigma_dev, sigma_dev) + r * r
    if d2 >= 0.0:
        return math.sqrt(d2), 0.0
    return 0.0, math.sqrt(-d2)


def phase_table(img: U16Image, *, grid: int = PHASE_GRID_N) -> F64Image:
    """Resampling-penalty table r(fx, fy) of the warp interpolation kernel.

    A uniform translation phase grid samples the same kernel phases a real
    warp drifts through; each cell measures the SIGNED common-support blur
    gap (``metrics.signed_blur_gap``) the warp path (``align.warp_image``)
    induces on real content from ``img`` (the moving side's scanner).
    r(0, 0) = 0 by construction.

    Cells are signed and some may be negative: on shifted correlated
    content the common edge support lands on edge flanks and the
    flank-gradient comparison reads a +-0.2..0.3 px support-selection
    systematic (diag16), so single cells are unreliable. The table is only
    meaningful through its phase average (``r_bar``) or the per-warp
    phase-field integral in ``resampling_penalty`` — real warps sweep the
    whole table and the systematic averages out.

    Returns:
        The ``grid`` x ``grid`` table of signed r values in pixels.

    """
    mask = _content_mask(img)
    base = as_f32(img.astype(np.float32) / 65535.0)
    table = np.zeros((grid, grid), dtype=np.float64)
    phases = np.linspace(0.0, 1.0, grid, endpoint=False)
    for iy, fy in enumerate(phases):
        for ix, fx in enumerate(phases):
            warp = np.array([[1.0, 0.0, fx], [0.0, 1.0, fy]], dtype=np.float64)
            shifted = warp_image(img, warp, shape2(img))
            shifted_f = as_f32(shifted.astype(np.float32) / 65535.0)
            table[iy, ix] = signed_blur_gap(base, shifted_f, mask)
    return table


def resampling_penalty(
    warp: npt.NDArray[np.float64], ref_shape: tuple[int, int], table: F64Image
) -> float:
    """Per-pair resampling penalty from the warp's phase field over the ROI.

    ``phi(x) = frac(A . x + t)`` is the sub-pixel sampling phase at each
    output pixel; ``r_i`` aggregates the table's per-phase ``r^2`` over the
    film ROI (the phase field is smooth, so ROI vs full content mask is
    immaterial).

    Returns:
        The phase-averaged resampling blur in pixels.

    """
    # numpy scalar indexing stubs are loose; a typed list keeps the
    # coefficient access exact.
    w = np.asarray(warp, dtype=np.float64).reshape(2, 3).tolist()
    height, width = ref_shape
    roi = film_roi_mask((height, width))
    grids = np.mgrid[0:height:_PHASE_SAMPLE_STRIDE, 0:width:_PHASE_SAMPLE_STRIDE]
    ys = cast("npt.NDArray[np.int64]", grids[0])
    xs = cast("npt.NDArray[np.int64]", grids[1])
    keep = roi[ys, xs]
    px = w[0][0] * xs + w[0][1] * ys + w[0][2]
    py = w[1][0] * xs + w[1][1] * ys + w[1][2]
    fx = np.abs(px[keep] % 1.0)
    fy = np.abs(py[keep] % 1.0)
    r2 = _sample_periodic(table**2, fy, fx)
    return math.sqrt(float(np.mean(r2)))


def write_blur_calibration(path: Path, cal: BlurCalibration) -> Path:
    """Write the blur calibration JSON (evidence + recommended constant).

    Returns:
        The written path.

    """
    payload = {
        "version": _BLUR_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "convention": "signed common-support bidirectional",
        "scale_correction": cal.scale_correction,
        "sigma_dev": cal.sigma_dev,
        "sigma_dev_forward": cal.sigma_dev_forward,
        "sigma_dev_reverse": cal.sigma_dev_reverse,
        "r_bar": cal.r_bar,
        "r_data": cal.r_data,
        "pairs": [
            {
                "name": p.name,
                "m_forward": p.m_forward,
                "m_reverse": p.m_reverse,
                "sigma_dev_sq_signed": p.sigma_dev_sq_signed,
                "r_data": p.r_data,
                "r_forward_table": p.r_forward_table,
                "r_reverse_table": p.r_reverse_table,
            }
            for p in cal.pairs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _written = path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_blur_calibration(path: Path) -> BlurCalibration:
    """Read a blur calibration JSON written by ``write_blur_calibration``.

    The pipeline consumes the constant via ``CompareConfig.blur_correction``
    (the JSON itself is an inspectable record); this reader exists so
    tooling can load the artifact symmetrically with the other
    calibrations.

    Returns:
        The parsed calibration.

    Raises:
        BlurCalibrationError: if the file is missing, malformed, or of an
            unsupported version.

    """
    try:
        raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Cannot read blur calibration: {path} ({exc})"
        raise BlurCalibrationError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"Unsupported blur calibration file: {path}"
        raise BlurCalibrationError(msg)
    payload = cast("dict[str, object]", raw)
    if payload.get("version") != _BLUR_VERSION:
        msg = f"Unsupported blur calibration file: {path}"
        raise BlurCalibrationError(msg)
    try:
        pairs = tuple(
            PairBlur(
                name=str(p["name"]),
                m_forward=float(cast("SupportsFloat", p["m_forward"])),
                m_reverse=float(cast("SupportsFloat", p["m_reverse"])),
                sigma_dev_sq_signed=float(
                    cast("SupportsFloat", p["sigma_dev_sq_signed"])
                ),
                r_data=float(cast("SupportsFloat", p["r_data"])),
                r_forward_table=float(cast("SupportsFloat", p["r_forward_table"])),
                r_reverse_table=float(cast("SupportsFloat", p["r_reverse_table"])),
            )
            for p in cast("list[dict[str, object]]", payload["pairs"])
        )
        calibration = BlurCalibration(
            sigma_dev=float(cast("SupportsFloat", payload["sigma_dev"])),
            r_bar=float(cast("SupportsFloat", payload["r_bar"])),
            scale_correction=float(
                cast("SupportsFloat", payload["scale_correction"])
            ),
            sigma_dev_forward=float(
                cast("SupportsFloat", payload["sigma_dev_forward"])
            ),
            sigma_dev_reverse=float(
                cast("SupportsFloat", payload["sigma_dev_reverse"])
            ),
            r_data=float(cast("SupportsFloat", payload["r_data"])),
            pairs=pairs,
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"Malformed blur calibration file: {path} ({exc})"
        raise BlurCalibrationError(msg) from exc
    return calibration


class BlurCorrector:
    """Applies a solved blur correction per pair on the rank images.

    Created once per run (the phase table is content-dependent and shared by
    all pairs of one direction); the per-pair resampling penalty ``r_i`` is
    computed from each pair's estimated warp.
    """

    def __init__(self, sigma_dev: float, table: F64Image) -> None:
        """Store the constant and the phase table (see ``phase_table``)."""
        self._sigma_dev: float = sigma_dev
        self._table: F64Image = table

    @classmethod
    def from_moving_image(
        cls, sigma_dev: float, moving: U16Image
    ) -> BlurCorrector:
        """Build a corrector, measuring the phase table on a moving scan.

        Returns:
            The corrector for runs warping images like ``moving``.

        """
        return cls(sigma_dev, phase_table(moving))

    def apply(
        self,
        ref_rank: F32Image,
        mov_rank: F32Image,
        mask: BoolMask,
        warp: npt.NDArray[np.float64],
    ) -> tuple[F32Image, F32Image, tuple[float, float, float]]:
        """Blur the sharper side's rank image by the per-pair net gap.

        Returns:
            ``(reference, moving, (r_i, sigma_ref, sigma_mov))``: the
            corrected rank images and the correction record (the per-pair
            resampling penalty and the two applied sigmas).

        """
        r_i = resampling_penalty(warp, shape2(ref_rank), self._table)
        sig_ref, sig_mov = correction_sigmas(self._sigma_dev, r_i)
        return (
            masked_gaussian(ref_rank, sig_ref, mask),
            masked_gaussian(mov_rank, sig_mov, mask),
            (r_i, sig_ref, sig_mov),
        )


def _content_mask(img: U16Image) -> BoolMask:
    """Film-content mask of a single scan (ROI minus dark surround), eroded.

    Returns:
        The eroded content mask.

    """
    roi = film_roi_mask(shape2(img))
    norm, _ = robust_rescale(img.astype(np.float32), roi)
    return erode_mask(surround_mask(norm, roi), _PROBE_ERODE_PX)


def _sample_periodic(
    grid: F64Image, fy: npt.NDArray[np.float64], fx: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Bilinear interpolation of the phase table, periodic in both axes.

    Returns:
        The interpolated values at ``(fy, fx)`` (fractional phases).

    """
    # ``ndarray.shape`` is loosely typed in numpy's stubs; narrow it.
    n = cast("int", grid.shape[0])
    pos_y = fy * n
    pos_x = fx * n
    y0 = np.floor(pos_y).astype(int) % n
    x0 = np.floor(pos_x).astype(int) % n
    y1 = (y0 + 1) % n
    x1 = (x0 + 1) % n
    wy = pos_y - np.floor(pos_y)
    wx = pos_x - np.floor(pos_x)
    top = grid[y0, x0] * (1.0 - wx) + grid[y0, x1] * wx
    bottom = grid[y1, x0] * (1.0 - wx) + grid[y1, x1] * wx
    result: npt.NDArray[np.float64] = top * (1.0 - wy) + bottom * wy
    return result
