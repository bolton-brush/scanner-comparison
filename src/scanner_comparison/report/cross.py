"""Cross-direction consistency metrics from a bidirectional run.

A forward run aligns DIR_B onto DIR_A; a reverse run aligns DIR_A onto
DIR_B. If both registrations are correct, the reverse warp inverts the
forward warp — the residual of their composition (the round-trip
displacement) measures registration self-consistency in pixels, and the
per-pair metric deltas expose direction-asymmetric effects (resampling
roles, per-scanner artifacts) that a single direction hides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scanner_comparison.records.results import RunSummary


@dataclass(frozen=True)
class CrossDirectionMetrics:
    """Per-pair comparison of the two alignment directions.

    Deltas are forward minus reverse. ``roundtrip_max_px`` is the largest
    displacement of the composed warp (reverse after forward, mapping the
    reference frame A -> B -> A) over the frame corners; it is 0 for
    perfectly inverse-consistent warps, so a value beyond registration
    precision flags an inconsistency between the directions.

    Note that ``blur_sigma`` is SIGNED (positive = the second scan
    blurrier), so consistent directions read opposite signs of the same
    magnitude and ``delta_blur_sigma`` reads ~2x the gap — for this metric
    the antisymmetry stat is the SUM (0 when consistent), not the delta.
    """

    name: str
    delta_rmse: float
    delta_ssim: float
    delta_grad_corr: float
    delta_grad_energy_ratio: float
    delta_blur_sigma: float
    roundtrip_max_px: float


def cross_direction_metrics(
    forward: RunSummary, reverse: RunSummary
) -> list[CrossDirectionMetrics]:
    """Match both directions' results by pair name and compute the deltas.

    Pairs missing metrics, a warp, or a frame shape in either direction
    (i.e. pairs that errored) are skipped.

    Returns:
        One row per pair present and complete in both directions.

    """
    by_name = {result.name: result for result in reverse.results}
    rows: list[CrossDirectionMetrics] = []
    for fwd in forward.results:
        rev = by_name.get(fwd.name)
        if rev is None:
            continue
        # Local copies so the None checks below narrow the types inline.
        fwd_m, rev_m = fwd.metrics, rev.metrics
        fwd_w, rev_w = fwd.reg_warp, rev.reg_warp
        shape = fwd.ref_shape
        if (
            fwd_m is None
            or rev_m is None
            or fwd_w is None
            or rev_w is None
            or shape is None
        ):
            continue
        rows.append(
            CrossDirectionMetrics(
                name=fwd.name,
                delta_rmse=fwd_m.rmse - rev_m.rmse,
                delta_ssim=fwd_m.ssim - rev_m.ssim,
                delta_grad_corr=fwd_m.grad_corr - rev_m.grad_corr,
                delta_grad_energy_ratio=(
                    fwd_m.grad_energy_ratio - rev_m.grad_energy_ratio
                ),
                delta_blur_sigma=fwd_m.blur_sigma - rev_m.blur_sigma,
                roundtrip_max_px=roundtrip_max_px(fwd_w, rev_w, shape),
            )
        )
    return rows


def roundtrip_max_px(
    forward_warp: list[float],
    reverse_warp: list[float],
    ref_shape: tuple[int, int],
) -> float:
    """Largest corner displacement of the warp round-trip, in pixels.

    Both warps follow the ``WARP_INVERSE_MAP`` convention (reference ->
    moving coordinates), so forward maps A -> B and reverse maps B -> A;
    their composition maps the reference frame onto itself and should be
    the identity.

    Returns:
        The maximum Euclidean displacement over the four frame corners.

    """
    forward3 = _as_homogeneous(forward_warp)
    reverse3 = _as_homogeneous(reverse_warp)
    composite = reverse3 @ forward3
    height, width = ref_shape
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [width - 1.0, 0.0, 1.0],
            [0.0, height - 1.0, 1.0],
            [width - 1.0, height - 1.0, 1.0],
        ]
    )
    mapped = (composite @ corners.T).T
    return float(np.linalg.norm(mapped - corners, axis=1).max())


def _as_homogeneous(warp: list[float]) -> np.ndarray:
    """Pack a row-major 2x3 warp into a 3x3 homogeneous matrix.

    Returns:
        The 3x3 matrix.

    """
    matrix = np.asarray(warp, dtype=np.float64).reshape(2, 3)
    return np.vstack([matrix, [0.0, 0.0, 1.0]])
