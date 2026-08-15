"""Intensity normalization of the overlapping region.

Two digitizers have different gain, offset, gamma, and possibly shading
characteristics. Normalization is computed only over the valid overlap so
that borders and fiducials unique to one scanner's field of view cannot bias
the comparison. The comparison itself runs on the percentile-rank (empirical
CDF) transform of each image, which removes any monotonic response
difference exactly; the linear gain/offset fit is kept as a scanner-response
diagnostic. Spatial variation is *reported* via tile statistics but never
corrected, so genuine information loss cannot be hidden by normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsFloat, cast

import cv2
import numpy as np

# scikit-image does not type this function precisely; the return value is
# narrowed with an explicit cast at the call site instead.
from skimage.filters import (
    threshold_otsu,  # pyright: ignore[reportUnknownVariableType]
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from scanner_comparison.core.imtypes import BoolMask, F32Image

_MAX_FIT_SAMPLES = 200_000
_FIT_SEED = 0
_MIN_TILE_FRACTION = 4
# Background exclusion never thresholds above this fraction of the normalized
# range, so unimodal content (no real surround) keeps all but its darkest
# tail.
_BACKGROUND_THRESHOLD_CAP = 0.10
# Marker value written by the flood fill into border-connected dark pixels.
_FLOOD_FILL_VALUE = 2


class NormalizationError(ValueError):
    """Raised when an image cannot be normalized over the given mask."""


@dataclass(frozen=True)
class RangeNormalization:
    """Percentile limits used to rescale one image to [0, 1]."""

    lower: float
    upper: float


@dataclass(frozen=True)
class TileResidualStats:
    """Spatial variation of the residual difference across the overlap.

    A large ``std`` relative to ``mean`` indicates spatially varying
    differences (e.g. scanner shading/vignetting) rather than a uniform
    response gap.
    """

    mean: float
    std: float
    tiles_used: int


def robust_rescale(
    img: F32Image,
    mask: BoolMask,
    *,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> tuple[F32Image, RangeNormalization]:
    """Rescale masked pixels to [0, 1] using robust percentile limits.

    Pixels outside the mask are set to zero in the output.

    Returns:
        The normalized image and the percentile limits that were used.

    Raises:
        NormalizationError: if the mask is empty or the range is degenerate.

    """
    values = img[mask]
    if values.size == 0:
        msg = "Cannot normalize: the overlap mask is empty"
        raise NormalizationError(msg)
    # Scalar percentiles return typed numpy scalars (a sequence of quantiles
    # would return an array whose indexing is loosely typed).
    lo = float(np.percentile(values, p_low))
    hi = float(np.percentile(values, p_high))
    if hi <= lo:
        msg = f"Degenerate intensity range in overlap: [{lo}, {hi}]"
        raise NormalizationError(msg)
    out = np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    out[~mask] = 0.0
    return out, RangeNormalization(lo, hi)


def percentile_rank(img: F32Image, mask: BoolMask) -> F32Image:
    """Empirical-CDF (percentile rank) transform over the masked pixels.

    Each masked pixel is replaced by the fractional average rank of its
    intensity among the masked values, mapping the masked histogram to a
    uniform (0, 1) distribution. Any strictly monotonic response difference
    between the scanners (gain, offset, gamma) is removed exactly, so the
    differences that remain are structural. Pixels outside the mask are 0.

    Returns:
        The rank image in (0, 1), zero outside the mask.

    Raises:
        NormalizationError: if the mask is empty.

    """
    values = img[mask]
    if values.size == 0:
        msg = "Cannot rank-transform: the overlap mask is empty"
        raise NormalizationError(msg)
    sorter = np.argsort(values)
    sorted_values = values[sorter]
    # Tied pixels share the average of the rank span they occupy.
    _unique, first, counts = np.unique(
        sorted_values, return_index=True, return_counts=True
    )
    average_rank = first.astype(np.float64) + (counts.astype(np.float64) - 1) / 2.0
    sorted_ranks = np.repeat(average_rank, counts)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[sorter] = sorted_ranks
    out = np.zeros(img.shape, dtype=np.float32)
    out[mask] = ((ranks + 0.5) / values.size).astype(np.float32)
    return out


def fit_gain_offset(
    src: F32Image,
    dst: F32Image,
    mask: BoolMask,
) -> tuple[float, float]:
    """Least-squares linear mapping ``dst ~= gain * src + offset`` on the mask.

    Removes any residual linear response difference left after independent
    percentile normalization, modeling the second scanner's overall response.

    Returns:
        The fitted ``(gain, offset)``.

    Raises:
        NormalizationError: if the mask is empty.

    """
    x = src[mask]
    y = dst[mask]
    if x.size == 0:
        msg = "Cannot fit gain/offset: the overlap mask is empty"
        raise NormalizationError(msg)
    if x.size > _MAX_FIT_SAMPLES:
        rng = np.random.default_rng(_FIT_SEED)
        idx = rng.choice(x.size, size=_MAX_FIT_SAMPLES, replace=False)
        x = x[idx]
        y = y[idx]
    design = np.stack([x, np.ones_like(x)], axis=1)
    design64 = np.asarray(design, dtype=np.float64)
    solution: npt.NDArray[np.float64] = np.linalg.lstsq(
        design64, np.asarray(y, dtype=np.float64), rcond=None
    )[0]
    gain = float(cast("SupportsFloat", solution[0]))
    offset = float(cast("SupportsFloat", solution[1]))
    return gain, offset


def apply_gain_offset(
    img: F32Image,
    gain: float,
    offset: float,
    mask: BoolMask,
) -> F32Image:
    """Apply a linear mapping, clip to [0, 1], and zero out non-mask pixels.

    Returns:
        The corrected image.

    """
    out = np.clip(img * gain + offset, 0.0, 1.0).astype(np.float32)
    out[~mask] = 0.0
    return out


def tile_residual_stats(
    a: F32Image,
    b: F32Image,
    mask: BoolMask,
    *,
    tiles: int = 8,
) -> TileResidualStats:
    """Summarize spatial variation of the residual ``b - a`` across a grid.

    The overlap bounding box is split into ``tiles`` x ``tiles`` cells; the
    per-cell residual standard deviations are averaged. Cells with little
    mask coverage are skipped.

    Returns:
        Mean and standard deviation of the per-cell residual std, and the
        number of cells used.

    """
    rows, cols = _mask_bbox(mask)
    cell_h = max(1, (rows[1] - rows[0]) // tiles)
    cell_w = max(1, (cols[1] - cols[0]) // tiles)
    min_pixels = cell_h * cell_w // _MIN_TILE_FRACTION
    stds: list[float] = []
    for r0 in range(rows[0], rows[1], cell_h):
        for c0 in range(cols[0], cols[1], cell_w):
            r1 = min(r0 + cell_h, rows[1])
            c1 = min(c0 + cell_w, cols[1])
            cell_mask = mask[r0:r1, c0:c1]
            if int(cell_mask.sum()) < min_pixels:
                continue
            residual = (b - a)[r0:r1, c0:c1][cell_mask]
            stds.append(float(cast("SupportsFloat", residual.std())))
    if not stds:
        return TileResidualStats(mean=0.0, std=0.0, tiles_used=0)
    arr = np.asarray(stds, dtype=np.float64)
    return TileResidualStats(
        mean=float(cast("SupportsFloat", arr.mean())),
        std=float(cast("SupportsFloat", arr.std())),
        tiles_used=int(arr.size),
    )


def surround_mask(ref_norm: F32Image, mask: BoolMask) -> BoolMask:
    """Exclude the dark non-film surround from a comparison mask.

    "Surround" means pixels that are both dark (below an Otsu threshold on
    the normalized reference, capped at ``_BACKGROUND_THRESHOLD_CAP``) and
    connected to the frame border. Dark regions enclosed by brighter film
    content are kept: intensity alone cannot distinguish them from the
    surround, but only border-connected darkness is certain to be outside
    the subject.

    Returns:
        ``mask`` minus the surround pixels.

    """
    values = ref_norm[mask]
    if values.size == 0:
        return mask
    otsu = float(cast("SupportsFloat", threshold_otsu(values)))
    cutoff = min(otsu, _BACKGROUND_THRESHOLD_CAP)
    dark: npt.NDArray[np.uint8] = (ref_norm < cutoff).astype(np.uint8)
    # Pad with a 1-px dark border so every border-connected dark region is
    # reachable from a single flood seed at the corner of the padding.
    padded: npt.NDArray[np.uint8] = np.pad(dark, 1, constant_values=1)
    flooded = padded.copy()
    flood_mask = np.zeros((flooded.shape[0] + 2, flooded.shape[1] + 2), dtype=np.uint8)
    _ = cv2.floodFill(flooded, flood_mask, (0, 0), _FLOOD_FILL_VALUE)
    surround = cast("BoolMask", (flooded == _FLOOD_FILL_VALUE)[1:-1, 1:-1])
    return mask & ~surround


def _mask_bbox(mask: BoolMask) -> tuple[tuple[int, int], tuple[int, int]]:
    """Bounding box of the True region.

    Returns:
        ``((row0, row1), (col0, col1))`` with exclusive upper bounds.

    Raises:
        NormalizationError: if the mask is empty.

    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        msg = "Cannot compute a bounding box: the overlap mask is empty"
        raise NormalizationError(msg)
    return (int(ys.min()), int(ys.max()) + 1), (int(xs.min()), int(xs.max()) + 1)
