"""Effective shared resolution via Fourier ring correlation (FRC).

FRC is the standard instrument for "at what spatial scale do these two
images actually agree" (the effective-resolution measure of microscopy and
cryo-EM): the two images' Fourier transforms are correlated in concentric
frequency rings, giving a curve that starts near 1.0 (coarse structure
agrees) and decays to ~0 (film grain / sensor noise — a different physical
realization in each scan — dominates). The first crossing below a
correlation criterion gives a spatial frequency ``f*``; the effective
shared resolution is ``1 / f*`` pixels, read as "structures larger than
this many pixels are conserved across both scans".

Two criteria are reported: 0.5 (conservative) and the classic 1/7 (van
Heel). The inputs are the prepared, fully corrected rank images, so the
measurement is the noise-limited shared resolution AFTER the MTF
equalization — the blur correction cannot inflate it. The metric is
symmetric by construction (``Re(F1 . conj(F2)) == Re(F2 . conj(F1))``), so
both directions of a bidirectional run read the same value.

Windowing is load-bearing: both spectra are computed through the SAME
apodization, and anything shared by the two windowed images leaks
correlation into every ring. The evaluation therefore (a) multiplies by a
2-D Hann window (frame edge) and (b) multiplies by the EDGE-SOFTENED mask
— the metric mask blurred by ``_MASK_SOFT_EDGE_SIGMA`` px — because the
rank images are zero outside the mask and a hard mask boundary would
appear in both spectra as an identical step edge (measured: with a hard
mask the curve floors at ~0.9 even for mismatched content). Rings are
capped at the inscribed Nyquist circle (0.5 cycles/px); the rectangular
FFT's corner regions are not evaluated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsFloat, SupportsInt, cast

import cv2
import numpy as np

from scanner_comparison.core.imtypes import BoolMask, F32Image, erode_mask, shape2

if TYPE_CHECKING:
    import numpy.typing as npt

# Correlation criteria for the resolution crossing: 0.5 (conservative) and
# the classic 1/7 (van Heel). Public: shared by metrics.compute_metrics.
FRC_THRESHOLD_HALF = 0.5
FRC_THRESHOLD_SEVENTH = 1.0 / 7.0
# Erosion of the evaluation mask: the windowed FFT sees the mask rim;
# dropping 2 px keeps the rim's spectral leakage out.
_MASK_ERODE_PX = 2
# The mask enters the window edge-softened by this sigma: the rank images
# are zero outside the mask, and a HARD mask boundary would be an identical
# step edge in both spectra, correlating every ring (see module docstring).
_MASK_SOFT_EDGE_SIGMA = 2.0
# A frequency ring enters the curve only with at least this many pixels.
_MIN_RING_PIXELS = 30
# Coarsest rings skipped: DC and its immediate neighborhood carry the
# window/mask envelope, not content agreement.
_FIRST_RING = 2
# Rings stop at the inscribed Nyquist circle (cycles/px).
_NYQUIST_FREQ = 0.5
# Reported when the curve never crosses the criterion: detail is conserved
# down to the Nyquist limit (2 px).
_NYQUIST_FLOOR_PX = 2.0
# A mask smaller than this after erosion cannot support a meaningful curve.
_MIN_MASK_PIXELS = 1_000


@dataclass(frozen=True)
class FrcCurve:
    """Per-ring Fourier ring correlation between two aligned images.

    ``freq`` holds the ring-center spatial frequencies in cycles per pixel
    (0.5 = Nyquist); ``corr`` the ring correlation in [-1, 1]. Empty arrays
    mean the mask was too small to evaluate.
    """

    freq: npt.NDArray[np.float64]
    corr: npt.NDArray[np.float64]


def frc_curve(a: F32Image, b: F32Image, mask: BoolMask) -> FrcCurve:
    """Compute the FRC curve of two aligned images over the masked overlap.

    The masked region is cropped to its bounding box, mean-subtracted, and
    apodized with a 2-D Hann window times the edge-softened mask (see the
    module docstring for why the mask edge must be soft) before the FFTs.
    Rings are one Fourier pixel wide (``1 / max(h, w)`` cycles/px) and stop
    at the inscribed Nyquist circle.

    Returns:
        The per-ring frequencies and correlations; empty arrays when the
        mask is too small to evaluate.

    """
    eval_mask = erode_mask(mask, _MASK_ERODE_PX)
    if int(eval_mask.sum()) < _MIN_MASK_PIXELS:
        return FrcCurve(freq=np.empty(0), corr=np.empty(0))
    (r0, r1), (c0, c1) = _mask_bbox(eval_mask)
    sub_mask = eval_mask[r0:r1, c0:c1]
    height, width = shape2(sub_mask)
    fa = _windowed_spectrum(a[r0:r1, c0:c1], sub_mask)
    fb = _windowed_spectrum(b[r0:r1, c0:c1], sub_mask)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    delta = 1.0 / max(height, width)
    rings: npt.NDArray[np.int64] = np.floor(
        np.sqrt(fy**2 + fx**2) / delta
    ).astype(np.int64)
    return _ring_correlations(fa, fb, rings, delta)


def _ring_correlations(
    fa: npt.NDArray[np.complex128],
    fb: npt.NDArray[np.complex128],
    rings: npt.NDArray[np.int64],
    delta: float,
) -> FrcCurve:
    """Per-ring normalized cross-correlation of two spectra.

    Returns:
        The FRC curve; ring centers are at ``(k + 0.5) * delta`` cycles/px.
        Rings past the inscribed Nyquist circle are excluded.

    """
    freqs: list[float] = []
    corrs: list[float] = []
    for k in range(_FIRST_RING, int(cast("SupportsInt", rings.max()))):
        freq = (k + 0.5) * delta
        if freq > _NYQUIST_FREQ:
            break
        # numpy's comparison-operator stubs resolve loosely; narrow it.
        sel = cast("npt.NDArray[np.bool_]", rings == k)
        if int(sel.sum()) < _MIN_RING_PIXELS:
            continue
        num, den = _ring_stats(fa[sel], fb[sel])
        freqs.append(freq)
        corrs.append(num / den if den > 0.0 else 0.0)
    return FrcCurve(
        freq=np.asarray(freqs, dtype=np.float64),
        corr=np.asarray(corrs, dtype=np.float64),
    )


def _ring_stats(
    xa: npt.NDArray[np.complex128], xb: npt.NDArray[np.complex128]
) -> tuple[float, float]:
    """Normalized-correlation numerator and denominator of one ring.

    Returns:
        ``(Re(sum(xa * conj(xb))), sqrt(sum|xa|^2 * sum|xb|^2))``.

    """
    num = float(np.real((xa * np.conj(xb)).sum()))
    den = math.sqrt(
        float(cast("SupportsFloat", (np.abs(xa) ** 2).sum()))
        * float(cast("SupportsFloat", (np.abs(xb) ** 2).sum()))
    )
    return num, den


def _windowed_spectrum(
    img: F32Image, mask: BoolMask
) -> npt.NDArray[np.complex128]:
    """FFT of the masked, mean-subtracted, softly apodized image.

    The window is the 2-D Hann window times the mask with its edge softened
    by a small Gaussian — the image is zero outside the mask, and a hard
    mask boundary would be an identical step edge in both spectra (see the
    module docstring).

    Returns:
        The complex spectrum of ``img`` over ``mask``.

    """
    sub = img.astype(np.float64)
    sub -= float(cast("SupportsFloat", sub[mask].mean()))
    height, width = shape2(mask)
    hann = np.hanning(height)[:, None] * np.hanning(width)[None, :]
    soft_mask = cv2.GaussianBlur(
        mask.astype(np.float64), (0, 0), _MASK_SOFT_EDGE_SIGMA
    )
    return np.fft.rfft2(sub * hann * soft_mask)


def frc_resolution(curve: FrcCurve, threshold: float) -> float | None:
    """Effective shared resolution in pixels from an FRC curve.

    The first ring whose correlation drops below ``threshold`` defines the
    crossing frequency ``f*`` (linearly interpolated between the
    neighboring rings); the resolution is ``1 / f*`` pixels — structures
    larger than that agree across both images.

    Returns:
        The resolution in pixels; ``_NYQUIST_FLOOR_PX`` when the curve
        never crosses (agreement down to the sampling limit); None when the
        curve is empty or already below the threshold at the coarsest ring
        (no agreement at any measured scale).

    """
    # Scalar indexing of numpy arrays is loosely typed in the stubs; typed
    # lists keep the crossing arithmetic exact (the arrays are small).
    freqs = cast("list[float]", curve.freq.tolist())
    corrs = cast("list[float]", curve.corr.tolist())
    if not freqs:
        return None
    below = [i for i, c in enumerate(corrs) if c < threshold]
    if not below:
        return _NYQUIST_FLOOR_PX
    i = below[0]
    if i == 0:
        return None
    # Linear interpolation of the crossing between rings i-1 and i.
    c0, c1 = corrs[i - 1], corrs[i]
    f0, f1 = freqs[i - 1], freqs[i]
    f_star = f0 + (f1 - f0) * (c0 - threshold) / (c0 - c1)
    return 1.0 / f_star


def _mask_bbox(mask: BoolMask) -> tuple[tuple[int, int], tuple[int, int]]:
    """Bounding box of the True region.

    Returns:
        ``((row0, row1), (col0, col1))`` with exclusive upper bounds; a
        full-frame box when the mask is empty (the caller guards on mask
        size first).

    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return (0, shape2(mask)[0]), (0, shape2(mask)[1])
    return (int(ys.min()), int(ys.max()) + 1), (int(xs.min()), int(xs.max()) + 1)
