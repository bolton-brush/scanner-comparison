"""Information-loss metrics computed on the aligned, rank-domain overlap.

The inputs are percentile-rank images (uniform (0, 1) histogram over the
masked overlap), so the metrics compare structure rather than each scanner's
intensity response. Three families of metrics are reported:

- value fidelity: MAE / RMSE / PSNR of the rank-transformed pixel values
- structure: SSIM (masked mean of the SSIM map)
- sharpness / fine detail: correlation and energy ratio of Sobel gradient
  magnitudes, which reveal blur even when pixel-wise metrics look acceptable
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsFloat, TypeAlias, cast

import cv2
import numpy as np

# scikit-image does not type this function precisely; the return value is
# narrowed with an explicit cast at the call site instead.
from skimage.metrics import (
    structural_similarity,  # pyright: ignore[reportUnknownVariableType]
)

from scanner_comparison.core.imtypes import BoolMask, F32Image, as_f32, erode_mask
from scanner_comparison.imaging.frc import (
    FRC_THRESHOLD_HALF,
    FRC_THRESHOLD_SEVENTH,
    FrcCurve,
    frc_curve,
    frc_resolution,
)

if TYPE_CHECKING:
    import numpy.typing as npt

_PSNR_CAP = 99.0
_GRADIENT_ERODE_PX = 3
_MIN_CORR_SAMPLES = 2
# Quantile (in percent) of the reference gradient magnitudes that defines the
# edge support for gradient metrics, and the minimum support size before
# falling back to the whole mask.
_EDGE_SUPPORT_QUANTILE = 70.0
_MIN_EDGE_PIXELS = 1_000
_EPS = 1e-12
# Locality probe: the signed difference is smoothed with a Gaussian of this
# sigma before measuring its magnitude. Spatially uncorrelated differences
# (film grain, focus/noise) average toward zero under the blur, so
# local_rmse << rmse; spatially coherent differences (a genuine local data
# mismatch) survive, so local_rmse ~ rmse.
_LOCAL_DIFF_BLUR_SIGMA = 4.0
# Differential-blur probe: search cap for the equivalent Gaussian blur on
# the reference that matches the moving scan's edge energy.
_BLUR_SIGMA_MAX = 3.0
_BLUR_SIGMA_BISECTION_ITERS = 8
# Structural and sharpness metrics are computed after a light Gaussian
# prefilter: film grain / sensor noise cannot be reproduced pixel-perfectly
# by a second scanner, so it would otherwise dominate SSIM and gradient
# scores without representing anatomical information loss. Value-fidelity
# metrics (MAE/RMSE/PSNR) intentionally remain unfiltered.
PREFILTER_SIGMA = 1.0
_SSIMFull: TypeAlias = "tuple[float, npt.NDArray[np.float64]]"


@dataclass(frozen=True)
class PairMetrics:
    """Metrics describing how faithfully one scan reproduces another.

    ``local_mse`` / ``local_rmse`` measure the signed difference after a
    Gaussian local-average blur (sigma ``_LOCAL_DIFF_BLUR_SIGMA``):
    ``local_rmse`` much smaller than ``rmse`` means the differences are
    spatially uncorrelated (grain/focus noise), while a ``local_rmse`` close
    to ``rmse`` means the error is a spatially coherent local data
    difference.     ``blur_sigma`` is the SIGNED common-support blur gap (in
    px): the equivalent Gaussian blur that, applied to the sharper image,
    matches the other's edge energy on the shared edge support — positive
    when the second scan is blurrier than the reference, negative when it
    is sharper, 0.0 when there is no measurable gap (see
    ``signed_blur_gap``). ``frc_resolution_px`` is the effective shared
    resolution (Fourier ring correlation, 0.5 criterion;
    ``frc_resolution_px_17`` uses the classic 1/7 criterion): structures
    larger than this many pixels agree across both scans (2.0 = conserved
    down to the Nyquist limit; None = no agreement at any measured scale).
    See ``frc.frc_resolution``.

    """

    mae: float
    rmse: float
    psnr: float
    local_mse: float
    local_rmse: float
    ssim: float
    grad_corr: float
    grad_energy_ratio: float
    blur_sigma: float
    frc_resolution_px: float | None
    frc_resolution_px_17: float | None
    overlap_fraction: float
    n_pixels: int


def compute_metrics(
    a: F32Image,
    b: F32Image,
    mask: BoolMask,
) -> tuple[PairMetrics, F32Image, F32Image, FrcCurve]:
    """Compare rank-transformed images ``a`` and ``b`` over the overlap.

    Returns:
        The metrics, plus the signed difference map (``a - b``), the SSIM
        map (both as full-size float images), and the FRC curve (for the
        per-pair curve artifact).

    """
    diff: F32Image = (a - b).astype(np.float32)
    mae, rmse = _value_errors(diff, mask)
    local_mse = _local_mse(diff, mask)
    a_f = prefilter(a)
    b_f = prefilter(b)
    ssim_map = _ssim_map(a_f, b_f)
    grad_corr, grad_ratio = _gradient_metrics(a_f, b_f, mask)
    blur_sigma = signed_blur_gap(a, b, mask)
    frc = frc_curve(a, b, mask)
    n_pixels = int(mask.sum())
    metrics = PairMetrics(
        mae=mae,
        rmse=rmse,
        psnr=_psnr(rmse),
        local_mse=local_mse,
        local_rmse=math.sqrt(local_mse),
        ssim=float(cast("SupportsFloat", ssim_map[mask].mean())),
        grad_corr=grad_corr,
        grad_energy_ratio=grad_ratio,
        blur_sigma=blur_sigma,
        frc_resolution_px=frc_resolution(frc, FRC_THRESHOLD_HALF),
        frc_resolution_px_17=frc_resolution(frc, FRC_THRESHOLD_SEVENTH),
        overlap_fraction=n_pixels / mask.size,
        n_pixels=n_pixels,
    )
    return metrics, diff, ssim_map, frc


def _value_errors(diff: F32Image, mask: BoolMask) -> tuple[float, float]:
    """Compute MAE and RMSE of a signed difference image over the mask.

    Returns:
        The ``(mae, rmse)`` pair.

    """
    values = np.abs(diff[mask])
    mae = float(cast("SupportsFloat", values.mean()))
    rmse = float(cast("SupportsFloat", np.sqrt(np.mean(values**2))))
    return mae, rmse


def local_mean_diff(diff: F32Image) -> F32Image:
    """Locally averaged signed difference (Gaussian, `_LOCAL_DIFF_BLUR_SIGMA`).

    Returns:
        The smoothed signed difference image (same shape).

    """
    return as_f32(cv2.GaussianBlur(diff, (0, 0), _LOCAL_DIFF_BLUR_SIGMA))


def local_diff_evaluation_mask(mask: BoolMask) -> BoolMask:
    """Region where the locally averaged difference is uncontaminated.

    The blur would mix in the zeros outside the mask and attenuate the local
    mean near mask edges, so both the metric and the rendered artifact use
    the mask eroded by three blur sigmas.

    Returns:
        The eroded evaluation mask.

    """
    return erode_mask(mask, math.ceil(3.0 * _LOCAL_DIFF_BLUR_SIGMA))


def _local_mse(diff: F32Image, mask: BoolMask) -> float:
    """Mean square of the locally averaged signed difference over the mask.

    Returns:
        The local mean-squared error.

    """
    values = local_mean_diff(diff)[local_diff_evaluation_mask(mask)]
    if values.size == 0:
        return 0.0
    return float(cast("SupportsFloat", np.mean(values**2)))


def _psnr(rmse: float) -> float:
    """Peak signal-to-noise ratio for unit-range data, capped for reports.

    Returns:
        PSNR in dB; ``_PSNR_CAP`` when the images are identical.

    """
    if rmse <= _EPS:
        return _PSNR_CAP
    return min(_PSNR_CAP, 20.0 * math.log10(1.0 / rmse))


def _ssim_map(a: F32Image, b: F32Image) -> F32Image:
    """Compute the full SSIM map between two normalized float images.

    scikit-image's return type for ``full=True`` is not precisely typed, so
    the result is narrowed with an explicit cast.

    Returns:
        Per-window SSIM values as a float32 image.

    """
    _mean, ssim_map64 = cast(
        "_SSIMFull",
        structural_similarity(a, b, data_range=1.0, full=True),
    )
    return ssim_map64.astype(np.float32)


def _gradient_metrics(
    a: F32Image,
    b: F32Image,
    mask: BoolMask,
) -> tuple[float, float]:
    """Gradient-magnitude correlation and energy ratio over the mask.

    Both quantities are computed only on the *edge support*: pixels where the
    reference image has an above-quantile gradient. Flat interior regions
    carry only irreproducible film grain after alignment, which would
    otherwise dilute the correlation toward zero.

    Returns:
        ``(pearson correlation of gradient magnitudes, mean(b)/mean(a))``
        over the edge support.

    """
    grad_mask = erode_mask(mask, _GRADIENT_ERODE_PX)
    grad_a = _gradient_magnitude(a)[grad_mask]
    grad_b = _gradient_magnitude(b)[grad_mask]
    support = edge_support(grad_a)
    if int(support.sum()) < _MIN_EDGE_PIXELS:
        return _pearson(grad_a, grad_b), _safe_ratio(
            float(cast("SupportsFloat", grad_b.mean())),
            float(cast("SupportsFloat", grad_a.mean())),
        )
    sup_a = grad_a[support]
    sup_b = grad_b[support]
    mean_a = float(cast("SupportsFloat", sup_a.mean()))
    mean_b = float(cast("SupportsFloat", sup_b.mean()))
    return _pearson(sup_a, sup_b), _safe_ratio(mean_b, mean_a)


def prefilter(img: F32Image) -> F32Image:
    """The light Gaussian prefilter shared by the structural metrics.

    Returns:
        The blurred image (``PREFILTER_SIGMA``).

    """
    return as_f32(cv2.GaussianBlur(img, (0, 0), PREFILTER_SIGMA))


def differential_blur_sigma(a: F32Image, b: F32Image, mask: BoolMask) -> float:
    """Equivalent Gaussian blur on ``a`` matching ``b``'s edge energy.

    Both inputs are the prefiltered rank images; the common prefilter cancels
    in quadrature, so the result estimates sqrt(sigma_b^2 - sigma_a^2) of the
    two scans. Energy is matched on ``a``'s OWN edge support (flat regions
    carry no blur signal). The evaluation mask is eroded past the reach of
    the largest trial blur so outside-mask zeros cannot attenuate the
    measurement.

    This is the previous (own-support, one-sided) measurement convention: it
    floors at 0.0 and inflates both directions (own-support selection plus
    interpolation ringing), so it cannot express "``b`` is sharper". The
    reported metric uses ``signed_blur_gap`` (common support, signed); this
    function stays public for cross-convention comparisons.

    Returns:
        The matching blur sigma in pixels; 0.0 when ``b`` is at least as
        sharp as ``a``; ``_BLUR_SIGMA_MAX`` when the gap exceeds the search
        range.

    """
    erode_px = _GRADIENT_ERODE_PX + math.ceil(
        3.0 * (PREFILTER_SIGMA + _BLUR_SIGMA_MAX)
    )
    grad_mask = erode_mask(mask, erode_px)
    grad_a = _gradient_magnitude(a)[grad_mask]
    support = edge_support(grad_a)
    if int(support.sum()) < _MIN_EDGE_PIXELS:
        support = np.ones(grad_a.shape, dtype=bool)
    return blur_sigma_on_support(a, b, grad_mask, support)


def blur_sigma_on_support(
    a_f: F32Image,
    b_f: F32Image,
    grad_mask: BoolMask,
    support: BoolMask,
) -> float:
    """Bisect the Gaussian sigma on ``a_f`` matching ``b_f``'s edge energy.

    Both inputs are the prefiltered images; ``support`` indexes into the
    grad_mask-flattened gradient values and chooses the edges the energies
    are averaged over (e.g. one image's own edge set, or the intersection
    of both). A support smaller than ``_MIN_EDGE_PIXELS`` falls back to the
    whole gradient mask (keeps the metric total on degenerate content).

    Returns:
        The matching blur sigma in pixels; 0.0 when ``b_f``'s mean edge
        energy on the support is not below ``a_f``'s; ``_BLUR_SIGMA_MAX``
        when the gap exceeds the search range.

    """
    grad_a = _gradient_magnitude(a_f)[grad_mask]
    grad_b = _gradient_magnitude(b_f)[grad_mask]
    if int(support.sum()) < _MIN_EDGE_PIXELS:
        support = np.ones(grad_a.shape, dtype=bool)
    sup_a = grad_a[support]
    sup_b = grad_b[support]
    target = float(cast("SupportsFloat", sup_b.mean()))
    base = float(cast("SupportsFloat", sup_a.mean()))
    if target <= _EPS or base <= target:
        return 0.0

    def energy_ratio(sigma: float) -> float:
        blurred = as_f32(cv2.GaussianBlur(a_f, (0, 0), sigma))
        grad = _gradient_magnitude(blurred)[grad_mask][support]
        energy = float(cast("SupportsFloat", grad.mean()))
        return energy / target

    if energy_ratio(_BLUR_SIGMA_MAX) > 1.0:
        return _BLUR_SIGMA_MAX
    # energy_ratio decreases monotonically with sigma: bisect for the crossing.
    lo, hi = 0.0, _BLUR_SIGMA_MAX
    for _ in range(_BLUR_SIGMA_BISECTION_ITERS):
        mid = 0.5 * (lo + hi)
        if energy_ratio(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def signed_blur_gap(a: F32Image, b: F32Image, mask: BoolMask) -> float:
    """Signed common-support blur gap in px: positive = ``b`` blurrier.

    Both inputs are the RAW rank images; the structural prefilter
    (``PREFILTER_SIGMA``) is applied internally, as is the metric's
    evaluation-mask erosion. The support is the intersection of BOTH
    images' above-quantile edge sets — required so the sign is unambiguous:
    on each image's own support the one-sided measurement floors at 0 and
    reads a positive gap in both directions (support-selection plus
    interpolation ringing), while on the common support a sharper ``b``
    reads as a clean negative gap.

    Sign convention: try blurring ``a`` to match ``b``; if that helps the
    gap is positive, else blur ``b`` to match ``a`` and negate (0.0 when
    neither helps). Exactly antisymmetric by construction:
    ``signed_blur_gap(a, b) == -signed_blur_gap(b, a)``.

    Caveats (diag16): the gap saturates toward the dominant blur term — a
    resampling penalty much smaller than the device gap is compressed away
    (do not solve the resampling component from this number; use the kernel
    phase table), and on shifted correlated content the support
    intersection lands on edge flanks and reads a phantom +-0.2..0.3 px.
    On aligned two-scanner pairs (the metric's operating point) it is the
    validated sharpness-gap instrument.

    Returns:
        The signed equivalent Gaussian blur in pixels.

    """
    a_f = prefilter(a)
    b_f = prefilter(b)
    erode_px = _GRADIENT_ERODE_PX + math.ceil(
        3.0 * (PREFILTER_SIGMA + _BLUR_SIGMA_MAX)
    )
    grad_mask = erode_mask(mask, erode_px)
    grad_a = _gradient_magnitude(a_f)[grad_mask]
    grad_b = _gradient_magnitude(b_f)[grad_mask]
    common = edge_support(grad_a) & edge_support(grad_b)
    forward = blur_sigma_on_support(a_f, b_f, grad_mask, common)
    if forward > 0.0:
        return forward
    backward = blur_sigma_on_support(b_f, a_f, grad_mask, common)
    return -backward if backward > 0.0 else 0.0


def edge_support(grad: npt.NDArray[np.float32]) -> BoolMask:
    """Pixels whose gradient exceeds its quantile over the mask.

    Returns:
        A boolean index into the flattened gradient values.

    """
    threshold = float(np.percentile(grad, _EDGE_SUPPORT_QUANTILE))
    return grad > threshold


def _gradient_magnitude(img: F32Image) -> F32Image:
    """Sobel gradient magnitude of a normalized float image.

    Returns:
        The per-pixel gradient magnitude.

    """
    grad_x = as_f32(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = as_f32(cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3))
    return as_f32(cv2.magnitude(grad_x, grad_y))


def _pearson(x: npt.NDArray[np.float32], y: npt.NDArray[np.float32]) -> float:
    """Pearson correlation with defined behavior for constant inputs.

    Computed manually rather than via ``np.corrcoef`` because corrcoef's
    overloads only accept >=64-bit float inputs.

    Returns:
        The correlation; 1.0 when both inputs are (near-)constant, 0.0 when
        only one is.

    """
    if x.size < _MIN_CORR_SAMPLES:
        return 1.0
    flat_x = float(cast("SupportsFloat", x.std())) <= _EPS
    flat_y = float(cast("SupportsFloat", y.std())) <= _EPS
    if flat_x or flat_y:
        return 1.0 if flat_x == flat_y else 0.0
    x64: npt.NDArray[np.float64] = x.astype(np.float64)
    y64: npt.NDArray[np.float64] = y.astype(np.float64)
    centered_x = x64 - float(cast("SupportsFloat", x64.mean()))
    centered_y = y64 - float(cast("SupportsFloat", y64.mean()))
    sum_xx = float(cast("SupportsFloat", np.sum(centered_x**2)))
    sum_yy = float(cast("SupportsFloat", np.sum(centered_y**2)))
    denom = math.sqrt(sum_xx * sum_yy)
    if denom <= _EPS:
        return 1.0
    return float(cast("SupportsFloat", np.sum(centered_x * centered_y))) / denom


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Ratio that stays defined when both sides are (near) zero.

    Returns:
        ``numerator / denominator``; 1.0 when both are ~0, inf when only the
        denominator is.

    """
    if abs(denominator) <= _EPS:
        return 1.0 if abs(numerator) <= _EPS else float("inf")
    return numerator / denominator
