"""Pair preparation: align two scans and build the comparison inputs.

``prepare_pair`` is the shared front half of every workflow that compares
or measures a pair of scans: it loads nothing itself (the caller loads the
images), registers the moving scan onto the reference, builds the metric
mask, rank-transforms both images, and applies the configured corrections
(blur, column gain). The comparison itself lives in
``pipeline.compare``; the calibration solves that reuse this chain live in
``pipeline.calibrate``.

The mask-construction order in ``prepare_pair`` is LOAD-BEARING — see the
inline comments (defect columns are excluded LAST, after normalization and
surround classification, so a masked run's compared region stays exactly
the unmasked run's region minus defect pixels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from scanner_comparison.calibration.blur import BlurCorrector
    from scanner_comparison.core.io import ImagePair
    from scanner_comparison.imaging.align import Registration
    from scanner_comparison.records.config import CompareConfig

from scanner_comparison.calibration.colgain import subtract_profile
from scanner_comparison.calibration.defects import defect_column_mask
from scanner_comparison.core.imtypes import (
    BoolMask,
    F32Image,
    U16Image,
    Warp,
    as_f32,
    as_u8,
    erode_mask,
    shape2,
)
from scanner_comparison.imaging.align import (
    AlignmentError,
    RoiOptions,
    film_roi_mask,
    geometric_overlap_mask,
    preprocess_log_dog,
    register,
    warp_image,
)
from scanner_comparison.imaging.normalize import (
    fit_gain_offset,
    percentile_rank,
    robust_rescale,
    surround_mask,
)

_UINT16_MAX = 65535
_MIN_OVERLAP_PIXELS = 10_000
_POST_SATURATION_ERODE_PX = 2
_MASK_ERODE_PX = 8


@dataclass(frozen=True)
class PreparedPair:
    """Aligned, rank-transformed images and registration details for a pair."""

    ref_rank: F32Image
    mov_rank: F32Image
    mask: BoolMask
    registration: Registration
    gain: float
    offset: float
    defect_columns_masked: int
    blur_applied: tuple[float, float, float] | None
    colgain_rms: float | None


@dataclass(frozen=True)
class FeatureImages:
    """Log-DoG feature images for the report: both sides + aligned moving."""

    reference: F32Image
    moving: F32Image
    aligned: F32Image


def prepare_pair(
    pair: ImagePair,
    ref: U16Image,
    moving: U16Image,
    config: CompareConfig,
    corrector: BlurCorrector | None = None,
) -> PreparedPair:
    """Align and rank-transform one pair of scans.

    ``ref``/``moving`` are the images behind ``pair`` (the caller loads
    them so the report path can reuse them for the feature images). The
    blur ``corrector`` (built once per run/direction by the caller) applies
    the configured blur correction to the rank images; a configured
    ``column_gain`` calibration is applied per pair (the profile is looked
    up by the pair's scan).

    Propagates ``ImageLoadError`` and ``NormalizationError`` from the
    loading and normalization stages.

    Returns:
        The percentile-rank images in the reference frame, the overlap mask,
        the registration, the residual gain/offset fit, and the correction
        records.

    Raises:
        AlignmentError: if the aligned overlap is too small to compare.

    """
    # Stationary column defects are detected in each scan's native frame,
    # pre-warp; the moving side's mask is warped into the reference frame by
    # the composite warp (nearest-neighbor) below. The masks also feed the
    # ECC criterion masks during registration.
    defects = _defect_masks(pair, config, shape2(ref), shape2(moving))
    # Registration always masks the scanner-dependent edges with the default
    # ROI; the configured margins only steer the metric masks below.
    registration = register(
        ref,
        moving,
        max_dim=config.max_dim,
        scale_correction=config.scale_correction,
        defect_masks=(defects.reference, defects.moving),
    )
    warped = warp_image(moving, registration.warp, shape2(ref))
    mask = geometric_overlap_mask(
        shape2(moving),
        registration.warp,
        shape2(ref),
        erode_px=_MASK_ERODE_PX,
    )
    mask &= _unsaturated(ref) & _unsaturated(warped)
    mask = erode_mask(mask, _POST_SATURATION_ERODE_PX)
    mask &= film_roi_mask(
        shape2(ref), margin=config.border_margin, corner=config.corner_margin
    )
    mask, gain, offset = _finalize_mask(mask, ref, warped, config)
    # Defect columns are excluded LAST, after normalization and surround
    # classification: the defect population is a few extreme-value columns,
    # and excluding it earlier moves the percentile/Otsu statistics enough
    # to flip the surround decision on borderline pairs (R055HM08y06m:
    # -18% coverage from 4 columns). Post-exclusion keeps a masked run's
    # compared region exactly the unmasked run's region minus defect pixels.
    if defects.reference is not None:
        mask &= ~defects.reference
    if defects.moving is not None:
        mask &= ~_warp_mask(defects.moving, registration.warp, shape2(ref))
    if int(mask.sum()) < _MIN_OVERLAP_PIXELS:
        msg = f"Insufficient overlap after alignment: {int(mask.sum())} px"
        raise AlignmentError(msg)
    # The comparison runs on percentile-rank images: identical physical
    # content then lands at identical values regardless of each scanner's
    # (monotonic) intensity response.
    ref_rank = percentile_rank(ref.astype(np.float32), mask)
    mov_rank = percentile_rank(warped.astype(np.float32), mask)
    # Blur correction: blur the sharper side's rank image by the net
    # device+resampling gap (mask-normalized Gaussian). The registration
    # feature path above is never blurred; the per-pair blur_sigma metric
    # below remains a reported residual on the corrected images.
    blur_applied: tuple[float, float, float] | None = None
    if corrector is not None:
        ref_rank, mov_rank, blur_applied = corrector.apply(
            ref_rank, mov_rank, mask, registration.warp
        )
    # Column-gain correction: subtract the reference scanner's stationary
    # column-coherent diff profile so the banding difference cannot
    # masquerade as coherent local error. Runs after the blur correction
    # so the sharpness gap cannot alias into the profiles.
    ref_rank, colgain_rms = _apply_column_gain(ref_rank, pair, config, mask)
    return PreparedPair(
        ref_rank=ref_rank,
        mov_rank=mov_rank,
        mask=mask,
        registration=registration,
        gain=gain,
        offset=offset,
        defect_columns_masked=defects.n_columns,
        blur_applied=blur_applied,
        colgain_rms=colgain_rms,
    )


def feature_images(ref: U16Image, moving: U16Image, warp: Warp) -> FeatureImages:
    """Log-DoG feature images of both sides, the moving one also warped.

    These are report artifacts only — registration computes the same
    features internally and the calibration solves never render them, so
    ``prepare_pair`` does not compute them; ``pipeline.compare`` calls this
    for the images it already loaded.

    Returns:
        The reference and moving feature images in their own frames and the
        moving one warped into the reference frame.

    """
    logdog_reference = preprocess_log_dog(ref, RoiOptions())
    logdog_moving = preprocess_log_dog(moving, RoiOptions())
    return FeatureImages(
        reference=logdog_reference,
        moving=logdog_moving,
        aligned=_warp_features(logdog_moving, warp, shape2(ref)),
    )


@dataclass(frozen=True)
class _DefectMasks:
    """Native-frame stationary column-defect masks for one pair."""

    reference: BoolMask | None
    moving: BoolMask | None
    n_columns: int


def _finalize_mask(
    mask: BoolMask,
    ref: U16Image,
    warped: U16Image,
    config: CompareConfig,
) -> tuple[BoolMask, float, float]:
    """Normalize both images and exclude the dark non-film surround.

    Returns:
        The metric mask and the diagnostic gain/offset fit.

    """
    ref_norm, _ = robust_rescale(ref.astype(np.float32), mask)
    mov_norm, _ = robust_rescale(warped.astype(np.float32), mask)
    gain, offset = fit_gain_offset(mov_norm, ref_norm, mask)
    if config.exclude_background:
        # Exclude the dark surround on BOTH images: one scanner's washed-out
        # histogram can make its surround pass the other's border-connected
        # darkness test, so a reference-only exclusion leaves a directional
        # coverage asymmetry (on R055FM07y06m: forward 41.6% vs reverse 62.5%
        # frame coverage; two-sided: 41.6% vs 43.0%).
        mask = surround_mask(ref_norm, mask)
        mask = surround_mask(mov_norm, mask)
    return mask, gain, offset


def _defect_masks(
    pair: ImagePair,
    config: CompareConfig,
    ref_shape: tuple[int, int],
    mov_shape: tuple[int, int],
) -> _DefectMasks:
    """Build the native-frame defect masks for one pair from the config.

    Returns:
        The reference and moving masks plus the total masked column count;
        both masks None when no defect mask is configured.

    """
    data = config.defect_mask
    if data is None:
        return _DefectMasks(None, None, 0)
    ref_cols = data.for_directory(pair.path_a.parent).defect_columns(pair.name)
    mov_cols = data.for_directory(pair.path_b.parent).defect_columns(pair.name)
    return _DefectMasks(
        reference=defect_column_mask(ref_shape, ref_cols),
        moving=defect_column_mask(mov_shape, mov_cols),
        n_columns=len(ref_cols) + len(mov_cols),
    )


def _apply_column_gain(
    ref_rank: F32Image, pair: ImagePair, config: CompareConfig, mask: BoolMask
) -> tuple[F32Image, float | None]:
    """Subtract the reference scan's stationary column-gain profile.

    The profile (estimated in the scanner's sensor frame) is shifted into
    this scan's frame by its defect-anchor crop offset. Scans unknown to
    or unanchored in the calibration get no correction.

    Returns:
        The corrected rank image and the applied rms (None when the
        correction is off or the scan is not covered).

    """
    if config.column_gain is None:
        return ref_rank, None
    entry = config.column_gain.for_directory(pair.path_a.parent)
    profile = entry.profile_for_scan(pair.name, shape2(ref_rank)[1])
    if profile is None:
        return ref_rank, None
    return subtract_profile(ref_rank, profile, mask)


def _warp_mask(mask: BoolMask, warp: Warp, out_shape: tuple[int, int]) -> BoolMask:
    """Warp a boolean mask into the reference frame (nearest-neighbor).

    Returns:
        The warped boolean mask with ``out_shape``; uncovered areas False.

    """
    height, width = out_shape
    warped = as_u8(
        cv2.warpAffine(
            mask.astype(np.uint8),
            warp,
            (width, height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    )
    return warped.astype(bool)


def _warp_features(feat: F32Image, warp: Warp, out_shape: tuple[int, int]) -> F32Image:
    """Warp a float feature image into the reference frame.

    Returns:
        The warped feature image with ``out_shape``; uncovered areas are 0.

    """
    height, width = out_shape
    return as_f32(
        cv2.warpAffine(
            feat,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    )


def _unsaturated(img: U16Image) -> BoolMask:
    """Mask of pixels that are not pinned at either 16-bit extreme.

    Returns:
        True where the pixel value is strictly inside ``(0, 65535)``.

    """
    return (img > 0) & (img < _UINT16_MAX)
