"""Euclidean registration of two scans of the same physical film.

Registration is performed in a log-DoG feature space: the log transform
turns the scanners' gamma-like (multiplicative) response differences into
additive offsets, and the difference of Gaussians then cancels that offset
(along with any smooth shading) and restricts the criterion to mid-frequency
structural content. OpenCV's enhanced correlation coefficient (ECC)
maximizer with per-image masks (``findTransformECCWithMask``) aligns those
feature images coarse-to-fine under a Euclidean (rotation + translation)
model; the film borders, beveled corners, and dark surround are masked out
of the criterion on both sides, since their position relative to the content
is scanner-dependent. ORB feature matching provides a fallback
initialization when ECC cannot converge from identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import SupportsFloat, cast

import cv2
import numpy as np

from scanner_comparison.core.imtypes import (
    BoolMask,
    F32Image,
    U8Image,
    U16Image,
    Warp,
    as_f32,
    as_f64,
    as_optional_u8,
    as_u8,
    as_u16,
    erode_mask,
    shape2,
    to_u8,
)
from scanner_comparison.imaging.normalize import surround_mask

_ECC_EPS_COARSE = 1e-6
_ECC_EPS_MID = 1e-7
_ECC_EPS_FINE = 1e-8
_ECC_ITERS_COARSE = 200
_ECC_ITERS_MID = 200
_ECC_ITERS_FINE = 400
_ECC_GAUSS_FILTER = 5
# Convergence gate on the final ECC correlation. Log-DoG feature images
# correlate lower than intensity images, so this sits well below 1.
_MIN_ECC_CORRELATION = 0.3
# Below this coarse-level correlation, ECC from identity is treated as
# probably stuck in a local optimum and the ORB initialization is tried as
# well; well-aligned log-DoG feature pairs reach 0.9+.
_GOOD_ECC_CORRELATION = 0.6
_ORB_FEATURES = 5000
_ORB_RATIO = 0.75
_ORB_MIN_MATCHES = 8
_ORB_RANSAC_THRESHOLD_PX = 3.0
_KNN_NEIGHBORS = 2
_BORDER_MARGIN = 0.03
_CORNER_MARGIN = 0.08
# Small positive shift so log() is defined at the black end of the scan.
_LOG_EPSILON = 1e-5
# DoG band limits: sigma_fine preserves structural detail, sigma_coarse
# removes the exposure offset and smooth shading (scanner vignetting).
_DOG_SIGMA_FINE = 2.0
_DOG_SIGMA_COARSE = 16.0
# The film edge is the strongest gradient in the frame and sits at a
# crop-dependent position, so an extra thin band inside the detected content
# is removed from the alignment masks.
_MASK_EDGE_BAND_FRAC = 0.005
# A scale correction within this of 1.0 is treated as disabled (exact float
# comparison of a user-supplied constant would be unreliable).
_SCALE_DISABLE_EPS = 1e-9
# Interpolation for the single resampling of the moving scan into the
# reference frame. Bilinear loses ~20% of gradient energy at the worst
# sub-pixel phase (measured), which would masquerade as moving-side blur in
# the sharpness metrics; Lanczos4 retains ~99%.
_WARP_INTERPOLATION = cv2.INTER_LANCZOS4
# Threshold for the area-resized defect mask: any positive residue marks the
# level pixel as defective.
_DEFECT_MASK_RESIZE_EPS = 1e-6


class AlignmentError(RuntimeError):
    """Raised when registration fails to converge to a usable transform."""


@dataclass(frozen=True)
class RoiOptions:
    """Border/corner/surround exclusion used throughout registration.

    Registration always masks the scanner-dependent film edges with the
    default margins, regardless of the (configurable) margins later used for
    the comparison metrics: edge apparatus carries no anatomical
    information, and its position relative to the content is
    scanner-dependent, so letting it into the alignment criterion can only
    bias or break the estimate.
    """

    border_margin: float = _BORDER_MARGIN
    corner_margin: float = _CORNER_MARGIN
    exclude_surround: bool = True


@dataclass(frozen=True)
class Registration:
    """Result of registering the moving image onto the reference image.

    ``warp`` is a 2x3 transform mapping reference-image coordinates to
    moving-image coordinates (OpenCV's ``WARP_INVERSE_MAP`` convention). The
    estimated part is Euclidean (rotation + translation); when a
    ``scale_correction`` was applied, the uniform scale is folded into
    ``warp`` as well and recorded in ``scale``. ``correlation`` is the final
    ECC coefficient in feature space.
    """

    warp: Warp
    correlation: float
    scale: float = 1.0


@dataclass(frozen=True)
class _Side:
    """One image prepared for registration: normalized intensity + features.

    ``defect`` is the image-frame stationary column-defect mask (already in
    the prescaled frame for the moving side); those columns are excluded
    from the ECC criterion at every pyramid level.
    """

    norm: F32Image
    feat: F32Image
    defect: BoolMask | None = None


@dataclass(frozen=True)
class _Masks:
    """ECC criterion masks for both images at one pyramid level."""

    reference: U8Image
    moving: U8Image


@dataclass(frozen=True)
class _Level:
    """One pyramid level: target size (None = full), eps, iteration cap."""

    target_dim: int | None
    eps: float
    iterations: int


@dataclass(frozen=True)
class _LevelInput:
    """Both images and masks resized to one pyramid level."""

    ref_norm: F32Image
    ref_feat: F32Image
    mov_norm: F32Image
    mov_feat: F32Image
    masks: _Masks
    scale: float


def film_roi_mask(
    shape: tuple[int, int],
    *,
    margin: float = _BORDER_MARGIN,
    corner: float = _CORNER_MARGIN,
) -> BoolMask:
    """Film-shaped region of interest: the frame minus border and corners.

    The scanned films have beveled corners and scanner-dependent borders,
    neither of which carries anatomical information; masking them keeps both
    registration and the difference metrics focused on film content.

    Returns:
        A boolean mask, True inside the ROI.

    """
    height, width = shape
    roi = np.zeros((height, width), dtype=bool)
    m = round(min(height, width) * margin)
    roi[m : height - m, m : width - m] = True
    c = round(min(height, width) * corner)
    if c > 0:
        # ramp[i] + ramp[j] <= c marks the beveled-off corner triangle;
        # np.mgrid is avoided because its stubs type the result loosely.
        ramp = np.arange(c, dtype=np.int32)
        bevel = (ramp[:, None] + ramp[None, :]) <= c
        roi[m : m + c, m : m + c] &= ~bevel  # top-left
        roi[m : m + c, width - m - c : width - m] &= ~np.fliplr(bevel)
        roi[height - m - c : height - m, m : m + c] &= ~np.flipud(bevel)
        roi[height - m - c : height - m, width - m - c : width - m] &= ~np.flip(bevel)
    return roi


def register(
    reference: U16Image,
    moving: U16Image,
    *,
    max_dim: int = 1200,
    defect_masks: tuple[BoolMask | None, BoolMask | None] | None = None,
    scale_correction: float = 1.0,
) -> Registration:
    """Estimate the transform aligning ``moving`` onto ``reference``.

    Each image is ROI-normalized, converted to a log-DoG band-pass feature
    image, and aligned coarse-to-fine with ECC, with both sides masked to
    informative film content. The ECC criterion stays Euclidean (rotation +
    translation); a known inter-scanner scale difference can instead be
    supplied as ``scale_correction``, a scanner-pair calibration constant
    that pre-resamples the moving image before estimation. The uniform scale
    then folds into the returned warp, so downstream code resamples the
    moving scan exactly once.

    ``defect_masks`` is an optional ``(reference, moving)`` pair of
    stationary column-defect masks in each scan's native frame (the moving
    one before any scale correction; it is resampled together with the
    moving image); those columns are excluded from the ECC criterion masks
    on both sides.

    Returns:
        The estimated warp, its final ECC correlation coefficient, and the
        applied scale correction.

    Raises:
        AlignmentError: if no usable transform can be found at all.

    """
    roi_opts = RoiOptions()
    ref_defect_mask, mov_defect_mask = (
        defect_masks if defect_masks is not None else (None, None)
    )
    scale_map: Warp | None = None
    if abs(scale_correction - 1.0) > _SCALE_DISABLE_EPS:
        scale_map = _scale_map(shape2(moving), scale_correction)
        moving = _prescale(moving, scale_map)
        if mov_defect_mask is not None:
            mov_defect_mask = _prescale_mask(mov_defect_mask, scale_map)
    ref = _side(reference, roi_opts, ref_defect_mask)
    mov = _side(moving, roi_opts, mov_defect_mask)
    levels = (
        _Level(max_dim, _ECC_EPS_COARSE, _ECC_ITERS_COARSE),
        _Level(max_dim * 2, _ECC_EPS_MID, _ECC_ITERS_MID),
        _Level(None, _ECC_EPS_FINE, _ECC_ITERS_FINE),
    )
    result = _coarse(ref, mov, levels[0], roi_opts)
    for level in levels[1:]:
        refined = _try_refine(ref, mov, result, level, roi_opts)
        if refined is None:
            break
        result = refined
    if result.correlation < _MIN_ECC_CORRELATION:
        msg = f"ECC converged poorly: final correlation {result.correlation:.3f}"
        raise AlignmentError(msg)
    if scale_map is not None:
        # ECC found ref(x) ~ moving_scaled(W x) = moving(S . W . x), so the
        # full reference -> original-moving map is the product S @ W.
        return Registration(
            warp=_compose_warps(scale_map, result.warp),
            correlation=result.correlation,
            scale=scale_correction,
        )
    return result


def warp_image(img: U16Image, warp: Warp, out_shape: tuple[int, int]) -> U16Image:
    """Warp ``img`` into the reference frame given as ``(height, width)``.

    Uses ``_WARP_INTERPOLATION`` so the one resampling the moving scan
    receives does not add measurable blur of its own.

    Returns:
        The warped image with ``out_shape``; uncovered areas are zero.

    """
    height, width = out_shape
    return as_u16(
        cv2.warpAffine(
            img,
            warp,
            (width, height),
            flags=_WARP_INTERPOLATION | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    )


def geometric_overlap_mask(
    moving_shape: tuple[int, int],
    warp: Warp,
    ref_shape: tuple[int, int],
    *,
    erode_px: int = 8,
) -> BoolMask:
    """Mask of reference-frame pixels covered by the warped moving image.

    The border is eroded by ``erode_px`` to remove interpolation artifacts
    along the warped edge.

    Returns:
        A boolean mask in the reference frame.

    """
    ones = np.ones(moving_shape, dtype=np.uint8)
    height, width = ref_shape
    warped = as_u8(
        cv2.warpAffine(
            ones,
            warp,
            (width, height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    )
    return erode_mask(warped.astype(bool), erode_px)


def _side(img: U16Image, roi: RoiOptions, defect: BoolMask | None = None) -> _Side:
    """Normalize one image and compute its log-DoG feature image.

    Returns:
        The normalized intensity image, its band-pass feature image, and
        the column-defect mask.

    """
    norm = norm_for_registration(img, roi)
    return _Side(norm=norm, feat=preprocess_log_dog(img, roi), defect=defect)


def _coarse(
    ref: _Side,
    mov: _Side,
    level: _Level,
    roi: RoiOptions,
) -> Registration:
    """Coarse-level ECC, from identity, with an ORB-initialized retry.

    ECC from identity can slide into a local optimum on band-pass features
    (e.g. when scanner line-noise banding dominates), so a mediocre result
    triggers a second run initialized from ORB keypoint matching; the better
    of the two wins.

    Returns:
        The best coarse registration.

    Raises:
        AlignmentError: if both the identity and ORB-initialized runs fail
            to converge at all.

    """
    level_input = _level_input(ref, mov, level, roi)
    identity: F32Image = np.eye(2, 3, dtype=np.float32)
    best: Registration | None = None
    try:
        result = _run_at_level(level_input, identity, level)
        if result.correlation >= _GOOD_ECC_CORRELATION:
            return result
        best = result
    except AlignmentError:
        pass
    try:
        init = _orb_initialization(level_input.ref_feat, level_input.mov_feat)
        orb_result = _run_at_level(level_input, init, level)
    except AlignmentError:
        if best is None:
            raise
        return best
    if best is None or orb_result.correlation > best.correlation:
        return orb_result
    return best


def _try_refine(
    ref: _Side,
    mov: _Side,
    current: Registration,
    level: _Level,
    roi: RoiOptions,
) -> Registration | None:
    """Attempt one finer-level ECC refinement of ``current``.

    Returns:
        The refined registration if it converged and did not regress, else
        None.

    """
    level_input = _level_input(ref, mov, level, roi)
    init = as_f32(_scale_warp_translation(current.warp, level_input.scale))
    try:
        refined = _run_at_level(level_input, init, level)
    except AlignmentError:
        return None
    return refined if refined.correlation >= current.correlation else None


def _level_input(ref: _Side, mov: _Side, level: _Level, roi: RoiOptions) -> _LevelInput:
    """Resize both images to one pyramid level and build the ECC masks.

    Returns:
        Both normalized images, both feature images, masks, and the scale.

    """
    scale = _level_scale(ref.norm, mov.norm, level.target_dim)
    ref_norm = _resize_by(ref.norm, scale)
    mov_norm = _resize_by(mov.norm, scale)
    ref_feat = _resize_by(ref.feat, scale)
    mov_feat = _resize_by(mov.feat, scale)
    masks = _alignment_masks(
        ref_norm,
        mov_norm,
        roi,
        _resize_defect_mask(ref.defect, scale),
        _resize_defect_mask(mov.defect, scale),
    )
    return _LevelInput(ref_norm, ref_feat, mov_norm, mov_feat, masks, scale)


def _run_at_level(
    level_input: _LevelInput,
    init: F32Image,
    level: _Level,
) -> Registration:
    """Run masked ECC at one level; the result is in full-resolution coords.

    ``init`` is given in level coordinates (identity and ORB estimates are
    resolution-independent or level-native).

    Returns:
        The refined warp mapped back to full-resolution coordinates.

    """
    warp, corr = _run_ecc(
        level_input.ref_feat,
        level_input.mov_feat,
        as_f32(init),
        level_input.masks,
        level,
    )
    return Registration(
        warp=_scale_warp_translation(warp, 1.0 / level_input.scale),
        correlation=corr,
    )


def norm_for_registration(img: U16Image, roi: RoiOptions) -> F32Image:
    """Robustly stretch a 16-bit image to float32 [0, 1] for registration.

    Percentile limits are computed only inside the film ROI so that
    scanner-specific border content cannot shift the stretch.

    Returns:
        The contrast-normalized image.

    Raises:
        AlignmentError: if the image has a degenerate intensity range.

    """
    values: F32Image = img.astype(np.float32)
    shape = shape2(img)
    mask = film_roi_mask(shape, margin=roi.border_margin, corner=roi.corner_margin)
    roi_values = values[mask]
    if roi_values.size == 0:
        msg = "Film ROI is empty; cannot normalize for registration"
        raise AlignmentError(msg)
    # Scalar percentiles return typed numpy scalars (a sequence of quantiles
    # would return an array whose indexing is loosely typed).
    lo = float(np.percentile(roi_values, 1.0))
    hi = float(np.percentile(roi_values, 99.0))
    if hi <= lo:
        msg = "Image has a degenerate intensity range"
        raise AlignmentError(msg)
    return as_f32(np.clip((values - lo) / (hi - lo), 0.0, 1.0))


def preprocess_log_dog(img: U16Image, roi: RoiOptions) -> F32Image:
    """Log-DoG band-pass feature image of a raw scan, masked to the film ROI.

    log() turns multiplicative (gamma-like) response differences between the
    scanners into additive offsets, which the difference of Gaussians then
    cancels along with smooth shading, isolating the mid-frequency
    structural content the alignment criterion matches on. Min/max
    normalization is restricted to the film ROI so that scanner-specific
    border apparatus cannot compress the feature range; pixels outside the
    ROI are zeroed.

    Returns:
        The feature image as float32 in [0, 1].

    Raises:
        AlignmentError: if the image has no usable structural content.

    """
    mask = film_roi_mask(
        shape2(img), margin=roi.border_margin, corner=roi.corner_margin
    )
    # The absolute intensity scale is immaterial: log() turns it into an
    # additive offset, which the difference of Gaussians cancels.
    img_float = img.astype(np.float32) / 255.0 + _LOG_EPSILON
    log_img = np.log(img_float)
    blur_fine = cv2.GaussianBlur(log_img, (0, 0), sigmaX=_DOG_SIGMA_FINE)
    blur_coarse = cv2.GaussianBlur(log_img, (0, 0), sigmaX=_DOG_SIGMA_COARSE)
    dog = as_f32(blur_fine - blur_coarse)
    # OpenCV's stubs require a concrete ``dst``; with a mask only masked
    # pixels are written, so the zero-filled buffer keeps the ROI exterior
    # at 0.
    normalized = as_f32(
        cv2.normalize(
            dog,
            np.zeros_like(dog),
            0,
            255,
            cv2.NORM_MINMAX,
            mask=mask.astype(np.uint8),
        )
    )
    if float(normalized.max()) <= 0.0:
        msg = "No structural content found for registration"
        raise AlignmentError(msg)
    return as_f32(normalized / 255.0)


def _alignment_masks(
    ref: F32Image,
    mov: F32Image,
    roi: RoiOptions,
    ref_defect: BoolMask | None = None,
    mov_defect: BoolMask | None = None,
) -> _Masks:
    """Build the ECC criterion masks for both images.

    Each mask is the film ROI minus (optionally) the border-connected dark
    surround and the stationary column defects, eroded by a thin band so the
    crop-dependent film edge cannot dominate the alignment criterion.

    Returns:
        The reference and moving masks as uint8 images.

    """
    return _Masks(
        reference=single_alignment_mask(ref, roi, ref_defect),
        moving=single_alignment_mask(mov, roi, mov_defect),
    )


def single_alignment_mask(
    img: F32Image, roi: RoiOptions, defect: BoolMask | None = None
) -> U8Image:
    """One image's ECC mask: film ROI, minus surround and defect columns.

    Returns:
        The mask as a uint8 image.

    """
    shape = shape2(img)
    mask = film_roi_mask(shape, margin=roi.border_margin, corner=roi.corner_margin)
    if roi.exclude_surround:
        mask &= surround_mask(img, mask)
    if defect is not None:
        mask &= ~defect
    mask = erode_mask(mask, round(min(shape) * _MASK_EDGE_BAND_FRAC))
    mask_u8: U8Image = mask.astype(np.uint8)
    return mask_u8


def _resize_defect_mask(mask: BoolMask | None, scale: float) -> BoolMask | None:
    """Resize a column-defect mask to a pyramid level.

    Area averaging spreads a defect column's influence over the level pixels
    it touches, so any positive residue marks the level pixel as defective;
    at full resolution (scale 1) the mask passes through unchanged.

    Returns:
        The resized boolean mask, or None when no defect mask was given.

    """
    if mask is None:
        return None
    if scale >= 1.0:
        return mask
    height, width = shape2(mask)
    resized = as_f32(
        cv2.resize(
            mask.astype(np.float32),
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    )
    return resized > _DEFECT_MASK_RESIZE_EPS


def _run_ecc(
    template: F32Image,
    moving: F32Image,
    init: F32Image,
    masks: _Masks,
    level: _Level,
) -> tuple[Warp, float]:
    """Single masked ECC run, converting OpenCV failures into ``AlignmentError``.

    Returns:
        The refined 2x3 warp (float64) and the correlation coefficient.

    Raises:
        AlignmentError: if OpenCV fails to converge.

    """
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        level.iterations,
        level.eps,
    )
    # OpenCV requires the warp matrix to be a contiguous float32/float64
    # single-channel matrix regardless of what the caller passed.
    init_f32 = np.ascontiguousarray(init, dtype=np.float32)
    try:
        corr, warped = cv2.findTransformECCWithMask(
            template,
            moving,
            masks.reference,
            masks.moving,
            init_f32,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            _ECC_GAUSS_FILTER,
        )
    except cv2.error as exc:
        msg = f"ECC alignment failed to converge: {exc}"
        raise AlignmentError(msg) from exc
    return as_f64(warped), float(corr)


def _orb_initialization(template: F32Image, moving: F32Image) -> F32Image:
    """Estimate a coarse template-to-moving transform from ORB keypoints.

    Returns:
        A 2x3 Euclidean transform as float32, suitable as an ECC start.

    Raises:
        AlignmentError: if too few usable keypoint matches are found.

    """
    orb = cv2.ORB.create(nfeatures=_ORB_FEATURES)
    kp_t, des_t_raw = orb.detectAndCompute(to_u8(template), None)
    kp_m, des_m_raw = orb.detectAndCompute(to_u8(moving), None)
    des_t = as_optional_u8(des_t_raw)
    des_m = as_optional_u8(des_m_raw)
    if des_t is None or des_m is None or len(kp_t) < _ORB_MIN_MATCHES:
        msg = "ORB initialization failed: not enough features"
        raise AlignmentError(msg)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_t, des_m, k=_KNN_NEIGHBORS)
    good = [
        m
        for pair in knn
        if len(pair) == _KNN_NEIGHBORS
        for m, n in [pair]
        if m.distance < _ORB_RATIO * n.distance
    ]
    if len(good) < _ORB_MIN_MATCHES:
        msg = f"ORB initialization failed: only {len(good)} good matches"
        raise AlignmentError(msg)
    src = np.asarray([kp_t[m.queryIdx].pt for m in good], dtype=np.float32)
    dst = np.asarray([kp_m[m.trainIdx].pt for m in good], dtype=np.float32)
    # RANSAC robustly flags inlier correspondences; the transform itself is
    # then refit on those inliers under the Euclidean model.
    inliers = as_optional_u8(
        cv2.estimateAffinePartial2D(
            src,
            dst,
            None,
            method=cv2.RANSAC,
            ransacReprojThreshold=_ORB_RANSAC_THRESHOLD_PX,
        )[1]
    )
    if inliers is None or int(inliers.sum()) < _ORB_MIN_MATCHES:
        msg = "ORB initialization failed: no consistent transform"
        raise AlignmentError(msg)
    keep = inliers.ravel().astype(bool)
    return _euclidean_from_correspondences(src[keep], dst[keep])


def _euclidean_from_correspondences(src: F32Image, dst: F32Image) -> F32Image:
    """Closed-form rigid fit (orthogonal Procrustes) between 2-D point sets.

    Returns:
        The 2x3 Euclidean warp mapping ``src`` points onto ``dst`` points.

    """
    src64 = src.astype(np.float64)
    dst64 = dst.astype(np.float64)
    src_mean = src64.mean(axis=0)
    dst_mean = dst64.mean(axis=0)
    # Optimal row-vector rotation Q for dst ~= src @ Q is U @ Vt from the
    # SVD of the cross-covariance; column-convention warps use its
    # transpose.
    cov = (src64 - src_mean).T @ (dst64 - dst_mean)
    u, _, vt = np.linalg.svd(cov)
    rot = u @ vt
    # numpy's det stubs resolve the return loosely; narrow before float().
    if float(cast("SupportsFloat", np.linalg.det(rot))) < 0.0:
        # Nearest proper rotation: flip the last singular vector to remove
        # the reflection.
        u[:, -1] *= -1.0
        rot = u @ vt
    warp = np.zeros((2, 3), dtype=np.float64)
    warp[:, :2] = rot.T
    warp[:, 2] = dst_mean - src_mean @ rot
    return as_f32(warp)


def _level_scale(ref: F32Image, mov: F32Image, max_dim: int | None) -> float:
    """Common downscale factor so both images share one coordinate grid.

    The two scans are at the same physical scale, so each pyramid level must
    resize them by the same factor: per-image factors would bake a spurious
    scale difference into the feature images that a Euclidean warp cannot
    absorb.

    Returns:
        The scale to apply to both images (1.0 = keep full resolution).

    """
    if max_dim is None:
        return 1.0
    largest = max(*shape2(ref), *shape2(mov))
    return min(1.0, max_dim / largest)


def _resize_by(img: F32Image, scale: float) -> F32Image:
    """Resize by an explicit scale factor (area-averaged downscale).

    Returns:
        The resized image; the input itself when ``scale`` is >= 1.

    """
    if scale >= 1.0:
        return img
    height, width = shape2(img)
    return as_f32(
        cv2.resize(
            img,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    )


def _scale_warp_translation(warp: Warp, factor: float) -> Warp:
    """Adapt a warp estimated at one resolution to a scaled resolution.

    Returns:
        The warp with its translation components multiplied by ``factor``.

    """
    scaled = warp.copy()
    scaled[0, 2] *= factor
    scaled[1, 2] *= factor
    return scaled


def _scale_map(shape: tuple[int, int], scale: float) -> Warp:
    """Uniform scale by ``scale`` about the image center, as a sampling map.

    Warping an image with this matrix (``WARP_INVERSE_MAP``) yields
    ``out(v) = img(S . v)``: content is magnified by ``1 / scale``, undoing
    an inter-scanner scale difference of ``scale`` in the warp's direction.

    Returns:
        The 2x3 scale matrix.

    """
    height, width = shape
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    return as_f64(
        np.array(
            [
                [scale, 0.0, center_x * (1.0 - scale)],
                [0.0, scale, center_y * (1.0 - scale)],
            ],
            dtype=np.float64,
        )
    )


def _prescale(img: U16Image, scale_map: Warp) -> U16Image:
    """Resample a scan by a uniform scale map, keeping the original size.

    Returns:
        The resampled image; uncovered areas are zero.

    """
    height, width = shape2(img)
    return as_u16(
        cv2.warpAffine(
            img,
            scale_map,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    )


def _prescale_mask(mask: BoolMask, scale_map: Warp) -> BoolMask:
    """Resample a boolean defect mask by a uniform scale map (nearest).

    Returns:
        The resampled boolean mask; uncovered areas are False.

    """
    height, width = shape2(mask)
    warped = as_u8(
        cv2.warpAffine(
            mask.astype(np.uint8),
            scale_map,
            (width, height),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    )
    return warped.astype(bool)


def _compose_warps(first: Warp, second: Warp) -> Warp:
    """Compose two 2x3 warps: apply ``second``, then ``first``.

    Returns:
        The 2x3 matrix of the composed transform.

    """
    first3 = np.vstack([first, [0.0, 0.0, 1.0]])
    second3 = np.vstack([second, [0.0, 0.0, 1.0]])
    return as_f64((first3 @ second3)[:2])
