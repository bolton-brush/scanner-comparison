"""Image math on typed arrays: normalization, registration, metrics, motion.

Everything here operates on plain arrays with masks passed explicitly; the
comparison pipeline (``scanner_comparison.pipeline``) is the consumer.
"""
from __future__ import annotations

from scanner_comparison.imaging.align import (
    AlignmentError,
    Registration,
    RoiOptions,
    film_roi_mask,
    geometric_overlap_mask,
    norm_for_registration,
    preprocess_log_dog,
    register,
    single_alignment_mask,
    warp_image,
)
from scanner_comparison.imaging.frc import (
    FRC_THRESHOLD_HALF,
    FRC_THRESHOLD_SEVENTH,
    FrcCurve,
    frc_curve,
    frc_resolution,
)
from scanner_comparison.imaging.metrics import (
    PREFILTER_SIGMA,
    PairMetrics,
    blur_sigma_on_support,
    compute_metrics,
    differential_blur_sigma,
    edge_support,
    local_diff_evaluation_mask,
    local_mean_diff,
    prefilter,
    signed_blur_gap,
)
from scanner_comparison.imaging.motion import motion_amplified_diff
from scanner_comparison.imaging.normalize import (
    NormalizationError,
    RangeNormalization,
    TileResidualStats,
    fit_gain_offset,
    percentile_rank,
    robust_rescale,
    surround_mask,
    tile_residual_stats,
)

__all__ = [
    "FRC_THRESHOLD_HALF",
    "FRC_THRESHOLD_SEVENTH",
    "PREFILTER_SIGMA",
    "AlignmentError",
    "FrcCurve",
    "NormalizationError",
    "PairMetrics",
    "RangeNormalization",
    "Registration",
    "RoiOptions",
    "TileResidualStats",
    "blur_sigma_on_support",
    "compute_metrics",
    "differential_blur_sigma",
    "edge_support",
    "film_roi_mask",
    "fit_gain_offset",
    "frc_curve",
    "frc_resolution",
    "geometric_overlap_mask",
    "local_diff_evaluation_mask",
    "local_mean_diff",
    "motion_amplified_diff",
    "norm_for_registration",
    "percentile_rank",
    "prefilter",
    "preprocess_log_dog",
    "register",
    "robust_rescale",
    "signed_blur_gap",
    "single_alignment_mask",
    "surround_mask",
    "tile_residual_stats",
    "warp_image",
]
