"""Compare duplicate film scans from two digitizers to detect information loss.

The package is layered, each subpackage depending only on the ones before
it: ``core`` (typed arrays, IO) ← ``imaging`` (normalization, registration,
metrics, motion) ← ``calibration`` (the device-calibration domains and
their JSON artifacts) ← ``records`` (run configuration and outcomes) ←
``report`` (artifacts and summaries) ← ``pipeline`` (the preparation
chain, the comparison run, and the prepare-dependent calibration solves)
← ``cli`` (the ``scanner-compare`` entry point). Each subpackage's
``__init__`` re-exports its public API.
"""
from __future__ import annotations

from scanner_comparison.calibration import (
    BlurCalibration,
    BlurCalibrationError,
    BlurCorrector,
    ColumnGainData,
    ColumnGainError,
    DefectMaskData,
    DefectMaskError,
    ScaleCalibration,
    ScaleCalibrationError,
    find_defects,
    read_blur_calibration,
    read_column_gain,
    read_defect_mask,
    read_scale_calibration,
    solve_scale_correction,
    write_blur_calibration,
    write_column_gain,
    write_defect_mask,
    write_scale_calibration,
)
from scanner_comparison.core import ImageLoadError, ImagePair, find_pairs, load_image
from scanner_comparison.imaging import (
    AlignmentError,
    NormalizationError,
    PairMetrics,
    Registration,
    compute_metrics,
    register,
)
from scanner_comparison.pipeline import (
    NoPairsError,
    PreparedPair,
    compare_pair,
    prepare_pair,
    run,
    solve_blur_constant,
    solve_column_gain,
)
from scanner_comparison.records import CompareConfig, PairResult, RunSummary, Thresholds
from scanner_comparison.report import CrossDirectionMetrics

__all__ = [
    "AlignmentError",
    "BlurCalibration",
    "BlurCalibrationError",
    "BlurCorrector",
    "ColumnGainData",
    "ColumnGainError",
    "CompareConfig",
    "CrossDirectionMetrics",
    "DefectMaskData",
    "DefectMaskError",
    "ImageLoadError",
    "ImagePair",
    "NoPairsError",
    "NormalizationError",
    "PairMetrics",
    "PairResult",
    "PreparedPair",
    "Registration",
    "RunSummary",
    "ScaleCalibration",
    "ScaleCalibrationError",
    "Thresholds",
    "compare_pair",
    "compute_metrics",
    "find_defects",
    "find_pairs",
    "load_image",
    "prepare_pair",
    "read_blur_calibration",
    "read_column_gain",
    "read_defect_mask",
    "read_scale_calibration",
    "register",
    "run",
    "solve_blur_constant",
    "solve_column_gain",
    "solve_scale_correction",
    "write_blur_calibration",
    "write_column_gain",
    "write_defect_mask",
    "write_scale_calibration",
]
