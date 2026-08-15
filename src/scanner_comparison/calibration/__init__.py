"""Device calibrations: the scanner-pair artifacts and their domain logic.

Each module owns one calibration domain, including its versioned JSON
artifact (readers and writers) and inspectable outputs. Solves that must
measure pairs through the full preparation chain (blur, column gain) live
one layer up in ``scanner_comparison.pipeline.calibrate``; what lives here
is everything that does not: the scale sweep, the defect detection, the
blur corrector and its phase tables, and the column-gain profile math.
"""
from __future__ import annotations

from scanner_comparison.calibration.blur import (
    PHASE_GRID_N,
    BlurCalibration,
    BlurCalibrationError,
    BlurCorrector,
    PairBlur,
    correction_sigmas,
    masked_gaussian,
    phase_table,
    read_blur_calibration,
    resampling_penalty,
    write_blur_calibration,
)
from scanner_comparison.calibration.colgain import (
    MIN_COLUMN_ROWS,
    MIN_PAIR_PROFILES,
    ColumnGainData,
    ColumnGainError,
    DirectoryColumnGain,
    PairColumnGain,
    aggregate_profiles,
    column_diff_profile,
    read_column_gain,
    shift_profile,
    subtract_profile,
    write_column_gain,
    write_column_gain_csvs,
    write_column_gain_maps,
)
from scanner_comparison.calibration.defects import (
    DefectMaskData,
    DefectMaskError,
    DefectScanInfo,
    DirectoryDefects,
    ScanCandidates,
    column_defect_candidates,
    defect_column_mask,
    find_defects,
    read_defect_mask,
    write_candidate_csvs,
    write_defect_map_pngs,
    write_defect_mask,
)
from scanner_comparison.calibration.scale import (
    ScaleCalibration,
    ScaleCalibrationError,
    masked_feature_ncc,
    read_scale_calibration,
    solve_scale_correction,
    write_scale_calibration,
)

__all__ = [
    "MIN_COLUMN_ROWS",
    "MIN_PAIR_PROFILES",
    "PHASE_GRID_N",
    "BlurCalibration",
    "BlurCalibrationError",
    "BlurCorrector",
    "ColumnGainData",
    "ColumnGainError",
    "DefectMaskData",
    "DefectMaskError",
    "DefectScanInfo",
    "DirectoryColumnGain",
    "DirectoryDefects",
    "PairBlur",
    "PairColumnGain",
    "ScaleCalibration",
    "ScaleCalibrationError",
    "ScanCandidates",
    "aggregate_profiles",
    "column_defect_candidates",
    "column_diff_profile",
    "correction_sigmas",
    "defect_column_mask",
    "find_defects",
    "masked_feature_ncc",
    "masked_gaussian",
    "phase_table",
    "read_blur_calibration",
    "read_column_gain",
    "read_defect_mask",
    "read_scale_calibration",
    "resampling_penalty",
    "shift_profile",
    "solve_scale_correction",
    "subtract_profile",
    "write_blur_calibration",
    "write_candidate_csvs",
    "write_column_gain",
    "write_column_gain_csvs",
    "write_column_gain_maps",
    "write_defect_map_pngs",
    "write_defect_mask",
    "write_scale_calibration",
]
