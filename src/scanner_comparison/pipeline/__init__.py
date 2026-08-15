"""Orchestration over the pair-preparation chain.

``prepare`` holds the shared front half of every pair workflow (align,
mask, rank-transform, apply corrections); ``compare`` measures, judges,
and renders one pair or a whole run; ``calibrate`` holds the calibration
solves that must measure pairs through the preparation chain (the blur and
column-gain constants). The mask-construction order in ``prepare_pair``
is load-bearing — defect columns are excluded LAST (see the module).
"""
from __future__ import annotations

from scanner_comparison.pipeline.calibrate import (
    solve_blur_constant,
    solve_column_gain,
)
from scanner_comparison.pipeline.compare import (
    NoPairsError,
    compare_pair,
    evaluate_thresholds,
    run,
)
from scanner_comparison.pipeline.prepare import (
    FeatureImages,
    PreparedPair,
    feature_images,
    prepare_pair,
)

__all__ = [
    "FeatureImages",
    "NoPairsError",
    "PreparedPair",
    "compare_pair",
    "evaluate_thresholds",
    "feature_images",
    "prepare_pair",
    "run",
    "solve_blur_constant",
    "solve_column_gain",
]
