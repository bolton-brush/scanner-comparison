"""Outputs: per-pair visual artifacts and run-level JSON/CSV summaries."""
from __future__ import annotations

from scanner_comparison.report.artifacts import PairImages, write_pair_artifacts
from scanner_comparison.report.cross import (
    CrossDirectionMetrics,
    cross_direction_metrics,
    roundtrip_max_px,
)
from scanner_comparison.report.summary import (
    write_cross_direction_summary,
    write_summary,
)

__all__ = [
    "CrossDirectionMetrics",
    "PairImages",
    "cross_direction_metrics",
    "roundtrip_max_px",
    "write_cross_direction_summary",
    "write_pair_artifacts",
    "write_summary",
]
