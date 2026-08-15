"""Run configuration and outcome records, plus the direction rules.

The forward/reverse rules shared by every bidirectional workflow live here
(``inverse_scale``, ``CompareConfig.reversed``, ``ImagePair.reversed`` in
``core.io``): the reverse direction inverts the scale correction and
negates the signed blur constant.
"""
from __future__ import annotations

from scanner_comparison.records.config import (
    BLUR_DISABLE_EPS,
    SCALE_DISABLE_EPS,
    CompareConfig,
    Thresholds,
    inverse_scale,
)
from scanner_comparison.records.results import PairResult, RunSummary

__all__ = [
    "BLUR_DISABLE_EPS",
    "SCALE_DISABLE_EPS",
    "CompareConfig",
    "PairResult",
    "RunSummary",
    "Thresholds",
    "inverse_scale",
]
