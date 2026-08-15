"""Outcome records of a comparison run: per-pair results and the summary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from scanner_comparison.imaging.metrics import PairMetrics


@dataclass
class PairResult:
    """Outcome of comparing one filename pair."""

    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    error: str | None = None
    metrics: PairMetrics | None = None
    reg_correlation: float = 0.0
    reg_scale: float = 1.0
    # The 2x3 registration warp (row-major, reference -> moving frame) and
    # the reference frame shape; both None when the pair errored out.
    # Carried so bidirectional runs can compose the two directions' warps
    # for the round-trip consistency check.
    reg_warp: list[float] | None = None
    ref_shape: tuple[int, int] | None = None
    gain: float = 1.0
    offset: float = 0.0
    tile_residual_mean: float = 0.0
    tile_residual_std: float = 0.0
    # Number of stationary column-defect pixels' source columns excluded
    # from the comparison (both sides, native frames; 0 without a mask).
    defect_columns_masked: int = 0
    # Blur correction applied to this pair: ``(r_i, sigma_ref, sigma_mov)``
    # — the per-pair resampling penalty and the Gaussian sigmas applied to
    # the reference/moving rank images. None when blur correction is off.
    blur_applied: tuple[float, float, float] | None = None
    # Column-gain correction applied to this pair: the rms of the
    # subtracted reference-frame profile over the masked pixels. None when
    # the column-gain correction is off.
    colgain_rms: float | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSummary:
    """Aggregate outcome of comparing two directories."""

    results: Sequence[PairResult]
    unmatched: Sequence[str]
    all_passed: bool
    out_dir: Path
