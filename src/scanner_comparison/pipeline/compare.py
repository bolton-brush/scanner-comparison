"""Pair comparison: measure a prepared pair, judge it, and render artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

from scanner_comparison.calibration.blur import BlurCorrector
from scanner_comparison.core.imtypes import shape2
from scanner_comparison.core.io import (
    ImageLoadError,
    ImagePair,
    find_pairs,
    load_image,
)
from scanner_comparison.imaging.align import AlignmentError
from scanner_comparison.imaging.metrics import (
    PairMetrics,
    compute_metrics,
    local_diff_evaluation_mask,
    local_mean_diff,
)
from scanner_comparison.imaging.motion import motion_amplified_diff
from scanner_comparison.imaging.normalize import (
    NormalizationError,
    tile_residual_stats,
)
from scanner_comparison.pipeline.prepare import feature_images, prepare_pair
from scanner_comparison.records.config import (
    BLUR_DISABLE_EPS,
    CompareConfig,
    Thresholds,
)
from scanner_comparison.records.results import PairResult, RunSummary
from scanner_comparison.report.artifacts import (
    PairImages,
    write_frc_csv,
    write_pair_artifacts,
)
from scanner_comparison.report.summary import write_summary

_TILE_GRID = 8


class NoPairsError(ValueError):
    """Raised when the two directories share no common image file names."""


def compare_pair(
    pair: ImagePair,
    out_dir: Path,
    config: CompareConfig,
    corrector: BlurCorrector | None = None,
) -> PairResult:
    """Run the full comparison pipeline for one pair of scans.

    ``corrector`` applies the configured blur correction to the rank images
    (``run`` builds it once per direction when ``config.blur_correction`` is
    nonzero); a configured ``column_gain`` calibration is applied per pair
    inside ``prepare_pair`` (the profile is looked up by the pair's scan).

    Propagates ``AlignmentError``, ``ImageLoadError``, and
    ``NormalizationError`` from the individual pipeline stages.

    Returns:
        The per-pair result including metrics, verdict, and artifact paths.

    """
    ref = load_image(pair.path_a)
    moving = load_image(pair.path_b)
    prepared = prepare_pair(pair, ref, moving, config, corrector)
    metrics, diff, ssim_map, frc = compute_metrics(
        prepared.ref_rank, prepared.mov_rank, prepared.mask
    )
    # The locally averaged diff artifact covers exactly the eroded region the
    # local_mse/local_rmse metrics are evaluated on: nearer the mask edge the
    # blur would mix in the zeros outside the mask and show attenuated values.
    local_diff = local_mean_diff(diff)
    local_diff[~local_diff_evaluation_mask(prepared.mask)] = 0.0
    motion_diff = motion_amplified_diff(
        prepared.ref_rank, prepared.mov_rank, prepared.mask
    )
    tile_stats = tile_residual_stats(
        prepared.ref_rank, prepared.mov_rank, prepared.mask, tiles=_TILE_GRID
    )
    failures = evaluate_thresholds(config.thresholds, metrics)
    features = feature_images(ref, moving, prepared.registration.warp)
    images = PairImages(
        reference=prepared.ref_rank,
        aligned=prepared.mov_rank,
        mask=prepared.mask,
        diff=diff,
        local_diff=local_diff,
        motion_diff=motion_diff,
        ssim_map=ssim_map,
        logdog_reference=features.reference,
        logdog_moving=features.moving,
        logdog_aligned=features.aligned,
    )
    artifacts = write_pair_artifacts(out_dir, pair.name, images)
    artifacts["frc_curve"] = write_frc_csv(out_dir, pair.name, frc)
    return PairResult(
        name=pair.name,
        passed=not failures,
        failures=failures,
        metrics=metrics,
        reg_correlation=prepared.registration.correlation,
        reg_scale=prepared.registration.scale,
        reg_warp=[float(v) for v in prepared.registration.warp.ravel()],
        ref_shape=shape2(prepared.ref_rank),
        gain=prepared.gain,
        offset=prepared.offset,
        tile_residual_mean=tile_stats.mean,
        tile_residual_std=tile_stats.std,
        defect_columns_masked=prepared.defect_columns_masked,
        blur_applied=prepared.blur_applied,
        colgain_rms=prepared.colgain_rms,
        artifacts=artifacts,
    )


def evaluate_thresholds(thresholds: Thresholds, metrics: PairMetrics) -> list[str]:
    """Check a pair's metrics against the pass/fail criteria.

    Returns:
        Human-readable descriptions of every threshold violation (empty when
        the pair passes).

    """
    failures: list[str] = []
    if metrics.rmse > thresholds.max_rmse:
        failures.append(f"rmse {metrics.rmse:.4f} > {thresholds.max_rmse:.4f}")
    if metrics.ssim < thresholds.min_ssim:
        failures.append(f"ssim {metrics.ssim:.4f} < {thresholds.min_ssim:.4f}")
    if metrics.grad_corr < thresholds.min_grad_corr:
        failures.append(
            f"grad_corr {metrics.grad_corr:.4f} < {thresholds.min_grad_corr:.4f}"
        )
    deviation = abs(metrics.grad_energy_ratio - 1.0)
    if deviation > thresholds.grad_energy_tolerance:
        detail = f"off by {deviation:.3f} > {thresholds.grad_energy_tolerance:.3f}"
        failures.append(f"grad_energy_ratio {metrics.grad_energy_ratio:.3f} {detail}")
    return failures


def run(
    dir_a: Path,
    dir_b: Path,
    out_dir: Path,
    config: CompareConfig,
    *,
    progress: Callable[[str], None] = print,
) -> RunSummary:
    """Compare every filename pair shared by the two directories.

    Pairs that raise a pipeline error are recorded as failed results rather
    than aborting the run. Propagates ``DefectMaskError`` when a configured
    defect mask does not cover both directories, and ``ColumnGainError``
    when a configured column-gain calibration does not.

    Returns:
        The aggregate run outcome.

    Raises:
        NoPairsError: if the directories share no PNG file names.

    """
    if config.defect_mask is not None:
        # Fail fast before processing any pair: the mask must cover both
        # directories (the per-pair lookup happens in prepare_pair).
        _ = config.defect_mask.for_directory(dir_a)
        _ = config.defect_mask.for_directory(dir_b)
    pairs, unmatched = find_pairs(dir_a, dir_b)
    if not pairs:
        msg = f"No matching file names between {dir_a} and {dir_b}"
        raise NoPairsError(msg)

    corrector: BlurCorrector | None = None
    if abs(config.blur_correction) > BLUR_DISABLE_EPS:
        # The resampling phase table is a kernel calibration measured on
        # this direction's moving side; built once per run.
        progress(f"blur correction sigma_dev={config.blur_correction:+.4f}")
        progress(f"  phase table probe: {pairs[0].name} (moving side)")
        corrector = BlurCorrector.from_moving_image(
            config.blur_correction, load_image(pairs[0].path_b)
        )

    if config.column_gain is not None:
        # Fail fast: the calibration must cover both directories (the
        # reverse-direction run of --both-directions looks up dir_b's).
        _ = config.column_gain.for_directory(dir_a)
        _ = config.column_gain.for_directory(dir_b)
        progress("column-gain correction: stationary profiles applied")

    results: list[PairResult] = []
    for index, pair in enumerate(pairs, start=1):
        progress(f"[{index}/{len(pairs)}] {pair.name}")
        try:
            results.append(compare_pair(pair, out_dir, config, corrector))
        except (AlignmentError, ImageLoadError, NormalizationError) as exc:
            results.append(PairResult(name=pair.name, passed=False, error=str(exc)))
    _ = write_summary(out_dir, results, unmatched, config)
    return RunSummary(
        results=results,
        unmatched=unmatched,
        all_passed=all(r.passed for r in results),
        out_dir=out_dir,
    )
