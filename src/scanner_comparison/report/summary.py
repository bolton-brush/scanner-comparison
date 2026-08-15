"""Run-level summaries: ``summary.json``/``summary.csv`` and cross-direction."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from scanner_comparison.records.config import CompareConfig
    from scanner_comparison.records.results import PairResult

from scanner_comparison.report.cross import CrossDirectionMetrics

_METRIC_FIELDS = (
    "mae",
    "rmse",
    "psnr",
    "local_mse",
    "local_rmse",
    "ssim",
    "grad_corr",
    "grad_energy_ratio",
    "blur_sigma",
    "frc_resolution_px",
    "frc_resolution_px_17",
    "overlap_fraction",
    "n_pixels",
)


def write_summary(
    out_dir: Path,
    results: Sequence[PairResult],
    unmatched: Sequence[str],
    config: CompareConfig,
) -> tuple[Path, Path]:
    """Write ``summary.json`` and ``summary.csv`` for a completed run.

    Returns:
        The paths of the JSON and CSV summaries.

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"

    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "config": _config_payload(config),
        "all_passed": all(r.passed for r in results),
        "unmatched_files": list(unmatched),
        "pairs": [_result_payload(r) for r in results],
    }
    _written = json_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "name",
            "passed",
            "failures",
            "error",
            "defect_columns_masked",
            "blur_r",
            "blur_sigma_ref_applied",
            "blur_sigma_mov_applied",
            "colgain_rms",
            *_METRIC_FIELDS,
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # csv's stubs type these returns as Any; narrow them with casts.
        _header = cast("int", writer.writeheader())
        for result in results:
            row = _result_payload(result)
            row["failures"] = "; ".join(result.failures)
            _row = cast(
                "int",
                writer.writerow({key: row.get(key, "") for key in fieldnames}),
            )
    return json_path, csv_path


def write_cross_direction_summary(
    out_dir: Path,
    rows: Sequence[CrossDirectionMetrics],
) -> tuple[Path, Path]:
    """Write ``cross_direction.json`` and ``cross_direction.csv``.

    Returns:
        The paths of the JSON and CSV cross-direction summaries.

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "cross_direction.json"
    csv_path = out_dir / "cross_direction.csv"

    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "pairs": [asdict(row) for row in rows],
    }
    _written = json_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(CrossDirectionMetrics.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        _header = cast("int", writer.writeheader())
        for row in rows:
            _row = cast("int", writer.writerow(asdict(row)))
    return json_path, csv_path


def _config_payload(config: CompareConfig) -> dict[str, object]:
    """Serialize the run configuration for ``summary.json``.

    Returns:
        The flat config mapping; a configured defect mask is summarized by
        its directories' stationary-group counts (the mask itself lives in
        its own inspectable JSON file).

    """
    payload: dict[str, object] = {
        "thresholds": asdict(config.thresholds),
        "max_dim": config.max_dim,
        "border_margin": config.border_margin,
        "corner_margin": config.corner_margin,
        "exclude_background": config.exclude_background,
        "scale_correction": config.scale_correction,
        "blur_correction": config.blur_correction,
    }
    mask = config.defect_mask
    payload["defect_mask"] = (
        None
        if mask is None
        else {
            "directories": {
                directory: len(entry.stationary_columns_ref_frame)
                for directory, entry in sorted(mask.directories.items())
            }
        }
    )
    colgain = config.column_gain
    payload["column_gain"] = (
        None
        if colgain is None
        else {
            "directories": {
                directory: {"width": entry.width, "rms": entry.rms}
                for directory, entry in sorted(colgain.directories.items())
            }
        }
    )
    return payload


def _result_payload(result: PairResult) -> dict[str, object]:
    """Flatten one pair result into a JSON/CSV-friendly dictionary.

    Returns:
        A flat mapping of result fields; metric fields are absent when the
        pair errored out before metrics were computed.

    """
    payload: dict[str, object] = {
        "name": result.name,
        "passed": result.passed,
        "failures": list(result.failures),
        "error": result.error,
        "reg_correlation": result.reg_correlation,
        "reg_scale": result.reg_scale,
        "reg_warp": result.reg_warp,
        "ref_shape": list(result.ref_shape) if result.ref_shape is not None else None,
        "gain": result.gain,
        "offset": result.offset,
        "tile_residual_mean": result.tile_residual_mean,
        "tile_residual_std": result.tile_residual_std,
        "defect_columns_masked": result.defect_columns_masked,
        "blur_r": None if result.blur_applied is None else result.blur_applied[0],
        "blur_sigma_ref_applied": (
            None if result.blur_applied is None else result.blur_applied[1]
        ),
        "blur_sigma_mov_applied": (
            None if result.blur_applied is None else result.blur_applied[2]
        ),
        "colgain_rms": result.colgain_rms,
    }
    if result.metrics is not None:
        for metric_field in _METRIC_FIELDS:
            payload[metric_field] = getattr(result.metrics, metric_field)
    return payload
