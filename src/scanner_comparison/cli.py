"""Command-line interface for the scanner comparison tool.

The first positional argument selects a mode subcommand:

- ``run_analysis`` (default when omitted, for backward compatibility):
  compare the two directories pair by pair.
- ``find_defect_mask``: detect stationary vertical column defects in both
  scanner directories and write an inspectable defect-mask JSON plus
  per-scanner defect-map PNGs / per-scan candidate CSVs.
- ``find_scale``: estimate the scanner-pair scale correction constant.
- ``find_blur``: estimate the scanner-pair signed device blur constant.
- ``find_column_gain``: estimate the stationary column-gain profiles.
- ``run_all``: chain all calibrations, then run_analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from scanner_comparison.calibration.blur import write_blur_calibration
from scanner_comparison.calibration.colgain import (
    ColumnGainError,
    read_column_gain,
    write_column_gain,
    write_column_gain_csvs,
    write_column_gain_maps,
)
from scanner_comparison.calibration.defects import (
    DefectMaskError,
    find_defects,
    read_defect_mask,
    write_candidate_csvs,
    write_defect_map_pngs,
    write_defect_mask,
)
from scanner_comparison.calibration.scale import (
    ScaleCalibrationError,
    solve_scale_correction,
    write_scale_calibration,
)
from scanner_comparison.core.io import find_pairs
from scanner_comparison.pipeline.calibrate import (
    solve_blur_constant,
    solve_column_gain,
)
from scanner_comparison.pipeline.compare import NoPairsError, run
from scanner_comparison.records.config import CompareConfig, Thresholds
from scanner_comparison.report.cross import cross_direction_metrics
from scanner_comparison.report.summary import write_cross_direction_summary

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scanner_comparison.calibration.colgain import ColumnGainData
    from scanner_comparison.calibration.defects import DefectMaskData
    from scanner_comparison.records.results import RunSummary
    from scanner_comparison.report.cross import CrossDirectionMetrics

# The scale correction is a *minor* calibration knob by design; anything
# outside this band indicates a unit error, not scanner miscalibration.
_SCALE_CORRECTION_MIN = 0.9
_SCALE_CORRECTION_MAX = 1.1
# A device blur beyond this (px) indicates a unit/usage error, not a real
# scanner gap (the metric's search cap is 3 px).
_BLUR_CORRECTION_MAX = 5.0

_MODES = (
    "run_analysis",
    "find_defect_mask",
    "find_scale",
    "find_blur",
    "find_column_gain",
    "run_all",
)


class _ParsedArgs(Protocol):
    """Structural view of the parsed namespace (argparse attributes are ``Any``)."""

    mode: str
    dir_a: Path
    dir_b: Path
    out: Path
    max_dim: int
    max_rmse: float
    min_ssim: float
    min_grad_corr: float
    grad_energy_tol: float
    border_margin: float
    corner_margin: float
    no_background_exclusion: bool
    scale_correction: float
    both_directions: bool
    defect_mask: Path | None
    blur_correction: float
    column_gain: Path | None
    scale_min: float
    scale_max: float
    scale_step: float
    max_pairs: int


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the selected mode, and return a process exit code.

    Returns:
        0 when every pair passes (or the defect scan completed), 1 when any
        pair fails, 2 on usage errors.

    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    if (
        args_list
        and not args_list[0].startswith("-")
        and args_list[0] not in _MODES
    ):
        # Legacy invocation without a mode: DIR_A DIR_B ...
        args_list.insert(0, "run_analysis")
    parser = _build_parser()
    # argparse.Namespace carries no attribute types; narrowing via ``object``
    # first because Namespace does not structurally overlap with the Protocol.
    args = cast("_ParsedArgs", cast("object", parser.parse_args(args_list)))
    if args.mode == "find_defect_mask":
        return _run_find_defect_mask(args)
    if args.mode == "find_scale":
        return _run_find_scale(args)
    if args.mode == "find_blur":
        return _run_find_blur(args)
    if args.mode == "find_column_gain":
        return _run_find_column_gain(args)
    if args.mode == "run_all":
        return _run_all(args)
    return _run_analysis(args)


def _valid_scale_value(scale_correction: float) -> bool:
    """Validate a scale correction against the calibration band.

    Returns:
        True when the constant is usable (errors are printed).

    """
    if not _SCALE_CORRECTION_MIN < scale_correction < _SCALE_CORRECTION_MAX:
        lo, hi = _SCALE_CORRECTION_MIN, _SCALE_CORRECTION_MAX
        print(f"error: --scale-correction {scale_correction} not in ({lo}, {hi})")
        return False
    return True


def _valid_correction_args(args: _ParsedArgs) -> bool:
    """Validate the scale/blur correction flags of a mode that has both.

    Returns:
        True when both constants are usable (errors are printed).

    """
    if not _valid_scale_value(args.scale_correction):
        return False
    if abs(args.blur_correction) > _BLUR_CORRECTION_MAX:
        print(f"error: --blur-correction {args.blur_correction} implausible")
        print(f"       (|B| <= {_BLUR_CORRECTION_MAX} px)")
        return False
    return True


def _compare_config(
    args: _ParsedArgs,
    *,
    scale_correction: float,
    blur_correction: float,
    defect_mask: DefectMaskData | None = None,
    column_gain: ColumnGainData | None = None,
) -> CompareConfig:
    """Assemble the run configuration from the parsed arguments.

    The correction constants are explicit: only the ``run_analysis`` parser
    defines the corresponding flags (``run_all`` feeds it the solved
    calibrations instead).

    Returns:
        The compare config.

    """
    return CompareConfig(
        thresholds=Thresholds(
            max_rmse=args.max_rmse,
            min_ssim=args.min_ssim,
            min_grad_corr=args.min_grad_corr,
            grad_energy_tolerance=args.grad_energy_tol,
        ),
        max_dim=args.max_dim,
        border_margin=args.border_margin,
        corner_margin=args.corner_margin,
        exclude_background=not args.no_background_exclusion,
        scale_correction=scale_correction,
        blur_correction=blur_correction,
        defect_mask=defect_mask,
        column_gain=column_gain,
    )


def _run_find_defect_mask(args: _ParsedArgs) -> int:
    """Detect stationary column defects and write the mask file.

    Writes the inspectable defect-mask JSON plus per-scanner defect-map
    PNGs and per-scan candidate CSVs next to it.

    Returns:
        0 on success, 2 on directory/IO errors.

    """
    try:
        data = find_defects(args.dir_a, args.dir_b)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}")
        return 2
    path = write_defect_mask(args.out, data)
    maps = write_defect_map_pngs(args.out, data)
    csvs = write_candidate_csvs(args.out, data)
    print()
    for key, entry in data.directories.items():
        anchored = sum(1 for s in entry.scans.values() if s.anchored)
        n_groups = len(entry.stationary_columns_ref_frame)
        masked = sorted(
            len(s.defect_columns_native) for s in entry.scans.values()
        )
        print(f"{key}:")
        print(f"  reference scan: {entry.reference_scan}")
        print(f"  {anchored}/{len(entry.scans)} anchored, {n_groups} stationary groups")
        if masked:
            med = masked[len(masked) // 2]
            print(f"  masked columns/scan: median {med}, max {masked[-1]}")
        print(f"  defect map: {maps[key]}")
        print(f"  candidate CSV: {csvs[key]}")
    print(f"\nDefect mask written to: {path}")
    return 0


def _scale_candidates(lo: float, hi: float, step: float) -> list[float]:
    """Inclusive candidate grid for a scale sweep.

    Returns:
        The candidate constants ``lo, lo+step, ...`` up to ``hi``.

    """
    n = round((hi - lo) / step) + 1
    return [lo + i * step for i in range(n)]


def _valid_scale_sweep(lo: float, hi: float, step: float) -> bool:
    """Validate sweep bounds against the calibration band; print errors.

    Returns:
        True when the sweep is usable.

    """
    if not _SCALE_CORRECTION_MIN < lo < hi < _SCALE_CORRECTION_MAX:
        print("error: --scale-min/--scale-max invalid:")
        print(f"  need {_SCALE_CORRECTION_MIN} < min < max < {_SCALE_CORRECTION_MAX}")
        print(f"  got min={lo}, max={hi}")
        return False
    if step <= 0.0:
        print(f"error: --scale-step must be positive (got {step})")
        return False
    return True


def _run_find_scale(args: _ParsedArgs) -> int:
    """Solve the scanner-pair scale constant by a registration sweep.

    Returns:
        0 on success, 2 on usage/data errors.

    """
    if not _valid_scale_sweep(args.scale_min, args.scale_max, args.scale_step):
        return 2
    candidates = _scale_candidates(args.scale_min, args.scale_max, args.scale_step)
    try:
        pairs, _unmatched = find_pairs(args.dir_a, args.dir_b)
        cal = solve_scale_correction(
            pairs,
            candidates=candidates,
            max_pairs=args.max_pairs,
            max_dim=args.max_dim,
        )
    except (ScaleCalibrationError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}")
        return 2
    path = write_scale_calibration(args.out, cal)
    print("\nScale calibration (mean masked feature NCC):")
    for s, ncc in zip(cal.candidates, cal.mean_ncc, strict=True):
        marker = " <-- best" if s == cal.scale else ""
        print(f"  {s:.5f}  {ncc:.4f}{marker}")
    print(f"\nRecommended: --scale-correction {cal.scale:.5f}")
    print(f"  (parabolic refinement: {cal.scale_refined:.5f})")
    print("  the curve is flat near the optimum; plateau values are fine")
    print(f"Calibration written to: {path}")
    return 0


def _load_defect_mask(args: _ParsedArgs) -> DefectMaskData | None:
    """Load the defect mask when ``--defect-mask`` is given.

    Propagates ``DefectMaskError`` when the file is malformed.

    Returns:
        The parsed mask, or None when no mask was supplied.

    """
    if args.defect_mask is None:
        return None
    return read_defect_mask(args.defect_mask)


def _run_find_blur(args: _ParsedArgs) -> int:
    """Solve the scanner-pair signed device blur constant.

    Returns:
        0 on success, 2 on usage/data errors.

    """
    if not _valid_scale_value(args.scale_correction):
        return 2
    try:
        defect_mask = _load_defect_mask(args)
    except DefectMaskError as exc:
        print(f"error: {exc}")
        return 2
    try:
        pairs, _unmatched = find_pairs(args.dir_a, args.dir_b)
        cal = solve_blur_constant(
            pairs,
            CompareConfig(
                scale_correction=args.scale_correction,
                max_dim=args.max_dim,
                defect_mask=defect_mask,
            ),
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}")
        return 2
    path = write_blur_calibration(args.out, cal)
    print(f"\nsigma_dev = {cal.sigma_dev:+.4f} px, r_bar = {cal.r_bar:.4f} px")
    print(f"  (positive sigma_dev = {args.dir_b.name} is blurrier)")
    print(f"  one-sided arms: forward {cal.sigma_dev_forward:+.4f}")
    print(f"                  reverse {cal.sigma_dev_reverse:+.4f}")
    print("  (the one-sided arms should bracket sigma_dev)")
    print(f"  data-solved r = {cal.r_data:.4f} px")
    print("  (under-reads the table by design: quadrature compression — use r_bar)")
    if cal.sigma_dev < 0.0:
        print("note: negative constant — the second directory is sharper;")
        print("  the signed value is valid as-is for --blur-correction")
    print(f"Recommended: --blur-correction {cal.sigma_dev:.4f}")
    print(f"Calibration written to: {path}")
    return 0


def _run_find_column_gain(args: _ParsedArgs) -> int:
    """Solve the stationary per-column gain profiles of the scanner pair.

    Returns:
        0 on success, 2 on usage/data errors.

    """
    if not _valid_correction_args(args):
        return 2
    if args.defect_mask is None:
        print("error: find_column_gain needs --defect-mask (the defect")
        print("       detection's crop anchoring defines the sensor frame)")
        return 2
    try:
        defect_mask = read_defect_mask(args.defect_mask)
    except DefectMaskError as exc:
        print(f"error: {exc}")
        return 2
    try:
        pairs, _unmatched = find_pairs(args.dir_a, args.dir_b)
        data = solve_column_gain(
            pairs,
            CompareConfig(
                scale_correction=args.scale_correction,
                blur_correction=args.blur_correction,
                max_dim=args.max_dim,
                defect_mask=defect_mask,
            ),
        )
    except (FileNotFoundError, NotADirectoryError, ColumnGainError) as exc:
        print(f"error: {exc}")
        return 2
    path = write_column_gain(args.out, data)
    maps = write_column_gain_maps(path, data)
    csvs = write_column_gain_csvs(path, data)
    print("\nStationary column-gain profiles (rank-domain offsets):")
    for key, entry in sorted(data.directories.items()):
        print(f"  {Path(key).name}: rms {entry.rms:.4f} ({entry.n_pairs} pairs)")
        print(f"    profile map: {maps[key]}")
        print(f"    profile CSV: {csvs[key]}")
    print("Apply with: run_analysis ... --column-gain <this file>")
    print(f"Calibration written to: {path}")
    return 0


def _run_all(args: _ParsedArgs) -> int:
    """Run find_scale + find_defect_mask + find_blur + find_column_gain, then analyze.

    The calibration JSONs (scale.json, defects.json + maps/CSVs, blur.json,
    colgain.json) are written into the output directory; the analysis then
    uses all of them (the defect mask and column-gain profiles are applied
    by default here). The defect mask is built before the blur solve so the
    solve runs on the masked pixel population.

    Returns:
        0 when every pair passes, 1 when any pair fails, 2 on usage errors.

    """
    if not _valid_scale_sweep(args.scale_min, args.scale_max, args.scale_step):
        return 2
    try:
        pairs, _unmatched = find_pairs(args.dir_a, args.dir_b)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}")
        return 2
    if not pairs:
        print(f"error: No matching file names between {args.dir_a} and {args.dir_b}")
        return 2

    print("== find_scale ==")
    candidates = _scale_candidates(args.scale_min, args.scale_max, args.scale_step)
    scale_cal = solve_scale_correction(
        pairs, candidates=candidates, max_pairs=args.max_pairs, max_dim=args.max_dim
    )
    _ = write_scale_calibration(args.out / "scale.json", scale_cal)
    print(f"scale constant: {scale_cal.scale:.5f}")

    print("== find_defect_mask ==")
    defect_data = find_defects(args.dir_a, args.dir_b)
    mask_path = write_defect_mask(args.out / "defects.json", defect_data)
    _ = write_defect_map_pngs(mask_path, defect_data)
    _ = write_candidate_csvs(mask_path, defect_data)

    print("== find_blur ==")
    blur_cal = solve_blur_constant(
        pairs,
        CompareConfig(
            scale_correction=scale_cal.scale,
            max_dim=args.max_dim,
            defect_mask=defect_data,
        ),
    )
    _ = write_blur_calibration(args.out / "blur.json", blur_cal)
    print(f"blur constant: {blur_cal.sigma_dev:+.4f} px")

    print("== find_column_gain ==")
    colgain_data: ColumnGainData | None = None
    try:
        colgain_data = solve_column_gain(
            pairs,
            CompareConfig(
                scale_correction=scale_cal.scale,
                blur_correction=blur_cal.sigma_dev,
                max_dim=args.max_dim,
                defect_mask=defect_data,
            ),
        )
    except ColumnGainError as exc:
        # The banding calibration needs enough defect-anchored scans; a
        # clean dataset may anchor none. Skip it rather than fail the run.
        print(f"warning: column-gain calibration skipped: {exc}")
    if colgain_data is not None:
        colgain_path = write_column_gain(args.out / "colgain.json", colgain_data)
        _ = write_column_gain_maps(colgain_path, colgain_data)
        _ = write_column_gain_csvs(colgain_path, colgain_data)

    print("== run_analysis ==")
    config = _compare_config(
        args,
        scale_correction=scale_cal.scale,
        blur_correction=blur_cal.sigma_dev,
        defect_mask=defect_data,
        column_gain=colgain_data,
    )
    if args.both_directions:
        return _run_both_directions(args, config)
    summary = run(args.dir_a, args.dir_b, args.out, config)
    _print_table(summary)
    print(f"\nArtifacts and summary written to: {summary.out_dir}")
    print(f"Result: {'PASS' if summary.all_passed else 'FAIL'}")
    return 0 if summary.all_passed else 1


def _load_corrections(
    args: _ParsedArgs,
) -> tuple[DefectMaskData | None, ColumnGainData | None] | None:
    """Load the optional defect-mask and column-gain calibration files.

    Returns:
        The parsed ``(defect_mask, column_gain)`` (either may be None), or
        None after printing the error when a file is malformed.

    """
    defect_mask: DefectMaskData | None = None
    if args.defect_mask is None:
        print("warning: no defect mask supplied; column defects not masked")
        print("         (run 'scanner-compare find_defect_mask' to build one)")
    else:
        try:
            defect_mask = read_defect_mask(args.defect_mask)
        except DefectMaskError as exc:
            print(f"error: {exc}")
            return None
    column_gain: ColumnGainData | None = None
    if args.column_gain is not None:
        try:
            column_gain = read_column_gain(args.column_gain)
        except ColumnGainError as exc:
            print(f"error: {exc}")
            return None
    return defect_mask, column_gain


def _run_analysis(args: _ParsedArgs) -> int:
    """Run the pair comparison (one or both directions).

    Returns:
        0 when every pair passes, 1 when any pair fails, 2 on usage errors.

    """
    if not _valid_correction_args(args):
        return 2
    loaded = _load_corrections(args)
    if loaded is None:
        return 2
    defect_mask, column_gain = loaded
    config = _compare_config(
        args,
        scale_correction=args.scale_correction,
        blur_correction=args.blur_correction,
        defect_mask=defect_mask,
        column_gain=column_gain,
    )
    try:
        if args.both_directions:
            return _run_both_directions(args, config)
        summary = run(args.dir_a, args.dir_b, args.out, config)
    except (
        NoPairsError,
        DefectMaskError,
        ColumnGainError,
        FileNotFoundError,
        NotADirectoryError,
    ) as exc:
        print(f"error: {exc}")
        return 2

    _print_table(summary)
    if summary.unmatched:
        print(f"\nUnmatched files (in only one directory): {len(summary.unmatched)}")
        for name in summary.unmatched:
            print(f"  {name}")
    print(f"\nArtifacts and summary written to: {summary.out_dir}")
    print(f"Result: {'PASS' if summary.all_passed else 'FAIL'}")
    return 0 if summary.all_passed else 1


def _run_both_directions(args: _ParsedArgs, config: CompareConfig) -> int:
    """Run the comparison in both alignment directions and report each.

    Forward: DIR_B aligned onto DIR_A (into ``OUT/forward``). Reverse: DIR_A
    aligned onto DIR_B (into ``OUT/reverse``) under the inverted
    configuration (``CompareConfig.reversed()``: scale inverted, blur
    sign-flipped). Direction-asymmetric metrics become a consistency check
    instead of a blind spot: ``blur_sigma`` is signed (positive = moving
    side blurrier), so consistent directions read opposite signs of the
    same magnitude.

    Returns:
        0 when both directions pass, 1 otherwise.

    """
    forward = run(args.dir_a, args.dir_b, args.out / "forward", config)
    reverse = run(args.dir_b, args.dir_a, args.out / "reverse", config.reversed())

    print(f"forward — {args.dir_b.name} aligned onto {args.dir_a.name}:")
    _print_table(forward)
    print(f"\nreverse — {args.dir_a.name} aligned onto {args.dir_b.name}:")
    _print_table(reverse)

    rows = cross_direction_metrics(forward, reverse)
    if rows:
        json_path, csv_path = write_cross_direction_summary(args.out, rows)
        _print_cross_table(rows)
        print(f"\nCross-direction summary: {json_path} and {csv_path}")
    print(f"\nArtifacts written to: {forward.out_dir} and {reverse.out_dir}")
    all_passed = forward.all_passed and reverse.all_passed
    print(f"Result: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


def _print_cross_table(rows: Sequence[CrossDirectionMetrics]) -> None:
    """Print the per-pair cross-direction deltas and round-trip error."""
    print("\ncross-direction (forward - reverse):")
    columns = f"{'d_rmse':>7} {'d_ssim':>7} {'d_gcorr':>7} {'d_gratio':>8}"
    columns = f"{columns} {'d_blur':>6} {'rt_px':>6}"
    header = f"{'name':<22} {columns}"
    print(header)
    print("-" * len(header))
    for row in rows:
        head = f"{row.name:<22} {row.delta_rmse:>7.4f} {row.delta_ssim:>7.4f}"
        tail = (
            f"{row.delta_grad_corr:>7.4f} {row.delta_grad_energy_ratio:>8.4f}"
            f" {row.delta_blur_sigma:>6.3f} {row.roundtrip_max_px:>6.2f}"
        )
        print(f"{head} {tail}")


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (separated for testability).

    Returns:
        The configured parser.

    """
    parser = argparse.ArgumentParser(
        prog="scanner-compare",
        description=(
            "Compare duplicate film scans digitized on two scanners. The "
            "first positional argument selects the mode; omitting it runs "
            "run_analysis (legacy behavior)."
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    analyze = subparsers.add_parser(
        "run_analysis",
        help="compare the two directories pair by pair",
        description=(
            "Compare duplicate film scans digitized on two scanners. Pairs "
            "PNGs by file name, aligns them, rank-transforms the overlap, "
            "and reports information-loss metrics plus difference heatmaps."
        ),
    )
    _add_io_args(analyze, Path("scanner-comparison-results"))
    _add_threshold_args(analyze)
    _ = analyze.add_argument(
        "--scale-correction",
        type=float,
        default=1.0,
        help=(
            "uniform scale of the second scanner relative to the reference, "
            "a scanner-pair calibration constant measured once (e.g. 0.9985, "
            "see find_scale); 1.0 disables the correction "
            "(default: %(default)s)"
        ),
    )
    _ = analyze.add_argument(
        "--both-directions",
        action="store_true",
        help=(
            "also run the reverse comparison (DIR_A aligned onto DIR_B) into "
            "OUT/reverse with the scale correction inverted and the blur "
            "correction sign-flipped; forward results go to OUT/forward and "
            "both metric sets are reported separately"
        ),
    )
    _ = analyze.add_argument(
        "--defect-mask",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "stationary column-defect mask JSON from find_defect_mask; "
            "defect columns are excluded from the metric mask and the "
            "alignment criterion (default: none, with a warning)"
        ),
    )
    _ = analyze.add_argument(
        "--column-gain",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "stationary column-gain (banding) calibration JSON from "
            "find_column_gain; each scanner's profile is subtracted from "
            "its rank image (default: none)"
        ),
    )
    _ = analyze.add_argument(
        "--blur-correction",
        type=float,
        default=0.0,
        metavar="B",
        help=(
            "signed device blur constant of the scanner pair from "
            "find_blur (positive = DIR_B blurrier; signed common-support "
            "convention); the sharper side's rank image is blurred by the "
            "net device+resampling gap (the resampling part from the "
            "kernel phase table) before metrics, so blur differences do "
            "not masquerade as other error; 0 disables "
            "(default: %(default)s)"
        ),
    )

    defects = subparsers.add_parser(
        "find_defect_mask",
        help="detect stationary column defects and write a mask file",
        description=(
            "Detect stationary vertical column (line) defects recurring "
            "across the scans of each directory and write an inspectable "
            "defect-mask JSON plus per-scanner defect-map PNGs and per-scan "
            "candidate CSVs. Feed the JSON to run_analysis --defect-mask."
        ),
    )
    _ = defects.add_argument("dir_a", type=Path, help="first scanner image directory")
    _ = defects.add_argument("dir_b", type=Path, help="second scanner image directory")
    _ = defects.add_argument(
        "--out",
        type=Path,
        default=Path("defects.json"),
        help=(
            "defect-mask JSON path; defect-map PNGs and candidate CSVs are "
            "written next to it (default: %(default)s)"
        ),
    )

    scale = subparsers.add_parser(
        "find_scale",
        help="estimate the scanner-pair scale correction constant",
        description=(
            "Estimate the uniform scale difference between the two scanners "
            "by a sweep: each candidate correction re-runs the Euclidean "
            "registration on a representative subset of pairs and is scored "
            "by masked log-DoG feature NCC at the resulting warp. Writes a "
            "sweep JSON and prints the recommended --scale-correction."
        ),
    )
    _ = scale.add_argument("dir_a", type=Path, help="reference scanner image directory")
    _ = scale.add_argument("dir_b", type=Path, help="second scanner image directory")
    _ = scale.add_argument(
        "--out",
        type=Path,
        default=Path("scale.json"),
        help="scale calibration JSON path (default: %(default)s)",
    )
    _ = scale.add_argument(
        "--max-dim",
        type=int,
        default=1200,
        help="coarse-registration downscale target, px (default: %(default)s)",
    )
    _add_sweep_args(scale)

    blur = subparsers.add_parser(
        "find_blur",
        help="estimate the scanner-pair device blur constant",
        description=(
            "Solve the signed device blur constant (positive = DIR_B "
            "blurrier) with the bidirectional signed common-support "
            "instrument: each pair is measured in both alignment "
            "directions (scale correction applied, inverted in reverse), "
            "the per-pair device variance is solved from the two signed "
            "gaps, and the resampling penalty is taken from the kernel "
            "phase table integrated over each warp's phase field. Median "
            "over pairs. Writes a blur JSON and prints the recommended "
            "--blur-correction."
        ),
    )
    _ = blur.add_argument("dir_a", type=Path, help="reference scanner image directory")
    _ = blur.add_argument("dir_b", type=Path, help="second scanner image directory")
    _ = blur.add_argument(
        "--out",
        type=Path,
        default=Path("blur.json"),
        help="blur calibration JSON path (default: %(default)s)",
    )
    _ = blur.add_argument(
        "--scale-correction",
        type=float,
        default=1.0,
        help=(
            "the scanner-pair scale constant from find_scale; blur must be "
            "solved on scale-corrected images (default: %(default)s)"
        ),
    )
    _ = blur.add_argument(
        "--defect-mask",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "stationary column-defect mask JSON from find_defect_mask; "
            "defect columns are excluded from the solve's metric masks and "
            "the alignment criterion (default: none)"
        ),
    )
    _ = blur.add_argument(
        "--max-dim",
        type=int,
        default=1200,
        help="coarse-registration downscale target, px (default: %(default)s)",
    )

    colgain = subparsers.add_parser(
        "find_column_gain",
        help="estimate the scanners' stationary column-gain (banding) profiles",
        description=(
            "Solve each scanner's stationary per-column gain profile (the "
            "full-height 'banding' of a line sensor): per pair and "
            "direction, the per-column median of the signed rank difference "
            "on fully corrected images is aggregated across films, so the "
            "stationary device pattern survives while film content and the "
            "other scanner's re-warped pattern average out. Writes a "
            "column-gain JSON; apply it with run_analysis --column-gain."
        ),
    )
    _ = colgain.add_argument(
        "dir_a", type=Path, help="reference scanner image directory"
    )
    _ = colgain.add_argument(
        "dir_b", type=Path, help="second scanner image directory"
    )
    _ = colgain.add_argument(
        "--out",
        type=Path,
        default=Path("colgain.json"),
        help="column-gain calibration JSON path (default: %(default)s)",
    )
    _ = colgain.add_argument(
        "--scale-correction",
        type=float,
        default=1.0,
        help=(
            "the scanner-pair scale constant from find_scale; the profiles "
            "must be solved on scale-corrected images (default: %(default)s)"
        ),
    )
    _ = colgain.add_argument(
        "--blur-correction",
        type=float,
        default=0.0,
        help=(
            "the signed device blur constant from find_blur; the profiles "
            "are solved on blur-corrected images so the sharpness gap does "
            "not alias into them (default: %(default)s = off)"
        ),
    )
    _ = colgain.add_argument(
        "--defect-mask",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "stationary column-defect mask JSON from find_defect_mask; "
            "defect columns are excluded from the profile estimation "
            "(default: none)"
        ),
    )
    _ = colgain.add_argument(
        "--max-dim",
        type=int,
        default=1200,
        help="coarse-registration downscale target, px (default: %(default)s)",
    )

    all_ = subparsers.add_parser(
        "run_all",
        help="run all calibrations, then run_analysis",
        description=(
            "Calibrate the scanner pair (scale, defect mask, then blur "
            "solved on the masked population, then the stationary "
            "column-gain profiles) and run the comparison with all "
            "corrections applied. Calibration JSONs (scale.json, "
            "defects.json plus defect maps/CSVs, blur.json, colgain.json) "
            "are written into the output directory."
        ),
    )
    _add_io_args(all_, Path("scanner-comparison-results"))
    _add_threshold_args(all_)
    _ = all_.add_argument(
        "--both-directions",
        action="store_true",
        help=(
            "also run the reverse comparison (DIR_A aligned onto DIR_B) into "
            "OUT/reverse with the scale correction inverted and the blur "
            "correction sign-flipped"
        ),
    )
    _add_sweep_args(all_)
    return parser


def _add_io_args(parser: argparse.ArgumentParser, default_out: Path) -> None:
    """Add the shared directory/output arguments to a subparser."""
    _ = parser.add_argument(
        "dir_a", type=Path, help="reference scanner image directory"
    )
    _ = parser.add_argument(
        "dir_b", type=Path, help="second scanner image directory"
    )
    _ = parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="output directory (default: %(default)s)",
    )


def _add_threshold_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared metric/registration knob arguments to a subparser."""
    _ = parser.add_argument(
        "--max-dim",
        type=int,
        default=1200,
        help="coarse-registration downscale target, px (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--max-rmse",
        type=float,
        default=0.05,
        help="max normalized RMSE (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--min-ssim",
        type=float,
        default=0.95,
        help="min masked SSIM (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--min-grad-corr",
        type=float,
        default=0.95,
        help="min gradient-magnitude correlation (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--grad-energy-tol",
        type=float,
        default=0.15,
        help="allowed |gradient energy ratio - 1| (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--border-margin",
        type=float,
        default=0.03,
        help="border band fraction excluded from comparisons (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--corner-margin",
        type=float,
        default=0.08,
        help="corner bevel leg fraction excluded (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--no-background-exclusion",
        action="store_true",
        help="keep the dark non-film surround in the metrics",
    )


def _add_sweep_args(parser: argparse.ArgumentParser) -> None:
    """Add the scale-sweep knob arguments to a subparser."""
    _ = parser.add_argument(
        "--scale-min",
        type=float,
        default=0.995,
        help="scale sweep lower bound (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--scale-max",
        type=float,
        default=1.005,
        help="scale sweep upper bound (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--scale-step",
        type=float,
        default=0.0005,
        help="scale sweep step (default: %(default)s)",
    )
    _ = parser.add_argument(
        "--max-pairs",
        type=int,
        default=5,
        help="pairs evaluated per candidate (default: %(default)s)",
    )


def _print_table(summary: RunSummary) -> None:
    """Print a compact per-pair metrics table to stdout."""
    columns = f"{'rmse':>7} {'lrmse':>7} {'ssim':>7} {'gcorr':>7} {'gratio':>7}"
    columns = f"{columns} {'blur':>5} {'frc':>5} {'dcols':>5}"
    header = f"{'name':<22} {columns}  verdict"
    print(header)
    print("-" * len(header))
    for result in summary.results:
        if result.metrics is None:
            blanks = (
                f"{'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7}"
                f" {'-':>5} {'-':>5} {'-':>5}"
            )
            print(f"{result.name:<22} {blanks}  ERROR")
            continue
        metrics = result.metrics
        frc = metrics.frc_resolution_px
        verdict = "PASS" if result.passed else "FAIL"
        row_head = f"{result.name:<22} {metrics.rmse:>7.4f} {metrics.local_rmse:>7.4f}"
        row_tail = (
            f"{metrics.ssim:>7.4f} {metrics.grad_corr:>7.4f}"
            f" {metrics.grad_energy_ratio:>7.3f} {metrics.blur_sigma:>5.2f}"
            f" {frc:>5.1f}"
            f" {result.defect_columns_masked:>5d}"
        )
        print(f"{row_head} {row_tail}  {verdict}")
        for failure in result.failures:
            print(f"    - {failure}")
        if result.error:
            print(f"    - {result.error}")
