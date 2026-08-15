"""Tests for stationary column-defect detection, masking, and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import make_phantom

from scanner_comparison.cli import main
from scanner_comparison.calibration.defects import (
    DefectMaskData,
    DefectMaskError,
    DefectScanInfo,
    DirectoryDefects,
    column_defect_candidates,
    defect_column_mask,
    find_defects,
    read_defect_mask,
    write_defect_mask,
)
from scanner_comparison.pipeline import run
from scanner_comparison.records import CompareConfig

CANVAS_SHAPE = (420, 700)
SCAN_WIDTH = 640
# Sensor-frame x positions of the planted stationary defect columns.
DEFECT_COLUMNS = (100, 300, 500)
_DEFECT_AMPLITUDE = 12_000
_N_SCANS = 8
# The anchor tolerance lets a scan's recovered offset differ from its true
# crop offset by up to this much.
_OFFSET_TOL_PX = 2


def _scan_from_canvas(
    canvas: np.ndarray,
    crop_left: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One scanner 'scan': the canvas cropped by an auto-crop offset, with
    the planted full-height sensor defect columns and fresh grain."""
    scan = canvas[:, crop_left : crop_left + SCAN_WIDTH].astype(np.int32)
    scan += rng.normal(0.0, 150.0, size=scan.shape).astype(np.int32)
    scan = np.clip(scan, 0, 65535)
    for col in DEFECT_COLUMNS:
        scan[:, col - crop_left] += _DEFECT_AMPLITUDE
    # A strong but non-stationary banding column at a random position: it
    # must never be reported as a stationary defect.
    band = int(rng.integers(20, SCAN_WIDTH - 20))
    scan[:, band] -= 9_000
    return np.clip(scan, 0, 65535).astype(np.uint16)


@pytest.fixture()
def scanner_dirs(tmp_path) -> tuple[Path, Path]:
    """Two scanner directories of jitter-cropped scans with planted defects."""
    canvas = make_phantom(seed=7, shape=CANVAS_SHAPE)
    dirs = []
    for s in ("scanner_a", "scanner_b"):
        directory = tmp_path / s
        directory.mkdir()
        rng = np.random.default_rng(len(s))
        for i in range(_N_SCANS):
            crop_left = int(rng.integers(0, CANVAS_SHAPE[1] - SCAN_WIDTH))
            img = _scan_from_canvas(canvas, crop_left, rng)
            cv2.imwrite(str(directory / f"scan{i}.png"), img)
        dirs.append(directory)
    return dirs[0], dirs[1]


def test_candidates_find_full_height_defects_only():
    """Full-height lines are candidates; part-height structures are not."""
    rng = np.random.default_rng(1)
    img = 30_000 + rng.normal(0.0, 400.0, size=(300, 400))
    img[:, 150] += 8_000  # bright full-height defect
    img[:, 250] -= 8_000  # dark full-height defect
    img[50:150, 320] += 20_000  # strong but only 100 px tall (2 of 6 bands)
    columns = column_defect_candidates(
        np.clip(img, 0.0, 65535.0).astype(np.uint16)
    ).columns
    assert 150 in columns
    assert 250 in columns
    assert 320 not in columns


def test_candidates_on_flat_image():
    """A nearly constant image still surfaces its isolated defect columns."""
    img = np.full((200, 300), 10_000, dtype=np.uint16)
    img[:, 100] = 20_000
    columns = column_defect_candidates(img).columns
    assert columns.tolist() == [100]


def test_defect_column_mask_clips_to_bounds():
    """Out-of-range columns are ignored; in-range columns are fully masked."""
    mask = defect_column_mask((10, 20), [3, 19, 20, -1])
    assert mask.shape == (10, 20)
    assert mask[:, 3].all() and mask[:, 19].all()
    assert int(mask.sum()) == 20


def test_find_defects_recovers_planted_columns(scanner_dirs):
    """Planted sensor columns are found despite per-scan crop jitter."""
    dir_a, dir_b = scanner_dirs
    data = find_defects(dir_a, dir_b, progress=lambda _s: None)
    for directory in (dir_a, dir_b):
        entry = data.for_directory(directory)
        assert len(entry.stationary_columns_ref_frame) == len(DEFECT_COLUMNS)
        # All anchored scans map their defect columns to the same
        # reference-frame positions (their anchor deltas absorb each scan's
        # crop offset relative to the reference scan, within tolerance).
        mapped_per_scan: list[list[int]] = []
        for info in entry.scans.values():
            assert info.anchored
            assert len(info.defect_columns_native) == len(DEFECT_COLUMNS)
            mapped_per_scan.append(
                sorted(c + info.x_offset for c in info.defect_columns_native)
            )
        for i in range(len(DEFECT_COLUMNS)):
            positions = [m[i] for m in mapped_per_scan]
            assert max(positions) - min(positions) <= 2 * _OFFSET_TOL_PX
            center = sum(positions) / len(positions)
            assert any(
                a - _OFFSET_TOL_PX <= center <= b + _OFFSET_TOL_PX
                for a, b, _n in entry.stationary_columns_ref_frame
            )


def test_find_defects_json_roundtrip(scanner_dirs, tmp_path):
    """The written JSON reads back into identical data."""
    dir_a, dir_b = scanner_dirs
    data = find_defects(dir_a, dir_b, progress=lambda _s: None)
    path = write_defect_mask(tmp_path / "defects.json", data)
    assert read_defect_mask(path) == data


def test_read_defect_mask_rejects_bad_files(tmp_path):
    """Missing/malformed/foreign-version files raise DefectMaskError."""
    with pytest.raises(DefectMaskError):
        read_defect_mask(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    _ = bad.write_text('{"version": 999}', encoding="utf-8")
    with pytest.raises(DefectMaskError):
        read_defect_mask(bad)


def _hand_made_mask(dir_a: Path, dir_b: Path) -> DefectMaskData:
    """A defect mask with known native columns for the pair_dirs fixture."""
    entries = {}
    for directory, cols, ref_cols in (
        (dir_a, [100, 200], [(100, 100, 1), (200, 200, 1)]),
        (dir_b, [150], [(150, 150, 1)]),
    ):
        key = str(directory.resolve())
        scans = {}
        for png in directory.glob("*.png"):
            scans[png.name] = DefectScanInfo(
                name=png.name,
                x_offset=0,
                anchored=True,
                candidate_count=len(cols),
                defect_columns_native=list(cols),
            )
        entries[key] = DirectoryDefects(
            directory=key,
            reference_scan=sorted(scans)[0] if scans else "",
            stationary_columns_ref_frame=ref_cols,
            scans=scans,
        )
    return DefectMaskData(params={}, directories=entries)


def test_run_with_defect_mask(pair_dirs, tmp_path):
    """The pipeline masks both sides' defect columns and reports the count."""
    dir_a, dir_b = pair_dirs
    config = CompareConfig(max_dim=512, defect_mask=_hand_made_mask(dir_a, dir_b))
    masked = run(dir_a, dir_b, tmp_path / "masked", config, progress=lambda _s: None)
    (result,) = [r for r in masked.results if r.name == "case1.png"]
    assert result.defect_columns_masked == 3

    plain = run(
        dir_a,
        dir_b,
        tmp_path / "plain",
        CompareConfig(max_dim=512),
        progress=lambda _s: None,
    )
    (plain_result,) = [r for r in plain.results if r.name == "case1.png"]
    assert plain_result.defect_columns_masked == 0

    mask = cv2.imread(str(tmp_path / "masked" / "case1_mask.png"), cv2.IMREAD_UNCHANGED)
    plain_mask = cv2.imread(
        str(tmp_path / "plain" / "case1_mask.png"), cv2.IMREAD_UNCHANGED
    )
    assert mask is not None and plain_mask is not None
    # The reference-side defect columns are entirely excluded.
    assert int(mask[:, 100].sum()) == 0
    assert int(mask[:, 200].sum()) == 0
    assert int(plain_mask[:, 100].sum()) > 0
    # Both sides' columns are excluded: besides the two reference columns,
    # the moving defect column lands warped (slightly tilted) around
    # x ~= 133-137 and disappears from the compared region.
    newly_excluded = (plain_mask > 0) & (mask == 0)
    assert int(newly_excluded[:, 128:144].sum()) > 300


def test_run_rejects_mask_for_wrong_directory(pair_dirs, tmp_path):
    """A defect mask that does not cover both directories fails fast."""
    dir_a, dir_b = pair_dirs
    other = tmp_path / "elsewhere"
    other.mkdir()
    mask_data = _hand_made_mask(other, dir_b)
    with pytest.raises(DefectMaskError):
        run(
            dir_a,
            dir_b,
            tmp_path / "out",
            CompareConfig(max_dim=512, defect_mask=mask_data),
            progress=lambda _s: None,
        )


def test_cli_find_defect_mask_and_run_analysis(scanner_dirs, tmp_path):
    """find_defect_mask writes JSON+PNG+CSV; run_analysis consumes it."""
    dir_a, dir_b = scanner_dirs
    mask_path = tmp_path / "defects.json"
    code = main(
        ["find_defect_mask", str(dir_a), str(dir_b), "--out", str(mask_path)]
    )
    assert code == 0
    payload = json.loads(mask_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["directories"]) == 2
    for key in payload["directories"]:
        safe = "".join(c if c.isalnum() else "-" for c in Path(key).name)
        assert (tmp_path / f"defects_{safe}.png").exists()
        assert (tmp_path / f"defects_{safe}.csv").exists()

    out = tmp_path / "analysis"
    code = main(
        [
            "run_analysis",
            str(dir_a),
            str(dir_b),
            "--out",
            str(out),
            "--max-dim",
            "512",
            "--defect-mask",
            str(mask_path),
        ]
    )
    assert code in (0, 1)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    for pair in summary["pairs"]:
        assert pair["defect_columns_masked"] == 2 * len(DEFECT_COLUMNS)


def test_cli_run_analysis_warns_without_defect_mask(pair_dirs, tmp_path, capsys):
    """Omitting --defect-mask prints a warning and runs unmasked."""
    dir_a, dir_b = pair_dirs
    code = main(
        [
            "run_analysis",
            str(dir_a),
            str(dir_b),
            "--out",
            str(tmp_path / "out"),
            "--max-dim",
            "512",
        ]
    )
    assert code in (0, 1)
    assert "no defect mask supplied" in capsys.readouterr().out


def test_cli_legacy_invocation_without_mode(pair_dirs, tmp_path):
    """DIR_A DIR_B without a mode still maps to run_analysis."""
    dir_a, dir_b = pair_dirs
    code = main(
        [str(dir_a), str(dir_b), "--out", str(tmp_path / "o"), "--max-dim", "512"]
    )
    assert code in (0, 1)


def test_cli_run_analysis_rejects_foreign_mask(pair_dirs, tmp_path):
    """run_analysis with a mask for other directories exits 2."""
    dir_a, dir_b = pair_dirs
    other = tmp_path / "elsewhere"
    other.mkdir()
    mask_path = write_defect_mask(
        tmp_path / "defects.json", _hand_made_mask(other, dir_b)
    )
    code = main(
        [
            "run_analysis",
            str(dir_a),
            str(dir_b),
            "--out",
            str(tmp_path / "o"),
            "--defect-mask",
            str(mask_path),
        ]
    )
    assert code == 2
