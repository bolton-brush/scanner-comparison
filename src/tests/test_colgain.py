"""Tests for the stationary column-gain (banding) calibration."""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest
from conftest import (
    SIM_FOV_WIDTH_FRAC,
    SIM_GAIN,
    SIM_OFFSET,
    ground_truth_warp,
    make_phantom,
)

from scanner_comparison.cli import main
from scanner_comparison.calibration.colgain import (
    ColumnGainError,
    aggregate_profiles,
    column_diff_profile,
    read_column_gain,
    shift_profile,
    subtract_profile,
    write_column_gain,
)
from scanner_comparison.calibration.defects import (
    DefectMaskData,
    DefectScanInfo,
    DirectoryDefects,
)
from scanner_comparison.core.io import find_pairs
from scanner_comparison.pipeline import run, solve_column_gain
from scanner_comparison.records import CompareConfig

# Planted banding amplitude (normalized intensity): clearly above grain,
# small enough to stay far from saturation.
_BAND_AMP = 0.02
# Per-pair translation jitter: the films land at different positions, as
# real hand-placed films do.
_JITTER = [(-6.0, -3.0), (-4.0, 2.0), (-1.0, -1.0), (2.0, 3.0), (4.0, -2.0), (6.0, 1.0)]


def _band_pattern(width: int) -> np.ndarray:
    """A smooth per-column pattern: sinusoid + slight ramp, zero mean."""
    x = np.arange(width, dtype=np.float64)
    pattern = np.sin(2.0 * np.pi * x / 37.0) + 0.5 * np.sin(2.0 * np.pi * x / 250.0)
    pattern += 0.3 * (x / width - 0.5)
    return _BAND_AMP * (pattern - pattern.mean())


def _plant_banding(img: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Add a per-column intensity offset (native frame) to a uint16 scan."""
    out = img.astype(np.float64) + pattern[None, :] * 65535.0
    return np.clip(out, 0.0, 65535.0).astype(np.uint16)


def _simulate_jittered(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """simulate_second_scanner with an extra per-pair translation jitter."""
    height, width = img.shape
    forward = ground_truth_warp((height, width)).copy()
    forward[0, 2] += dx
    forward[1, 2] += dy
    out_width = int(round(width * SIM_FOV_WIDTH_FRAC))
    warped = cv2.warpAffine(
        img.astype(np.float32),
        forward.astype(np.float64),
        (out_width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    rescaled = warped / 65535.0 * SIM_GAIN + SIM_OFFSET
    return np.clip(rescaled, 0.0, 65535.0).astype(np.uint16)


def _write_pairs(dir_a, dir_b, *, banded: bool) -> np.ndarray | None:
    """Six jittered phantom pairs; scanner B optionally banded."""
    pattern: np.ndarray | None = None
    for i, (dx, dy) in enumerate(_JITTER):
        phantom = make_phantom(seed=42 + i)
        b_scan = _simulate_jittered(phantom, dx, dy)
        if banded:
            if pattern is None:
                pattern = _band_pattern(b_scan.shape[1])
            b_scan = _plant_banding(b_scan, pattern)
        cv2.imwrite(str(dir_a / f"case{i}.png"), phantom)
        cv2.imwrite(str(dir_b / f"case{i}.png"), b_scan)
    return pattern


def _identity_mask(dir_a, dir_b) -> DefectMaskData:
    """A defect mask anchoring every scan at offset 0 (no defect columns)."""
    entries = {}
    for directory in (dir_a, dir_b):
        key = str(directory.resolve())
        scans = {
            png.name: DefectScanInfo(
                name=png.name,
                x_offset=0,
                anchored=True,
                candidate_count=0,
                defect_columns_native=[],
            )
            for png in directory.glob("*.png")
        }
        entries[key] = DirectoryDefects(
            directory=key,
            reference_scan=sorted(scans)[0] if scans else "",
            stationary_columns_ref_frame=[],
            scans=scans,
        )
    return DefectMaskData(params={}, directories=entries)


@pytest.fixture()
def banded_pair_dirs(tmp_path):
    """Six jittered pairs; scanner B carries planted stationary banding."""
    dir_a = tmp_path / "scanner_a"
    dir_b = tmp_path / "scanner_b"
    dir_a.mkdir()
    dir_b.mkdir()
    pattern = _write_pairs(dir_a, dir_b, banded=True)
    return dir_a, dir_b, pattern


@pytest.fixture()
def plain_pair_dirs(tmp_path):
    """Six jittered pairs without banding (the null case)."""
    dir_a = tmp_path / "scanner_a"
    dir_b = tmp_path / "scanner_b"
    dir_a.mkdir()
    dir_b.mkdir()
    _ = _write_pairs(dir_a, dir_b, banded=False)
    return dir_a, dir_b


def _pairs(dir_a, dir_b):
    pairs, _unmatched = find_pairs(dir_a, dir_b)
    return pairs


def _solve(dir_a, dir_b):
    """Solve with the identity-anchored defect mask (required)."""
    config = CompareConfig(max_dim=512, defect_mask=_identity_mask(dir_a, dir_b))
    return solve_column_gain(_pairs(dir_a, dir_b), config, progress=lambda _s: None)


def test_column_diff_profile_recovers_planted_offset():
    """The per-column median of the diff tracks a planted column offset."""
    rng = np.random.default_rng(7)
    height, width = 300, 400
    ref = rng.uniform(0.0, 1.0, size=(height, width)).astype(np.float32)
    offset = _band_pattern(width)
    mov = (ref - offset[None, :]).astype(np.float32)
    mask = np.ones((height, width), dtype=bool)
    profile = column_diff_profile(ref, mov, mask)
    expected = offset - offset.mean()
    assert np.all(np.isfinite(profile))
    assert np.max(np.abs(profile - expected)) < 0.01


def test_column_diff_profile_marks_sparse_columns_nan():
    """Columns with too few masked rows are NaN (no estimate)."""
    height, width = 300, 100
    img = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    mask[:10, :] = True  # 10 rows < MIN_COLUMN_ROWS everywhere
    profile = column_diff_profile(img, img, mask)
    assert not np.isfinite(profile).any()


def test_shift_profile_moves_scan_frame_to_sensor_frame():
    """sensor column = scan column + offset; uncovered columns are NaN."""
    profile = np.arange(10, dtype=np.float64)
    shifted = shift_profile(profile, 3, 14)
    assert np.allclose(shifted[3:13], profile)
    assert not np.isfinite(shifted[:3]).any()
    assert not np.isfinite(shifted[13:]).any()
    # Negative offsets crop the left edge.
    shifted_neg = shift_profile(profile, -2, 8)
    assert np.allclose(shifted_neg[:8], profile[2:10])


def test_aggregate_profiles_median_and_min_pairs():
    """Median across pairs; columns with too few contributors go to 0."""
    base = _band_pattern(200)
    rng = np.random.default_rng(3)
    profiles = [base + rng.normal(0.0, 1e-4, size=200) for _ in range(4)]
    agg, width = aggregate_profiles(profiles)
    assert width == 200
    assert np.max(np.abs(agg - (base - base.mean()))) < 1e-3
    # Fewer than MIN_PAIR_PROFILES profiles: no valid estimate anywhere.
    agg2, _ = aggregate_profiles(profiles[:2])
    assert np.all(agg2 == 0.0)


def test_aggregate_profiles_rejects_empty():
    """Aggregating nothing raises ColumnGainError."""
    with pytest.raises(ColumnGainError):
        aggregate_profiles([])


def test_subtract_profile_within_mask_only():
    """The profile is subtracted inside the mask only; rms is reported."""
    height, width = 200, 300
    ref = np.full((height, width), 0.5, dtype=np.float32)
    mask = np.ones((height, width), dtype=bool)
    mask[:50] = False
    pattern = _band_pattern(width)
    ref_out, rms = subtract_profile(ref, pattern, mask)
    expected = 0.5 - pattern
    assert np.allclose(ref_out[50:], expected[None, :], atol=1e-6)
    assert np.all(ref_out[:50] == 0.5)  # outside the mask: untouched
    assert rms == pytest.approx(math.sqrt(float(np.mean(pattern**2))), abs=1e-6)


def test_subtract_profile_rejects_width_mismatch():
    """A profile of the wrong width raises ColumnGainError."""
    ref = np.zeros((50, 100), dtype=np.float32)
    mask = np.ones((50, 100), dtype=bool)
    with pytest.raises(ColumnGainError):
        subtract_profile(ref, np.zeros(120, dtype=np.float64), mask)


def test_profile_for_scan_crops_by_offset(banded_pair_dirs, tmp_path):
    """The per-scan subtract-profile is the sensor profile at its offset."""
    dir_a, dir_b, _pattern = banded_pair_dirs
    data = _solve(dir_a, dir_b)
    entry = data.for_directory(dir_a)
    name = sorted(entry.scan_offsets)[0]
    full = np.asarray(entry.profile)
    cropped = entry.profile_for_scan(name, 500)
    assert cropped is not None
    assert cropped.shape == (500,)
    assert np.allclose(cropped, full[:500])
    assert entry.profile_for_scan("unknown.png", 500) is None


def test_column_gain_json_roundtrip(banded_pair_dirs, tmp_path):
    """The calibration JSON round-trips losslessly."""
    dir_a, dir_b, _pattern = banded_pair_dirs
    data = _solve(dir_a, dir_b)
    path = write_column_gain(tmp_path / "colgain.json", data)
    assert read_column_gain(path) == data


def test_read_column_gain_rejects_bad_files(tmp_path):
    """Missing/malformed calibration files raise ColumnGainError."""
    with pytest.raises(ColumnGainError):
        read_column_gain(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    _ = bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ColumnGainError):
        read_column_gain(bad)


def test_solve_column_gain_requires_defect_mask(banded_pair_dirs):
    """The solve needs the defect mask's crop anchoring."""
    dir_a, dir_b, _pattern = banded_pair_dirs
    with pytest.raises(ColumnGainError, match="defect mask"):
        solve_column_gain(
            _pairs(dir_a, dir_b), CompareConfig(max_dim=512), progress=lambda _s: None
        )


def test_solve_column_gain_recovers_planted_banding(banded_pair_dirs):
    """B's stationary profile matches the planted pattern (reverse pairs)."""
    dir_a, dir_b, pattern = banded_pair_dirs
    data = _solve(dir_a, dir_b)
    entry_b = data.for_directory(dir_b)
    assert pattern is not None
    recovered = np.asarray(entry_b.profile)
    # B's frame profile comes from the reverse pairs (B the reference):
    # diff = B - A carries B's banding with a positive (pdf-like) scale.
    # (All synthetic scans share the defect reference frame at offset 0, so
    # the sensor frame coincides with B's native columns.)
    interior = slice(40, -40)
    corr = np.corrcoef(pattern[interior], recovered[interior])[0, 1]
    assert corr > 0.9
    assert entry_b.rms > 0.001
    # The forward (A-frame) profile estimates the diff combination
    # bandA - bandB(w) = -bandB(w) here: NOT flat by design — it is the
    # correction the forward run needs.
    entry_a = data.for_directory(dir_a)
    assert entry_a.rms > 0.001


def test_solve_column_gain_null_without_banding(plain_pair_dirs):
    """No planted banding: the aggregated profiles stay near zero."""
    dir_a, dir_b = plain_pair_dirs
    data = _solve(dir_a, dir_b)
    for entry in data.directories.values():
        assert entry.rms < 0.005


def test_run_with_column_gain_reduces_coherent_error(banded_pair_dirs, tmp_path):
    """Applying the calibration lowers local_rmse and records the amount."""
    dir_a, dir_b, _pattern = banded_pair_dirs
    data = _solve(dir_a, dir_b)

    def quiet(_s: str) -> None:
        return None

    plain = run(
        dir_a, dir_b, tmp_path / "plain", CompareConfig(max_dim=512), progress=quiet
    )
    fixed = run(
        dir_a,
        dir_b,
        tmp_path / "fixed",
        CompareConfig(max_dim=512, column_gain=data),
        progress=quiet,
    )
    plain_lrmse = [r.metrics.local_rmse for r in plain.results if r.metrics]
    fixed_lrmse = [r.metrics.local_rmse for r in fixed.results if r.metrics]
    assert len(plain_lrmse) == 6
    assert sum(fixed_lrmse) < sum(plain_lrmse) * 0.8
    for result in fixed.results:
        assert result.colgain_rms is not None
        assert result.colgain_rms > 0.0


def test_cli_find_column_gain_and_run_analysis(banded_pair_dirs, tmp_path):
    """find_column_gain writes the calibration; run_analysis consumes it."""
    dir_a, dir_b, _pattern = banded_pair_dirs
    from scanner_comparison.calibration.defects import write_defect_mask

    mask_path = write_defect_mask(
        tmp_path / "defects.json", _identity_mask(dir_a, dir_b)
    )
    cal_path = tmp_path / "colgain.json"
    code = main(
        [
            "find_column_gain",
            str(dir_a),
            str(dir_b),
            "--out",
            str(cal_path),
            "--max-dim",
            "512",
            "--defect-mask",
            str(mask_path),
        ]
    )
    assert code == 0
    payload = json.loads(cal_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["directories"]) == 2
    assert payload["pairs"]
    # Inspectable artifacts: one profile map PNG + CSV per directory.
    maps = sorted(tmp_path.glob("colgain_*.png"))
    csvs = sorted(tmp_path.glob("colgain_*.csv"))
    assert len(maps) == 2
    assert len(csvs) == 2

    out = tmp_path / "out"
    code = main(
        [
            "run_analysis",
            str(dir_a),
            str(dir_b),
            "--out",
            str(out),
            "--max-dim",
            "512",
            "--column-gain",
            str(cal_path),
        ]
    )
    assert code in (0, 1)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["column_gain"] is not None
    pair = summary["pairs"][0]
    assert pair["colgain_rms"] is not None


def test_cli_find_column_gain_rejects_missing_defect_mask(
    banded_pair_dirs, tmp_path
):
    """find_column_gain without --defect-mask exits 2."""
    dir_a, dir_b, _pattern = banded_pair_dirs
    code = main(
        [
            "find_column_gain",
            str(dir_a),
            str(dir_b),
            "--out",
            str(tmp_path / "colgain.json"),
            "--max-dim",
            "512",
        ]
    )
    assert code == 2


def test_cli_run_analysis_rejects_bad_column_gain(pair_dirs, tmp_path):
    """A malformed --column-gain file exits 2."""
    dir_a, dir_b = pair_dirs
    bad = tmp_path / "bad.json"
    _ = bad.write_text("{}", encoding="utf-8")
    code = main(["run_analysis", str(dir_a), str(dir_b), "--column-gain", str(bad)])
    assert code == 2
