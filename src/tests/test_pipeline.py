"""End-to-end pipeline tests on simulated scanner pairs."""

from __future__ import annotations

import json

import cv2
import pytest

from scanner_comparison.cli import main
from scanner_comparison.core.io import find_pairs
from scanner_comparison.pipeline import NoPairsError, compare_pair, run
from scanner_comparison.records import CompareConfig, Thresholds

from conftest import simulate_second_scanner

STRICT = CompareConfig(
    thresholds=Thresholds(
        max_rmse=0.03,
        min_ssim=0.97,
        min_grad_corr=0.97,
        grad_energy_tolerance=0.10,
    ),
    max_dim=512,
)
NO_MARGINS = CompareConfig(
    thresholds=STRICT.thresholds,
    max_dim=512,
    border_margin=0.0,
    corner_margin=0.0,
    exclude_background=False,
)


def _single_pair(tmp_path, subdir, a_img, b_img):
    """Write one image pair into fresh dirs and return the ImagePair."""
    dir_a = tmp_path / f"{subdir}_a"
    dir_b = tmp_path / f"{subdir}_b"
    dir_a.mkdir()
    dir_b.mkdir()
    cv2.imwrite(str(dir_a / "case.png"), a_img)
    cv2.imwrite(str(dir_b / "case.png"), b_img)
    pairs, _ = find_pairs(dir_a, dir_b)
    return pairs[0]


def test_lossless_simulation_passes(pair_dirs, tmp_path):
    dir_a, dir_b = pair_dirs
    out_dir = tmp_path / "results"
    summary = run(dir_a, dir_b, out_dir, STRICT, progress=lambda _: None)
    assert summary.all_passed
    assert summary.unmatched == ["only_in_a.png"]
    result = summary.results[0]
    assert result.metrics is not None
    assert result.metrics.ssim > STRICT.thresholds.min_ssim
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "summary.csv").is_file()
    assert result.artifacts["diff_heatmap"].is_file()
    assert result.artifacts["local_diff_heatmap"].is_file()
    assert result.artifacts["motion_amplified_diff"].is_file()
    assert result.artifacts["frc_curve"].is_file()
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["all_passed"]
    assert payload["pairs"][0]["name"] == "case1.png"
    assert payload["config"]["border_margin"] == STRICT.border_margin
    # The lossless synthetic pair shares its (smooth) texture, so detail is
    # conserved to a fine scale (both FRC criteria reported in the summary).
    frc_px = payload["pairs"][0]["frc_resolution_px"]
    assert frc_px is None or frc_px >= 2.0
    assert "frc_resolution_px_17" in payload["pairs"][0]


def test_blurred_simulation_fails(pair_dirs, tmp_path, phantom):
    dir_a, dir_b = pair_dirs
    blurry_b = simulate_second_scanner(phantom, blur_sigma=2.5)
    cv2.imwrite(str(dir_b / "case1.png"), blurry_b)
    summary = run(dir_a, dir_b, tmp_path / "results", STRICT, progress=lambda _: None)
    assert not summary.all_passed
    assert any(
        "grad_energy_ratio" in failure for failure in summary.results[0].failures
    )


def test_run_raises_without_pairs(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    with pytest.raises(NoPairsError):
        run(dir_a, dir_b, tmp_path / "out", STRICT, progress=lambda _: None)


def test_cli_exit_codes(pair_dirs, tmp_path):
    dir_a, dir_b = pair_dirs
    out_dir = tmp_path / "results"
    code = main([str(dir_a), str(dir_b), "--out", str(out_dir), "--max-dim", "512"])
    assert code == 0
    assert main([str(dir_a), str(tmp_path / "missing"), "--out", str(out_dir)]) == 2


def test_cli_both_directions(pair_dirs, tmp_path):
    dir_a, dir_b = pair_dirs
    out_dir = tmp_path / "bi"
    code = main(
        [
            str(dir_a),
            str(dir_b),
            "--out",
            str(out_dir),
            "--max-dim",
            "512",
            "--both-directions",
        ]
    )
    assert code == 0
    fwd = json.loads((out_dir / "forward" / "summary.json").read_text())
    rev = json.loads((out_dir / "reverse" / "summary.json").read_text())
    assert fwd["all_passed"] and rev["all_passed"]
    # Direction asymmetry check: the simulated scanner B carries the
    # simulator's interpolation blur, so forward (B moving onto sharp A)
    # measures a positive signed blur gap, while reverse (sharp A moving
    # onto blurred B) reads the same gap with the opposite sign — the
    # signed convention has no floor.
    assert fwd["pairs"][0]["blur_sigma"] > 0.2
    assert rev["pairs"][0]["blur_sigma"] < -0.2
    # Cross-direction summary: the round-trip warp composition should be
    # near-identity on the lossless pair, and the blur delta spans both
    # signed readings (forward - reverse ~ 2x the gap magnitude).
    cross = json.loads((out_dir / "cross_direction.json").read_text())
    assert (out_dir / "cross_direction.csv").is_file()
    row = cross["pairs"][0]
    assert row["name"] == "case1.png"
    assert row["roundtrip_max_px"] < 3.0
    assert row["delta_blur_sigma"] == pytest.approx(
        fwd["pairs"][0]["blur_sigma"] - rev["pairs"][0]["blur_sigma"]
    )


def test_cli_both_directions_inverts_scale_correction(pair_dirs, tmp_path):
    dir_a, dir_b = pair_dirs
    out_dir = tmp_path / "bi_scale"
    _ = main(
        [
            str(dir_a),
            str(dir_b),
            "--out",
            str(out_dir),
            "--max-dim",
            "512",
            "--both-directions",
            "--scale-correction",
            "0.9985",
        ]
    )
    fwd = json.loads((out_dir / "forward" / "summary.json").read_text())
    rev = json.loads((out_dir / "reverse" / "summary.json").read_text())
    assert fwd["config"]["scale_correction"] == pytest.approx(0.9985)
    assert rev["config"]["scale_correction"] == pytest.approx(1.0 / 0.9985)


def test_cli_rejects_absurd_scale_correction(pair_dirs, tmp_path):
    dir_a, dir_b = pair_dirs
    code = main(
        [str(dir_a), str(dir_b), "--out", str(tmp_path / "o"), "--scale-correction", "0.5"]
    )
    assert code == 2


def test_compare_pair_reports_second_scanner_response(pair_dirs, tmp_path):
    dir_a, dir_b = pair_dirs
    pairs, _ = find_pairs(dir_a, dir_b)
    result = compare_pair(pairs[0], tmp_path / "results", STRICT)
    assert result.passed
    # Each image is independently percentile-normalized first, so the
    # residual gain/offset fit should be near identity even though scanner
    # B's simulated histogram is compressed and offset.
    assert result.gain == pytest.approx(1.0, abs=0.05)
    assert abs(result.offset) < 0.05
    assert result.reg_correlation > 0.9


def test_border_junk_deprioritized_when_margins_enabled(tmp_path, phantom):
    clean_b = simulate_second_scanner(phantom)
    junk_b = simulate_second_scanner(phantom, junk_border=True)
    clean_pair = _single_pair(tmp_path, "clean", phantom, clean_b)
    junk_pair = _single_pair(tmp_path, "junk", phantom, junk_b)
    clean = compare_pair(clean_pair, tmp_path / "rc", STRICT)
    junk = compare_pair(junk_pair, tmp_path / "rj", STRICT)
    junk_no_margin = compare_pair(junk_pair, tmp_path / "rjn", NO_MARGINS)
    # The warp's offset/scale pushes a sliver of B's junk band past A's
    # margin, so junk is mitigated rather than fully eliminated. (Tolerance
    # is looser since the Lanczos4 warp: bilinear used to smooth the leaked
    # sliver, the sharper kernel renders it faithfully.)
    assert junk.metrics.rmse < clean.metrics.rmse + 0.015
    assert junk.metrics.ssim > clean.metrics.ssim - 0.02
    assert junk.metrics.rmse < 0.5 * junk_no_margin.metrics.rmse


def test_border_junk_pollutes_when_margins_disabled(tmp_path, phantom):
    junk_b = simulate_second_scanner(phantom, junk_border=True)
    junk_pair = _single_pair(tmp_path, "junk0", phantom, junk_b)
    with_margin = compare_pair(junk_pair, tmp_path / "r_on", STRICT)
    without = compare_pair(junk_pair, tmp_path / "r_off", NO_MARGINS)
    # Registration is unaffected (edges are always masked there); with the
    # metric margins disabled, the junk frame lands in the overlap and
    # pollutes the metrics. The thin junk bands move the masked-mean SSIM
    # only slightly, so the SSIM gap is smaller than the RMSE gap.
    assert without.metrics.rmse > with_margin.metrics.rmse + 0.02
    assert without.metrics.ssim < with_margin.metrics.ssim - 0.01


def test_dark_surround_excluded_by_default(tmp_path, phantom):
    # Dark surround on all four sides of the reference; mismatched mid-gray
    # junk on all four sides of the second scan. With background exclusion
    # on, the junk landing in the surround region must not affect the
    # metrics. (The junk is mid-gray rather than clipped-bright because an
    # extreme band deep inside the frame would also dominate the log-DoG
    # registration features; registration only guarantees masking of the
    # default border/corner margins.)
    a = phantom.copy()
    band = round(min(a.shape) * 0.10)
    a[:band, :] = 800
    a[-band:, :] = 700
    a[:, :band] = 750
    a[:, -band:] = 720
    junk_b = simulate_second_scanner(phantom)
    junk_b[:band, :] = 30_000
    junk_b[-band:, :] = 28_000
    junk_b[:, :band] = 29_000
    junk_b[:, -band:] = 27_000
    pair = _single_pair(tmp_path, "surround", a, junk_b)
    with_exclusion = compare_pair(pair, tmp_path / "r_on", STRICT)
    without = compare_pair(pair, tmp_path / "r_off", NO_MARGINS)
    assert with_exclusion.metrics.rmse < without.metrics.rmse
