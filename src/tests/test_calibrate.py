"""Tests for the scale/blur calibration solvers and their CLI wiring."""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest
from conftest import SIM_SCALE, make_phantom, simulate_second_scanner

from scanner_comparison.calibration.blur import (
    BlurCorrector,
    correction_sigmas,
    masked_gaussian,
    phase_table,
    resampling_penalty,
)
from scanner_comparison.cli import main
from scanner_comparison.calibration.defects import (
    DefectMaskData,
    DefectScanInfo,
    DirectoryDefects,
    write_defect_mask,
)
from scanner_comparison.core.imtypes import shape2
from scanner_comparison.core.io import find_pairs, load_image
from scanner_comparison.pipeline import compare_pair, solve_blur_constant
from scanner_comparison.records import CompareConfig
from scanner_comparison.calibration.scale import solve_scale_correction


@pytest.fixture()
def blurred_pair_dirs(tmp_path):
    """pair_dirs variant: scanner B additionally blurred by a known sigma."""
    dir_a = tmp_path / "scanner_a"
    dir_b = tmp_path / "scanner_b"
    dir_a.mkdir()
    dir_b.mkdir()
    for i, seed in enumerate((42, 43)):
        phantom = make_phantom(seed=seed)
        blurred_b = simulate_second_scanner(phantom, blur_sigma=1.2)
        cv2.imwrite(str(dir_a / f"case{i}.png"), phantom)
        cv2.imwrite(str(dir_b / f"case{i}.png"), blurred_b)
    return dir_a, dir_b


def test_correction_sigmas_forward_blurs_reference():
    """Moving blurrier by sigma_dev + resampling: blur the reference."""
    sig_ref, sig_mov = correction_sigmas(0.685, 0.482)
    assert sig_ref == pytest.approx(math.hypot(0.685, 0.482))
    assert sig_mov == 0.0


def test_correction_sigmas_reverse_blurs_moving():
    """Negated gap (sharper moving side) stronger than r: blur the moving."""
    sig_ref, sig_mov = correction_sigmas(-0.685, 0.482)
    assert sig_ref == 0.0
    assert sig_mov == pytest.approx(math.sqrt(0.685**2 - 0.482**2))


def test_correction_sigmas_resampling_dominant_blurs_reference():
    """A sharper moving side that loses more than its edge to resampling."""
    sig_ref, sig_mov = correction_sigmas(-0.3, 0.482)
    assert sig_ref == pytest.approx(math.sqrt(0.482**2 - 0.3**2))
    assert sig_mov == 0.0


def test_correction_sigmas_zero_gap():
    assert correction_sigmas(0.0, 0.0) == (0.0, 0.0)
    # Pure resampling penalty: blur the reference by r.
    sig_ref, sig_mov = correction_sigmas(0.0, 0.482)
    assert sig_ref == pytest.approx(0.482)
    assert sig_mov == 0.0


def test_masked_gaussian_matches_plain_blur_in_the_interior():
    """Deep inside the mask the masked blur equals the ordinary blur."""
    rng = np.random.default_rng(3)
    img = rng.random((120, 160), dtype=np.float32)
    mask = np.zeros(img.shape, dtype=bool)
    mask[10:110, 10:150] = True
    img[~mask] = 0.0  # rank images are zero outside the mask
    masked = masked_gaussian(img, 2.0, mask)
    plain = cv2.GaussianBlur(img, (0, 0), 2.0)
    interior = np.zeros_like(mask)
    interior[30:90, 30:130] = True
    assert np.allclose(masked[interior], plain[interior], atol=1e-5)
    assert masked[~mask].sum() == 0.0
    # The rim is renormalized, not darkened by outside zeros bleeding in.
    rim = mask & ~interior
    assert float(masked[rim].mean()) > float(plain[rim].mean())


def test_masked_gaussian_zero_sigma_is_identity():
    img = np.full((8, 8), 0.5, dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    assert masked_gaussian(img, 0.0, mask) is img


def test_phase_table_zero_phase_and_bounds(phantom):
    """r(0,0) = 0 by construction; signed cells are small blurs.

    Cells are signed under the common-support instrument: negative
    (apparent sharpening) cells occur from the support-selection
    systematic on shifted correlated content, so only the magnitude is
    bounded — the table is consumed through phase averages/integrals.
    """
    table = phase_table(phantom, grid=3)
    assert table[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert float(np.abs(table).max()) < 1.5


def test_resampling_penalty_samples_the_table(phantom):
    """Identity warp -> r(0,0); a constant fractional shift -> that cell."""
    table = phase_table(phantom, grid=3)
    shape = shape2(phantom)
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert resampling_penalty(identity, shape, table) == pytest.approx(0.0, abs=1e-3)
    shift_x = np.array([[1.0, 0.0, 1.0 / 3.0], [0.0, 1.0, 0.0]])
    r = resampling_penalty(shift_x, shape, table)
    # The penalty is a quadrature average of the table, so it is unsigned
    # even where a table cell itself reads negative (signed instrument).
    assert r == pytest.approx(abs(float(table[0, 1])), abs=0.05)


def _flat_phantom(seed: int, shape: tuple[int, int] = (512, 640)) -> np.ndarray:
    """Uniform band-limited texture phantom.

    The sweep criterion needs spatially uniform content: the blob phantom's
    central contrast concentration biases the masked NCC toward smaller
    scales (shrinking overlap concentrates on strong features); flat texture
    decorrelates honestly at the wrong scale, so the sweep peaks at the
    truth.
    """
    rng = np.random.default_rng(seed)
    texture = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    img = cv2.GaussianBlur(texture, (0, 0), 1.5)
    img = 0.5 + 0.35 * img / (3.0 * float(img.std()))
    return (np.clip(img, 0.0, 1.0) * 65535.0).astype(np.uint16)


def test_solve_scale_correction_recovers_known_scale(tmp_path):
    """The sweep peaks at the simulator's known 0.9985 scale mismatch."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    for i, seed in enumerate((42, 43)):
        phantom = _flat_phantom(seed)
        cv2.imwrite(str(dir_a / f"case{i}.png"), phantom)
        cv2.imwrite(
            str(dir_b / f"case{i}.png"),
            simulate_second_scanner(phantom, scale=SIM_SCALE),
        )
    pairs, _ = find_pairs(dir_a, dir_b)
    candidates = [0.9965, 0.9975, 0.9985, 0.9995, 1.0005]
    cal = solve_scale_correction(
        pairs, candidates=candidates, max_pairs=5, max_dim=512,
        progress=lambda _s: None,
    )
    assert abs(cal.scale - SIM_SCALE) <= 0.001
    assert abs(cal.scale_refined - SIM_SCALE) <= 0.002
    assert len(cal.mean_ncc) == len(candidates)


def test_solve_blur_constant_recovers_planted_blur(blurred_pair_dirs):
    """A planted 1.2 px device blur is recovered clearly above the floor.

    The signed common-support blur gap is an edge-energy equivalence, so a
    Gaussian blurred in the intensity domain does not read as exactly its
    nominal sigma; empirically the 1.2 px plant + the simulator's own
    resampling blur solve to ~1.04 px under the bidirectional signed
    convention. The assertion guards the order of magnitude and the clear
    separation from the simulator-only floor (~0.45, see the lossless
    test).
    """
    dir_a, dir_b = blurred_pair_dirs
    pairs, _ = find_pairs(dir_a, dir_b)
    cal = solve_blur_constant(
        pairs, CompareConfig(max_dim=512), grid=3, progress=lambda _s: None
    )
    assert 0.85 <= cal.sigma_dev <= 1.25
    assert cal.r_bar > 0.0
    # The one-sided arms should bracket the combined constant.
    assert cal.sigma_dev_forward < cal.sigma_dev < cal.sigma_dev_reverse
    assert len(cal.pairs) == 2


def test_solve_blur_constant_antisymmetric_under_swap(blurred_pair_dirs):
    """solve(A, B) == -solve(B, A): the signed solve is direction-honest."""
    dir_a, dir_b = blurred_pair_dirs
    pairs_ab, _ = find_pairs(dir_a, dir_b)
    pairs_ba, _ = find_pairs(dir_b, dir_a)
    cal_ab = solve_blur_constant(
        pairs_ab, CompareConfig(max_dim=512), grid=3, progress=lambda _s: None
    )
    cal_ba = solve_blur_constant(
        pairs_ba, CompareConfig(max_dim=512), grid=3, progress=lambda _s: None
    )
    assert cal_ab.sigma_dev > 0.0 > cal_ba.sigma_dev
    assert cal_ab.sigma_dev == pytest.approx(-cal_ba.sigma_dev, abs=0.05)


def test_solve_blur_constant_identical_pair_is_zero(tmp_path, phantom):
    """True null: the same image in both directories solves to ~0.

    Registration converges to the identity (no phase drift -> r ~ r(0,0) =
    0) and the signed gap vanishes, so the device constant is 0. (A
    shifted same-content pair is NOT a usable null probe under the
    common-support instrument — the support-selection systematic reads a
    phantom gap of a few tenths of a px there; diag16.)
    """
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    cv2.imwrite(str(dir_a / "case1.png"), phantom)
    cv2.imwrite(str(dir_b / "case1.png"), phantom)
    pairs, _ = find_pairs(dir_a, dir_b)
    cal = solve_blur_constant(
        pairs, CompareConfig(max_dim=512), grid=3, progress=lambda _s: None
    )
    assert abs(cal.sigma_dev) <= 0.05


def test_solve_blur_constant_attributes_simulator_blur(pair_dirs):
    """The 'lossless' simulator pair is NOT a null: scanner B's image
    carries the simulator's own INTER_LINEAR content blur, which the honest
    signed instrument reports as B's device blur (~0.45 px on the phantom).

    Our pipeline's OWN resampling penalty is still accounted separately
    (the table r_i at apply time); what changed vs the previous convention
    is that the instrument no longer absorbs B's intrinsic content blur
    into r_i.
    """
    dir_a, dir_b = pair_dirs
    pairs, _ = find_pairs(dir_a, dir_b)
    cal = solve_blur_constant(
        pairs, CompareConfig(max_dim=512), grid=3, progress=lambda _s: None
    )
    assert 0.3 <= cal.sigma_dev <= 0.6


def test_blur_correction_shrinks_residual_blur(blurred_pair_dirs, tmp_path):
    """Applying the solved constant shrinks the signed residual blur gap."""
    dir_a, dir_b = blurred_pair_dirs
    pairs, _ = find_pairs(dir_a, dir_b)
    cal = solve_blur_constant(
        pairs, CompareConfig(max_dim=512), grid=3, progress=lambda _s: None
    )
    pair = pairs[0]
    uncorrected = compare_pair(pair, tmp_path / "raw", CompareConfig(max_dim=512))
    assert uncorrected.metrics is not None
    corrector = BlurCorrector(cal.sigma_dev, phase_table(load_image(pair.path_b), grid=3))
    corrected = compare_pair(
        pair,
        tmp_path / "corr",
        CompareConfig(max_dim=512),
        corrector=corrector,
    )
    assert corrected.metrics is not None
    # The residual is signed: it shrinks in MAGNITUDE (and may overshoot
    # into a small negative value when the correction slightly exceeds the
    # gap — that is the closure behavior, not a floor).
    assert abs(corrected.metrics.blur_sigma) < 0.5 * abs(
        uncorrected.metrics.blur_sigma
    )
    assert corrected.blur_applied is not None
    # Forward rule (DIR_B blurrier): the reference side is blurred.
    _r_i, sig_ref, sig_mov = corrected.blur_applied
    assert sig_ref > 0.0
    assert sig_mov == 0.0


def test_cli_find_scale_writes_calibration(tmp_path, pair_dirs):
    """find_scale exits 0 and writes a sweep JSON with a recommendation."""
    dir_a, dir_b = pair_dirs
    out = tmp_path / "scale.json"
    code = main(
        [
            "find_scale",
            str(dir_a),
            str(dir_b),
            "--out",
            str(out),
            "--max-dim",
            "512",
            "--scale-min",
            "0.9975",
            "--scale-max",
            "0.9995",
            "--scale-step",
            "0.001",
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["candidates"]) == 3
    assert abs(payload["scale"] - SIM_SCALE) <= 0.001


def test_cli_find_scale_rejects_bad_sweep(pair_dirs):
    """Out-of-band sweep bounds exit 2."""
    dir_a, dir_b = pair_dirs
    code = main(
        ["find_scale", str(dir_a), str(dir_b), "--scale-min", "0.5", "--scale-max", "1.0"]
    )
    assert code == 2


def test_cli_find_blur_writes_calibration(blurred_pair_dirs, tmp_path):
    """find_blur exits 0 and writes the solved constant and evidence."""
    dir_a, dir_b = blurred_pair_dirs
    out = tmp_path / "blur.json"
    code = main(
        ["find_blur", str(dir_a), str(dir_b), "--out", str(out), "--max-dim", "512"]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["convention"] == "signed common-support bidirectional"
    assert payload["sigma_dev"] > 0.8
    assert payload["r_bar"] > 0.0
    assert payload["sigma_dev_forward"] > 0.0
    assert payload["sigma_dev_reverse"] > 0.0
    assert len(payload["pairs"]) == 2
    for pair in payload["pairs"]:
        assert pair["m_forward"] > 0.0 > pair["m_reverse"]
        assert pair["r_forward_table"] > 0.0
        assert pair["r_reverse_table"] > 0.0


def _hand_made_mask(dir_a, dir_b) -> DefectMaskData:
    """A defect mask with known native columns for small synthetic dirs."""
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


def test_cli_find_blur_with_defect_mask(blurred_pair_dirs, tmp_path):
    """find_blur --defect-mask solves on the masked pixel population."""
    dir_a, dir_b = blurred_pair_dirs
    mask_path = write_defect_mask(
        tmp_path / "defects.json", _hand_made_mask(dir_a, dir_b)
    )
    out = tmp_path / "blur.json"
    code = main(
        [
            "find_blur",
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
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    # Same planted blur, solved through the masked path.
    assert payload["sigma_dev"] > 0.8


def test_cli_find_blur_rejects_bad_defect_mask(blurred_pair_dirs, tmp_path):
    """A malformed --defect-mask exits 2."""
    dir_a, dir_b = blurred_pair_dirs
    bad = tmp_path / "bad.json"
    _ = bad.write_text("{}", encoding="utf-8")
    code = main(
        [
            "find_blur",
            str(dir_a),
            str(dir_b),
            "--out",
            str(tmp_path / "blur.json"),
            "--defect-mask",
            str(bad),
        ]
    )
    assert code == 2


def test_cli_run_analysis_blur_correction(pair_dirs, tmp_path):
    """--blur-correction lands in the summary config and per-pair fields."""
    dir_a, dir_b = pair_dirs
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
            "--blur-correction",
            "0.685",
        ]
    )
    assert code in (0, 1)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["blur_correction"] == 0.685
    (pair,) = [p for p in summary["pairs"] if p["name"] == "case1.png"]
    assert pair["blur_r"] > 0.0
    # Positive sigma_dev = DIR_B blurrier: the forward rule blurs the
    # reference side by hypot(sigma_dev, r).
    assert pair["blur_sigma_ref_applied"] > 0.0
    assert pair["blur_sigma_mov_applied"] == 0.0


def test_cli_run_analysis_blur_correction_sign_flip(pair_dirs, tmp_path):
    """--both-directions negates the blur constant for the reverse run."""
    dir_a, dir_b = pair_dirs
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
            "--both-directions",
            "--blur-correction",
            "0.685",
        ]
    )
    assert code in (0, 1)
    fwd = json.loads((out / "forward" / "summary.json").read_text(encoding="utf-8"))
    rev = json.loads((out / "reverse" / "summary.json").read_text(encoding="utf-8"))
    assert fwd["config"]["blur_correction"] == 0.685
    assert rev["config"]["blur_correction"] == -0.685


def test_cli_run_analysis_rejects_absurd_blur(pair_dirs, tmp_path):
    """An implausible blur constant exits 2."""
    dir_a, dir_b = pair_dirs
    code = main(
        ["run_analysis", str(dir_a), str(dir_b), "--blur-correction", "99.0"]
    )
    assert code == 2


def test_cli_run_all_end_to_end(pair_dirs, tmp_path):
    """run_all calibrates all four constants, then analyzes with them."""
    dir_a, dir_b = pair_dirs
    out = tmp_path / "all"
    code = main(
        [
            "run_all",
            str(dir_a),
            str(dir_b),
            "--out",
            str(out),
            "--max-dim",
            "512",
            "--scale-min",
            "0.9975",
            "--scale-max",
            "0.9995",
            "--scale-step",
            "0.001",
        ]
    )
    assert code in (0, 1)
    assert (out / "scale.json").exists()
    assert (out / "blur.json").exists()
    assert (out / "defects.json").exists()
    assert (out / "colgain.json").exists()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    config = summary["config"]
    assert abs(config["scale_correction"] - SIM_SCALE) <= 0.001
    assert config["blur_correction"] != 0.0  # the simulator's own ~0.4 px
    assert config["defect_mask"] is not None
    assert config["column_gain"] is not None
