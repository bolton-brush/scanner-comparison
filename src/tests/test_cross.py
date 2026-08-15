"""Tests for cross-direction consistency metrics."""

from __future__ import annotations

import numpy as np
import pytest

from scanner_comparison.report.cross import cross_direction_metrics, roundtrip_max_px
from scanner_comparison.imaging.metrics import PairMetrics
from scanner_comparison.records import PairResult, RunSummary

_SHAPE = (3520, 2990)


def _warp(angle_deg: float = 0.4, scale: float = 0.9985) -> np.ndarray:
    height, width = _SHAPE
    import cv2

    rot = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_deg, scale)
    rot[:, 2] += [15.3, -10.2]
    return rot


def test_roundtrip_exact_inverse_is_zero():
    fwd = _warp()
    fwd3 = np.vstack([fwd, [0.0, 0.0, 1.0]])
    rev = np.linalg.inv(fwd3)[:2]
    residual = roundtrip_max_px(fwd.ravel().tolist(), rev.ravel().tolist(), _SHAPE)
    assert residual < 1e-9


def test_roundtrip_detects_inconsistency():
    fwd = _warp()
    rev = np.linalg.inv(np.vstack([fwd, [0.0, 0.0, 1.0]]))[:2]
    rev[0, 2] += 5.0  # 5 px disagreement in x
    residual = roundtrip_max_px(fwd.ravel().tolist(), rev.ravel().tolist(), _SHAPE)
    assert residual == pytest.approx(5.0, rel=0.01)


def _pair_result(name: str, rmse: float, ssim: float, blur: float) -> PairResult:
    metrics = PairMetrics(
        mae=rmse * 0.7,
        rmse=rmse,
        psnr=30.0,
        local_mse=rmse * rmse * 0.1,
        local_rmse=rmse * 0.3,
        ssim=ssim,
        grad_corr=0.8,
        grad_energy_ratio=0.9,
        blur_sigma=blur,
        frc_resolution_px=4.0,
        frc_resolution_px_17=3.5,
        overlap_fraction=0.5,
        n_pixels=1000,
    )
    fwd = _warp()
    return PairResult(
        name=name,
        passed=True,
        metrics=metrics,
        reg_warp=fwd.ravel().tolist(),
        ref_shape=_SHAPE,
    )


def _summary(*results: PairResult) -> RunSummary:
    from pathlib import Path

    return RunSummary(
        results=list(results), unmatched=[], all_passed=True, out_dir=Path("out")
    )


def test_cross_direction_metrics_matches_by_name():
    fwd = _pair_result("a.png", rmse=0.04, ssim=0.9, blur=0.9)
    rev = _pair_result("a.png", rmse=0.03, ssim=0.92, blur=0.4)
    rows = cross_direction_metrics(_summary(fwd), _summary(rev))
    assert len(rows) == 1
    row = rows[0]
    assert row.name == "a.png"
    assert row.delta_rmse == pytest.approx(0.01)
    assert row.delta_ssim == pytest.approx(-0.02)
    assert row.delta_blur_sigma == pytest.approx(0.5)
    # Both warps are identical here, so the round-trip sees a real
    # (non-inverse) composition: just check it is finite and positive.
    assert row.roundtrip_max_px > 0.0


def test_cross_direction_metrics_skips_incomplete_pairs():
    complete = _pair_result("ok.png", rmse=0.04, ssim=0.9, blur=0.9)
    errored = PairResult(name="bad.png", passed=False, error="ECC diverged")
    rows = cross_direction_metrics(
        _summary(complete, errored), _summary(_pair_result("ok.png", 0.03, 0.92, 0.4))
    )
    assert [row.name for row in rows] == ["ok.png"]
