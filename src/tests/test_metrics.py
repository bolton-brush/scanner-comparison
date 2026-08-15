"""Tests for the metric computations on controlled inputs."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from scanner_comparison.imaging.metrics import compute_metrics


def _full_mask(shape: tuple[int, int]) -> np.ndarray:
    return np.ones(shape, dtype=bool)


def test_identical_images_score_perfectly(phantom):
    img = phantom.astype(np.float32) / 65535.0
    mask = _full_mask(img.shape)
    metrics, diff, ssim_map, _frc = compute_metrics(img, img, mask)
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.ssim > 0.999
    assert metrics.grad_corr > 0.999
    assert metrics.grad_energy_ratio == pytest.approx(1.0)
    assert diff.max() == 0.0
    assert ssim_map.shape == img.shape


def test_blur_lowers_gradient_energy_and_ssim(phantom):
    img = phantom.astype(np.float32) / 65535.0
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    mask = _full_mask(img.shape)
    metrics, _, _, _frc = compute_metrics(img, blurred, mask)
    assert metrics.grad_energy_ratio < 0.9
    assert metrics.ssim < 0.99
    assert metrics.rmse > 0.005


def test_grad_corr_handles_constant_images():
    img = np.full((32, 32), 0.5, dtype=np.float32)
    mask = _full_mask(img.shape)
    metrics, _, _, _frc = compute_metrics(img, img, mask)
    assert metrics.grad_corr == 1.0
    assert metrics.grad_energy_ratio == 1.0


def test_metrics_respect_mask(phantom):
    img = phantom.astype(np.float32) / 65535.0
    other = img.copy()
    other[:128, :128] = 1.0 - other[:128, :128]  # big local difference
    mask = _full_mask(img.shape)
    mask[:128, :128] = False  # ...but masked out
    masked, _, _, _frc = compute_metrics(img, other, mask)
    assert masked.rmse == pytest.approx(0.0, abs=1e-7)
    unmasked, _, _, _frc = compute_metrics(img, other, _full_mask(img.shape))
    assert unmasked.rmse > 0.05


def test_local_mse_vanishes_for_pixel_noise():
    # Sign-alternating per-pixel differences (grain/focus-like) average to
    # zero under the local blur, so local_rmse collapses while rmse stays.
    img = np.full((96, 96), 0.5, dtype=np.float32)
    checker = np.indices(img.shape).sum(axis=0) % 2 * 2 - 1  # +-1 checkerboard
    other = img + 0.05 * checker.astype(np.float32)
    metrics, _, _, _frc = compute_metrics(img, other, _full_mask(img.shape))
    assert metrics.rmse == pytest.approx(0.05, rel=1e-5)
    assert metrics.local_rmse < 0.2 * metrics.rmse


def test_local_mse_persists_for_coherent_difference():
    # A uniform local bias is spatially coherent: the local average keeps
    # it, so local_rmse stays at the raw rmse level.
    img = np.full((96, 96), 0.5, dtype=np.float32)
    other = img + 0.05
    metrics, _, _, _frc = compute_metrics(img, other, _full_mask(img.shape))
    assert metrics.rmse == pytest.approx(0.05, rel=1e-5)
    assert metrics.local_rmse == pytest.approx(metrics.rmse, rel=0.05)


def test_local_mse_zero_for_identical_images(phantom):
    img = phantom.astype(np.float32) / 65535.0
    metrics, _, _, _frc = compute_metrics(img, img, _full_mask(img.shape))
    assert metrics.local_mse == 0.0
    assert metrics.local_rmse == 0.0


def test_blur_sigma_recovers_known_blur(phantom):
    img = phantom.astype(np.float32) / 65535.0
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    metrics, _, _, _frc = compute_metrics(img, blurred, _full_mask(img.shape))
    assert 1.5 < metrics.blur_sigma < 2.5


def test_blur_sigma_negative_when_second_scan_is_sharper(phantom):
    img = phantom.astype(np.float32) / 65535.0
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    # Reference blurred, second scan sharp: the signed gap flips sign
    # (blurring the reference cannot match; blurring the second scan can).
    metrics, _, _, _frc = compute_metrics(blurred, img, _full_mask(img.shape))
    assert -2.5 < metrics.blur_sigma < -1.5


def test_blur_gap_is_exactly_antisymmetric(phantom):
    img = phantom.astype(np.float32) / 65535.0
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    mask = _full_mask(img.shape)
    forward, _, _, _frc = compute_metrics(img, blurred, mask)
    reverse, _, _, _frc = compute_metrics(blurred, img, mask)
    assert forward.blur_sigma == pytest.approx(-reverse.blur_sigma, abs=1e-9)


def test_blur_sigma_zero_for_identical_images(phantom):
    img = phantom.astype(np.float32) / 65535.0
    metrics, _, _, _frc = compute_metrics(img, img, _full_mask(img.shape))
    assert metrics.blur_sigma == 0.0
