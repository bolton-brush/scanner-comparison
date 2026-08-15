"""Tests for overlap normalization and gain/offset fitting."""

from __future__ import annotations

import numpy as np
import pytest

from scanner_comparison.imaging.normalize import (
    NormalizationError,
    apply_gain_offset,
    fit_gain_offset,
    percentile_rank,
    robust_rescale,
    surround_mask,
    tile_residual_stats,
)


def _ramp(shape=(64, 64)) -> np.ndarray:
    yy, _ = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (yy / (shape[0] - 1) * 65535.0).astype(np.float32)


def test_robust_rescale_maps_percentiles_to_unit_range():
    img = _ramp()
    mask = np.ones_like(img, dtype=bool)
    out, norm = robust_rescale(img, mask, p_low=0.0, p_high=100.0)
    assert norm.lower == 0.0
    assert norm.upper == 65535.0
    assert out.min() == 0.0
    assert out.max() == 1.0


def test_robust_rescale_zeros_outside_mask():
    img = _ramp()
    mask = np.zeros_like(img, dtype=bool)
    mask[16:48, 16:48] = True
    out, _ = robust_rescale(img, mask, p_low=0.0, p_high=100.0)
    assert out[~mask].sum() == 0.0
    assert out[mask].max() == 1.0


def test_fit_gain_offset_recovers_known_mapping():
    rng = np.random.default_rng(1)
    src = rng.uniform(0.1, 0.9, size=(128, 128)).astype(np.float32)
    gain, offset = 1.4, -0.2
    dst = src * gain + offset
    mask = np.ones_like(src, dtype=bool)
    est_gain, est_offset = fit_gain_offset(src, dst, mask)
    assert est_gain == pytest.approx(gain, rel=1e-3)
    assert est_offset == pytest.approx(offset, abs=1e-3)


def test_apply_gain_offset_clips_and_masks():
    img = np.full((8, 8), 0.8, dtype=np.float32)
    mask = np.ones_like(img, dtype=bool)
    mask[:2] = False
    out = apply_gain_offset(img, gain=2.0, offset=0.0, mask=mask)
    assert out[mask].max() == 1.0
    assert out[~mask].sum() == 0.0


def test_tile_residual_stats_uniform_residual():
    a = np.full((64, 64), 0.5, dtype=np.float32)
    b = a + 0.01
    mask = np.ones_like(a, dtype=bool)
    stats = tile_residual_stats(a, b, mask, tiles=4)
    assert stats.tiles_used == 16
    assert stats.mean < 0.01  # near-constant residual -> tiny per-tile std


def test_percentile_rank_uniformizes_histogram():
    rng = np.random.default_rng(0)
    img = rng.gamma(2.0, 1000.0, size=(64, 64)).astype(np.float32)  # skewed
    mask = np.ones_like(img, dtype=bool)
    out = percentile_rank(img, mask)
    values = out[mask]
    assert values.min() > 0.0
    assert values.max() < 1.0
    # Continuous inputs have no ties, so the midranks fall exactly uniformly.
    hist, _ = np.histogram(values, bins=8, range=(0.0, 1.0))
    assert (hist == hist[0]).all()


def test_percentile_rank_invariant_to_monotonic_transform():
    img = _ramp()
    mask = np.ones_like(img, dtype=bool)
    base = percentile_rank(img, mask)
    # Any strictly increasing response (here: gamma-like log) leaves the
    # rank image unchanged.
    transformed = percentile_rank(np.log1p(img), mask)
    np.testing.assert_allclose(base, transformed, atol=1e-6)


def test_percentile_rank_ties_share_average_rank():
    img = np.zeros((4, 4), dtype=np.float32)
    img[2, 2] = 5.0
    mask = np.ones_like(img, dtype=bool)
    out = percentile_rank(img, mask)
    # 15 tied zeros share the average rank of 0..14; the outlier takes 15.
    assert out[0, 0] == pytest.approx((7.0 + 0.5) / 16.0)
    assert out[2, 2] == pytest.approx((15.0 + 0.5) / 16.0)


def test_percentile_rank_zeros_outside_mask():
    img = _ramp()
    mask = np.zeros_like(img, dtype=bool)
    mask[16:48, 16:48] = True
    out = percentile_rank(img, mask)
    assert out[~mask].sum() == 0.0
    assert out[mask].min() > 0.0


def test_percentile_rank_empty_mask_raises():
    img = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros_like(img, dtype=bool)
    with pytest.raises(NormalizationError):
        percentile_rank(img, mask)


def test_surround_mask_excludes_border_dark_keeps_interior_dark():
    img = np.full((200, 200), 0.7, dtype=np.float32)
    img[:20, :] = 0.01  # dark band connected to the border (surround)
    img[80:120, 80:120] = 0.02  # dark pocket enclosed by bright content
    mask = np.ones_like(img, dtype=bool)
    result = surround_mask(img, mask)
    assert not result[5, 100]  # surround excluded
    assert result[100, 100]  # interior dark pocket kept
    assert result[150, 150]  # bright content kept


def test_surround_mask_respects_input_mask():
    img = np.full((100, 100), 0.7, dtype=np.float32)
    mask = np.ones_like(img, dtype=bool)
    mask[:50] = False
    result = surround_mask(img, mask)
    assert not result[:50].any()
    assert result[50:].all()
