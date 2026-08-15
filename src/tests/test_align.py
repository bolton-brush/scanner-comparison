"""Tests for Euclidean registration against a known ground-truth transform."""

from __future__ import annotations

import numpy as np

from scanner_comparison.imaging.align import geometric_overlap_mask, register, warp_image

from conftest import SIM_SCALE, ground_truth_warp, simulate_second_scanner


def test_register_recovers_known_transform(phantom, phantom_b):
    registration = register(phantom, phantom_b, max_dim=512)
    expected = ground_truth_warp(phantom.shape)
    # Linear part: within 1% relative; translation: within 2 px.
    np.testing.assert_allclose(
        registration.warp[:, :2], expected[:, :2], rtol=0.01, atol=0.01
    )
    np.testing.assert_allclose(registration.warp[:, 2], expected[:, 2], atol=2.0)
    assert registration.correlation > 0.9


def test_warped_images_match_after_registration(phantom, phantom_b):
    registration = register(phantom, phantom_b, max_dim=512)
    warped = warp_image(phantom_b, registration.warp, phantom.shape)
    mask = geometric_overlap_mask(phantom_b.shape, registration.warp, phantom.shape)
    a = phantom[mask].astype(np.float64)
    b = warped[mask].astype(np.float64)
    corr = np.corrcoef(a, b)[0, 1]
    assert corr > 0.99


def test_overlap_mask_excludes_borders(phantom, phantom_b):
    registration = register(phantom, phantom_b, max_dim=512)
    mask = geometric_overlap_mask(phantom_b.shape, registration.warp, phantom.shape)
    assert 0.7 < mask.mean() < 1.0
    # Translation (15, -10) and the cropped field of view: the top row and
    # right column of the reference frame fall outside the moving image's
    # coverage.
    assert not mask[0, :].any()
    assert not mask[:, -1].any()


def test_border_junk_does_not_change_recovered_warp(phantom):
    clean_b = simulate_second_scanner(phantom)
    junk_b = simulate_second_scanner(phantom, junk_border=True)
    clean = register(phantom, clean_b, max_dim=512)
    junk = register(phantom, junk_b, max_dim=512)
    # Border apparatus is masked out of the criterion, so the estimate must
    # be nearly identical to the clean case; the DoG coarse blur smears the
    # junk band edge past the mask by a pixel or so, hence the 1 px
    # translation tolerance.
    np.testing.assert_allclose(clean.warp[:, :2], junk.warp[:, :2], atol=1e-3)
    np.testing.assert_allclose(clean.warp[:, 2], junk.warp[:, 2], atol=1.0)


def test_scale_correction_recovers_known_similarity(phantom):
    scaled_b = simulate_second_scanner(phantom, scale=SIM_SCALE)
    registration = register(phantom, scaled_b, max_dim=512, scale_correction=SIM_SCALE)
    expected = ground_truth_warp(phantom.shape, SIM_SCALE)
    # Linear part: within 1% relative; translation: within 2 px.
    np.testing.assert_allclose(
        registration.warp[:, :2], expected[:, :2], rtol=0.01, atol=0.01
    )
    np.testing.assert_allclose(registration.warp[:, 2], expected[:, 2], atol=2.0)
    assert registration.scale == SIM_SCALE
    assert registration.correlation > 0.9


def test_scale_correction_improves_correlation(phantom):
    scaled_b = simulate_second_scanner(phantom, scale=SIM_SCALE)
    uncorrected = register(phantom, scaled_b, max_dim=512)
    corrected = register(phantom, scaled_b, max_dim=512, scale_correction=SIM_SCALE)
    # A Euclidean warp cannot absorb a uniform scale difference, so applying
    # the matching calibration constant must align measurably better.
    assert corrected.correlation > uncorrected.correlation
    assert uncorrected.scale == 1.0


def test_scale_correction_defaults_to_plain_euclidean(phantom, phantom_b):
    registration = register(phantom, phantom_b, max_dim=512)
    assert registration.scale == 1.0
    # The linear part stays orthonormal: no scale leaks in by default.
    linear = registration.warp[:, :2]
    np.testing.assert_allclose(
        linear @ linear.T, np.eye(2), rtol=1e-4, atol=1e-4
    )
