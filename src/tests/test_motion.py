"""Tests for the motion-amplified difference artifact."""

from __future__ import annotations

import numpy as np

from scanner_comparison.imaging.motion import motion_amplified_diff


def test_motion_amplified_diff_vanishes_for_identical_images(phantom):
    img = phantom.astype(np.float32) / 65535.0
    mask = np.ones(img.shape, dtype=bool)
    diff = motion_amplified_diff(img, img, mask)
    assert np.abs(diff).max() < 0.02


def test_motion_amplification_lights_up_a_line_shift(phantom):
    # A 3 px horizontal shift of one row band — the "line shift midway
    # through the scan" pattern. The amplified diff in the band must carry
    # clearly more energy than the plain signed diff there.
    img = phantom.astype(np.float32) / 65535.0
    shifted = img.copy()
    shifted[200:252, :] = np.roll(img[200:252, :], 3, axis=1)
    mask = np.zeros(img.shape, dtype=bool)
    mask[50:450, 30:-30] = True
    plain = img - shifted
    amplified = motion_amplified_diff(img, shifted, mask)
    # Band interior, clear of the mask erosion and the band boundaries.
    interior = np.zeros_like(mask)
    interior[212:240, 60:-60] = True
    plain_energy = float(np.mean(np.abs(plain[interior])))
    amplified_energy = float(np.mean(np.abs(amplified[interior])))
    assert amplified_energy > 2.0 * plain_energy


def test_motion_amplified_diff_respects_mask(phantom):
    img = phantom.astype(np.float32) / 65535.0
    shifted = np.roll(img, 2, axis=1)
    mask = np.zeros(img.shape, dtype=bool)
    mask[100:400, 100:500] = True
    diff = motion_amplified_diff(img, shifted, mask)
    # Everything outside the mask is exactly zero, and the artifact erodes
    # the mask further so the amplified warp cannot pull in uncovered
    # pixels: the band just inside the raw mask is zero too.
    assert diff[~mask].max() == 0.0
    assert diff[100:116, :].max() == 0.0
    # The interior (phantom's vertical bar is at columns 300-304) lights up.
    assert np.abs(diff[200:300, 200:400]).max() > 0.05
