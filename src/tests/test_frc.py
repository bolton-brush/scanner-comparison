"""Fourier ring correlation / effective shared resolution tests.

The instrument is exercised on aligned phantom pairs with INDEPENDENT grain
(the real-data condition): the conserved-detail scale must react to the
simulated device blur but stay in a sane band.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from tests.conftest import make_phantom

from scanner_comparison.core.imtypes import as_f32
from scanner_comparison.imaging.frc import (
    FRC_THRESHOLD_HALF,
    FRC_THRESHOLD_SEVENTH,
    frc_curve,
    frc_resolution,
)

_DEVICE_BLUR_SIGMA = 1.2
_GRAIN_SIGMA = 1.0
_GRAIN_AMPLITUDE = 0.02
# Expected crossing band (px) for the fixture below; validated by the
# instrument probe (0.5 criterion crossed at 4.5 px, 1/7 at 3.9 px).
_RESOLUTION_LO_PX = 2.5
_RESOLUTION_HI_PX = 8.0


def _grainy_pair(
    device_blur: float = _DEVICE_BLUR_SIGMA, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligned phantom pair: second scan blurred, each with its own grain.

    Returns:
        The two float images and a full-frame mask minus a border band.

    """
    rng = np.random.default_rng(seed)
    a = make_phantom().astype(np.float32) / 65535.0
    b = cv2.GaussianBlur(a, (0, 0), device_blur) if device_blur > 0 else a

    def grain() -> np.ndarray:
        return cv2.GaussianBlur(
            rng.normal(0.0, _GRAIN_AMPLITUDE, a.shape).astype(np.float32),
            (0, 0),
            _GRAIN_SIGMA,
        )

    height, width = a.shape
    mask = np.zeros((height, width), dtype=bool)
    mask[80 : height - 80, 80 : width - 80] = True
    a_out = as_f32(np.clip(a + grain(), 0.0, 1.0))
    b_out = as_f32(np.clip(b + grain(), 0.0, 1.0))
    return a_out, b_out, mask


def test_frc_resolution_in_expected_band() -> None:
    """The 0.5 crossing lands in the probe-validated band for the fixture."""
    a, b, mask = _grainy_pair()
    curve = frc_curve(a, b, mask)
    resolution = frc_resolution(curve, FRC_THRESHOLD_HALF)
    assert resolution is not None
    assert _RESOLUTION_LO_PX <= resolution <= _RESOLUTION_HI_PX


def test_frc_resolution_17_le_half() -> None:
    """The 1/7 criterion crosses at higher frequency -> finer px value."""
    a, b, mask = _grainy_pair()
    curve = frc_curve(a, b, mask)
    half = frc_resolution(curve, FRC_THRESHOLD_HALF)
    seventh = frc_resolution(curve, FRC_THRESHOLD_SEVENTH)
    assert half is not None and seventh is not None
    assert seventh <= half


def test_frc_antisymmetry() -> None:
    """FRC is symmetric by construction: swapping the inputs is a no-op."""
    a, b, mask = _grainy_pair()
    forward = frc_resolution(frc_curve(a, b, mask), FRC_THRESHOLD_HALF)
    reverse = frc_resolution(frc_curve(b, a, mask), FRC_THRESHOLD_HALF)
    assert forward is not None and reverse is not None
    assert forward == pytest.approx(reverse)


def test_frc_dose_response() -> None:
    """A heavier device blur forces a coarser conserved-detail scale."""
    a, b_sharp, mask = _grainy_pair(device_blur=0.5)
    _, b_blurred, _ = _grainy_pair(device_blur=2.5)
    fine = frc_resolution(frc_curve(a, b_sharp, mask), FRC_THRESHOLD_HALF)
    coarse = frc_resolution(frc_curve(a, b_blurred, mask), FRC_THRESHOLD_HALF)
    assert fine is not None and coarse is not None
    assert coarse > fine


def test_frc_identical_images_hit_nyquist_floor() -> None:
    """Identical images never cross the criterion: resolution = 2 px."""
    a, _, mask = _grainy_pair(device_blur=0.0)
    resolution = frc_resolution(frc_curve(a, a, mask), FRC_THRESHOLD_HALF)
    assert resolution == 2.0


def test_frc_degenerate_mask_returns_none() -> None:
    """A mask too small to evaluate yields no resolution (not a crash)."""
    a, b, mask = _grainy_pair()
    tiny = np.zeros_like(mask)
    resolution = frc_resolution(frc_curve(a, b, tiny), FRC_THRESHOLD_HALF)
    assert resolution is None
