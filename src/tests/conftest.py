"""Shared fixtures: synthetic phantoms and a second-scanner simulator.

No real scan data is committed; the phantom has smooth regions, sharp edges,
and fine texture so that SSIM and gradient metrics behave realistically.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

PHANTOM_SHAPE = (512, 640)
# Fraction of the reference width covered by the simulated scanner's field
# of view; the cropped columns keep the pair partially overlapping.
SIM_FOV_WIDTH_FRAC = 0.96
SIM_ROTATION_DEG = 0.4
SIM_TRANSLATE_X = 15.0
SIM_TRANSLATE_Y = -10.0
SIM_GAIN = 45_000.0
SIM_OFFSET = 16_000.0
# Uniform scale difference standing in for a scanner-pair ppi miscalibration;
# matches the ~0.15% measured between the two real digitizers.
SIM_SCALE = 0.9985


def make_phantom(seed: int = 42, shape: tuple[int, int] = PHANTOM_SHAPE) -> np.ndarray:
    """Deterministic uint16 phantom with blobs, edges, and fine texture."""
    rng = np.random.default_rng(seed)
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    img = 0.15 + 0.10 * (xx / width) + 0.05 * (yy / height)
    for _ in range(24):
        cy = rng.uniform(0.1, 0.9) * height
        cx = rng.uniform(0.1, 0.9) * width
        sigma = rng.uniform(8.0, 60.0)
        amp = rng.uniform(0.2, 0.8)
        img = img + amp * np.exp(
            -(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))
        )
    img[100:104, 50:-50] = 0.9  # sharp horizontal bar
    img[50:-50, 300:304] = 0.85  # sharp vertical bar
    rr = (yy - 380.0) ** 2 + (xx - 480.0) ** 2
    img[rr < 30.0**2] = 0.95  # disk
    img[rr < 12.0**2] = 0.05  # disk core
    # Band-limited texture: fine enough for SSIM/gradient metrics to react
    # to, but smooth enough to survive interpolation during registration.
    texture = rng.normal(0.0, 0.02, size=shape).astype(np.float32)
    img = img + cv2.GaussianBlur(texture, (0, 0), 1.0)
    return (np.clip(img, 0.0, 1.0) * 65535.0).astype(np.uint16)


def ground_truth_warp(shape: tuple[int, int], scale: float = 1.0) -> np.ndarray:
    """Forward similarity transform (reference -> second scanner coords)."""
    height, width = shape
    rot = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), SIM_ROTATION_DEG, scale
    )
    rot3 = np.vstack([rot, [0.0, 0.0, 1.0]])
    translate = np.array(
        [[1.0, 0.0, SIM_TRANSLATE_X], [0.0, 1.0, SIM_TRANSLATE_Y], [0.0, 0.0, 1.0]],
    )
    forward = translate @ rot3
    return forward[:2]


def simulate_second_scanner(
    img: np.ndarray,
    *,
    blur_sigma: float = 0.0,
    junk_border: bool = False,
    scale: float = 1.0,
) -> np.ndarray:
    """Simulate a second digitizer: known similarity warp + gain/offset.

    ``warpAffine`` without ``WARP_INVERSE_MAP`` internally inverts the given
    matrix, so passing the content-mapping transform ``F`` (reference ->
    scanner B coordinates) directly yields ``B(x) = A(F^-1 x)`` as intended.

    ``scale`` adds a uniform scale difference between the scanners (a ppi
    miscalibration); ``register(scale_correction=scale)`` should undo it.

    With ``junk_border``, a bright frame and fake corner content are added,
    mimicking scanner-specific border apparatus that must not affect the
    comparison when border/corner margins are enabled.
    """
    height, width = img.shape
    forward = ground_truth_warp((height, width), scale)
    out_width = int(round(width * SIM_FOV_WIDTH_FRAC))
    warped = cv2.warpAffine(
        img.astype(np.float32),
        forward.astype(np.float64),
        (out_width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    if junk_border:
        # Kept inside the default border/corner margins (3% / 8%) so the
        # margins-enabled test exercises exactly the excluded region.
        out_height = height
        band = round(min(out_height, out_width) * 0.02)
        warped[:band, :] = 60_000.0
        warped[-band:, :] = 58_000.0
        warped[:, :band] = 59_000.0
        warped[:, -band:] = 57_000.0
        corner = round(min(out_height, out_width) * 0.05)
        warped[:corner, :corner] = 55_000.0
        warped[-corner:, -corner:] = 54_000.0
    if blur_sigma > 0.0:
        warped = cv2.GaussianBlur(warped, (0, 0), blur_sigma)
    rescaled = warped / 65535.0 * SIM_GAIN + SIM_OFFSET
    return np.clip(rescaled, 0.0, 65535.0).astype(np.uint16)


@pytest.fixture()
def phantom() -> np.ndarray:
    """The reference 'scan A'."""
    return make_phantom()


@pytest.fixture()
def phantom_b(phantom: np.ndarray) -> np.ndarray:
    """The same phantom as digitized by 'scanner B' (lossless simulation)."""
    return simulate_second_scanner(phantom)


@pytest.fixture()
def pair_dirs(tmp_path, phantom, phantom_b):
    """Two directories holding one matching pair plus one unmatched file."""
    dir_a = tmp_path / "scanner_a"
    dir_b = tmp_path / "scanner_b"
    dir_a.mkdir()
    dir_b.mkdir()
    cv2.imwrite(str(dir_a / "case1.png"), phantom)
    cv2.imwrite(str(dir_b / "case1.png"), phantom_b)
    cv2.imwrite(str(dir_a / "only_in_a.png"), phantom)
    return dir_a, dir_b
