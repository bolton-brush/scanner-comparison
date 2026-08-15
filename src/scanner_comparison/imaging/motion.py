"""Residual-motion visualization via a motion-amplified difference image.

After Euclidean (+ optional scale) registration, small residual
displacements can remain — e.g. non-linear scanner transport artifacts like
a line shift midway through a scan. At 1-2 px these barely show in a plain
difference image. This module estimates the residual displacement field
between the aligned rank images with dense optical flow (DIS), then
re-warps the moving image by an AMPLIFIED multiple of that field: wherever
a residual displacement exists, the signed difference against the reference
lights up as strong red/blue structure, while intensity-only differences
(grain, response noise) are unaffected because only geometry is amplified.

Strictly a diagnostic artifact: the amplified image is never fed to the
metrics, so the visualization cannot hide or manufacture information-loss
signal.
"""

from __future__ import annotations

import cv2
import numpy as np

from scanner_comparison.core.imtypes import (
    BoolMask,
    F32Image,
    U8Image,
    as_f32,
    erode_mask,
    shape2,
    to_u8,
)

# Light prefilter before flow estimation: the rank images carry each
# scanner's irreproducible film grain, which would otherwise drive spurious
# per-pixel flow.
_FLOW_PREFILTER_SIGMA = 1.0
# Multiplier applied to the residual displacement field before re-warping
# the moving image. 5x turns a 1-2 px residual into a 5-10 px displacement,
# which the signed difference then renders as strong red/blue structure.
_MOTION_AMPLIFICATION = 5.0
# Extra mask erosion for the artifact: the amplified warp samples the moving
# image up to ~amplification * max-residual pixels beyond the nominal
# position, and the metric mask's own erosion barely covers that excursion.
_ARTIFACT_MASK_ERODE_PX = 16


def motion_amplified_diff(
    reference: F32Image,
    aligned: F32Image,
    mask: BoolMask,
    *,
    amplification: float = _MOTION_AMPLIFICATION,
) -> F32Image:
    """Signed difference after amplifying the residual motion field.

    Estimates the dense residual displacement from ``reference`` to
    ``aligned`` with DIS optical flow, re-warps ``aligned`` by
    ``amplification`` times that field, and returns the signed difference
    ``reference - warped`` restricted to the (further eroded) mask.

    Returns:
        The motion-amplified signed difference image (same shape).

    """
    flow = _residual_flow(reference, aligned)
    height, width = shape2(reference)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x = as_f32(grid_x + amplification * flow[..., 0])
    map_y = as_f32(grid_y + amplification * flow[..., 1])
    warped = as_f32(
        cv2.remap(
            aligned,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    )
    diff = as_f32(reference - warped)
    diff[~erode_mask(mask, _ARTIFACT_MASK_ERODE_PX)] = 0.0
    return diff


def _residual_flow(reference: F32Image, aligned: F32Image) -> F32Image:
    """Dense residual displacement field from reference to aligned.

    DIS requires 8-bit input; the rank images are (0, 1)-uniform over the
    overlap, so the quantization costs little. ``DISOpticalFlow_create`` is
    absent from OpenCV's stubs despite existing at runtime, hence the narrow
    ignores and the ``object`` hop that contains the unknown types.

    Returns:
        The flow field as an (H, W, 2) float32 image (dx, dy per pixel).

    """
    dis = cv2.DISOpticalFlow_create(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    )
    flow: object = dis.calc(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        _flow_input(reference), _flow_input(aligned), None
    )
    return as_f32(flow)  # pyright: ignore[reportUnknownArgumentType]


def _flow_input(img: F32Image) -> U8Image:
    """Prefilter and quantize a rank image to 8-bit for the flow estimator.

    Returns:
        The blurred uint8 image.

    """
    smoothed = as_f32(cv2.GaussianBlur(img, (0, 0), _FLOW_PREFILTER_SIGMA))
    return to_u8(smoothed)
