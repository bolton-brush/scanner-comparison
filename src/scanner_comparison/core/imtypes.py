"""Shared numpy array type aliases and OpenCV-to-numpy converters.

OpenCV's stub return type ``MatLike`` does not assign cleanly to typed numpy
arrays; the small ``as_*`` helpers contain that boundary in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, cast

if TYPE_CHECKING:
    import numpy.typing as npt

import cv2
import numpy as np

U16Image: TypeAlias = "npt.NDArray[np.uint16]"
U8Image: TypeAlias = "npt.NDArray[np.uint8]"
F32Image: TypeAlias = "npt.NDArray[np.float32]"
F64Image: TypeAlias = "npt.NDArray[np.float64]"
BoolMask: TypeAlias = "npt.NDArray[np.bool_]"
Warp: TypeAlias = "npt.NDArray[np.float64]"


def as_u16(mat: object) -> U16Image:
    """View an OpenCV matrix result as a typed uint16 array.

    Returns:
        The buffer as ``NDArray[uint16]`` (no copy when already correct).

    """
    return np.asarray(mat, dtype=np.uint16)


def as_u8(mat: object) -> U8Image:
    """View an OpenCV matrix result as a typed uint8 array.

    Returns:
        The buffer as ``NDArray[uint8]`` (no copy when already correct).

    """
    return np.asarray(mat, dtype=np.uint8)


def as_f32(mat: object) -> F32Image:
    """View an OpenCV matrix result as a typed float32 array.

    Returns:
        The buffer as ``NDArray[float32]`` (no copy when already correct).

    """
    return np.asarray(mat, dtype=np.float32)


def as_f64(mat: object) -> F64Image:
    """View an OpenCV matrix result as a typed float64 array.

    Returns:
        The buffer as ``NDArray[float64]`` (no copy when already correct).

    """
    return np.asarray(mat, dtype=np.float64)


def as_optional_array(mat: object) -> npt.NDArray[np.generic] | None:
    """Narrow an OpenCV result that is ``None`` at runtime on failure.

    OpenCV's stubs declare always-present returns, but e.g.
    ``Feature2D.detectAndCompute`` yields ``None`` descriptors when no
    keypoints are found, and ``estimateAffinePartial2D`` yields ``None``
    when no consistent transform exists.

    Returns:
        The matrix as an ndarray, or None when OpenCV returned None.

    """
    if mat is None:
        return None
    result: npt.NDArray[np.generic] = np.asarray(mat)
    return result


def as_optional_u8(mat: object) -> U8Image | None:
    """Narrow an OpenCV uint8 result that is ``None`` at runtime on failure.

    ``Feature2D.detectAndCompute`` yields ``None`` descriptors when no
    keypoints are found, despite stubs declaring an always-present return.

    Returns:
        The matrix as ``NDArray[uint8]``, or None when OpenCV returned None.

    """
    if mat is None:
        return None
    return np.asarray(mat, dtype=np.uint8)


def shape2(img: npt.NDArray[np.generic]) -> tuple[int, int]:
    """Concrete ``(height, width)`` of a 2-D image array.

    ``npt.NDArray``'s shape type parameter defaults to ``Any``, so ``.shape``
    is loosely typed; this narrows it in one place.

    Returns:
        ``(height, width)`` as plain Python ints.

    """
    return cast("tuple[int, int]", img.shape)


def to_u8(img: F32Image) -> U8Image:
    """Convert a normalized float image to 8-bit, clipping to [0, 1] first.

    Returns:
        The image quantized to uint8.

    """
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def erode_mask(mask: BoolMask, pixels: int) -> BoolMask:
    """Erode a boolean mask by ``pixels`` using a 3x3 kernel.

    Returns:
        The eroded mask; unchanged when ``pixels`` is zero.

    """
    if pixels <= 0:
        return mask
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = as_u8(cv2.erode(mask.astype(np.uint8), kernel, iterations=pixels))
    return eroded.astype(bool)
