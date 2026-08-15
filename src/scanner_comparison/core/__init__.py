"""Foundations: typed image arrays, the OpenCV boundary, and image IO."""
from __future__ import annotations

from scanner_comparison.core.imtypes import (
    BoolMask,
    F32Image,
    F64Image,
    U8Image,
    U16Image,
    Warp,
    as_f32,
    as_f64,
    as_optional_array,
    as_optional_u8,
    as_u8,
    as_u16,
    erode_mask,
    shape2,
    to_u8,
)
from scanner_comparison.core.io import (
    ImageLoadError,
    ImagePair,
    find_pairs,
    load_image,
)

__all__ = [
    "BoolMask",
    "F32Image",
    "F64Image",
    "ImageLoadError",
    "ImagePair",
    "U8Image",
    "U16Image",
    "Warp",
    "as_f32",
    "as_f64",
    "as_optional_array",
    "as_optional_u8",
    "as_u8",
    "as_u16",
    "erode_mask",
    "find_pairs",
    "load_image",
    "shape2",
    "to_u8",
]
