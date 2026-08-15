"""Image loading and cross-directory pairing of duplicate scans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

import cv2
import numpy as np

from scanner_comparison.core.imtypes import U16Image, as_u16, shape2

_GRAYSCALE_NDIM = 2


class ImageLoadError(ValueError):
    """Raised when an image cannot be read or has an unexpected format."""


@dataclass(frozen=True)
class ImagePair:
    """Two paths referring to the same physical scan digitized twice."""

    name: str
    path_a: Path
    path_b: Path

    def reversed(self) -> ImagePair:
        """The same pair with the directories exchanged (reverse direction).

        Returns:
            The pair with ``path_a`` and ``path_b`` swapped.

        """
        return ImagePair(name=self.name, path_a=self.path_b, path_b=self.path_a)


def load_image(path: Path) -> U16Image:
    """Load a 16-bit grayscale image from ``path``.

    Returns:
        The image as a 2-D ``uint16`` array.

    Raises:
        ImageLoadError: if the file is unreadable or not 16-bit grayscale.

    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        msg = f"Could not read image: {path}"
        raise ImageLoadError(msg)
    arr: npt.NDArray[np.generic] = np.asarray(raw)
    if arr.ndim != _GRAYSCALE_NDIM or arr.dtype != np.uint16:
        msg = (
            f"Expected a 16-bit grayscale image, got shape {shape2(arr)} "
            f"dtype {arr.dtype}: {path}"
        )
        raise ImageLoadError(msg)
    return as_u16(arr)


def find_pairs(dir_a: Path, dir_b: Path) -> tuple[list[ImagePair], list[str]]:
    """Pair PNGs present in both directories by identical file name.

    Returns:
        The matched pairs sorted by name, and the names found in only one of
        the two directories.

    """
    files_a = {p.name: p for p in sorted(dir_a.iterdir()) if _is_png(p)}
    files_b = {p.name: p for p in sorted(dir_b.iterdir()) if _is_png(p)}
    common = sorted(files_a.keys() & files_b.keys())
    unmatched = sorted(files_a.keys() ^ files_b.keys())
    pairs = [ImagePair(name, files_a[name], files_b[name]) for name in common]
    return pairs, unmatched


def _is_png(path: Path) -> bool:
    """Return whether ``path`` is a regular file with a .png extension.

    Returns:
        True for regular files ending in ``.png`` (any case).

    """
    return path.is_file() and path.suffix.lower() == ".png"
