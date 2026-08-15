"""Tests for image loading and filename pairing."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from scanner_comparison.core.io import ImageLoadError, find_pairs, load_image


def test_find_pairs_matches_by_name(pair_dirs):
    dir_a, dir_b = pair_dirs
    pairs, unmatched = find_pairs(dir_a, dir_b)
    assert [p.name for p in pairs] == ["case1.png"]
    assert unmatched == ["only_in_a.png"]
    assert pairs[0].path_a.parent == dir_a
    assert pairs[0].path_b.parent == dir_b


def test_load_image_reads_uint16(pair_dirs):
    dir_a, _ = pair_dirs
    img = load_image(dir_a / "case1.png")
    assert img.dtype == np.uint16
    assert img.ndim == 2


def test_load_image_rejects_8bit(tmp_path):
    path = tmp_path / "eight.png"
    cv2.imwrite(str(path), np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(ImageLoadError, match="16-bit grayscale"):
        load_image(path)


def test_load_image_missing_file(tmp_path):
    with pytest.raises(ImageLoadError, match="Could not read"):
        load_image(tmp_path / "missing.png")
