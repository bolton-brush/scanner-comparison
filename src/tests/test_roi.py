"""Tests for the film region-of-interest mask (border + beveled corners)."""

from __future__ import annotations

from scanner_comparison.imaging.align import film_roi_mask


def test_roi_excludes_borders_and_corners():
    shape = (1000, 1200)
    margin = 0.03
    corner = 0.08
    mask = film_roi_mask(shape, margin=margin, corner=corner)
    m = round(min(shape) * margin)  # 30
    c = round(min(shape) * corner)  # 80

    assert not mask[0, :].any()
    assert not mask[-1, :].any()
    assert not mask[:, 0].any()
    assert not mask[:, -1].any()
    # Deep inside the beveled-off corner triangle: excluded.
    assert not mask[m + 5, m + 5]
    assert not mask[m + 5, mask.shape[1] - m - 6]
    assert not mask[mask.shape[0] - m - 6, m + 5]
    assert not mask[mask.shape[0] - m - 6, mask.shape[1] - m - 6]
    # Just past the bevel hypotenuse (legs sum to > c): kept.
    assert mask[m + 50, m + 40]
    assert mask[m + 40, mask.shape[1] - m - 51]
    # Center is kept.
    assert mask[500, 600]
    # The ROI still covers the bulk of the frame.
    assert 0.85 < mask.mean() < 0.97


def test_roi_zero_margins_keep_full_frame():
    mask = film_roi_mask((500, 700), margin=0.0, corner=0.0)
    assert mask.all()


def test_roi_corner_only():
    mask = film_roi_mask((1000, 1000), margin=0.0, corner=0.10)
    assert mask[0, 500]
    assert mask[500, 0]
    assert not mask[20, 20]
