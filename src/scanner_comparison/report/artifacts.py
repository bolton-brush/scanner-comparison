"""Per-pair visual artifacts: heatmaps, overlays, and intermediate images."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

from scanner_comparison.core.imtypes import as_u8, to_u8

if TYPE_CHECKING:
    from scanner_comparison.core.imtypes import BoolMask, F32Image, U8Image
    from scanner_comparison.imaging.frc import FrcCurve

_DIFF_AMPLIFICATION = 5.0
# The locally averaged diff is a few times smaller than the raw diff
# (uncorrelated grain averages out), so it gets a stronger amplification to
# stay visible: saturation at |local diff| = 0.05.
_LOCAL_DIFF_AMPLIFICATION = 20.0
_SSIM_DEFECT_AMPLIFICATION = 10.0


@dataclass(frozen=True)
class PairImages:
    """Aligned, rank-transformed images and maps belonging to one pair.

    ``reference`` and ``aligned`` are percentile-rank images (uniform (0, 1)
    histogram over the masked overlap), and the signed diff
    (``reference - aligned``), its locally averaged version ``local_diff``
    (zeroed outside the erosion band where the blur would be contaminated),
    the motion-amplified signed diff ``motion_diff`` (residual displacement
    field amplified and re-warped; diagnostic only), SSIM map, and overlay
    are derived from them. The ``logdog_*`` fields are the band-pass feature
    images the registration matched, each in its own frame except
    ``logdog_aligned``, which is the moving one warped into the reference
    frame.
    """

    reference: F32Image
    aligned: F32Image
    mask: BoolMask
    diff: F32Image
    local_diff: F32Image
    motion_diff: F32Image
    ssim_map: F32Image
    logdog_reference: F32Image
    logdog_moving: F32Image
    logdog_aligned: F32Image


def write_pair_artifacts(
    out_dir: Path,
    name: str,
    images: PairImages,
) -> dict[str, Path]:
    """Write artifact images for one pair.

    Writes the signed difference heatmaps (red: reference brighter, blue:
    second scan brighter; raw, locally averaged, and motion-amplified), the
    alignment overlay, SSIM defect map, and the intermediate images
    (percentile-rank images, log-DoG registration features, and the metric
    mask).

    Returns:
        A mapping of artifact kind to the written path.

    Raises:
        OSError: if an artifact file cannot be written.

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem

    heatmap = _signed_heatmap(images.diff, images.mask, _DIFF_AMPLIFICATION)
    local_heatmap = _signed_heatmap(
        images.local_diff, images.mask, _LOCAL_DIFF_AMPLIFICATION
    )
    motion_heatmap = _signed_heatmap(
        images.motion_diff, images.mask, _DIFF_AMPLIFICATION
    )

    overlay = np.zeros((*images.mask.shape, 3), dtype=np.uint8)
    overlay[..., 2] = to_u8(images.reference)  # red channel: reference scan
    overlay[..., 1] = to_u8(images.aligned)  # green channel: second scan

    ssim_defect = cv2.applyColorMap(
        to_u8((1.0 - images.ssim_map) * _SSIM_DEFECT_AMPLIFICATION),
        cv2.COLORMAP_JET,
    )
    ssim_defect[~images.mask] = 0

    mask_u8 = as_u8(images.mask.astype(np.uint8) * 255)

    artifacts = {
        "diff_heatmap": out_dir / f"{stem}_diff.png",
        "local_diff_heatmap": out_dir / f"{stem}_local_diff.png",
        "motion_amplified_diff": out_dir / f"{stem}_motion_diff.png",
        "overlay": out_dir / f"{stem}_overlay.png",
        "ssim_map": out_dir / f"{stem}_ssim.png",
        "rank_reference": out_dir / f"{stem}_rank_ref.png",
        "rank_aligned": out_dir / f"{stem}_rank_mov.png",
        "logdog_reference": out_dir / f"{stem}_logdog_ref.png",
        "logdog_moving": out_dir / f"{stem}_logdog_mov.png",
        "logdog_aligned": out_dir / f"{stem}_logdog_aligned.png",
        "mask": out_dir / f"{stem}_mask.png",
    }
    images_by_kind = {
        "diff_heatmap": heatmap,
        "local_diff_heatmap": local_heatmap,
        "motion_amplified_diff": motion_heatmap,
        "overlay": overlay,
        "ssim_map": ssim_defect,
        "rank_reference": to_u8(images.reference),
        "rank_aligned": to_u8(images.aligned),
        "logdog_reference": to_u8(images.logdog_reference),
        "logdog_moving": to_u8(images.logdog_moving),
        "logdog_aligned": to_u8(images.logdog_aligned),
        "mask": mask_u8,
    }
    for kind, path in artifacts.items():
        if not cv2.imwrite(str(path), images_by_kind[kind]):
            msg = f"Failed to write artifact: {path}"
            raise OSError(msg)
    return artifacts


def write_frc_csv(out_dir: Path, name: str, curve: FrcCurve) -> Path:
    """Write the per-pair FRC curve CSV (``freq_cyc_per_px``, ``frc`` rows).

    Returns:
        The written path (``<name>_frc.csv`` next to the pair artifacts).

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    out = out_dir / f"{stem}_frc.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        _header = cast("int", writer.writerow(["freq_cyc_per_px", "frc"]))
        # Scalar indexing of numpy arrays is loosely typed; go via lists.
        freqs = cast("list[float]", curve.freq.tolist())
        corrs = cast("list[float]", curve.corr.tolist())
        for freq, corr in zip(freqs, corrs, strict=True):
            _row = cast("int", writer.writerow([f"{freq:.6f}", f"{corr:.6f}"]))
    return out


def _signed_heatmap(
    values: F32Image, mask: BoolMask, amplification: float
) -> U8Image:
    """Render a signed float map as a diverging red/blue heatmap.

    Red where the value is positive (reference brighter), blue where
    negative (second scan brighter), black at zero / outside the mask.
    (``to_u8`` clips to [0, 1], so each channel keeps only its own sign.)

    Returns:
        The BGR heatmap image.

    """
    amplified = values * amplification
    heatmap = np.zeros((*mask.shape, 3), dtype=np.uint8)
    heatmap[..., 2] = to_u8(amplified)
    heatmap[..., 0] = to_u8(-amplified)
    heatmap[~mask] = 0
    return heatmap
