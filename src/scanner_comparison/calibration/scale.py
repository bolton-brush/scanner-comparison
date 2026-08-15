"""Scale calibration: estimate the scanner-pair scale correction constant.

Two nominally identical-resolution digitizers typically differ by a small
uniform scale (~0.1-0.2%). The ECC criterion stays Euclidean by design, so
the constant is solved by a sweep: for each candidate correction, the
library's registration (pre-scale + Euclidean ECC) is run on a representative
subset of pairs, and the masked log-DoG feature NCC at the resulting
composite warp is averaged. The best candidate — refined by a parabola
through its neighbors — is the calibration constant (the curve is flat near
the optimum, so the refined value is reported alongside the grid best).

This productizes the diag5-diag9 investigation: scale couples with
rotation/translation at first order, so post-hoc sweeps on a converged warp
or per-pair ORB estimates are unreliable; the sweep re-optimizes the full
registration per candidate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, SupportsFloat, cast

import cv2
import numpy as np

from scanner_comparison.core.imtypes import (
    BoolMask,
    F32Image,
    Warp,
    as_f32,
    as_u8,
    shape2,
)
from scanner_comparison.core.io import load_image
from scanner_comparison.imaging.align import (
    RoiOptions,
    geometric_overlap_mask,
    norm_for_registration,
    preprocess_log_dog,
    register,
    single_alignment_mask,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from scanner_comparison.core.io import ImagePair

_SCALE_VERSION = 1
_MIN_NCC_PIXELS = 10_000
# Below this determinant the three-point parabola fit is degenerate.
_PARABOLA_DENOM_EPS = 1e-18


class ScaleCalibrationError(ValueError):
    """Raised when a scale calibration cannot be computed from the data."""


@dataclass(frozen=True)
class ScaleCalibration:
    """A solved scanner-pair scale constant with its sweep evidence."""

    scale: float
    scale_refined: float
    candidates: tuple[float, ...]
    mean_ncc: tuple[float, ...]
    pairs_used: tuple[str, ...]


def masked_feature_ncc(
    ref_feat: F32Image,
    ref_mask: BoolMask,
    mov_feat: F32Image,
    mov_mask: BoolMask,
    warp: Warp,
) -> float:
    """Pearson correlation of log-DoG features over the aligned overlap.

    The moving feature image is warped by the (composite) registration warp;
    the correlation runs over the intersection of both alignment masks and
    the geometric overlap.

    Returns:
        The correlation; 0.0 when the overlap is empty or degenerate.

    """
    height, width = shape2(ref_feat)
    warped = as_f32(
        cv2.warpAffine(
            mov_feat,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    )
    warped_mask = (
        as_u8(
            cv2.warpAffine(
                mov_mask.astype(np.uint8),
                warp,
                (width, height),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        )
        > 0
    )
    overlap = geometric_overlap_mask(shape2(mov_feat), warp, shape2(ref_feat))
    mask = ref_mask & warped_mask & overlap
    if int(mask.sum()) < _MIN_NCC_PIXELS:
        return 0.0
    a = ref_feat[mask].astype(np.float64)
    b = warped[mask].astype(np.float64)
    a -= float(cast("SupportsFloat", a.mean()))
    b -= float(cast("SupportsFloat", b.mean()))
    denom = math.sqrt(
        float(cast("SupportsFloat", a @ a)) * float(cast("SupportsFloat", b @ b))
    )
    return float(cast("SupportsFloat", a @ b)) / denom if denom > 0 else 0.0


def solve_scale_correction(
    pairs: Sequence[ImagePair],
    *,
    candidates: Sequence[float],
    max_pairs: int = 5,
    max_dim: int = 1200,
    progress: Callable[[str], None] = print,
) -> ScaleCalibration:
    """Solve the scale correction constant by a registration sweep.

    Per candidate constant, every selected pair (up to ``max_pairs``,
    evenly spaced by name) is registered with that pre-scale and scored by
    masked feature NCC at the returned composite warp.

    Returns:
        The calibration: the best candidate, its parabolic refinement, and
        the sweep evidence.

    Raises:
        ScaleCalibrationError: if there are no pairs or no candidates.

    """
    if not pairs:
        msg = "No pairs to solve scale from"
        raise ScaleCalibrationError(msg)
    if not candidates:
        msg = "No scale candidates to evaluate"
        raise ScaleCalibrationError(msg)
    step = max(1, math.ceil(len(pairs) / max_pairs))
    selected = list(pairs[::step][:max_pairs])
    progress(f"  scale sweep: {len(candidates)} candidates x {len(selected)} pairs")
    totals = _ncc_totals(selected, candidates, max_dim, progress)
    mean_ncc = tuple(totals[s] / len(selected) for s in candidates)
    best_idx = max(range(len(candidates)), key=lambda i: mean_ncc[i])
    best = float(candidates[best_idx])
    return ScaleCalibration(
        scale=best,
        scale_refined=_parabolic_peak(candidates, mean_ncc, best_idx),
        candidates=tuple(float(c) for c in candidates),
        mean_ncc=mean_ncc,
        pairs_used=tuple(p.name for p in selected),
    )


def _ncc_totals(
    selected: Sequence[ImagePair],
    candidates: Sequence[float],
    max_dim: int,
    progress: Callable[[str], None],
) -> dict[float, float]:
    """Summed masked feature NCC per candidate over the selected pairs.

    Returns:
        A mapping of candidate constant to summed NCC.

    """
    totals = dict.fromkeys(candidates, 0.0)
    roi = RoiOptions()
    for pair in selected:
        ref = load_image(pair.path_a)
        mov = load_image(pair.path_b)
        ref_feat = preprocess_log_dog(ref, roi)
        mov_feat = preprocess_log_dog(mov, roi)
        ref_mask = single_alignment_mask(norm_for_registration(ref, roi), roi)
        mov_mask = single_alignment_mask(norm_for_registration(mov, roi), roi)
        progress(f"  {pair.name}")
        for s in candidates:
            reg = register(ref, mov, max_dim=max_dim, scale_correction=s)
            ncc = masked_feature_ncc(
                ref_feat,
                ref_mask.astype(bool),
                mov_feat,
                mov_mask.astype(bool),
                reg.warp,
            )
            totals[s] += ncc
            progress(f"    s={s:.5f} ncc={ncc:.4f} corr={reg.correlation:.4f}")
    return totals


def write_scale_calibration(path: Path, cal: ScaleCalibration) -> Path:
    """Write the scale calibration JSON (sweep curve + recommended constant).

    Returns:
        The written path.

    """
    payload = {
        "version": _SCALE_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "scale": cal.scale,
        "scale_refined": cal.scale_refined,
        "candidates": list(cal.candidates),
        "mean_ncc": list(cal.mean_ncc),
        "pairs_used": list(cal.pairs_used),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _written = path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_scale_calibration(path: Path) -> ScaleCalibration:
    """Read a scale calibration JSON written by ``write_scale_calibration``.

    The pipeline consumes the constant via ``CompareConfig.scale_correction``
    (the JSON itself is an inspectable record); this reader exists so
    tooling can load the artifact symmetrically with the other
    calibrations.

    Returns:
        The parsed calibration.

    Raises:
        ScaleCalibrationError: if the file is missing, malformed, or of an
            unsupported version.

    """
    try:
        raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Cannot read scale calibration: {path} ({exc})"
        raise ScaleCalibrationError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"Unsupported scale calibration file: {path}"
        raise ScaleCalibrationError(msg)
    payload = cast("dict[str, object]", raw)
    if payload.get("version") != _SCALE_VERSION:
        msg = f"Unsupported scale calibration file: {path}"
        raise ScaleCalibrationError(msg)
    try:
        calibration = ScaleCalibration(
            scale=float(cast("SupportsFloat", payload["scale"])),
            scale_refined=float(cast("SupportsFloat", payload["scale_refined"])),
            candidates=tuple(
                float(cast("SupportsFloat", c))
                for c in cast("list[object]", payload["candidates"])
            ),
            mean_ncc=tuple(
                float(cast("SupportsFloat", v))
                for v in cast("list[object]", payload["mean_ncc"])
            ),
            pairs_used=tuple(
                str(p) for p in cast("list[object]", payload["pairs_used"])
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"Malformed scale calibration file: {path} ({exc})"
        raise ScaleCalibrationError(msg) from exc
    return calibration


def _parabolic_peak(
    candidates: Sequence[float], mean_ncc: Sequence[float], best_idx: int
) -> float:
    """Parabola vertex through the best candidate and its two neighbors.

    Falls back to the grid best at the sweep edges or when the parabola is
    degenerate/ill-conditioned (a flat curve, which is expected near the
    optimum).

    Returns:
        The refined scale constant.

    """
    if best_idx == 0 or best_idx == len(candidates) - 1:
        return float(candidates[best_idx])
    x0, x1, x2 = (
        float(candidates[best_idx - 1]),
        float(candidates[best_idx]),
        float(candidates[best_idx + 1]),
    )
    y0, y1, y2 = mean_ncc[best_idx - 1], mean_ncc[best_idx], mean_ncc[best_idx + 1]
    # Vertex of the parabola through three points (equispaced or not).
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < _PARABOLA_DENOM_EPS:
        return x1
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    if a >= 0.0:
        return x1  # no maximum (flat or convex curve): keep the grid best
    b = (x0 * x0 * (y2 - y1) + x1 * x1 * (y0 - y2) + x2 * x2 * (y1 - y0)) / denom
    vertex = -b / (2.0 * a)
    if not (x0 <= vertex <= x2):
        return x1
    return vertex
