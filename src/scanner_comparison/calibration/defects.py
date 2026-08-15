"""Detection of stationary vertical column (line) defects in a scanner.

A sensor/readout defect produces a bright or dark column spanning the whole
image height, recurring at the same sensor x in every scan. Per scan,
detection uses a full-height coherence test: the frame is split into
horizontal bands, each band's column-median profile is high-pass filtered
(median filter) and MAD z-scored, and a column is a candidate only when most
bands agree in sign and significance. Film content (trabecular texture,
label-strip edges) is not full-height coherent and is rejected by
construction; per-scan banding noise is not coherent either.

Scanners auto-crop, so PNG x is not a stable sensor coordinate: cross-scan
stationarity is established in two passes. Pass 1 anchors each scan's
high-z candidates onto the scan with the clearest defect set by a 1-D
translation search, and recurring columns are found by per-scan boolean
voting. Pass 2 re-anchors every scan against the pass-1 stationary
recurrence peaks (a noise-free target, tolerating weaker per-scan
candidates) and the vote is repeated. Only columns recurring in at least
half the anchored scans are reported as stationary hardware defects; each
scan's mask is the set of its own native candidate columns landing in those
groups, so per-scan banding (the scanner's information-loss character) is
never masked.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, SupportsFloat, SupportsInt, cast

import cv2
import numpy as np

# scipy does not type this function precisely; the return value is narrowed
# with an explicit cast at the call site instead.
from scipy.ndimage import (
    median_filter,  # pyright: ignore[reportUnknownVariableType]
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt

from scanner_comparison.core.imtypes import BoolMask, U16Image, shape2
from scanner_comparison.core.io import load_image

_MASK_VERSION = 1
_Z_THRESHOLD = 8.0
_ANCHOR_GATE_Z = 15.0
_MEDIAN_FILTER_PX = 11
_MAX_GROUP_WIDTH_PX = 4
_GROUP_GAP_PX = 2
_ANCHOR_TOL_PX = 2
_ANCHOR_SEARCH_PX = 60
_ANCHOR_MIN_MATCHES = 3
_ANCHOR_MIN_MARGIN = 2  # best delta must beat the runner-up by this many
_MIN_ANCHORED_SCANS = 5
_PASS2_MIN_MATCHES = 2
# The runner-up anchor score is taken at least this far from the best delta,
# so the guard measures a genuine alternative offset, not the peak's flank.
_RUNNER_UP_MIN_DISTANCE_PX = 5
_BANDS = 6
_BANDS_MIN_PASS = 4
_STATIONARITY_FRACTION = 0.5


class DefectMaskError(ValueError):
    """Raised when a defect-mask file is malformed or misses a directory."""


@dataclass(frozen=True)
class ScanCandidates:
    """Stage-1 result for one scan: candidate columns and their peak |z|."""

    columns: npt.NDArray[np.int64]
    peaks: npt.NDArray[np.float64]


@dataclass(frozen=True)
class _AnchorResult:
    """Per-scan anchoring outcome relative to the anchor target set."""

    delta: int
    matches: int
    runner_up: int
    anchored: bool


@dataclass(frozen=True)
class DefectScanInfo:
    """Per-scan defect record: anchoring offset and native defect columns."""

    name: str
    x_offset: int
    anchored: bool
    candidate_count: int
    defect_columns_native: list[int]


@dataclass(frozen=True)
class DirectoryDefects:
    """Stationary defect columns found in one scanner directory.

    ``stationary_columns_ref_frame`` holds ``(start, end, recurrence)``
    groups in the reference scan's frame; each scan's
    ``defect_columns_native`` are its own columns falling inside them.
    """

    directory: str
    reference_scan: str
    stationary_columns_ref_frame: list[tuple[int, int, int]]
    scans: dict[str, DefectScanInfo]

    def defect_columns(self, name: str) -> list[int]:
        """Native defect columns of one scan (empty when unknown/unanchored).

        Returns:
            The scan's native defect column indices, or an empty list when
            the scan is unknown or was not anchored.

        """
        info = self.scans.get(name)
        if info is None or not info.anchored:
            return []
        return info.defect_columns_native


@dataclass(frozen=True)
class DefectMaskData:
    """A two-directory defect mask as written to/read from JSON."""

    params: dict[str, float | int]
    directories: dict[str, DirectoryDefects]

    def for_directory(self, directory: Path) -> DirectoryDefects:
        """Look up the entry for ``directory`` by absolute path.

        Returns:
            The directory's defect record.

        Raises:
            DefectMaskError: if the mask file does not cover the directory.

        """
        key = str(directory.resolve())
        entry = self.directories.get(key)
        if entry is None:
            known = ", ".join(sorted(self.directories))
            msg = f"Defect mask has no entry for {key} (covers: {known})"
            raise DefectMaskError(msg)
        return entry


def column_defect_candidates(img: U16Image) -> ScanCandidates:
    """Stage 1: defect-candidate columns of one scan (full-height coherence).

    Returns:
        Candidate group-center column indices and each group's peak |z|.

    """
    return _candidates_from_z(_bands_z(img))


def defect_column_mask(shape: tuple[int, int], columns: Sequence[int]) -> BoolMask:
    """Boolean mask that is True on the given columns (full height).

    Returns:
        A boolean mask of ``shape``; out-of-range columns are ignored.

    """
    mask = np.zeros(shape, dtype=bool)
    for c in columns:
        if 0 <= c < shape[1]:
            mask[:, int(c)] = True
    return mask


def find_defects(
    dir_a: Path,
    dir_b: Path,
    *,
    progress: Callable[[str], None] = print,
) -> DefectMaskData:
    """Detect stationary column defects in both scanner directories.

    Returns:
        The defect-mask data for both directories.

    """
    params: dict[str, float | int] = {
        "z_threshold": _Z_THRESHOLD,
        "anchor_gate_z": _ANCHOR_GATE_Z,
        "median_filter_px": _MEDIAN_FILTER_PX,
        "max_group_width_px": _MAX_GROUP_WIDTH_PX,
        "group_gap_px": _GROUP_GAP_PX,
        "anchor_tol_px": _ANCHOR_TOL_PX,
        "anchor_search_px": _ANCHOR_SEARCH_PX,
        "anchor_min_matches": _ANCHOR_MIN_MATCHES,
        "anchor_min_margin": _ANCHOR_MIN_MARGIN,
        "min_anchored_scans": _MIN_ANCHORED_SCANS,
        "pass2_min_matches": _PASS2_MIN_MATCHES,
        "bands": _BANDS,
        "bands_min_pass": _BANDS_MIN_PASS,
        "stationarity_fraction": _STATIONARITY_FRACTION,
    }
    directories = {
        str(d.resolve()): _find_directory_defects(d, progress=progress)
        for d in (dir_a, dir_b)
    }
    return DefectMaskData(params=params, directories=directories)


def write_defect_mask(path: Path, data: DefectMaskData) -> Path:
    """Write the defect mask JSON (inspectable detection record).

    Returns:
        The written path.

    """
    payload = {
        "version": _MASK_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "params": data.params,
        "directories": {
            key: {
                "reference_scan": entry.reference_scan,
                "stationary_columns_ref_frame": [
                    [a, b, n] for a, b, n in entry.stationary_columns_ref_frame
                ],
                "scans": {
                    name: {
                        "x_offset": info.x_offset,
                        "anchored": info.anchored,
                        "candidate_count": info.candidate_count,
                        "defect_columns_native": info.defect_columns_native,
                    }
                    for name, info in sorted(entry.scans.items())
                },
            }
            for key, entry in sorted(data.directories.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _written = path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_defect_mask(path: Path) -> DefectMaskData:
    """Read a defect mask JSON written by ``write_defect_mask``.

    Returns:
        The parsed defect-mask data.

    Raises:
        DefectMaskError: if the file is missing, malformed, or of an
            unsupported version.

    """
    try:
        raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Cannot read defect mask: {path} ({exc})"
        raise DefectMaskError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"Unsupported defect mask file: {path}"
        raise DefectMaskError(msg)
    payload = cast("dict[str, object]", raw)
    if payload.get("version") != _MASK_VERSION:
        msg = f"Unsupported defect mask file: {path}"
        raise DefectMaskError(msg)
    try:
        params = cast("dict[str, float | int]", payload["params"])
        directories = {
            key: _read_directory_defects(key, entry)
            for key, entry in cast(
                "dict[str, dict[str, object]]", payload["directories"]
            ).items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"Malformed defect mask file: {path} ({exc})"
        raise DefectMaskError(msg) from exc
    return DefectMaskData(params=params, directories=directories)


def _read_directory_defects(key: str, entry: dict[str, object]) -> DirectoryDefects:
    """Parse one directory entry of the defect-mask JSON.

    Returns:
        The parsed directory defect record.

    """
    scans_raw = cast("dict[str, dict[str, object]]", entry["scans"])
    scans = {
        name: DefectScanInfo(
            name=name,
            x_offset=int(cast("SupportsInt", scan["x_offset"])),
            anchored=bool(scan["anchored"]),
            candidate_count=int(cast("SupportsInt", scan["candidate_count"])),
            defect_columns_native=[
                int(cast("SupportsInt", c))
                for c in cast("list[object]", scan["defect_columns_native"])
            ],
        )
        for name, scan in scans_raw.items()
    }
    groups_raw = cast("list[list[object]]", entry["stationary_columns_ref_frame"])
    groups = [
        (
            int(cast("SupportsInt", g[0])),
            int(cast("SupportsInt", g[1])),
            int(cast("SupportsInt", g[2])),
        )
        for g in groups_raw
    ]
    return DirectoryDefects(
        directory=key,
        reference_scan=str(entry["reference_scan"]),
        stationary_columns_ref_frame=groups,
        scans=scans,
    )


def write_defect_map_pngs(
    path: Path, data: DefectMaskData, *, dilate_px: int = 1
) -> dict[str, Path]:
    """Write one defect-map PNG per directory.

    The stationary defect groups are overlaid in red on the directory's
    reference scan, next to the mask JSON.

    Returns:
        A mapping of directory key to the written PNG path.

    Raises:
        OSError: if an image cannot be written.

    """
    written: dict[str, Path] = {}
    for key, entry in data.directories.items():
        img = load_image(Path(key) / entry.reference_scan).astype(np.float32)
        lo = float(np.percentile(img, 1.0))
        hi = float(np.percentile(img, 99.0))
        norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        vis = cv2.cvtColor((norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for a, b, _n in entry.stationary_columns_ref_frame:
            vis[:, max(0, a - dilate_px) : b + dilate_px + 1] = (0, 0, 255)
        safe = "".join(c if c.isalnum() else "-" for c in Path(key).name)
        out = path.with_name(f"{path.stem}_{safe}.png")
        if not cv2.imwrite(str(out), vis):
            msg = f"Failed to write defect map: {out}"
            raise OSError(msg)
        written[key] = out
    return written


def write_candidate_csvs(
    path: Path, data: DefectMaskData
) -> dict[str, Path]:
    """Write one per-scan candidate/anchoring CSV per directory.

    Returns:
        A mapping of directory key to the written CSV path.

    """
    written: dict[str, Path] = {}
    for key, entry in data.directories.items():
        safe = "".join(c if c.isalnum() else "-" for c in Path(key).name)
        out = path.with_name(f"{path.stem}_{safe}.csv")
        rows = [
            {
                "name": info.name,
                "candidate_count": info.candidate_count,
                "x_offset": info.x_offset,
                "anchored": info.anchored,
                "defect_columns_native": ";".join(
                    str(c) for c in info.defect_columns_native
                ),
            }
            for info in sorted(entry.scans.values(), key=lambda i: i.name)
        ]
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            # csv's stubs type this return as Any; narrow it with a cast.
            _header = cast("int", writer.writeheader())
            writer.writerows(rows)
        written[key] = out
    return written


def _find_directory_defects(
    directory: Path, *, progress: Callable[[str], None]
) -> DirectoryDefects:
    """Two-stage stationary column-defect detection for one directory.

    Returns:
        The directory's defect record (empty stationary set when too few
        scans can be anchored).

    """
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".png")
    per_scan: dict[str, ScanCandidates] = {}
    for path in paths:
        per_scan[path.name] = column_defect_candidates(load_image(path))
        progress(f"  {path.name}: {per_scan[path.name].columns.size} candidates")

    gated = {
        n: c.columns[c.peaks > _ANCHOR_GATE_Z]
        for n, c in per_scan.items()
    }
    # Reference = clearest defect set: most gated candidates, tiebreak z sum.
    ref_name = max(
        per_scan,
        key=lambda n: (
            gated[n].size,
            float(per_scan[n].peaks[per_scan[n].peaks > _ANCHOR_GATE_Z].sum()),
        ),
    )
    anchors: dict[str, _AnchorResult] = {}
    for name in per_scan:
        anchors[name] = (
            _AnchorResult(0, int(gated[name].size), 0, True)
            if name == ref_name
            else _anchor_offset(
                gated[name], gated[ref_name],
                min_matches=_ANCHOR_MIN_MATCHES, min_margin=_ANCHOR_MIN_MARGIN,
            )
        )
    anchored1 = sum(1 for a in anchors.values() if a.anchored)
    if anchored1 < _MIN_ANCHORED_SCANS:
        progress(f"  warning: only {anchored1} anchored scans")
        progress("  no stationary defects reported (too few anchored)")
        return DirectoryDefects(
            directory=str(directory.resolve()),
            reference_scan=ref_name,
            stationary_columns_ref_frame=[],
            scans={
                name: DefectScanInfo(name, 0, False, int(c.columns.size), [])
                for name, c in per_scan.items()
            },
        )

    _stationary1, peaks1 = _vote_stationary(per_scan, anchors)
    # Pass 2: re-anchor every scan against the stationary peaks using its
    # full candidate set; the clean target tolerates weaker candidates.
    for name, cand in per_scan.items():
        anchors[name] = _anchor_offset(
            cand.columns, peaks1,
            min_matches=_PASS2_MIN_MATCHES, min_margin=_ANCHOR_MIN_MARGIN,
        )
    stationary, _peaks = _vote_stationary(per_scan, anchors)
    anchored2 = sum(1 for a in anchors.values() if a.anchored)
    progress(f"  anchor reference: {ref_name}; {anchored2}/{len(per_scan)} anchored")
    progress(f"  stationary groups: {stationary}")

    scans: dict[str, DefectScanInfo] = {}
    for name, cand in per_scan.items():
        anchor = anchors[name]
        defect: list[int] = []
        if anchor.anchored:
            shifted = cast("list[int]", (cand.columns + anchor.delta).tolist())
            defect = sorted(
                c - anchor.delta
                for c in shifted
                if any(
                    a - _ANCHOR_TOL_PX <= c <= b + _ANCHOR_TOL_PX
                    for a, b, _n in stationary
                )
            )
        scans[name] = DefectScanInfo(
            name=name,
            x_offset=anchor.delta,
            anchored=anchor.anchored,
            candidate_count=int(cand.columns.size),
            defect_columns_native=defect,
        )
    return DirectoryDefects(
        directory=str(directory.resolve()),
        reference_scan=ref_name,
        stationary_columns_ref_frame=stationary,
        scans=scans,
    )


def _bands_z(img: U16Image) -> npt.NDArray[np.float32]:
    """Per-column defect z-score: full-height band-coherent MAD outlier.

    The frame is split into horizontal bands; per band the column-median
    profile is median-filter high-passed and MAD z-scored. A column's score
    is the signed minimum |z| over the bands that pass the z threshold with
    a consistent sign, or 0 when fewer than ``_BANDS_MIN_PASS`` bands pass:
    sensor defects span the whole image height; content texture does not.

    Returns:
        The per-column z array (float32, length = width).

    """
    height, width = shape2(img)
    edges = np.linspace(0, height, _BANDS + 1).astype(int)
    band_z: list[npt.NDArray[np.float32]] = []
    for b in range(_BANDS):
        band = img[edges[b] : edges[b + 1]]
        profile = np.median(band.astype(np.float32), axis=0)
        band_z.append(_profile_z(profile))
    zs = np.stack(band_z)  # (bands, width)
    # np.sign and fancy indexing are loosely typed in numpy's stubs; narrow
    # them with casts.
    signs = np.sign(zs)
    dominant = np.argmax(np.abs(zs), axis=0)
    sign = cast("npt.NDArray[np.float32]", signs[dominant, np.arange(width)])
    sign[~sign.astype(bool)] = 1.0  # exact zeros (flat columns) pick +
    passes = cast(
        "npt.NDArray[np.bool_]",
        (signs == sign[None, :]) & (np.abs(zs) > _Z_THRESHOLD),
    )
    strength = cast(
        "npt.NDArray[np.float32]",
        np.where(passes, np.abs(zs), np.inf).min(axis=0),
    )
    n_pass = cast("npt.NDArray[np.int64]", passes.sum(axis=0))
    out = np.where(n_pass >= _BANDS_MIN_PASS, sign * strength, 0.0)
    return np.asarray(out, dtype=np.float32)


def _profile_z(profile: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Median-filter high-pass MAD z-scores of one per-column profile.

    Returns:
        The z array (same length as ``profile``).

    """
    base = cast(
        "npt.NDArray[np.float32]",
        median_filter(profile, size=_MEDIAN_FILTER_PX, mode="nearest"),
    )
    resid = profile - base
    return np.asarray(resid / _mad_scale(resid), dtype=np.float32)


def _mad_scale(resid: npt.NDArray[np.float32]) -> float:
    """Robust residual scale, immune to degenerate constant runs.

    Flat surround regions put long runs of the residual at exactly 0.0; once
    they pass half the columns the plain MAD collapses to ~0 and z-scores
    explode, so the MAD is computed over nonzero residuals only (exact-zero
    test: the zeros come from constant runs, not from float noise). When the
    nonzero pool is too small to hold noise (nearly constant profile with
    isolated spikes), or degenerates itself, the noise floor is ~0 and any
    residual is significant: a tiny scale is returned.

    Returns:
        The residual's Gaussian-equivalent standard deviation estimate.

    """
    nonzero = resid[resid.astype(bool)]
    if nonzero.size < max(32, resid.size // 100):
        return 1e-12
    # numpy's scalar-reduction stubs are loose; narrow with casts.
    center = float(cast("SupportsFloat", np.median(nonzero)))
    mad = float(cast("SupportsFloat", np.median(np.abs(nonzero - center))))
    if mad <= 0.0:
        return 1e-12
    return 1.4826 * mad + 1e-12


def _candidates_from_z(z: npt.NDArray[np.float32]) -> ScanCandidates:
    """Group z-thresholded columns into narrow candidate groups.

    Returns:
        Candidate group-center column indices and each group's peak |z|.

    """
    hits = np.flatnonzero(np.abs(z) > _Z_THRESHOLD).tolist()
    columns: list[int] = []
    peaks: list[float] = []
    if hits:
        start = hits[0]
        prev = hits[0]
        for h in hits[1:]:
            if h - prev > _GROUP_GAP_PX:
                _keep_group(z, start, prev, columns, peaks)
                start = h
            prev = h
        _keep_group(z, start, prev, columns, peaks)
    return ScanCandidates(
        np.asarray(columns, dtype=np.int64), np.asarray(peaks, dtype=np.float64)
    )


def _keep_group(
    z: npt.NDArray[np.float32],
    start: int,
    end: int,
    columns: list[int],
    peaks: list[float],
) -> None:
    """Record a contiguous outlier group if it is narrow enough."""
    if end - start + 1 > _MAX_GROUP_WIDTH_PX:
        return
    group = cast("list[float]", np.abs(z[start : end + 1]).tolist())
    peak = max(range(len(group)), key=lambda i: group[i])
    columns.append(start + peak)
    peaks.append(group[peak])


def _anchor_offset(
    cand_cols: npt.NDArray[np.int64],
    ref_cols: npt.NDArray[np.int64],
    *,
    min_matches: int,
    min_margin: int,
) -> _AnchorResult:
    """x-offset of the candidates maximizing overlap with the target set.

    Scores every delta in +-``_ANCHOR_SEARCH_PX`` by matched-candidate count
    (+-``_ANCHOR_TOL_PX`` per match). The best delta must clear the match
    minimum and beat the runner-up (best count more than
    ``_RUNNER_UP_MIN_DISTANCE_PX`` away) by ``min_margin`` matches,
    otherwise the scan is reported unanchored at delta 0.

    Returns:
        The anchoring outcome (delta, matches, runner-up, anchored flag).

    """
    if cand_cols.size == 0 or ref_cols.size == 0:
        return _AnchorResult(0, 0, 0, False)
    counts: list[int] = []
    for delta in range(-_ANCHOR_SEARCH_PX, _ANCHOR_SEARCH_PX + 1):
        shifted = cand_cols + delta
        nearest = np.abs(shifted[:, None] - ref_cols[None, :]).argmin(axis=1)
        dist = np.abs(shifted - ref_cols[nearest])
        # numpy's count_nonzero stub is loose; narrow with a cast.
        counts.append(
            int(cast("SupportsInt", np.count_nonzero(dist <= _ANCHOR_TOL_PX)))
        )
    best_idx = max(range(len(counts)), key=lambda i: counts[i])
    best = counts[best_idx]
    runner_up = max(
        (
            c
            for i, c in enumerate(counts)
            if abs(i - best_idx) > _RUNNER_UP_MIN_DISTANCE_PX
        ),
        default=0,
    )
    ok = best >= min_matches and best - runner_up >= min_margin
    return _AnchorResult(
        delta=(best_idx - _ANCHOR_SEARCH_PX) if ok else 0,
        matches=best,
        runner_up=runner_up,
        anchored=ok,
    )


def _vote_stationary(
    per_scan: dict[str, ScanCandidates], anchors: dict[str, _AnchorResult]
) -> tuple[list[tuple[int, int, int]], npt.NDArray[np.int64]]:
    """Stationary column groups from per-scan boolean votes.

    One vote per anchored scan per reference-frame column (candidates
    dilated by the anchor tolerance); columns recurring in at least half the
    anchored scans are stationary hardware defects.

    Returns:
        ``(groups, peaks)``: ``(start, end, max_recurrence)`` groups and
        each group's recurrence-peak column.

    """
    anchored = [n for n in per_scan if anchors[n].anchored]
    max_col = max(
        (
            int(cast("SupportsInt", c.columns.max()))
            if c.columns.size
            else 0
            for c in per_scan.values()
        ),
        default=0,
    )
    width = max_col + _ANCHOR_SEARCH_PX + 10
    hit_matrix = np.zeros((len(anchored), width), dtype=bool)
    for i, name in enumerate(anchored):
        cols = cast(
            "list[int]", (per_scan[name].columns + anchors[name].delta).tolist()
        )
        for c in cols:
            lo = max(0, c - _ANCHOR_TOL_PX)
            hi_c = min(width, c + _ANCHOR_TOL_PX + 1)
            hit_matrix[i, lo:hi_c] = True
    # numpy's reduction stubs are loose; a typed list keeps this exact.
    recurrence = cast("list[int]", hit_matrix.sum(axis=0).tolist())
    min_scans = max(2, math.ceil(_STATIONARITY_FRACTION * len(anchored)))
    hits = [i for i, n in enumerate(recurrence) if n >= min_scans]
    stationary: list[tuple[int, int, int]] = []
    peaks: list[int] = []
    if hits:
        start = hits[0]
        prev = hits[0]
        for h in hits[1:]:
            if h - prev > _GROUP_GAP_PX:
                group = recurrence[start : prev + 1]
                stationary.append((start, prev, max(group)))
                peaks.append(start + max(range(len(group)), key=lambda i: group[i]))
                start = h
            prev = h
        group = recurrence[start : prev + 1]
        stationary.append((start, prev, max(group)))
        peaks.append(start + max(range(len(group)), key=lambda i: group[i]))
    return stationary, np.asarray(peaks, dtype=np.int64)
