"""Per-column gain ("banding") calibration and correction.

Line-sensor digitizers map each image column to one sensor element, so
per-element gain/offset differences (photo-response non-uniformity plus
illumination falloff along the scan line) appear as full-height vertical
banding. The pattern differs between devices, survives the local average
behind ``local_rmse`` as spatially coherent error, and is NOT covered by
the defect mask (which removes only discrete defect columns).

Model (rank domain, reference frame of one comparison direction)::

    diff(x) = band_ref(x) - band_mov(w(x)) + content residual

where the registration warp ``w`` differs per film. The two scanners'
stationary bandings are NOT separately observable from pair differences:
only the combination ``band_ref - band_mov(w)`` enters, and for smooth
profiles the warp diversity between films is far too small to disentangle
them. So the calibration estimates, per comparison direction, exactly the
observable quantity: the per-column median diff profile of each pair,
aggregated across films in the scanner's SENSOR frame (film content
changes per scan and averages out; the banding combination does not). The
correction then subtracts that profile from the reference side's rank
image — the moving side is never touched. Running both alignment
directions yields one profile per scanner; a run picks the profile of its
reference directory.

The sensor frame per directory is the defect mask's anchor frame: the
scanner auto-crops films to different widths, so a pair's reference-frame
columns map to sensor columns by the scan's integer ``x_offset`` (the
defect detection's crop anchoring, reused here — the calibration
therefore requires a defect mask, and copies the offsets into its own
JSON so applying it is self-contained).

The profiles are additive offsets in the rank domain — the first-order
image of a small per-column intensity gain after the percentile-rank
transform. Like the blur correction, the goal is to remove a known
device systematic so it cannot masquerade as information loss; the
per-pair applied magnitude is reported.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, SupportsFloat, SupportsInt, cast

import cv2
import numpy as np

from scanner_comparison.core.imtypes import F32Image, shape2

if TYPE_CHECKING:
    import numpy.typing as npt

    from scanner_comparison.core.imtypes import BoolMask

_COLGAIN_VERSION = 1
# A per-pair column estimate needs at least this many masked rows.
MIN_COLUMN_ROWS = 200
# A sensor column enters the aggregated profile only when at least this
# many pairs contributed a valid estimate of it.
MIN_PAIR_PROFILES = 3


class ColumnGainError(ValueError):
    """Raised when a column-gain calibration cannot be built or applied."""


@dataclass(frozen=True)
class PairColumnGain:
    """Per-pair banding evidence: the pair's column-profile strength."""

    name: str
    direction: str
    rms: float
    n_columns: int


@dataclass(frozen=True)
class DirectoryColumnGain:
    """Stationary per-column rank-offset profile of one scanner directory.

    ``profile`` is the value to SUBTRACT from that scanner's rank images,
    per SENSOR-frame column (0 where the evidence was insufficient); the
    sensor frame is the defect mask's anchor frame (the directory's
    ``reference_scan``). ``scan_offsets`` maps each anchored scan name to
    its integer crop offset: sensor column = scan column + offset.
    """

    directory: str
    reference_scan: str
    width: int
    n_pairs: int
    rms: float
    profile: list[float]
    scan_offsets: dict[str, int]

    def profile_for_scan(self, name: str, width: int) -> npt.NDArray[np.float64] | None:
        """The subtract-profile in one scan's native frame.

        Returns:
            The profile cropped to the scan's columns (length ``width``,
            zero-filled outside the sensor frame's coverage), or None when
            the scan is unknown or was not anchored.

        """
        offset = self.scan_offsets.get(name)
        if offset is None:
            return None
        sensor = np.asarray(self.profile, dtype=np.float64)
        out = np.zeros(width, dtype=np.float64)
        lo = max(0, -offset)
        hi = min(width, sensor.shape[0] - offset)
        if hi > lo:
            out[lo:hi] = sensor[lo + offset : hi + offset]
        return out


@dataclass(frozen=True)
class ColumnGainData:
    """A two-directory column-gain calibration as written to/read from JSON."""

    params: dict[str, float | int]
    directories: dict[str, DirectoryColumnGain]
    pairs: tuple[PairColumnGain, ...]

    def for_directory(self, directory: Path) -> DirectoryColumnGain:
        """Look up the profile for ``directory`` by absolute path.

        Returns:
            The directory's column-gain record.

        Raises:
            ColumnGainError: if the calibration does not cover the
                directory.

        """
        key = str(directory.resolve())
        entry = self.directories.get(key)
        if entry is None:
            known = ", ".join(sorted(self.directories))
            msg = f"Column-gain calibration has no entry for {key} (covers: {known})"
            raise ColumnGainError(msg)
        return entry


def column_diff_profile(
    ref_rank: F32Image, mov_rank: F32Image, mask: BoolMask
) -> npt.NDArray[np.float64]:
    """Per-column median of ``ref - mov`` over the masked rows.

    Returns:
        The column-coherent difference profile (float64, length = width);
        NaN on columns with fewer than ``MIN_COLUMN_ROWS`` masked pixels.
        The valid entries are mean-subtracted (the global gain/offset is
        already handled upstream).

    """
    diff = (ref_rank - mov_rank).astype(np.float64)
    masked = np.where(mask, diff, np.nan)
    count = mask.sum(axis=0)
    # Columns with no masked rows at all make nanmedian warn (all-NaN
    # slice); they are marked NaN below anyway.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmedian(masked, axis=0)
    profile[count < MIN_COLUMN_ROWS] = np.nan
    valid = np.isfinite(profile)
    if valid.any():
        profile -= float(np.mean(profile[valid]))
    return np.asarray(profile, dtype=np.float64)


def shift_profile(
    profile: npt.NDArray[np.float64], offset: int, width: int
) -> npt.NDArray[np.float64]:
    """Move a scan-frame profile into the sensor frame (sensor = scan + offset).

    Returns:
        The profile on ``[0, width)`` (NaN where the scan does not cover
        the sensor frame).

    """
    out = np.full(width, np.nan, dtype=np.float64)
    lo = max(0, offset)
    hi = min(width, offset + int(cast("SupportsInt", profile.shape[0])))
    if hi > lo:
        out[lo:hi] = profile[lo - offset : hi - offset]
    return out


def aggregate_profiles(
    profiles: list[npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], int]:
    """Median-aggregate shifted per-pair profiles into the stationary one.

    All profiles share one sensor frame (equal widths required — enforced
    by the caller); NaN entries are excluded per column.

    Returns:
        ``(profile, width)``: the aggregated profile (0 on sensor columns
        with fewer than ``MIN_PAIR_PROFILES`` valid estimates),
        mean-subtracted over the valid columns, and the frame width.

    Raises:
        ColumnGainError: if there are no profiles to aggregate.

    """
    if not profiles:
        msg = "No column profiles to aggregate"
        raise ColumnGainError(msg)
    width = int(cast("SupportsInt", profiles[0].shape[0]))
    stack = np.stack(profiles)
    n_valid = np.isfinite(stack).sum(axis=0)
    # Columns no pair estimated (all-NaN slices) are handled by the
    # MIN_PAIR_PROFILES guard below; silence nanmedian's warning for them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        agg = np.nanmedian(stack, axis=0)
    enough = n_valid >= MIN_PAIR_PROFILES
    if enough.any():
        mean = float(np.mean(agg[enough]))
        agg = np.where(enough, agg - mean, 0.0)
    else:
        agg = np.zeros(width, dtype=np.float64)
    return np.asarray(np.nan_to_num(agg), dtype=np.float64), width


def subtract_profile(
    ref_rank: F32Image, profile: npt.NDArray[np.float64], mask: BoolMask
) -> tuple[F32Image, float]:
    """Subtract a column profile from the reference rank image in the mask.

    Returns:
        ``(reference, rms)``: the corrected rank image and the rms of the
        applied offset over the masked pixels.

    Raises:
        ColumnGainError: if the profile does not match the image width.

    """
    _height, width = shape2(ref_rank)
    if profile.shape[0] != width:
        msg = f"Profile width {profile.shape[0]} != image width {width}"
        raise ColumnGainError(msg)
    corr = np.broadcast_to(profile[None, :], ref_rank.shape)
    ref_out = ref_rank - np.where(mask, corr, 0.0).astype(np.float32)
    n = float(mask.sum())
    rms = math.sqrt(float((corr**2)[mask].sum()) / max(n, 1.0))
    return ref_out, rms


def write_column_gain(path: Path, data: ColumnGainData) -> Path:
    """Write the column-gain calibration JSON (inspectable profiles).

    Returns:
        The written path.

    """
    payload = {
        "version": _COLGAIN_VERSION,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "params": data.params,
        "directories": {
            key: {
                "reference_scan": entry.reference_scan,
                "width": entry.width,
                "n_pairs": entry.n_pairs,
                "rms": entry.rms,
                "profile": entry.profile,
                "scan_offsets": entry.scan_offsets,
            }
            for key, entry in sorted(data.directories.items())
        },
        "pairs": [
            {
                "name": p.name,
                "direction": p.direction,
                "rms": p.rms,
                "n_columns": p.n_columns,
            }
            for p in data.pairs
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _written = path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_column_gain_maps(
    path: Path, data: ColumnGainData, *, amplify: float = 20.0
) -> dict[str, Path]:
    """Write one profile-map PNG per directory, next to the JSON.

    The stationary profile is rendered as a vertical-stripe image (what it
    removes from the rank domain, amplified): red = positive (the
    reference reads brighter in those sensor columns), blue = negative —
    the same sign convention as the ``*_diff.png`` artifacts.

    Returns:
        A mapping of directory key to the written PNG path.

    Raises:
        OSError: if an image cannot be written.

    """
    written: dict[str, Path] = {}
    for key, entry in data.directories.items():
        profile = np.asarray(entry.profile, dtype=np.float32)
        stripe = np.broadcast_to(profile[None, :], (256, profile.shape[0]))
        amplified = stripe * amplify
        vis = np.zeros((*stripe.shape, 3), dtype=np.uint8)
        vis[..., 2] = (np.clip(amplified, 0.0, 1.0) * 255.0).astype(np.uint8)
        vis[..., 0] = (np.clip(-amplified, 0.0, 1.0) * 255.0).astype(np.uint8)
        safe = "".join(c if c.isalnum() else "-" for c in Path(key).name)
        out = path.with_name(f"{path.stem}_{safe}.png")
        if not cv2.imwrite(str(out), vis):
            msg = f"Failed to write column-gain map: {out}"
            raise OSError(msg)
        written[key] = out
    return written


def write_column_gain_csvs(path: Path, data: ColumnGainData) -> dict[str, Path]:
    """Write one per-column profile CSV per directory (column, offset).

    Returns:
        A mapping of directory key to the written CSV path.

    """
    written: dict[str, Path] = {}
    for key, entry in data.directories.items():
        safe = "".join(c if c.isalnum() else "-" for c in Path(key).name)
        out = path.with_name(f"{path.stem}_{safe}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            _header = cast("int", writer.writerow(["column", "offset"]))
            for column, offset in enumerate(entry.profile):
                _row = cast("int", writer.writerow([column, f"{offset:.6f}"]))
        written[key] = out
    return written


def read_column_gain(path: Path) -> ColumnGainData:
    """Read a column-gain calibration JSON written by ``write_column_gain``.

    Returns:
        The parsed calibration data.

    Raises:
        ColumnGainError: if the file is missing, malformed, or of an
            unsupported version.

    """
    try:
        raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Cannot read column-gain calibration: {path} ({exc})"
        raise ColumnGainError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"Unsupported column-gain calibration file: {path}"
        raise ColumnGainError(msg)
    payload = cast("dict[str, object]", raw)
    if payload.get("version") != _COLGAIN_VERSION:
        msg = f"Unsupported column-gain calibration file: {path}"
        raise ColumnGainError(msg)
    try:
        params = cast("dict[str, float | int]", payload["params"])
        directories = {
            key: _read_directory_colgain(key, entry)
            for key, entry in cast(
                "dict[str, dict[str, object]]", payload["directories"]
            ).items()
        }
        pairs = tuple(
            PairColumnGain(
                name=str(p["name"]),
                direction=str(p["direction"]),
                rms=float(cast("SupportsFloat", p["rms"])),
                n_columns=int(cast("SupportsInt", p["n_columns"])),
            )
            for p in cast("list[dict[str, object]]", payload["pairs"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"Malformed column-gain calibration file: {path} ({exc})"
        raise ColumnGainError(msg) from exc
    return ColumnGainData(params=params, directories=directories, pairs=pairs)


def _read_directory_colgain(key: str, entry: dict[str, object]) -> DirectoryColumnGain:
    """Parse one directory entry of the column-gain JSON.

    Returns:
        The parsed directory column-gain record.

    """
    offsets = {
        name: int(cast("SupportsInt", v))
        for name, v in cast("dict[str, object]", entry["scan_offsets"]).items()
    }
    return DirectoryColumnGain(
        directory=key,
        reference_scan=str(entry["reference_scan"]),
        width=int(cast("SupportsInt", entry["width"])),
        n_pairs=int(cast("SupportsInt", entry["n_pairs"])),
        rms=float(cast("SupportsFloat", entry["rms"])),
        profile=[
            float(cast("SupportsFloat", v))
            for v in cast("list[object]", entry["profile"])
        ],
        scan_offsets=offsets,
    )
