"""Calibration solves that run through the pair-preparation chain.

Unlike ``calibration.scale`` (a registration sweep) and
``calibration.defects`` (raw-scan statistics), the blur and column-gain
solves must measure pairs through the full preparation chain —
registration, the metric mask, the rank transform, and the upstream
corrections — so the constants are measured on exactly the images the
comparison will see. Both solvers are bidirectional: the reverse direction
is the same pair with the directories exchanged
(``ImagePair.reversed()``) under the inverted configuration
(``CompareConfig.reversed()``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, SupportsFloat, SupportsInt, cast

import numpy as np

from scanner_comparison.calibration.blur import (
    PHASE_GRID_N,
    BlurCalibration,
    BlurCalibrationError,
    BlurCorrector,
    PairBlur,
    phase_table,
    resampling_penalty,
)
from scanner_comparison.calibration.colgain import (
    MIN_COLUMN_ROWS,
    MIN_PAIR_PROFILES,
    ColumnGainData,
    ColumnGainError,
    DirectoryColumnGain,
    PairColumnGain,
    aggregate_profiles,
    column_diff_profile,
    shift_profile,
)
from scanner_comparison.core.imtypes import F64Image, shape2
from scanner_comparison.core.io import load_image
from scanner_comparison.imaging.metrics import signed_blur_gap
from scanner_comparison.pipeline.prepare import prepare_pair
from scanner_comparison.records.config import BLUR_DISABLE_EPS

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy.typing as npt

    from scanner_comparison.calibration.defects import DirectoryDefects
    from scanner_comparison.core.io import ImagePair
    from scanner_comparison.records.config import CompareConfig


def _signed_square(value: float) -> float:
    """sign(v) * v^2 — the signed variance of a signed blur gap.

    Returns:
        The squared magnitude with the input's sign preserved.

    """
    return math.copysign(value * value, value)


def solve_blur_constant(
    pairs: list[ImagePair],
    config: CompareConfig,
    *,
    grid: int = PHASE_GRID_N,
    progress: Callable[[str], None] = print,
) -> BlurCalibration:
    """Solve the signed device blur constant of a scanner pair.

    Bidirectional signed common-support measurement per pair (the diag16
    convention): the forward direction per the config, plus the swapped
    reverse direction with the scale correction inverted. The signed gaps
    decompose as ``m_f^2 = sigma_dev^2 + r_f^2`` and ``m_r^2 = r_r^2 -
    sigma_dev^2`` (signed squares), so per pair ``sigma_dev_i^2 = (m_f^2 -
    m_r^2 - r_f^2 + r_r^2) / 2`` and the constant is the signed square root
    of the per-pair median. The resampling penalties come from the kernel
    phase tables (one per side's content) integrated over each warp's phase
    field: the edge-energy gap cannot see the resampling component when the
    device gap dominates (diag16's quadrature compression), so the
    data-solved ``r_data`` is reported only as a cross-check against the
    table. Positive ``sigma_dev`` means the second directory's scanner is
    intrinsically blurrier. Swapping the two directories negates the solved
    constant (antisymmetry).

    Returns:
        The calibration: signed ``sigma_dev``, the one-sided arm constants
        (forward/reverse gaps alone), the phase-averaged ``r_bar``, and
        the per-pair evidence.

    Raises:
        BlurCalibrationError: if there are no pairs to solve from.

    """
    if not pairs:
        msg = "No pairs to solve blur from"
        raise BlurCalibrationError(msg)
    # One phase table per side's content: the forward direction warps the
    # second scanner's image, the reverse warps the first scanner's.
    progress("  resampling phase tables (both sides' content)...")
    table_b = phase_table(load_image(pairs[0].path_b), grid=grid)
    table_a = phase_table(load_image(pairs[0].path_a), grid=grid)
    both_tables = np.concatenate((table_a.ravel(), table_b.ravel()))
    r_bar = math.sqrt(float(np.mean(both_tables**2)))
    progress(f"  phase tables built (r_bar={r_bar:.4f} px)")
    reverse_config = config.reversed()
    solved: list[PairBlur] = []
    for index, pair in enumerate(pairs, start=1):
        progress(f"  [{index}/{len(pairs)}] {pair.name}")
        solved.append(
            _measure_pair_gaps(
                pair, config, reverse_config, (table_b, table_a), progress
            )
        )

    def _signed_sqrt_median(values: list[float]) -> float:
        median = float(np.median(values))
        return math.copysign(math.sqrt(abs(median)), median)

    sigma_dev = _signed_sqrt_median([p.sigma_dev_sq_signed for p in solved])
    # One-sided arms: forward gaps alone minus their table penalty, and the
    # reverse-gaps-alone equivalent — they should bracket the combined solve.
    sigma_dev_forward = _signed_sqrt_median(
        [_signed_square(p.m_forward) - p.r_forward_table**2 for p in solved]
    )
    sigma_dev_reverse = _signed_sqrt_median(
        [p.r_reverse_table**2 - _signed_square(p.m_reverse) for p in solved]
    )
    r_data = float(np.median([p.r_data for p in solved]))
    progress(f"  sigma_dev = {sigma_dev:+.4f} px")
    progress(f"  arms: forward {sigma_dev_forward:+.4f}")
    progress(f"        reverse {sigma_dev_reverse:+.4f} (should bracket it)")
    progress(f"  r: table r_bar={r_bar:.4f} px, data-solved r_data={r_data:.4f}")
    progress("  (r_data under-reads by design: quadrature compression)")
    return BlurCalibration(
        sigma_dev=sigma_dev,
        r_bar=r_bar,
        scale_correction=config.scale_correction,
        sigma_dev_forward=sigma_dev_forward,
        sigma_dev_reverse=sigma_dev_reverse,
        r_data=r_data,
        pairs=tuple(solved),
    )


def _measure_pair_gaps(
    pair: ImagePair,
    config: CompareConfig,
    reverse_config: CompareConfig,
    tables: tuple[F64Image, F64Image],
    progress: Callable[[str], None],
) -> PairBlur:
    """Measure the bidirectional signed blur gaps of one pair.

    Forward per ``config``, reverse per the swapped pair with the inverted
    scale correction; the resampling penalties of each direction's warp are
    integrated from ``tables`` (forward side's table first, reverse side's
    second).

    Returns:
        The per-pair evidence: both signed gaps, the signed per-pair
        ``sigma_dev^2``, the data-solved ``r`` (cross-check only), and both
        table penalties.

    """
    prep_f = prepare_pair(
        pair, load_image(pair.path_a), load_image(pair.path_b), config
    )
    m_f = signed_blur_gap(prep_f.ref_rank, prep_f.mov_rank, prep_f.mask)
    r_f = resampling_penalty(
        prep_f.registration.warp, shape2(prep_f.ref_rank), tables[0]
    )
    reverse_pair = pair.reversed()
    prep_r = prepare_pair(
        reverse_pair,
        load_image(reverse_pair.path_a),
        load_image(reverse_pair.path_b),
        reverse_config,
    )
    m_r = signed_blur_gap(prep_r.ref_rank, prep_r.mov_rank, prep_r.mask)
    r_r = resampling_penalty(
        prep_r.registration.warp, shape2(prep_r.ref_rank), tables[1]
    )
    s2_f = _signed_square(m_f)
    s2_r = _signed_square(m_r)
    s2 = (s2_f - s2_r - r_f**2 + r_r**2) / 2.0
    r_data = math.sqrt(max(0.0, (s2_f + s2_r) / 2.0))
    progress(f"    m_f={m_f:+.4f} m_r={m_r:+.4f} r_table=({r_f:.4f}, {r_r:.4f})")
    progress(f"    signed sigma_dev^2={s2:+.4f} r_data={r_data:.4f}")
    return PairBlur(
        name=pair.name,
        m_forward=m_f,
        m_reverse=m_r,
        sigma_dev_sq_signed=s2,
        r_data=r_data,
        r_forward_table=r_f,
        r_reverse_table=r_r,
    )


def solve_column_gain(
    pairs: list[ImagePair],
    config: CompareConfig,
    *,
    progress: Callable[[str], None] = print,
) -> ColumnGainData:
    """Solve the stationary per-column gain ("banding") profiles of a pair.

    Per pair and direction, the per-column median of the signed rank diff
    over the metric mask is the column-coherent error profile: the
    reference scanner's stationary banding, plus the moving scanner's
    banding re-warped by that film's registration, plus content. The
    scanner auto-crops each film differently, so profiles are aggregated
    in the scanner's SENSOR frame (the defect mask's anchor frame; each
    pair's profile is shifted by its reference scan's ``x_offset``):
    across films the content averages out and the banding combination
    persists; the two directions yield one sensor-frame profile per
    scanner. The solve runs on fully corrected images (scale, blur, defect
    mask per ``config``; any ``config.column_gain`` is ignored so the
    solve never measures pre-corrected images) so sharpness/scale
    systematics do not alias into the profiles. Pairs whose reference scan
    is unanchored in the defect mask are skipped for that direction.

    Returns:
        The calibration: one sensor-frame subtract-profile per directory
        (with the per-scan crop offsets, so applying it is
        self-contained), plus the per-pair evidence.

    Raises:
        ColumnGainError: if there are no pairs to solve from, or no defect
            mask is configured (the crop anchoring comes from it).

    """
    if not pairs:
        msg = "No pairs to solve column gain from"
        raise ColumnGainError(msg)
    if config.defect_mask is None:
        msg = "find_column_gain needs the crop anchoring of a defect mask"
        raise ColumnGainError(msg)
    setup = _column_gain_setup(pairs, config)
    shifted_a: list[npt.NDArray[np.float64]] = []
    shifted_b: list[npt.NDArray[np.float64]] = []
    evidence: list[PairColumnGain] = []
    for index, pair in enumerate(pairs, start=1):
        progress(f"  [{index}/{len(pairs)}] {pair.name}")
        s_f, s_r, pair_evidence = _accumulate_column_profiles(pair, setup)
        evidence.extend(pair_evidence)
        progress(f"    profile rms: forward={pair_evidence[0].rms:.4f}")
        progress(f"                   reverse={pair_evidence[1].rms:.4f}")
        if s_f is not None:
            shifted_a.append(s_f)
        if s_r is not None:
            shifted_b.append(s_r)
    data = _column_gain_data(pairs[0], setup, shifted_a, shifted_b, evidence)
    for key, entry in sorted(data.directories.items()):
        progress(f"  stationary profile rms: {key} = {entry.rms:.4f}")
    return data


def _scan_width(directory: Path, name: str) -> int:
    """Width of one scan in a directory (the sensor frame's extent).

    Returns:
        The image width in pixels.

    """
    return shape2(load_image(directory / name))[1]


@dataclass(frozen=True)
class _ColumnGainSetup:
    """Everything the per-pair column-profile measurement needs."""

    config: CompareConfig  # solve config (column_gain stripped)
    reverse_config: CompareConfig
    corrector_f: BlurCorrector | None
    corrector_r: BlurCorrector | None
    defects_a: DirectoryDefects
    defects_b: DirectoryDefects
    width_a: int  # sensor frame widths = the defect reference scans' widths
    width_b: int


def _column_gain_setup(
    pairs: list[ImagePair], config: CompareConfig
) -> _ColumnGainSetup:
    """Build the per-direction configs, blur correctors, and anchor data.

    Returns:
        The shared solve context.

    Raises:
        ColumnGainError: if no defect mask is configured.

    """
    defect_mask = config.defect_mask
    if defect_mask is None:  # solve_column_gain guards this; be total anyway
        msg = "find_column_gain needs the crop anchoring of a defect mask"
        raise ColumnGainError(msg)
    solve_config = replace(config, column_gain=None)
    reverse_config = solve_config.reversed()
    corrector_f, corrector_r = _blur_correctors(pairs[0], config)
    defects_a = defect_mask.for_directory(pairs[0].path_a.parent)
    defects_b = defect_mask.for_directory(pairs[0].path_b.parent)
    return _ColumnGainSetup(
        config=solve_config,
        reverse_config=reverse_config,
        corrector_f=corrector_f,
        corrector_r=corrector_r,
        defects_a=defects_a,
        defects_b=defects_b,
        width_a=_scan_width(pairs[0].path_a.parent, defects_a.reference_scan),
        width_b=_scan_width(pairs[0].path_b.parent, defects_b.reference_scan),
    )


def _accumulate_column_profiles(
    pair: ImagePair, setup: _ColumnGainSetup
) -> tuple[
    npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None, list[PairColumnGain]
]:
    """Measure one pair's profiles and shift them into the sensor frames.

    Returns:
        The sensor-frame forward profile (None when the pair's DIR_A scan
        is unanchored), the reverse likewise, and the pair's evidence
        records (forward first).

    """
    p_f, p_r, evidence = _measure_column_profiles(
        pair, setup.config, setup.reverse_config, setup.corrector_f, setup.corrector_r
    )
    s_f: npt.NDArray[np.float64] | None = None
    info_a = setup.defects_a.scans.get(pair.name)
    if info_a is not None and info_a.anchored:
        s_f = shift_profile(p_f, info_a.x_offset, setup.width_a)
    s_r: npt.NDArray[np.float64] | None = None
    info_b = setup.defects_b.scans.get(pair.name)
    if info_b is not None and info_b.anchored:
        s_r = shift_profile(p_r, info_b.x_offset, setup.width_b)
    return s_f, s_r, evidence


def _blur_correctors(
    pair: ImagePair, config: CompareConfig
) -> tuple[BlurCorrector | None, BlurCorrector | None]:
    """Build the per-direction blur correctors for a bidirectional solve.

    The phase table is content-dependent: each direction's corrector is
    measured on that direction's moving-side content, the reverse with the
    negated constant. Both None when the blur correction is disabled.

    Returns:
        The ``(forward, reverse)`` correctors.

    """
    if abs(config.blur_correction) <= BLUR_DISABLE_EPS:
        return None, None
    corrector_f = BlurCorrector.from_moving_image(
        config.blur_correction, load_image(pair.path_b)
    )
    corrector_r = BlurCorrector.from_moving_image(
        -config.blur_correction, load_image(pair.path_a)
    )
    return corrector_f, corrector_r


def _column_gain_data(
    pair0: ImagePair,
    setup: _ColumnGainSetup,
    shifted_a: list[npt.NDArray[np.float64]],
    shifted_b: list[npt.NDArray[np.float64]],
    evidence: list[PairColumnGain],
) -> ColumnGainData:
    """Aggregate the sensor-frame profiles and package the calibration.

    A side with no anchored pairs gets an all-zero profile (the correction
    is a no-op for that direction) with ``n_pairs`` 0 — visible in the
    JSON rather than failing the whole solve.

    Returns:
        The two-directory calibration (profiles keyed by directory,
        self-contained via the embedded per-scan crop offsets).

    """
    params: dict[str, float | int] = {
        "scale_correction": setup.config.scale_correction,
        "blur_correction": setup.config.blur_correction,
        "min_column_rows": MIN_COLUMN_ROWS,
        "min_pair_profiles": MIN_PAIR_PROFILES,
    }
    directories: dict[str, DirectoryColumnGain] = {}
    for path, defects, shifted, frame_width in (
        (pair0.path_a, setup.defects_a, shifted_a, setup.width_a),
        (pair0.path_b, setup.defects_b, shifted_b, setup.width_b),
    ):
        if shifted:
            profile, width = aggregate_profiles(shifted)
        else:
            profile, width = np.zeros(frame_width, dtype=np.float64), frame_width
        key = str(path.parent.resolve())
        directories[key] = DirectoryColumnGain(
            directory=key,
            reference_scan=defects.reference_scan,
            width=width,
            n_pairs=len(shifted),
            rms=math.sqrt(float(np.mean(profile**2))),
            profile=[float(v) for v in cast("list[float]", profile.tolist())],
            scan_offsets={
                name: info.x_offset
                for name, info in defects.scans.items()
                if info.anchored
            },
        )
    return ColumnGainData(params=params, directories=directories, pairs=tuple(evidence))


def _measure_column_profiles(
    pair: ImagePair,
    config: CompareConfig,
    reverse_config: CompareConfig,
    corrector_f: BlurCorrector | None,
    corrector_r: BlurCorrector | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[PairColumnGain]]:
    """Measure one pair's column-coherent diff profile in both directions.

    Forward per ``config`` (profile in DIR_A's native frame), reverse per
    the swapped pair with the inverted scale correction (profile in
    DIR_B's native frame).

    Returns:
        The forward profile, the reverse profile, and the pair's evidence
        records (forward first).

    """
    prep_f = prepare_pair(
        pair, load_image(pair.path_a), load_image(pair.path_b), config, corrector_f
    )
    p_f = column_diff_profile(prep_f.ref_rank, prep_f.mov_rank, prep_f.mask)
    reverse_pair = pair.reversed()
    prep_r = prepare_pair(
        reverse_pair,
        load_image(reverse_pair.path_a),
        load_image(reverse_pair.path_b),
        reverse_config,
        corrector_r,
    )
    p_r = column_diff_profile(prep_r.ref_rank, prep_r.mov_rank, prep_r.mask)
    evidence: list[PairColumnGain] = []
    for direction, profile in (("forward", p_f), ("reverse", p_r)):
        valid = np.isfinite(profile)
        rms = (
            math.sqrt(float(cast("SupportsFloat", np.mean(profile[valid] ** 2))))
            if valid.any()
            else 0.0
        )
        evidence.append(
            PairColumnGain(
                name=pair.name,
                direction=direction,
                rms=rms,
                n_columns=int(cast("SupportsInt", valid.sum())),
            )
        )
    return p_f, p_r, evidence
