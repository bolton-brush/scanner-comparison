"""Run configuration records: pass/fail thresholds and the compare config.

The direction helpers centralize the forward/reverse rules used by every
bidirectional workflow (``--both-directions`` runs and the bidirectional
calibration solves): the reverse direction inverts the scale correction (if
the forward map scales by s, the inverse map scales by 1/s) and negates the
signed blur constant (the device blur gap of ``moving - reference`` negates
when the directories swap).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scanner_comparison.calibration.colgain import ColumnGainData
    from scanner_comparison.calibration.defects import DefectMaskData

# A scale correction within this of 1.0 is treated as disabled (exact float
# comparison of a user-supplied constant would be unreliable).
SCALE_DISABLE_EPS = 1e-9
# A blur correction within this of 0 is treated as disabled.
BLUR_DISABLE_EPS = 1e-9


def inverse_scale(scale: float) -> float:
    """Scale correction for the reversed direction: 1/s (1.0 stays 1.0).

    Returns:
        The inverse of ``scale``.

    """
    if abs(scale - 1.0) < SCALE_DISABLE_EPS:
        return 1.0
    return 1.0 / scale


@dataclass(frozen=True)
class Thresholds:
    """Pass/fail criteria applied to each pair's metrics."""

    max_rmse: float = 0.05
    min_ssim: float = 0.95
    min_grad_corr: float = 0.95
    grad_energy_tolerance: float = 0.15


@dataclass(frozen=True)
class CompareConfig:
    """Tunable knobs shared by one comparison run."""

    thresholds: Thresholds = field(default_factory=Thresholds)
    max_dim: int = 1200
    border_margin: float = 0.03
    corner_margin: float = 0.08
    exclude_background: bool = True
    # Scanner-pair calibration constant: uniform scale of the moving scan
    # relative to the reference (1.0 = identical sampling pitch). Measured
    # once per scanner pair; estimation stays Euclidean.
    scale_correction: float = 1.0
    # Signed device blur constant of the scanner pair (the ``find_blur``
    # solve; positive = the second scanner is intrinsically blurrier;
    # signed common-support convention). 0.0 = off. When set, the sharper
    # side's rank image is blurred by the net gap (device gap combined in
    # quadrature with the per-pair resampling penalty from the kernel phase
    # table) so the metrics compare equal-MTF images; the registration
    # feature path is never blurred, and per-pair blur_sigma remains a
    # reported (signed) residual.
    blur_correction: float = 0.0
    # Stationary column-defect mask from the ``find_defect_mask`` CLI mode;
    # defect columns are excluded from the metric mask and the alignment
    # criterion. None = no defect masking (a CLI warning is printed).
    defect_mask: DefectMaskData | None = None
    # Stationary per-column gain ("banding") calibration from the
    # ``find_column_gain`` CLI mode; the reference directory's profile is
    # subtracted from the reference rank image so the banding difference
    # cannot masquerade as coherent local error. None = no correction.
    column_gain: ColumnGainData | None = None

    def reversed(self) -> CompareConfig:
        """Configuration for the reverse-direction run.

        The scale correction is inverted (if the forward map scales by s,
        the inverse map scales by 1/s) and the signed blur constant negated
        (the device blur gap of ``moving - reference`` negates when the
        directories swap). The calibration artifacts are direction-free.

        Returns:
            The reverse-direction configuration.

        """
        return replace(
            self,
            scale_correction=inverse_scale(self.scale_correction),
            blur_correction=-self.blur_correction,
        )
