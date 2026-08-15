# Calibration: measuring the device constants

The analysis pipeline equalizes four device systematics — sampling pitch, stationary
column defects, intrinsic sharpness (MTF), and per-column gain banding — but none of the
constants are assumed: each is **measured from the data by the tool itself**, via a
dedicated CLI mode that writes an inspectable, versioned JSON artifact. `run_all` chains
all four solves and then the analysis in the dependency order:

```
find_scale  →  find_defect_mask  →  find_blur  →  find_column_gain  →  run_analysis
```

The order matters: blur is solved on scale-corrected images; column gain is solved on
fully corrected images (scale + blur + defect mask) so sharpness/scale systematics
cannot alias into the banding profiles; and the column-gain calibration *requires* the
defect mask — it reuses the defect detection's per-scan crop anchoring as its
sensor-frame coordinate system.

All calibration JSONs are versioned, self-describing records (parameters, per-pair
evidence, timestamps) meant to be inspected, not just consumed; each has a matching
reader (`read_scale_calibration`, `read_defect_mask`, `read_blur_calibration`,
`read_column_gain`).

______________________________________________________________________

## `find_scale` — the sampling-pitch constant

**What it measures.** Two nominally identical-resolution digitizers typically differ by
a small uniform scale (~0.1–0.2 %). For the development pair: **s = 0.9985** (SC113's
pitch is ~0.15 % finer than SC108's).

**How.** A sweep over candidate corrections (default 0.9965–1.0005 in 0.0005 steps). For
each candidate, the *full* registration (pre-scale + masked Euclidean ECC) is re-run on
a representative subset of pairs (up to 5, evenly spaced by name), and scored by
**masked log-DoG feature NCC** at the resulting composite warp — the Pearson correlation
of the band-pass feature images over the intersection of both alignment masks and the
geometric overlap. The best candidate is refined by a **parabola through it and its
neighbors** (the curve is flat near the optimum, so both the grid best and the refined
vertex are reported).

**Why this design.** Scale couples with rotation/translation at first order in the ECC
objective, so every cheaper estimator fails: post-hoc sweeps on a converged warp measure
nothing (the optimizer already absorbed the scale error), and per-pair ORB scale
estimates are too fragile to ship (3/15 pairs failed outright in the investigation;
banding-heavy pairs gave wild estimates). Re-optimizing the full registration per
candidate is the robust criterion. A single global constant is kept (not per-pair)
because real per-film scale variation (~0.1 %) sits at the estimator noise floor.

The correction is applied by pre-resampling the moving scan and folding the scale into
the returned warp; `--both-directions` inverts it (1/s). See
[registration.md](registration.md#the-scale-pre-correction).

## `find_defect_mask` — stationary column defects

**What it measures.** A sensor/readout defect produces a bright or dark vertical line
spanning the whole scan height, recurring at the same *sensor* x in every scan. These
must be masked so they cannot masquerade as information loss — while per-scan banding
and film content must *never* be masked (banding is the scanner's information-loss
character; masking it would hide signal).

**Stage 1 — per-scan candidates (full-height coherence).** The frame is split into **6
horizontal bands**; per band, the column-median profile is high-pass filtered (median
filter, k = 11) and robustly z-scored (MAD over *nonzero* residuals — flat surround runs
would otherwise collapse the MAD and explode the z-scores). A column is a candidate only
when **≥ 4 of 6 bands** exceed z > 8 *with a consistent sign*, grouped into clusters ≤ 4
px wide. A sensor defect spans the whole image height; trabecular texture and
label-strip edges do not — this coherence test is what rejects film content by
construction.

**Stage 2 — cross-scan stationarity, in the sensor frame.** Scanners auto-crop, so PNG x
is not a stable sensor coordinate (the development set varies 2828–3552 px wide). Each
scan is *anchored* to a common frame by a 1-D translation search over its candidate set:

- **Pass 1:** anchor each scan's strong candidates (z > 15) onto the scan with the
  clearest defect set (±60 px search, ±2 px match tolerance, z-weighted; guarded: ≥ 3
  matches and the best offset must beat the runner-up — at least 5 px away — by ≥ 2
  matches, else the scan is *unanchored*).
- **Vote:** one boolean vote per anchored scan per column (dilated ±2 px); columns
  recurring in **≥ 50 %** of anchored scans are stationary.
- **Pass 2:** re-anchor *every* scan against the pass-1 recurrence **peaks** (a
  noise-free target that tolerates weaker per-scan candidates; ≥ 2 matches + margin ≥
  2), then vote again. Fewer than 5 pass-1 anchored scans ⇒ the directory conservatively
  gets an empty stationary set.

Each scan's mask is the set of *its own native* candidate columns landing in the final
stationary groups; unanchored scans get no mask and are flagged in the JSON. On the
development dataset: SC108 has 20 stationary groups (~19 columns/scan masked, 0.54 % of
the frame), SC113 has 4 (0.12 %); 14/15 scans anchored per side. Known limitation: a
weak defect cluster near the per-scan detection floor (z 5–10, passing the gate in only
3/15 scans) can never pass the 50 % recurrence gate and stays unmasked — deliberately
conservative.

Outputs: `defects.json` (versioned; per-directory stationary groups in the reference
scan's frame, per-scan crop offsets `x_offset`, anchored flags, native defect columns),
plus a defect-map PNG and candidate CSV per directory for inspection. Applying the mask:
defect columns are excluded from the ECC criterion masks during registration and —
**last**, after normalization and surround classification — from the metric mask (the
load-bearing ordering explained in
[analysis-pipeline.md](analysis-pipeline.md#stage-3--metric-mask-construction)).

## `find_blur` — the device sharpness (MTF) constant

**What it measures.** The signed intrinsic sharpness gap between the two scanners,
**σ_dev** in pixels (positive = the second directory's scanner is blurrier), separated
from the resampling blur `r` that the pipeline's own warp incurs. For the development
pair (defect-masked): **σ_dev = +0.5332 px, r̄ = 0.2832 px**.

**The measurement model.** Two blur variables combine per comparison direction:

- `σ_dev` — the device blur constant (a property of the scanner *pair*),
- `r` — the resampling penalty incurred by warping the moving scan; it depends on the
  interpolation kernel and the *sub-pixel sampling phase* of each output pixel.

Per side, variances add: `σ_eff² = σ_intrinsic² + r²` (r only on the warped side). With
the signed gap `m` measured per direction, the signed-square decomposition is

```
m_f² = σ_dev² + r_f²        m_r² = r_r² − σ_dev²        (signed squares)
```

so per pair `σ_dev_i² = (m_f² − m_r² − r_f² + r_r²)/2`, and the constant is the **signed
square root of the per-pair median**. The solve is **bidirectional**: the reverse
direction is the same pair with the directories exchanged and the scale correction
inverted, measured through the identical preparation chain — so swapping the two input
directories negates the solved constant (verified exactly on real data).

**Where `r` comes from — the phase table.** The resampling penalty cannot be solved from
the edge-energy gaps: when the device gap dominates (~2:1 in the real regime), the
instrument reads the dominant term and the resampling contribution is *compressed away*
(quadrature compression — measured: a planted 0.6 px device blur solved back accurately
while its known r = 0.29 read as 0.00). Instead, `r` is measured **once per kernel**: a
6×6 sub-pixel translation grid is pushed through the real warp path on real content, and
each cell's signed common-support blur gap is recorded (r(0,0) = 0 by construction). Per
pair, the penalty `r_i` integrates that table over the warp's phase field
`φ(x) = frac(A·x + t)` on the film ROI. Real warps sweep the whole table uniformly (the
cross-axis rotation/scale terms drift through many phase cycles across a 3000 px frame),
so `r_i ≈ r̄` everywhere in practice — but the per-warp integral is the guard against
degenerate near-integer warps. Individual table cells can be negative (a ±0.2–0.3 px
support-selection systematic on shifted correlated content); the table is only
meaningful through phase averages. Regenerate the table if the warp interpolation kernel
ever changes (`_WARP_INTERPOLATION`).

**Applying the correction** (`--blur-correction`, `BlurCorrector`): per pair, the
sharper side's rank image is blurred by the net signed gap
`d2 = sign(σ_dev)·σ_dev² + r_i²` — the reference by `√d2` when positive, the moving side
by `√−d2` otherwise — using the **mask-normalized Gaussian** so mask-edge zeros never
bleed in. Metric path only; registration features are never blurred, and the per-pair
`blur_sigma` remains a reported residual (solving blur per pair at comparison time would
erase real per-film focus variation).

**The JSON** (`version: 2`, convention `"signed common-support bidirectional"`) records
the constant, the **one-sided arm solves** (`sigma_dev_forward` / `sigma_dev_reverse` —
each direction's gaps alone, table-corrected; they should bracket the combined constant,
and a wide split flags direction asymmetry), `r_bar` (phase-averaged penalty over both
sides' content), `r_data` (the data-solved resampling component — a **cross-check
only**, expected near zero by quadrature compression), and the per-pair evidence
(`m_forward`/`m_reverse`, signed variances, both table penalties).

*Convention note:* these constants supersede the previous own-support convention (0.6864
/ 0.482), kept only for reproducing pre-2026-08 results. The conventions differ because
the old own-support instrument inflated both directions (support-selection + ringing);
the physical decomposition r ≈ 0.29 / σ_dev ≈ 0.53 is the common-support one.

## `find_column_gain` — stationary banding profiles

**What it measures.** Line-sensor digitizers map each image column to one sensor
element; per-element gain differences (photo-response non-uniformity plus illumination
falloff) appear as full-height vertical **banding** that differs between devices,
survives the local average behind `local_rmse` as *spatially coherent* error, and is not
covered by the defect mask (which removes only discrete defect columns).

**The observability constraint (important).** In the rank domain, one pair's column
difference is `diff(x) = band_ref(x) − band_mov(w(x)) + content`, with a different warp
`w` per film. The two scanners' bandings are **not separately observable** from pair
differences: only that warp-composed combination enters, and the per-film warp jitter
(even ±6 px) cannot decorrelate smooth profiles. (An earlier draft that estimated and
subtracted per-scanner profiles from *both* sides was caught by the synthetic tests: the
leaked profile cancels the true one and the net correction is ≈ 0 for smooth banding.)
So the calibration estimates exactly the observable quantity, and the correction
subtracts it from the **reference side only**; the moving side is never touched.

**How.** Per pair and direction, on fully corrected images (scale + blur + defect mask;
any pre-existing column-gain config is stripped so the solve never measures
pre-corrected images): the **per-column median of the signed rank difference** over the
metric mask (columns with < 200 masked rows are dropped; the profile is
mean-subtracted). Because the scanners auto-crop each film differently, profiles are
aggregated in the scanner's **sensor frame** — the defect mask's anchor frame: each
pair's profile is shifted by its reference scan's `x_offset` (sensor column = scan
column + offset), then **median across pairs** per sensor column (columns with < 3
contributing pairs get 0). Across films the content averages out; the stationary banding
combination persists. The two alignment directions yield one profile per scanner; a run
applies the profile of its reference directory, shifted back by the scan's crop offset.
The applied magnitude is reported per pair as `colgain_rms`.

**Development-dataset results.** Stationary profile rms 0.0028 (SC108) / 0.0031 (SC113),
dominated by *smooth* long-wavelength components (illumination/optics falloff), not
per-element spikes. Applying it reduced `local_rmse` by ~8–10 % (means, both directions)
with value/structural metrics essentially unchanged — and a key finding for
interpretation: per-pair banding profiles correlate only 0.35–0.66 with the stationary
profile, i.e. **most visible banding is per-scan varying**
(illumination/film-dependent), not sensor-fixed. A stationary device calibration can
only remove the stationary share; removing the rest would need per-pair estimation,
which risks absorbing real content differences — deliberately not done.

Outputs: `colgain.json` (versioned, self-contained: profiles plus the per-scan crop
offsets), a stripe-map PNG per directory (red = the reference reads brighter in those
sensor columns — the same sign convention as the `*_diff.png` artifacts), and a
per-column CSV.

## Bidirectionality and consistency rules

All direction rules are centralized (`CompareConfig.reversed()`,
`ImagePair.reversed()`): the reverse direction **inverts** the scale correction and
**negates** the signed blur constant; the calibration artifacts themselves (defect
groups, banding profiles) are direction-free per scanner. The blur solve's exact
antisymmetry under directory swap (forward +0.5353 / reverse −0.5353 with independent
registrations) is the instrument's strongest internal consistency check, and the
one-sided arms bracketing the combined constant is the second.
