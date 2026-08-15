# The analysis pipeline, stage by stage

This document walks through one comparison run (`run_analysis`, or the analysis stage of
`run_all`) in the order the data flows, and explains *why* each stage exists and why it
sits where it does. The implementation lives in `pipeline/prepare.py` (stages 1–6),
`pipeline/compare.py` (stages 7–9), and the `imaging/` subpackage.

Inputs: two directories of 16-bit grayscale PNG scans (`DIR_A` = reference, `DIR_B` =
moving), paired by **identical file name**. A pair is one physical film digitized on
both scanners.

______________________________________________________________________

## Stage 0 — pairing and loading

`core.io.find_pairs` intersects the two directories' PNG file names; names present on
only one side are listed as `unmatched` in the summary. `load_image` enforces 16-bit
grayscale (anything else fails the pair with a recorded error rather than aborting the
run).

## Stage 1 — registration (estimate the transform)

Detailed in [registration.md](registration.md). In short:

- Each image is robustly contrast-normalized (p1–p99 within the film ROI) and converted
  to a **log-DoG band-pass feature image** (σ = 2 minus σ = 16 of the log intensity).
- **Masked Euclidean ECC** (`cv2.findTransformECCWithMask`) aligns those feature images
  coarse-to-fine over a 3-level pyramid (1200 px → 2400 px → full resolution), with both
  sides masked to informative film content (borders, beveled corners, dark surround, and
  stationary defect columns excluded from the criterion).
- If ECC from identity converges poorly (correlation < 0.6), an **ORB keypoint
  initialization** (RANSAC inliers + closed-form Procrustes fit) is tried as well and
  the better result wins.
- A known inter-scanner **scale difference** is applied as a pre-correction constant
  (resampling the moving scan before ECC) and folded into the returned warp — the ECC
  criterion itself stays Euclidean.

The output is a single 2×3 warp mapping reference coordinates to moving-image
coordinates, plus the final ECC correlation (recorded per pair as `reg_correlation`; on
the development dataset, 0.97–0.99).

## Stage 2 — the single resampling

The moving scan is warped **once** into the reference frame with **Lanczos4**
interpolation (`align.warp_image`).

Interpolation choice is a measurement-integrity decision, not a cosmetic one: bilinear
resampling loses ~20 % of gradient energy at the worst sub-pixel phase — equivalent to
~0.41 px of extra blur — which would *masquerade as moving-side blur* in the sharpness
metrics. Lanczos4 retains ~99 % of gradient energy (measured effective penalty r̄ =
0.283 px), and that residual penalty is itself calibrated and accounted for in the blur
correction ([calibration.md](calibration.md)). The feature-path warps (pre-scale,
feature images) use plain bilinear because they never touch the metric images.

## Stage 3 — metric mask construction

Only pixels carrying comparable anatomical content may enter the metrics. The mask is
the intersection of:

1. **Geometric overlap** — where the warped moving frame actually covers the reference
   frame (warped-ones mask, eroded 8 px to remove interpolation edge artifacts).
1. **Unsaturated pixels** — pixels pinned at 0 or 65535 in *either* image carry no
   intensity information; excluded, then the mask is eroded a further 2 px so the
   saturation boundary cannot bleed in.
1. **Film ROI** — the frame minus a border band (`--border-margin`, default 3 %) and
   beveled corners (`--corner-margin`, default 8 %): scanner-dependent edge apparatus
   carries no anatomical information.
1. **Dark non-film surround, on both images** — pixels that are dark (below an Otsu
   threshold capped at 10 % of the normalized range) *and* connected to the frame border
   by a flood fill are excluded. Dark regions enclosed by brighter anatomy are kept:
   intensity alone cannot distinguish interior dark anatomy from the surround, but only
   border-connected darkness is certain to be outside the subject. The exclusion runs on
   **both** images (not just the reference): one scanner's washed-out histogram
   otherwise passes the other's darkness test, leaving a directional coverage asymmetry
   (measured: 41.6 % vs 62.5 % frame coverage forward/reverse with one-sided exclusion;
   41.6 % vs 43.0 % two-sided).
1. **Stationary column defects — excluded LAST.** This ordering is load-bearing: the
   defect population is a handful of extreme-value columns, and excluding it *before*
   the normalization/surround statistics moves the percentile and Otsu thresholds enough
   to flip the surround classification on borderline pairs (observed: −18 % coverage
   from 4 columns on one pair). Excluding defects last guarantees a masked run compares
   exactly the unmasked run's pixel population minus the defect pixels. (The defect
   masks *also* feed the ECC criterion masks at stage 1 — upstream there, where biasing
   the warp estimate is the concern.)

A pair with fewer than 10 000 surviving overlap pixels is recorded as a failure
(`Insufficient overlap`).

## Stage 4 — percentile-rank transform

Both images (reference, and warped moving) are replaced pixel-by-pixel by the
**fractional rank of their intensity among the masked pixels** (average ranks for ties,
midrank convention), mapping the masked histogram to a uniform (0, 1) distribution.

This is the central normalization of the pipeline: any **strictly monotonic** response
difference between the scanners — gain, offset, gamma, any smooth tone curve — is
removed *exactly*, because ranks are invariant under monotonic transforms. The
differences that remain are structural: real differences in what the two scans resolved.

Two honest costs, documented for interpretation:

- The rank transform stretches flat-region grain noise along with the signal (a
  compressed histogram region gets expanded). The structural metrics therefore prefilter
  lightly (σ = 1 Gaussian) before measuring, and value-fidelity thresholds should be
  calibrated against a same-scanner rescan pair when one is available.
- Rank equality is necessary, not sufficient, for content equality — which is exactly
  why the sharpness and FRC instruments exist alongside the value metrics.

A least-squares **gain/offset fit** of moving → reference is still computed and reported
— as a scanner-response *diagnostic* only; it is never applied. Likewise per-tile
residual statistics are reported but never corrected. The tool's philosophy throughout:
corrections that could hide information loss are never applied silently.

## Stage 5 — device corrections (rank domain)

With the images now comparable, two *measured* device systematics are equalized so they
cannot masquerade as information loss. Both corrections are calibration-driven
(constants come from `find_blur` / `find_column_gain`; see
[calibration.md](calibration.md)) and both are recorded per pair (`blur_applied`,
`colgain_rms`).

1. **Blur (MTF) equalization.** The sharper side's rank image is blurred by the net
   signed gap: `d2 = sign(σ_dev)·σ_dev² + r_i²`, where `σ_dev` is the device blur
   constant of the scanner pair and `r_i` is this pair's resampling penalty (integrated
   from the warp kernel's measured phase-response table over the pair's own warp phase
   field). If `d2` is positive the reference is blurred by `√d2`; otherwise the moving
   side by `-√d2`. The blur is **mask-normalized** (`G(img·m)/G(m)`), so the zeros
   outside the mask never bleed into the rim and perturb value metrics. Registration
   features are never blurred — the correction applies to the metric path only. The
   per-pair `blur_sigma` metric remains a *reported residual*: blur is never re-solved
   per pair at comparison time, because that would erase real per-film focus variation.
1. **Column-gain (banding) subtraction.** Line-sensor digitizers map each image column
   to one sensor element; per-element gain differences appear as full-height vertical
   banding that differs between devices and survives local averaging as coherent error.
   The reference scanner's stationary per-column profile (estimated in the sensor frame,
   shifted into this scan's frame by its crop offset) is subtracted from the reference
   rank image. It runs *after* the blur correction so the sharpness gap cannot alias
   into the profiles. The moving side is never touched (the two scanners' bandings are
   not separately observable from pair differences; the measured profile is exactly
   their warp-composed combination — see [calibration.md](calibration.md)).

## Stage 6 — metrics

`compute_metrics` on the corrected rank images over the final mask. Full definitions and
rationales in [metrics.md](metrics.md); in brief:

| Family          | Metrics                                     | Question answered                                                                 |
| --------------- | ------------------------------------------- | --------------------------------------------------------------------------------- |
| Value fidelity  | `mae`, `rmse`, `psnr`                       | How different are the pixel values (rank domain)?                                 |
| Coherence split | `local_mse`, `local_rmse`                   | Is the error spatially coherent (real local data difference) or incoherent grain? |
| Structure       | `ssim` (+ map)                              | Does local structure/luminance/contrast agree?                                    |
| Fine detail     | `grad_corr`, `grad_energy_ratio`            | Are edges reproduced with the same strength? (the most sensitive blur detector)   |
| Sharpness       | `blur_sigma` (signed)                       | Residual sharpness gap in pixels, after correction                                |
| Resolution      | `frc_resolution_px`, `frc_resolution_px_17` | Above what structure size do the two scans agree?                                 |
| Coverage        | `overlap_fraction`, `n_pixels`              | How much of the frame was compared?                                               |

Diagnostics computed alongside (reported, never fed to pass/fail): the **gain/offset**
fit, **tile residual statistics** (8×8 grid; spatial variation of the residual flags
shading/vignetting), and the per-pair correction records.

Pass/fail is judged against `Thresholds` (defaults: rmse ≤ 0.05, ssim ≥ 0.95, grad_corr
≥ 0.95, grad_energy_ratio within ±0.15 of 1) — deliberately near-lossless bars; the exit
code is 0 only when every pair passes.

## Stage 7 — diagnostic visualizations

Three signed difference renderings, each isolating a different error axis (red =
reference brighter, blue = second scan brighter, black = zero / outside mask):

- `*_diff.png` — the raw signed rank difference, ×5.
- `*_local_diff.png` — the signed difference locally averaged with a σ = 4 Gaussian,
  ×20. Uncorrelated grain averages toward zero; what remains is spatially coherent
  error. The map is zeroed outside the mask eroded by 3σ so the image covers exactly the
  region the `local_rmse` metric is evaluated on (the blur would otherwise mix in
  outside zeros).
- `*_motion_diff.png` — **motion-amplified** difference: the residual displacement field
  (DIS dense optical flow between the aligned rank images) is multiplied ×5 and the
  moving image re-warped by it before differencing. Sub-pixel misregistration lights up
  as strong red/blue structure; pure intensity noise does not (only geometry is
  amplified). Strictly a diagnostic artifact — it never feeds any metric.

Plus the alignment **overlay** (reference in red, aligned moving in green; identical
well-aligned content appears yellow, fringes show structural differences or
misregistration), the **SSIM defect map** (×10), and all intermediates: the two rank
images, the three log-DoG feature images (reference / moving / aligned moving), and the
final mask. The per-pair **FRC curve** is written as `*_frc.csv`.

## Stage 8 — run-level outputs

- `summary.json` / `summary.csv` — per-pair metrics, verdicts, registration details
  (warp, correlation, scale), correction records, diagnostics, unmatched files, and the
  full config block.
- Console table with the headline metrics per pair.
- Exit code 0/1 on threshold failure (2 on usage/calibration-coverage errors).
- With `--both-directions`: the whole run repeats with the directories exchanged (scale
  correction inverted, blur constant negated — handled centrally by
  `ImagePair.reversed()` / `CompareConfig.reversed()`), writing `forward/` and
  `reverse/` plus `cross_direction.json`/`.csv`: per-pair metric deltas and
  `roundtrip_max_px`, the largest frame-corner displacement of the composed
  forward∘reverse warp (identity for inverse-consistent registrations; ~0.05–0.22 px on
  the development dataset). The round-trip validates the global transforms' mutual
  consistency only — it cannot see local (non-rigid) residuals;
  `motion_diff`/`local_diff` are the lenses for those.

## Error handling philosophy

A pair that fails at any pipeline stage (unreadable image, degenerate intensity range,
alignment failure, insufficient overlap) is recorded as a failed `PairResult` with the
error message, and the run continues. Structural problems fail fast *before* any pair is
processed: a configured defect mask or column-gain calibration that does not cover both
directories aborts the run immediately, and a missing blur/scale calibration version is
rejected at load.
