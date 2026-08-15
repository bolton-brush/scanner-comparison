# Report — Comparing duplicate film digitizations across two scanners

> Status: data-filled outline for review. All numbers below come from the cited run
> directories and calibration JSONs; `[TODO]` marks anything that still needs a
> decision, prose, or a figure. Method wording for §3 can be adapted from
> `docs/analysis-pipeline.md`, `docs/registration.md`, `docs/metrics.md`, and
> `docs/calibration.md`.
>
> **Primary evidence on disk:**
>
> - Golden baseline run (all corrections, both directions): `tmpout/run-colgain/`
> - Signed-convention baseline (blur constant 0.5528): `tmpout/run-all-signed/`
> - FRC evidence run (identical config to run-colgain + FRC metric): `tmpout/frc-check/`
> - Earlier ablation stages: `tmpout/run-rank/` (Euclidean only),
>   `tmpout/run-both-surround2/` (+ scale, two-sided surround),
>   `tmpout/run-both-defects-masked/` (+ defect mask)
> - Calibration artifacts:
>   `tmpout/{scale,blur-signed-masked,blur-signed-rev,defects,colgain}.json`,
>   per-scanner defect maps/CSVs (`tmpout/defects_BBGSC*.{png,csv}`), colgain stripe
>   maps/CSVs (`tmpout/colgain_BBGSC*.{png,csv}`)
> - Before/after blur-correction figure sets:
>   `agents/scratch/diag16/artifacts/forward_baseline/` and `.../forward_corrected/`
> - Scanner previews: `agents/BBGSC108_NDT_Dosimetry_Pro_preview.png`,
>   `agents/BBGSC113_Dental_Film_Digitizer_preview.png`

______________________________________________________________________

## 1. Introduction

The goal of this research is to compare and identify differences between two digital
cephalometric film scanners. The two digital scanners used in this project have vastly
different brightness response curves, contain artifacts of physical scanning such as
varying crop zones or orientations, as well as differing physical scanning hardware
defects. A naive comparison of the images would not sufficiently explain similarities
and differences between the digital images.

We found that after analysing matched images from both scanners with a sample size of 15
pairs, that artifacts larger then 10px (0.016 in) are maintained across both scanners.
Our tuned metrics place the outputs of both scanners statistically close to each other
but not close enough to be statistically identical. After masking and pre-processing
images, we obtained MSE scores mostly under 0.05 with one outlier, and SSIM scores
clustered around 0.9.

## 2. Materials

### 2.1 Films and scans

- 15 PA cephalometric films, each digitized on both devices; pairs matched by identical
  file name, in 600ppi, 16-bit PNGs
- Content: PA cephalograms with fiducial apparatus (ear-rod posts, marker dots, label
  strips, ruler edge) — stable features that aid registration; same polarity on both
  devices (bone white, background dark).
- All data is sourced from the Bolton Brush Growth Study Center (BBGSC) at Case Western
  Reserve University (CWRU)

|                                       | BBGSC108 NDT Dosimetry Pro ("SC108") | BBGSC113 Dental Film Digitizer ("SC113") |
| ------------------------------------- | ------------------------------------ | ---------------------------------------- |
| Role                                  | reference (DIR_A)                    | moving (DIR_B)                           |
| Files                                 | 15 PNG, 16-bit grayscale             | 15 PNG, 16-bit grayscale                 |
| Widths                                | 2828–3552 px                         | 2722–3455 px                             |
| Heights                               | 2383–2996 px                         | 2380–2997 px                             |
| Observed intensity extrema (one pair) | 31 – 65535 (full range)              | 15909 – 63300 (compressed/offset)        |
| Histogram character                   | dark background, high contrast       | flat / washed out                        |

- Both devices auto-crop per scan: sizes vary within each scanner's set, so PNG pixel
  coordinates are not stable sensor coordinates.

### 2.2 Software

- The `scanner-comparison` tool built by BBGSC, version 0.1.0 [1]; Python ≥ 3.13.
- Locked dependency versions (uv.lock): OpenCV headless 5.0.0.93, NumPy 2.5.1,
  scikit-image 0.26.0, SciPy 1.18.0.
- Entry point: `scanner-compare`; full pipeline of one `run_all` invocation (§A.1).
- [TODO] decide citation form for the repository (URL / archive / supplementary) —
  update reference [1] accordingly.

## 3. Methods

The comparison pipeline follows a fairly linear process:

1. Load Images from directory
1. Solve for scale constant
1. Align log-DoG images
1. Solve for calibration constants (defects, blur, column gain differences)
1. Equalize images with solved constants
1. Calculate comparison metrics
1. Generate a report and write output files
1. Optionally repeat in both directions to test for consistency

The calibration constants are solved in a fixed dependency order — scale, then defects,
then blur, then column gain — because each solve is performed through the same
processing chain used for the final comparison, with all previously solved corrections
active. The exclusion mask is likewise built up in stages; the mask in use is stated at
each step.

### 3.1 Masking

The images obtained from each scanner had differences in the borders of the image along
with unequal cropping around the scanned images themselves. We opted to mask out the
areas surrounding the scan as they are not relevant to the contents of the scan.

We utilized a 3% border mask on each edge of the image with beveled corners at 8%, and a
content mask removing the dark non-film surround: pixels darker than an Otsu threshold
(capped at 10% of the normalized intensity range) that are connected to the frame border
by a flood fill. Dark regions enclosed by brighter anatomy are kept, since only
border-connected darkness is certain to lie outside the subject. A final erosion band of
0.5% of the frame size is removed just inside the detected content edge, because the
film edge is the strongest gradient in the frame and sits at a crop-dependent position.

This base mask — the film region of interest, less the surround and the content-edge
band — is applied in every subsequent step, and is extended by the stationary defect
mask once the latter has been solved (Section 3.4).

### 3.2 Registration

Both images are first transformed into a band-pass feature space. Taking the logarithm
of the intensities converts the multiplicative, gamma-like response differences between
the two devices into additive offsets; the difference of Gaussians (DoG) — the image
blurred with σ = 2 px minus a copy blurred with σ = 16 px — then cancels those offsets
together with any smooth shading, isolating the mid-frequency structural content of the
anatomy. The resulting log-DoG feature images are min–max normalized within the film
region of interest [2].

Alignment is estimated by masked enhanced correlation coefficient (ECC) maximization
under a Euclidean (rotation and translation) motion model, computed coarse-to-fine over
a three-level pyramid (Table 1). The base mask serves as the criterion mask on both
images at every level, so that scanner-dependent borders and surround content cannot
influence the estimate. When the coarsest level converges poorly from an identity
initialization (correlation below 0.6), the estimate is restarted from an ORB keypoint
matching initialization (5,000 features, ratio test 0.75, RANSAC at 3 px, closed-form
Procrustes fit on at least 8 inliers) and the better of the two results is retained;
pairs whose final correlation falls below 0.3 are rejected.

Table 1. Pyramid levels and termination criteria for the ECC registration.

| Level  | Max dimension   | ε    | Iterations |
| ------ | --------------- | ---- | ---------- |
| coarse | 1200 px         | 1e-6 | 200        |
| mid    | 2400 px         | 1e-7 | 200        |
| fine   | full resolution | 1e-8 | 400        |

### 3.3 Scale calibration

Although both devices report a nominal resolution of 600 ppi, we found that their true
sampling pitches differ slightly. Because scale couples with rotation and translation at
first order in the ECC objective, the difference cannot be recovered reliably from a
converged registration; instead, we sweep candidate scale corrections from 0.9965 to
1.0005 in steps of 0.0005, re-running the full registration of Section 3.2 for each
candidate on three evenly spaced pairs and scoring it by the masked correlation (NCC) of
the log-DoG feature images at the resulting warp [3]. The optimum is refined by fitting
a parabola through the best candidate and its neighbors. The sweep is scored over the
base masks of both images, warped into correspondence.

The solved constant (s = 0.9985; Section 4.2) is applied to the moving scan as a uniform
pre-scaling about its center and folded into the estimated warp, so that the moving scan
is resampled exactly once, using Lanczos4 interpolation (measured kernel-induced blur r̄
= 0.287 px, versus ~0.41 px for bilinear).

### 3.4 Stationary column-defect detection

Sensor and readout defects appear as bright or dark vertical lines that span the full
height of a scan and recur at the same sensor position in every scan. To detect them,
each raw scan is divided into six horizontal bands; within each band, the per-column
median intensity profile is high-pass filtered (median filter, 11 px kernel) and
converted to robust z-scores using the median absolute deviation. A column is retained
as a candidate only when at least four of the six bands exceed |z| = 8 with a consistent
sign, and adjacent outlier columns are grouped into clusters of at most 4 px. This
full-height coherence requirement rejects film content and label edges by construction,
and per-scan banding variations are deliberately never masked, since they constitute
part of each device's intrinsic character.

Because each device auto-crops films to different widths, pixel column indices are not
stable sensor coordinates. Each scan is therefore anchored to a common sensor frame by a
one-dimensional translation search over its strongest candidates (|z| > 15; ±60 px
search range; ±2 px match tolerance; accepted only with at least three matched
candidates and a margin of at least two over the next-best offset). Columns recurring in
at least half of the anchored scans are classified as stationary hardware defects, and a
second anchoring pass against the resulting recurrence peaks recovers scans with weaker
candidate sets [3].

From this point onward, the detected defect columns are excluded from the registration
criterion masks on both sides and, in the analysis path below, from the comparison mask,
so that known defective sensor columns can neither bias the alignment nor masquerade as
information loss. Detection results are given in Section 4.3.

### 3.5 Metric mask and rank normalization

All comparison metrics are computed over a per-pair metric mask: the geometric overlap
of the two aligned frames, eroded by 8 px to remove interpolation artifacts, restricted
to pixels unsaturated in both scans (with a further 2 px erosion), and intersected with
the base mask — with the surround exclusion evaluated independently on both images,
since a washed-out histogram on one device would otherwise pass the other's darkness
test and produce a directional coverage asymmetry. Defect columns are excluded last,
after the intensity normalization and surround classification, so that their exclusion
cannot alter the statistics used to classify the remaining pixels; pairs with fewer than
10,000 surviving pixels are rejected [4]. This mask covered on average 49.4% of the
frame (range 34.8–61.1%).

Within the metric mask, each pixel is replaced by the fractional rank of its intensity
among the masked pixels (average ranks for ties), mapping both images to a uniform (0,
1\) histogram. This rank transform removes any strictly monotonic response difference
between the devices — gain, offset, and gamma — exactly, so that the remaining
differences are structural. A least-squares gain/offset fit and per-tile (8 × 8)
residual statistics are computed as diagnostics but are never applied to the images.

### 3.6 Blur calibration

To keep the known sharpness difference between the devices from masquerading as
information loss, we measure and equalize the device blur gap. Each pair is prepared in
both alignment directions with the scale correction and defect mask active, and the
equivalent Gaussian blur that matches the sharper image's edge energy to the other's is
measured on the edge support common to both images. Because resampling the moving scan
adds a small, sub-pixel-phase-dependent blur of its own, a per-pair resampling penalty
is integrated from the interpolation kernel's measured phase-response table (a 6 × 6
sub-pixel grid) over the pair's estimated warp, and the device blur constant is
separated from it by a signed-square decomposition of the two directional gaps; the
median across pairs is taken as the calibration constant (Section 4.2) [3]. At
comparison time, the sharper side's rank image is blurred by the net gap with a
mask-normalized Gaussian, so that masked-out pixels never bleed into the measured
region. Registration feature images are never blurred, and the per-pair residual blur
gap is reported rather than corrected.

### 3.7 Column-gain calibration

Line-sensor digitizers map each image column to one sensor element, so per-element gain
differences appear as full-height vertical banding that differs between devices and
survives local averaging as spatially coherent error. For each pair and direction, the
per-column median of the signed rank difference is taken over the metric mask (columns
with fewer than 200 masked pixels are excluded), and the resulting profiles are
aggregated in the scanner's sensor coordinate frame — each profile shifted by its scan's
defect-anchoring offset — taking the median across pairs, with sensor columns supported
by fewer than three pairs set to zero. Because only the warp-composed combination of the
two devices' bandings is observable from pairwise differences, the aggregated profile is
subtracted from the reference side's rank image only. The solve is performed on fully
corrected images (scale, defect mask, and blur correction active) so that sharpness and
scale systematics do not alias into the profiles [3]. Results are given in Section 4.4.

### 3.8 The comparison run

In the full analysis, each pair is registered with the scale pre-correction and
defect-extended criterion masks; the moving scan is resampled once into the reference
frame; the metric mask is constructed and both images are rank-transformed. The solved
device corrections are then applied to the rank images — first the blur equalization,
then the column-gain subtraction — and the comparison metrics (Section 3.9) and
diagnostic visualizations are computed on the corrected images [4].

### 3.9 Metrics

Value fidelity is measured by the mean absolute error (MAE), root-mean-square error
(RMSE), and peak signal-to-noise ratio (PSNR) of the signed rank difference, computed
without prefiltering. To separate spatially coherent error from independent film grain,
the signed difference is locally averaged with a Gaussian of σ = 4 px and its magnitude
measured over the mask eroded by three standard deviations (local RMSE): uncorrelated
grain averages toward zero under this blur, while coherent local differences survive.

Structural agreement is measured by the structural similarity index (SSIM), and
fine-detail reproduction is assessed on the Sobel gradient magnitudes over the reference
image's edge support (pixels above the 70th percentile of gradient magnitude): the
Pearson correlation tests whether the same edges appear in both scans, and the ratio of
mean edge energies tests whether they are reproduced at the same strength. These
structural metrics are computed after a light Gaussian prefilter (σ = 1 px) that
suppresses irreproducible grain. The residual sharpness gap after correction is reported
as a signed equivalent Gaussian blur in pixels (positive when the second scan is
blurrier), measured on the edge support common to both images.

The effective shared resolution is estimated by Fourier ring correlation (FRC) of the
two corrected rank images: the images' Fourier transforms are correlated in concentric
frequency rings after apodization with a Hann window multiplied by the edge-softened
mask, rings are evaluated up to the inscribed Nyquist circle, and the reported value is
the first crossing — linearly interpolated — of the 0.5 correlation criterion (with the
classic 1/7 criterion reported alongside), expressed in pixels as the smallest structure
size at which the two scans still agree [5].

Pass/fail verdicts are computed against fixed thresholds (RMSE ≤ 0.05, SSIM ≥ 0.95,
gradient correlation ≥ 0.95, gradient energy ratio within ±0.15 of unity), chosen to
describe near-lossless agreement; the remaining quantities are reported without
thresholds. As diagnostics, the residual displacement field between the aligned images
is estimated by DIS dense optical flow and visualized after five-fold amplification, and
per-tile residual statistics summarize spatial variation; neither feeds the pass/fail
decision.

### 3.10 Bidirectional protocol

Every analysis is performed in both directions: forward (the SC113 scan warped onto the
SC108 frame) and reverse (the directories exchanged, with the scale correction inverted
and the blur constant negated). Cross-direction consistency is quantified by per-pair
metric deltas and by the round-trip closure of the composed forward-then-reverse warp,
measured as the largest displacement of the frame corners in pixels.

## 4. Results

### 4.1 Registration quality

- ECC correlation per pair (forward, with defect mask): 0.9694–0.9938 (all 15 pairs in
  the per-pair table, §4.6).
- Defect-mask effect: all 30 registrations' ECC correlations improved vs the unmasked
  run (e.g. FM08y06m 0.9772 → 0.9789).
- Round-trip closure (reverse ∘ forward, largest frame-corner displacement): 0.02–0.22
  px across ~3000 px frames (per-pair values in §4.6).

### 4.2 Calibration constants

| Constant                                                          | Value                                        | Source artifact                  |
| ----------------------------------------------------------------- | -------------------------------------------- | -------------------------------- |
| Scale correction s                                                | 0.9985 (refined 0.99850)                     | `tmpout/scale.json`              |
| Device blur σ_dev (signed; positive = SC113 blurrier)             | **+0.5353 px**                               | `tmpout/blur-signed-masked.json` |
| — one-sided arms (should bracket σ_dev)                           | +0.4521 (fwd) / +0.6072 (rev)                | same                             |
| Resampling penalty r̄ (Lanczos4, phase-averaged)                   | 0.2866 px                                    | same                             |
| r_data (data-solved; cross-check only, compressed to 0 by design) | 0.0                                          | same                             |
| Antisymmetry check (directories swapped)                          | σ_dev = −0.5353 (exact negation; arms swap)  | `tmpout/blur-signed-rev.json`    |
| Column-gain stationary profile rms                                | 0.00282 (SC108) / 0.00308 (SC113) rank units | `tmpout/colgain.json`            |
| Column-gain per-pair coverage                                     | 14/15 pairs anchored per side                | same                             |

- Scale sweep curve (mean masked feature NCC, 3 pairs): 0.9481 @ 0.9965 → 0.9809 @
  0.9980 → **0.9826 @ 0.9985 (peak)** → 0.9593 @ 1.0000 → 0.9436 @ 1.0005.
- Per-pair blur-solve evidence (`blur-signed-masked.json`): m_forward +0.463…+0.592,
  m_reverse −0.650…−0.475, per-pair σ_dev,i² median maps to the 0.47–0.62 px spread;
  per-warp table penalties r_f = 0.2827–0.2828, r_r = 0.2904–0.2905.
- Independent scratch re-validation (diag16, same convention): σ_dev = +0.5332 px, r̄ =
  0.2832 px; correction-sweep zero crossings of the signed residual at 0.459 (fwd) /
  0.624 (rev), bracketing the solve.
- Convention note: constants follow the signed common-support convention (blur.json
  version 2). The superseded own-support convention values (σ_dev 0.6864, r̄ 0.482) are
  retained only for reproducing pre-2026-08 results.

### 4.3 Defect detection

|                                        | SC108                                                                                  | SC113                                  |
| -------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------- |
| Stationary groups                      | 20                                                                                     | 4                                      |
| Group locations (reference-scan frame) | 20 groups between columns 1131–1135 and 2677–2681 (full list in `tmpout/defects.json`) | 178–182, 193–197, 1058–1062, 2547–2551 |
| Anchored scans                         | 14/15 (R055HM08y06m unanchored)                                                        | 14/15 (R055FM09y01m unanchored)        |
| Masked columns per scan                | median 19 (range 3–22)                                                                 | median 4 (range 3–4)                   |
| Masked frame fraction                  | ≈ 0.54 %                                                                               | ≈ 0.12 %                               |

- Cross-scanner verification: no SC108 stationary column has a counterpart in the SC113
  scan of the same film (|z| ≤ 8 at the mapped positions) and vice versa (|z| ≤ 5) —
  consistent with hardware origin, not film content.
- Figures: `tmpout/defects_BBGSC108-NDT-Dosimetry-Pro.png`,
  `tmpout/defects_BBGSC113-Dental-Film-Digitizer.png` (+ per-scan candidate CSVs).

### 4.4 Column-gain (banding) calibration

- Stationary profiles: rms 0.00282 (SC108) / 0.00308 (SC113), smooth long-wavelength
  shape \[figure: `tmpout/colgain_BBGSC*.png` stripe maps / `.csv` columns\].
- Per-pair banding profiles correlate only 0.35–0.66 with the stationary profile.
- Applied per pair (reference side): mean applied rms 0.0032 (fwd) / 0.0035 (rev); 14/15
  pairs corrected per direction (the unanchored scans receive none).
- Effect: mean local_rmse 0.0110 → 0.0102 (fwd) and 0.0113 → 0.0103 (rev) vs the
  no-colgain baseline; RMSE/SSIM/gradient metrics unchanged to the third decimal
  (ablation table, §4.5).

### 4.5 Ablation over correction stages (15-pair means)

| Stage                              | Direction | RMSE   | SSIM  | grad_corr | grad_energy_ratio | local_rmse | blur_sigma |
| ---------------------------------- | --------- | ------ | ----- | --------- | ----------------- | ---------- | ---------- |
| Euclidean only (no scale)          | fwd       | 0.0406 | 0.871 | 0.630     | 0.707             | —          | —          |
| + scale 0.9985, two-sided surround | fwd       | 0.0382 | 0.914 | 0.713     | 0.779             | 0.0111     | +0.843     |
|                                    | rev       | 0.0386 | 0.911 | 0.712     | 0.944             | 0.0114     | +0.469     |
| + defect mask                      | fwd       | 0.0385 | 0.915 | 0.717     | 0.780             | 0.0110     | +0.835     |
|                                    | rev       | 0.0388 | 0.913 | 0.716     | 0.940             | 0.0113     | +0.474     |
| + blur correction                  | fwd       | 0.0303 | 0.927 | 0.738     | 0.901             | 0.0110     | −0.442     |
|                                    | rev       | 0.0345 | 0.919 | 0.730     | 0.887             | 0.0113     | −0.404     |
| + column gain (final baseline)     | fwd       | 0.0301 | 0.927 | 0.738     | 0.896             | 0.0102     | −0.425     |
|                                    | rev       | 0.0347 | 0.919 | 0.729     | 0.897             | 0.0103     | −0.435     |

Notes on the cells (factual):

- The "+ blur correction" stage ran with σ_dev = +0.5528 (run-all-signed's in-chain,
  unmasked solve); the final baseline applied +0.5353. The signed residual then reads
  negative past the zero crossing — documented instrument behavior, not over-correction
  (diag16 sweep: residual −0.436/−0.407 at sd = 0.55, crossing means 0.459/0.624).
- local_rmse is unchanged by the blur correction to ≤ 3e-5 (diag16 acceptance
  criterion).
- Blur-correction artifact pair (before/after figures):
  `agents/scratch/diag16/artifacts/forward_baseline|forward_corrected/` — at the solved
  constant: fwd RMSE 0.0383 → 0.0305, SSIM 0.9155 → 0.9265; rev 0.0387 → 0.0351, 0.9130
  → 0.9183; cross-direction grad-ratio asymmetry −0.160 → −0.002.
- Applied blur sigmas (final baseline): forward reference side mean 0.6054 px (=
  hypot(0.5353, r_i)); reverse moving side mean 0.4497 px.

### 4.6 Per-pair results (final baseline: scale + defects + blur + column gain)

FRC columns from `tmpout/frc-check/` (same correction config); all other columns from
`tmpout/run-colgain/`. "blur resid" = post-correction signed `blur_sigma`; dcols =
masked defect columns (both sides); round-trip = composed warp closure in px.

| Pair     | ECC corr | round-trip px | RMSE fwd/rev    | SSIM fwd/rev  | grad_corr fwd/rev | grad_ratio fwd/rev | blur resid fwd/rev | lrmse fwd/rev   | FRC 0.5 fwd/rev (px) | FRC 1/7 fwd/rev (px) | dcols |
| -------- | -------- | ------------- | --------------- | ------------- | ----------------- | ------------------ | ------------------ | --------------- | -------------------- | -------------------- | ----- |
| FM07y06m | 0.9925   | 0.05          | 0.0269 / 0.0307 | 0.929 / 0.921 | 0.838 / 0.828     | 0.862 / 0.863      | −0.416 / −0.439    | 0.0076 / 0.0075 | 6.8 / 6.8            | 4.5 / 4.3            | 22    |
| FM08y01m | 0.9938   | 0.09          | 0.0276 / 0.0317 | 0.951 / 0.948 | 0.732 / 0.720     | 0.942 / 0.918      | −0.463 / −0.393    | 0.0083 / 0.0081 | 3.9 / 3.7            | 2.3 / 2.0            | 23    |
| FM08y06m | 0.9789   | 0.18          | 0.0480 / 0.0548 | 0.824 / 0.806 | 0.558 / 0.522     | 0.805 / 0.779      | −0.475 / −0.393    | 0.0162 / 0.0158 | 9.2 / 8.9            | 6.7 / 6.7            | 26    |
| FM09y01m | 0.9876   | 0.04          | 0.0221 / 0.0254 | 0.946 / 0.942 | 0.791 / 0.779     | 0.897 / 0.891      | −0.439 / −0.416    | 0.0057 / 0.0064 | 5.0 / 4.8            | 3.2 / 3.2            | 14    |
| FM09y06m | 0.9916   | 0.11          | 0.0331 / 0.0378 | 0.911 / 0.907 | 0.533 / 0.532     | 0.885 / 0.898      | −0.416 / −0.439    | 0.0079 / 0.0075 | 6.5 / 6.5            | 3.6 / 3.6            | 24    |
| HM07y06m | 0.9867   | 0.07          | 0.0197 / 0.0241 | 0.965 / 0.958 | 0.852 / 0.854     | 0.924 / 0.914      | −0.439 / −0.416    | 0.0062 / 0.0064 | 6.0 / 6.0            | 3.8 / 3.7            | 7     |
| HM08y01m | 0.9694   | 0.06          | 0.0252 / 0.0314 | 0.952 / 0.940 | 0.725 / 0.711     | 0.926 / 0.907      | −0.451 / −0.404    | 0.0088 / 0.0097 | 6.4 / 6.4            | 4.6 / 4.6            | 11    |
| HM08y06m | 0.9810   | 0.09          | 0.0265 / 0.0360 | 0.936 / 0.900 | 0.819 / 0.819     | 0.856 / 0.916      | −0.334 / −0.557    | 0.0120 / 0.0150 | 7.8 / 7.8            | 5.5 / 5.6            | 4     |
| HM09y01m | 0.9860   | 0.16          | 0.0586 / 0.0639 | 0.858 / 0.859 | 0.474 / 0.493     | 0.919 / 0.864      | −0.498 / −0.357    | 0.0229 / 0.0216 | 4.2 / 4.2            | 2.8 / 2.6            | 9     |
| HM09y06m | 0.9865   | 0.06          | 0.0408 / 0.0455 | 0.870 / 0.860 | 0.770 / 0.763     | 0.879 / 0.866      | −0.463 / −0.416    | 0.0163 / 0.0153 | 5.7 / 5.7            | 3.7 / 3.7            | 9     |
| LM07y06m | 0.9883   | 0.02          | 0.0196 / 0.0222 | 0.967 / 0.964 | 0.883 / 0.867     | 0.927 / 0.941      | −0.404 / −0.451    | 0.0065 / 0.0065 | 3.5 / 3.5            | 2.0 / 2.0            | 25    |
| LM08y01m | 0.9891   | 0.22          | 0.0212 / 0.0240 | 0.963 / 0.959 | 0.863 / 0.847     | 0.919 / 0.938      | −0.393 / −0.463    | 0.0076 / 0.0076 | 4.0 / 3.8            | 2.2 / 2.0            | 24    |
| LM08y06m | 0.9898   | 0.09          | 0.0326 / 0.0378 | 0.940 / 0.934 | 0.775 / 0.748     | 0.907 / 0.935      | −0.393 / −0.463    | 0.0117 / 0.0114 | 4.3 / 4.3            | 2.8 / 2.8            | 24    |
| LM09y01m | 0.9836   | 0.03          | 0.0254 / 0.0271 | 0.945 / 0.941 | 0.675 / 0.680     | 0.894 / 0.913      | −0.393 / −0.463    | 0.0076 / 0.0075 | 5.7 / 5.4            | 2.0 / 2.0            | 20    |
| LM09y06m | 0.9887   | 0.05          | 0.0245 / 0.0276 | 0.948 / 0.943 | 0.783 / 0.775     | 0.901 / 0.914      | −0.404 / −0.451    | 0.0075 / 0.0075 | 5.4 / 5.4            | 3.2 / 2.7            | 24    |

Cross-direction deltas (fwd − rev, means): ΔRMSE −0.0045, ΔSSIM +0.0084, Δgrad_corr
+0.0099, Δgrad_ratio −0.0017, Δblur_sigma +0.009 (per-pair values in
`tmpout/run-colgain/cross_direction.csv`).

### 4.7 Effective shared resolution (FRC)

- 0.5 criterion: 3.5–9.2 px across the 15 pairs (forward mean 5.63 px; reverse max
  |delta| vs forward 0.23 px). Finest: LM07y06m 3.5 px. Coarsest: FM08y06m 9.2 px.
- 1/7 criterion: 2.0–6.7 px; the value 2.0 is the Nyquist floor (agreement down to the
  sampling limit) reached by 2/15 pairs (LM07y06m; LM09y01m reverse).
- Per-pair curves: `tmpout/frc-check/{forward,reverse}/*_frc.csv`.
- [TODO] pick 1–2 example curve figures for the report.

### 4.8 Threshold verdicts

- 0/15 pairs pass all default thresholds in either direction. Failure counts, forward:
  grad_corr 15/15, ssim 10/15, rmse 1/15, grad_energy_ratio 1/15. Reverse: grad_corr
  15/15, ssim 12/15, rmse 2/15, grad_energy_ratio 1/15.
- [TODO] one sentence of context for readers (the defaults are near-lossless bars; a
  lossless synthetic pair passes with wide margin — Appendix D).

### 4.9 Notable per-pair findings

- **HM09y01m** is the worst pair on both directions: RMSE 0.0586/0.0639, SSIM
  0.858/0.859, grad_corr 0.474/0.493, local_rmse 0.0229/0.0216 (local/raw ratio ≈ 0.39
  vs the typical 0.2–0.3), with a coherent density blob in `local_diff` and a saturated
  full-height bright line near the right edge; registration healthy (corr 0.986). It is
  also the scale-sweep outlier (prefers s ≈ 0.9992 vs the global 0.9985).
- **FM08y06m** has the coarsest FRC (9.2 px) and the lowest SSIM (0.824 fwd) with the
  largest defect count (26 columns) and an unanchored SC108 scan (no defect mask on one
  side in that pair's direction chain — [TODO] double-check which side/direction before
  writing this up).
- motion_diff structures: skull-edge fringe and orbit-height horizontal line candidates
  on FM07y06m (residual non-rigid geometry; unchanged by the blur correction). [TODO]
  decide whether the tiled phase-correlation line-shift probe belongs here or in Future
  Work; it has not been run as a library feature.

## 5. Discussion

[TODO — interpretation lives here; data to draw on is all in §4. Placeholder prompts:]

- What does the FRC range mean for the practical question (which structures / clinical
  features survive both digitizations)?
- The device characterization: SC113 intrinsically ~0.54 px blurrier than SC108 (σ_dev),
  with a real per-scan spread (0.47–0.62 px) attributable to focus/transport.
- grad_corr 0.47–0.88 as grain-level irreproducibility between physically different
  digitizers — expected, and the reason thresholds need a same-scanner rescan reference.
- Banding: stationary share removed (local_rmse −7…−9 %); the larger per-scan-varying
  share (per-pair profile correlations 0.35–0.66 with the stationary profile) remains by
  design.
- The bidirectional protocol as methodology: exact blur antisymmetry, symmetric FRC,
  sub-quarter-pixel warp round-trip.

## 6. Known limitations

- **Defect mask**: a real but weak SC113 defect cluster (columns ~679–683, z 5–10, above
  threshold in only 3/15 scans) can never pass the ≥ 50 % recurrence gate and is not
  masked (conservative by design; metric impact tiny).
- **Column gain**: only the stationary share of the banding is removable; per-scan
  banding remains visible in local_diff by design (it is scanner character, and per-pair
  correction would risk absorbing real content differences).
- **Thresholds**: defaults are near-lossless bars, not deployment acceptance criteria; a
  same-scanner rescan pair for calibration does not exist in this dataset.
- **blur_sigma instrument**: saturates toward the dominant blur term (the resampling
  component cannot be solved from it — hence the phase table); reads a phantom ±0.2–0.3
  px on shifted correlated content (not a valid null test on translation probes; a true
  null needs a same-scanner rescan pair).
- **FRC**: one unexplained fine-scale recovery on FM08y06m (~0.6 correlation at 2–2.2
  px; suspected mask-rim renormalization of the blur correction; source unknown). The
  first-crossing rule makes the reported value robust to it; full curves are preserved.
- **Round-trip check** validates the global (similarity) transforms only;
  local/non-rigid residuals are visible in motion_diff but have no shipped quantitative
  metric.
- **Dataset**: 15 films from one batch, one device pair; PNG DPI metadata is wrong on
  both devices (96 vs nominal 600 ppi) — all measurements are therefore in pixel units.

## 7. Conclusion

[TODO] 3–4 sentences.

## 8. Future work

- Same-scanner rescan pair → data-driven pass/fail thresholds (the long-standing top
  item).
- Tiled phase-correlation line-shift probe on the aligned log-DoG images (decide
  piecewise-Euclidean correction vs report-only; motion_diff already shows candidates).
- Per-pair column-gain estimation with an absorption guard (for the non-stationary
  banding share).
- [TODO] anything else you want to promise (e.g. non-rigid registration research).

______________________________________________________________________

## References

1. BBGSC, *scanner-comparison* (software), version 0.1.0. \[TODO: repository URL /
   archive DOI\]
1. "Registration: log-DoG features and masked Euclidean ECC," scanner-comparison
   technical documentation, `docs/registration.md` [1].
1. "Calibration: measuring the device constants," scanner-comparison technical
   documentation, `docs/calibration.md` [1].
1. "The analysis pipeline, stage by stage," scanner-comparison technical documentation,
   `docs/analysis-pipeline.md` [1].
1. "Metrics: definitions, purposes, and how to read them," scanner-comparison technical
   documentation, `docs/metrics.md` [1].

## Appendix A — Reproduction

### A.1 Commands

```bash
# full calibration + analysis, both directions (what produced run-colgain):
scanner-compare run_all \
  "test-images/235_Scanner_Comparison-selected/BBGSC108 NDT Dosimetry Pro" \
  "test-images/235_Scanner_Comparison-selected/BBGSC113 Dental Film Digitizer" \
  --out tmpout/run-colgain --both-directions

# step-by-step equivalent (the constants used in the reported runs):
scanner-compare find_scale DIR_A DIR_B --out tmpout/scale.json
scanner-compare find_defect_mask DIR_A DIR_B --out tmpout/defects.json
scanner-compare find_blur DIR_A DIR_B --scale-correction 0.9985 \
  --defect-mask tmpout/defects.json --out tmpout/blur-signed-masked.json
scanner-compare find_column_gain DIR_A DIR_B --scale-correction 0.9985 \
  --blur-correction 0.5353 --defect-mask tmpout/defects.json --out tmpout/colgain.json
scanner-compare run_analysis DIR_A DIR_B --out tmpout/run-colgain \
  --scale-correction 0.9985 --blur-correction 0.5353 \
  --defect-mask tmpout/defects.json --column-gain tmpout/colgain.json --both-directions
```

### A.2 Run directory inventory

| Directory                         | Config                                                          | Role                       |
| --------------------------------- | --------------------------------------------------------------- | -------------------------- |
| `tmpout/run-colgain/`             | scale 0.9985 + defects + blur 0.5353 + colgain, both directions | golden baseline            |
| `tmpout/frc-check/`               | identical + FRC metric                                          | FRC evidence               |
| `tmpout/full-check/`              | identical (blur 0.53527 unrounded)                              | parity/checkpoint evidence |
| `tmpout/run-all-signed/`          | scale + defects + blur 0.5528 (in-chain unmasked solve)         | ablation stage             |
| `tmpout/run-both-defects-masked/` | scale + defects                                                 | ablation stage             |
| `tmpout/run-both-surround2/`      | scale only, two-sided surround                                  | ablation stage             |
| `tmpout/run-rank/`                | Euclidean-only, single direction, pre-local_metrics             | earliest baseline          |

## Appendix B — Calibration artifact schemas

Field-level documentation of each artifact is in the calibration reference [3].

- `scale.json` (version 1): `scale`, `scale_refined`, `candidates`, `mean_ncc`,
  `pairs_used`, `generated_at`.
- `defects.json` (version 1): `params` (all detection constants),
  `directories{<abs path>{reference_scan, stationary_columns_ref_frame[[start,end,recurrence]…], scans{<name>{x_offset, anchored, candidate_count, defect_columns_native}}}}`.
- `blur.json` (version 2, `"convention": "signed common-support bidirectional"`):
  `sigma_dev`, `sigma_dev_forward`, `sigma_dev_reverse`, `r_bar`, `r_data`,
  `scale_correction`,
  `pairs[{name, m_forward, m_reverse, sigma_dev_sq_signed, r_data, r_forward_table, r_reverse_table}]`.
- `colgain.json` (version 1):
  `params{scale_correction, blur_correction, min_column_rows, min_pair_profiles}`,
  `directories{<abs path>{reference_scan, width, n_pairs, rms, profile[], scan_offsets{<name>: x_offset}}}`,
  `pairs[{name, direction, rms, n_columns}]`.

## Appendix C — Instrument validation on synthetic fixtures

From the test suite (113 tests) and the diag16 control probes:

| Probe                                                      | Expected                                           | Measured                                                                         |
| ---------------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------- |
| Identical pair, blur solve                                 | σ_dev = 0                                          | +0.0000 px                                                                       |
| Planted device blur 1.2 px (intensity domain)              | ≈ +1.0–1.2 (simulator's own content blur included) | +1.0396 px                                                                       |
| Same solve, directories swapped                            | exact negation                                     | −1.0396 px                                                                       |
| Planted 2.0 px blur (rank domain)                          | ±2.0                                               | ±1.998 px                                                                        |
| Planted 0.6 px device blur through the real warp path      | σ_dev ≈ 0.6, r under-read                          | σ_dev = +0.6035, r_data = 0.00 (quadrature compression)                          |
| Same-scan sub-pixel resample (own-support metric)          | 0                                                  | +0.43…0.69 both ways (the support artifact that motivated common support)        |
| Same-scan sub-pixel resample (common support)              | fwd ≈ r, rev ≈ 0                                   | fwd 0.28–0.30, rev 0.000                                                         |
| FRC synthetic fixture                                      | crossing at planted scale                          | 4.57 px (0.5 criterion) / 3.96 px (1/7)                                          |
| FRC 7 px-rolled control                                    | decorrelation at fine scales                       | decorrelates (vs the silent-0.9 failure of the hard-mask first implementation)   |
| FRC pixel-shuffled control                                 | ≈ 0 everywhere                                     | ≈ 0                                                                              |
| FRC forward/reverse symmetry on real data                  | ≈ equal                                            | max                                                                              |
| Blur antisymmetry on real data (independent registrations) | m_f ≈ −m_r                                         | per-pair sums ≤ 0.08 px; constant-level ±0.5353 exact                            |
| Blur sweep closure (diag16)                                | signed residual crosses 0 at ≈ the solve           | crossings 0.459 (fwd) / 0.624 (rev) vs solve 0.533 (within the 0.1 px stop rule) |

## Appendix D — Glossary

[TODO] draft definitions: DoG (difference of Gaussians), log-DoG, ECC (enhanced
correlation coefficient), MTF, percentile-rank / empirical-CDF transform, edge support,
FRC (Fourier ring correlation) + the 0.5 and 1/7 criteria, PRNU / column banding,
stationary defect, sensor frame (auto-crop anchoring), rank units.
