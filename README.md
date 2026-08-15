# scanner-comparison

Compare duplicate film scans digitized on two different scanners to detect information
loss. Pairs images by filename, aligns them (Euclidean ECC registration), transforms the
overlapping region to percentile ranks (removing any monotonic brightness/gamma
difference), and reports pixel-fidelity, structural, and sharpness metrics plus
difference heatmaps.

The method is documented in depth in [docs/](docs/README.md): the pipeline stage by
stage, the log-DoG registration feature space, every metric and its purpose, the
calibration modes, and the outline of the accompanying report.

## Usage

```bash
scanner-compare run_analysis DIR_A DIR_B --out results/
# the mode may be omitted (legacy form):
python -m scanner_comparison DIR_A DIR_B --out results/
# both alignment directions (B onto A into results/forward, A onto B into
# results/reverse, with the scale correction inverted automatically):
python -m scanner_comparison run_analysis DIR_A DIR_B --out results/ --both-directions
```

A `--both-directions` run additionally writes `cross_direction.json` /
`cross_direction.csv` in the output root: per-pair metric deltas (forward − reverse)
plus `roundtrip_max_px`, the largest frame-corner displacement of the composed
forward-then-reverse warp. The round-trip is identity for inverse-consistent
registrations, so a value beyond ~1 px flags an inconsistency between the directions;
note it checks the global transform only, not local (non-rigid) residuals. If the two
scanners have a known sampling-pitch mismatch, pass it as `--scale-correction S` (e.g.
0.9985): the moving scan is resampled by the uniform factor S before Euclidean ECC
registration. A systematic sharpness difference can be matched with
`--blur-correction B` (e.g. 0.533): the sharper side's rank image is blurred by the net
device+resampling gap (sign: positive = DIR_B blurrier; a negative value is valid and
means DIR_B is sharper; reversed automatically in `--both-directions`), so blur
differences do not masquerade as other error; the per-pair `blur_sigma` column then
reports the signed *residual* gap. Exit code is 0 when every pair passes its thresholds,
1 otherwise. See `scanner-compare --help` for threshold options.

## Calibration modes

The calibration constants are measured by the tool itself:

```bash
scanner-compare find_scale DIR_A DIR_B --out scale.json
scanner-compare find_defect_mask DIR_A DIR_B --out defects.json
scanner-compare find_blur DIR_A DIR_B --scale-correction 0.9985 \
  --defect-mask defects.json --out blur.json
scanner-compare find_column_gain DIR_A DIR_B --scale-correction 0.9985 \
  --blur-correction 0.5332 --defect-mask defects.json --out colgain.json
```

`find_scale` sweeps candidate corrections (registration re-optimized per candidate,
scored by masked log-DoG feature NCC at the resulting warp) and prints the recommended
`--scale-correction`. `find_blur` solves the signed device blur constant on
scale-corrected images by a bidirectional signed common-support measurement: each pair's
blur gap is measured in both directions (the reverse built by swapping the paths and
inverting the scale correction) and the signed gaps are combined with the per-warp
resampling penalties (from the interpolation kernel's measured phase response,
integrated over each warp's sub-pixel phase field) in quadrature, median over pairs. The
constant is antisymmetric — swapping the two directories negates it. The JSON
(`version: 2`, `"convention": "signed common-support bidirectional"`) also reports the
one-sided arm solves (they should bracket the constant), the phase-averaged `r_bar` used
for the apply-time resampling penalty, and the data-solved `r_data` — a cross-check
only: the edge-energy instrument cannot see the resampling component when the device gap
dominates, so expect `r_data` near zero while the true per-warp penalty comes from the
table. All four write inspectable JSONs plus printed recommendations.

Measured constants for the development scanner pair (BBGSC108 NDT Dosimetry Pro vs
BBGSC113 Dental Film Digitizer, `--scale-correction 0.9985`, defect-masked):
`sigma_dev = +0.5332 px`, `r_bar = 0.2832 px` — i.e. `--blur-correction 0.5332`. These
supersede the previous own-support convention's constants (0.6864 / 0.482), which are
kept only for reproducing pre-2026-08 results; regenerate them if the warp interpolation
kernel changes.

`run_all` chains everything — find_scale, find_defect_mask, find_blur, find_column_gain,
then run_analysis with all corrections applied:

```bash
scanner-compare run_all DIR_A DIR_B --out results/ --both-directions
```

## Column-gain (banding) calibration

Line-sensor digitizers map each image column to one sensor element, so per-element gain
differences appear as full-height vertical banding that differs between the two devices
and survives the `local_diff` average as spatially coherent error (the defect mask
removes only discrete defect columns, not this smooth column structure).
`find_column_gain` measures the stationary column-coherent difference profile on fully
corrected images (scale + blur + defect mask): per pair and alignment direction it takes
the per-column median of the signed rank difference over the metric mask, then
median-aggregates the per-pair profiles across all films in the reference scanner's
native frame — film content changes per scan and averages out, the device banding does
not. (The two scanners' bandings are not separately observable from pair differences;
the profile is exactly their warp-composed combination, which is what the correction
needs.) Applying it with `run_analysis ... --column-gain colgain.json` subtracts the
profile from the reference side's rank image; the per-pair `colgain_rms` column reports
the applied magnitude. `find_column_gain` writes the inspectable `colgain.json` plus,
per scanner directory, a profile-map PNG (`colgain_<directory>.png` — the stripe image
of what is subtracted, red = reference reads brighter there, same sign convention as the
`*_diff.png` artifacts) and a per-column CSV.

## Column-defect mask

A sensor/readout column defect appears as a bright or dark vertical line spanning the
whole scan height and recurring at the same sensor position in every scan. Detect and
mask them so they cannot masquerade as information loss:

```bash
scanner-compare find_defect_mask DIR_A DIR_B --out defects.json
scanner-compare run_analysis DIR_A DIR_B --out results/ --defect-mask defects.json
```

`find_defect_mask` writes the inspectable `defects.json` (per-directory stationary
column groups, per-scan crop offsets and masked columns) plus a defect-map PNG and
candidate CSV per scanner directory. Detection requires full-height band coherence per
scan and recurrence across at least half of the (auto-crop-realigned) scans, so film
content (label edges, texture) and per-scan banding noise are never masked; scans that
cannot be anchored confidently are left unmasked and flagged in the JSON. Without
`--defect-mask`, `run_analysis` prints a warning and masks nothing. The reported
per-pair `defect_columns_masked` count records how many columns were excluded.

## Reading the output

Per pair, these artifacts are written next to `summary.json`/`summary.csv`:

- `*_overlay.png` — red = reference scan, green = aligned second scan, both
  percentile-rank transformed; well-aligned identical content appears yellow, red/green
  fringes show structural differences or misalignment.
- `*_diff.png` — signed rank-domain difference (reference - second scan), amplified 5x:
  red = reference brighter, blue = second scan brighter.
- `*_local_diff.png` — the signed difference locally averaged with a 4 px Gaussian
  (amplified 20x): uncorrelated grain averages away, so what remains is spatially
  coherent error. `local_rmse` / `rmse` quantifies this split.
- `*_motion_diff.png` — motion-amplified signed difference: the residual displacement
  field (dense optical flow) is exaggerated 5x before warping, so sub-pixel
  misregistration lights up while intensity noise does not.
- `*_ssim.png` — heatmap of local SSIM defects (amplified 10x).
- `*_rank_ref.png` / `*_rank_mov.png` — the percentile-rank images the metrics are
  computed on (masked pixels form a uniform (0, 1) histogram by construction).
- `*_logdog_ref.png` / `*_logdog_mov.png` — the log-DoG band-pass feature images the
  registration matched, each in its own frame.
- `*_logdog_aligned.png` — the moving log-DoG image warped into the reference frame;
  comparing it against `*_logdog_ref.png` shows the alignment quality in feature space.
- `*_mask.png` — the final comparison mask (metric overlap region).
- `*_frc.csv` — the per-pair Fourier ring correlation curve (`freq_cyc_per_px`, `frc`).

All metrics (MAE/RMSE/PSNR, SSIM, gradient correlation and energy ratio) are computed in
the rank domain: each masked pixel is replaced by the fractional rank of its intensity
among the masked values, so identical physical content lands at identical values
regardless of each scanner's brightness/gamma response. Note that the rank transform
stretches flat-region sensor/grain noise along with the signal; the structural metrics
prefilter lightly, but calibrate thresholds against a same-scanner rescan pair if one is
available.

Default thresholds (rmse ≤ 0.05, ssim ≥ 0.95, gradient correlation ≥ 0.95, gradient
energy ratio within ±0.15 of 1) describe *excellent* agreement; a lossless synthetic
pair passes them with wide margin. The most sensitive indicator of fine-detail loss is
the gradient energy ratio: values well below 1 mean the second scan is blurrier than the
reference; `blur_sigma` expresses the same gap in pixels as a signed common-support
measurement — the equivalent Gaussian blur that, applied to the sharper image, matches
the other's edge energy on the shared edge set (positive = second scan blurrier,
negative = second scan sharper), so the two directions of a `--both-directions` run read
as approximate negatives of each other. `grad_corr` near zero means grain-level texture
does not reproduce at all — expected between physically different digitizers; calibrate
`--min-grad-corr` against a same-scanner rescan pair if one is available.

For a direct "what size of detail survives both scanners" statement, use
`frc_resolution_px`: the effective shared resolution from the Fourier ring correlation
of the two corrected rank images — the FRC curve starts near 1.0 at coarse scales and
decays to ~0 where independent film grain dominates, and the reported value is the
crossing of the 0.5 correlation criterion, in pixels (`frc_resolution_px_17` uses the
classic 1/7 criterion). Read it as: structures larger than this many pixels agree across
both scans. The value is symmetric by construction (both directions of a
`--both-directions` run read the same), and 2.0 means detail is conserved down to the
Nyquist limit. The underlying curve is written per pair as `*_frc.csv`.

## Region of interest

Registration runs in a log-DoG band-pass feature space with both images masked to the
film content; metrics are computed over the same region. Excluded by default: a
`--border-margin` band (3%), `--corner-margin` beveled corners (8%), and the dark
non-film surround (disable with `--no-background-exclusion`).

## Development

Environment is provided by the nix flake (`direnv allow`, or `nix develop`). Python
dependencies live in `pyproject.toml` at the repository root (managed with `uv add`);
the flake mirrors them via uv2nix. Run `uv lock` after changing dependencies, and
`git add` the changed files so nix can see them.

The package (`src/scanner_comparison/`) is layered — each subpackage depends only on the
ones before it, and each `__init__.py` re-exports its public API:

```
core         typed image arrays, the OpenCV boundary, image IO
imaging      normalization, registration, metrics, motion — pure functions on arrays
calibration  the device-calibration domains and their JSON artifacts
records      run configuration and outcome records (plus the direction rules)
report       per-pair artifacts and run-level summaries
pipeline     the preparation chain, the comparison run, and the
             prepare-dependent calibration solves (blur, column gain)
cli          the scanner-compare entry point
```

Checks: `nix flake check` runs formatting, `ruff`, `basedpyright`, and `pytest`. Run
`nix fmt` to auto-format.
