# Metrics: definitions, purposes, and how to read them

All metrics are computed on the **percentile-rank images** over the **final metric
mask** (see [analysis-pipeline.md](analysis-pipeline.md), stages 3–5), i.e. after
registration, masking, rank normalization, and the configured device corrections. The
rank domain is what makes them meaningful: identical physical content lands at identical
values regardless of each scanner's (monotonic) intensity response, so every metric
compares *structure*, not scanner response curves.

A structural distinction up front:

- **Value-fidelity metrics** (`mae`, `rmse`, `psnr`) intentionally run on the **raw**
  rank images — no prefilter — so they see everything, including grain.
- **Structural and sharpness metrics** (`ssim`, `grad_corr`, `grad_energy_ratio`,
  `blur_sigma`) run after a light **Gaussian prefilter (σ = 1 px)**. Film grain / sensor
  noise is a different physical realization in each scan and cannot be reproduced
  pixel-perfectly by a second scanner; without the prefilter it would dominate the
  structural scores without representing anatomical information loss.

______________________________________________________________________

## Value fidelity: `mae`, `rmse`, `psnr`

The signed difference image is `diff = ref_rank − mov_rank`; MAE and RMSE are its mean
absolute and root-mean-square values over the mask; PSNR is `20·log10(1/rmse)`
(unit-range data, capped at 99 dB for identical images).

**Purpose:** the baseline "how different are the pixel values" answer. On the
development dataset, corrected forward runs land at rmse ≈ 0.03 (PSNR ≈ 30 dB).

**Reading:** the default pass bar is `rmse ≤ 0.05` — a deliberately near-lossless
threshold; calibrate against a same-scanner rescan pair when one is available. RMSE
alone cannot distinguish *where* the error lives (grain vs. anatomy) — that is what
`local_rmse`, the gradient metrics, and FRC are for.

## The coherence split: `local_mse`, `local_rmse`

The signed diff is blurred with a Gaussian of **σ = 4 px** and measured (MSE and its
root) over the mask **eroded by 3σ**, so the blur never mixes in the zeros outside the
mask.

**Purpose:** this is the *spatial-coherence* instrument, and one of the most informative
numbers the tool produces:

- Differences that are **spatially uncorrelated** (film grain, independent sensor noise)
  average toward zero under the local blur → `local_rmse ≪ rmse`.
- Differences that are **spatially coherent** (a real local data mismatch — a shading
  cast, a banding residual, a region one scanner rendered differently) survive the
  averaging → `local_rmse ≈ rmse`.

**Reading:** the ratio `local_rmse / rmse` is the diagnostic. On the development dataset
it is typically 0.2–0.3 (most of the raw error is irreproducible grain); the known
outlier pair reads ~0.46 and its `local_diff` map shows a coherent density blob — a real
per-film difference, not device noise. The blur correction provably leaves `local_rmse`
unchanged (≤ 3e-5 shift), confirming it removes only the sharpness axis. No threshold is
enforced — it is a reported diagnostic.

## Structure: `ssim`

**SSIM** (structural similarity, scikit-image, `data_range = 1`) is computed on the
prefiltered rank images; the reported value is the mean of the per-window SSIM map over
the mask, and the full map is rendered as `*_ssim.png` (defects amplified ×10).

**Purpose:** the perceptual-structure check — local luminance/contrast/structure
agreement per window rather than per pixel. Complementary to RMSE: SSIM tolerates smooth
response differences better and penalizes structural ones harder.

**Reading:** default pass bar `ssim ≥ 0.95`. Corrected development-dataset runs land at
0.92–0.95; again, these bars describe *excellent* agreement and await same-scanner
rescan calibration.

## Fine detail: `grad_corr` and `grad_energy_ratio`

Both metrics operate on **Sobel gradient magnitudes** of the prefiltered rank images,
restricted to the **edge support**: pixels where the *reference* gradient exceeds its
70th percentile over the mask (with a 1000-pixel minimum and a whole-mask fallback on
degenerate content). Restricting to edges matters: flat interior regions carry only
irreproducible grain after alignment, which would dilute the correlation toward zero
without indicating any loss.

- `grad_corr` — **Pearson correlation** of the two images' gradient magnitudes over the
  edge support. *Are the same edges present at the same places?*
- `grad_energy_ratio` — **mean(grad_mov) / mean(grad_ref)** over the edge support. *Are
  the edges reproduced with the same strength?* A value well below 1 means the second
  scan is blurrier than the reference; above 1 means sharper.

**Purpose:** the **most sensitive detectors of fine-detail (MTF) loss**. Pixel-value
metrics can look acceptable while edge energy is silently attenuated; these two cannot
be flattered that way.

**Reading:** on the development dataset `grad_corr` is 0.47–0.88 — grain-level texture
genuinely does not reproduce between two physically different digitizers, so the 0.95
default is a near-lossless bar awaiting same-scanner rescan calibration.
`grad_energy_ratio` near 1 with `blur_sigma` near 0 after correction confirms the MTF
equalization worked. Default pass bars: `grad_corr ≥ 0.95`, energy ratio within ±0.15 of
1\.

## Sharpness gap: `blur_sigma` (signed, common support)

The equivalent **Gaussian blur σ (in pixels)** that, applied to the sharper image,
matches the other's mean edge energy — bisected over σ ∈ [0, 3 px]. Sign convention:
**positive = the second (moving) scan is blurrier**; negative = it is sharper; 0 = no
measurable gap.

Two instrument decisions are load-bearing:

- **Common support.** The energy is averaged over the *intersection of both images'*
  edge sets (each its own q70 support). On each image's own support the one-sided
  measurement floors at 0 and reads a *phantom positive gap in both directions*
  (support-selection asymmetry plus interpolation ringing — measured: a sub-pixel
  resample of identical content read +0.43…0.69 px both ways on own support, vs +0.29 /
  0.00 on common support). The common support makes the sign meaningful.
- **Signed by construction.** Blur A→B; if that helps, the gap is positive, else blur
  B→A and negate. Exactly antisymmetric:
  `signed_blur_gap(a, b) = −signed_blur_gap(b, a)`, so the two directions of a
  `--both-directions` run read as approximate negatives of each other — a built-in
  consistency check (in `cross_direction.json`, the *sum* of the two directions is the
  antisymmetry stat; the delta reads ~2× the gap).

**Purpose:** before correction, it measures the device sharpness gap in physical units;
after `--blur-correction`, it reports the **residual** gap — the primary evidence that
the correction closed the gap it intended to (and, past the crossing, it goes slightly
negative: the signed instrument is sensitive to over-blur, which the old one-sided
floored metric was blind to).

**Caveats (measured during validation):** the instrument **saturates toward the dominant
blur term** — a resampling penalty much smaller than the device gap is compressed away,
so never solve the resampling component from this number (it comes from the kernel phase
table instead); and on shifted-but-correlated content the support intersection lands on
edge flanks and reads a phantom ±0.2–0.3 px — i.e. it is not a valid null test on
single-phase translation probes. On aligned two-scanner pairs (its operating point) it
is the validated sharpness-gap instrument. Never thresholded — always reported.

## Effective shared resolution: `frc_resolution_px`, `frc_resolution_px_17`

**Fourier ring correlation (FRC)** — the standard effective-resolution instrument of
microscopy/cryo-EM, adopted here to answer: *"at what pixel level are details conserved
across both scans?"*

The two corrected rank images' Fourier transforms are correlated in concentric
one-pixel-wide frequency rings. The curve starts near 1.0 (coarse structure agrees) and
decays toward 0 as independent film grain takes over. The first crossing below a
correlation criterion (linearly interpolated between rings) defines a frequency `f*`;
the reported resolution is `1/f*` pixels — **structures larger than this many pixels
agree across both scans**. Two criteria are reported: the conservative **0.5** and the
classic **1/7** (van Heel). A curve that never crosses reports 2.0 — conserved down to
the Nyquist limit; a curve below threshold at the coarsest ring reports no value.

**Windowing is load-bearing** (both lessons learned the hard way, from a first
implementation that was silently wrong): the images are cropped to the mask bounding
box, mean-subtracted, and apodized by a **2-D Hann window times the edge-softened mask**
(2 px Gaussian rim). A hard mask boundary is an *identical* step edge in both spectra
and correlates every ring (measured: curves floored at ~0.9 even for mismatched
content). Rings stop at the inscribed Nyquist circle — the rectangular FFT's corner
regions are sampling artifacts, not data. The first two rings (DC neighborhood) are
skipped: they carry the window envelope, not content agreement.

**Purpose:** the headline, symmetric, resolution statement. Because the inputs are the
*corrected* rank images, it is the noise-limited shared resolution **after** MTF
equalization — the blur correction cannot inflate it. Symmetric by construction
(`Re(F₁·conj(F₂)) = Re(F₂·conj(F₁))`), so both directions of a bidirectional run read
the same value (measured: max |Δ| = 0.23 px). On the development dataset: **3.5–9.2 px**
across the 15 pairs (mean ≈ 5.6 px; 1/7 criterion: 2.0–6.7 px, flooring at Nyquist on
the best pairs). The full curve per pair is preserved in `*_frc.csv`. Reported only — no
pass/fail threshold.

## Coverage: `overlap_fraction`, `n_pixels`

The fraction of the frame (and absolute count) that survived the metric mask. Guards the
interpretation of everything else: a metric computed over 20 % of a frame is a different
statement than over 45 %. Typical development-dataset coverage is ~42 % (borders,
corners, surround, saturation, and defect columns removed).

## Reported-only diagnostics (never pass/fail, never corrected)

These exist so that *spatial* response differences are visible without ever being
silently removed:

- **`gain`, `offset`** — least-squares linear fit of moving → reference over the mask:
  the residual linear response gap after independent percentile normalization. A
  scanner-response diagnostic.
- **`tile_residual_mean` / `tile_residual_std`** — the overlap bounding box is split
  into an 8×8 grid; the per-cell residual standard deviations are summarized (mean and
  spread). A large spread relative to the mean flags spatially varying differences
  (shading, vignetting) rather than a uniform response gap.
- **`reg_correlation` / `reg_scale`** — the final ECC correlation and applied scale
  correction per pair.
- **`defect_columns_masked`**, **`blur_applied`** `(r_i, σ_ref, σ_mov)`,
  **`colgain_rms`** — the correction records: exactly what was equalized for this pair.

## The diagnostic artifacts (image-space companions)

The signed heatmaps are not metrics, but each is built to isolate one error axis (all
red = reference brighter, blue = second scan brighter):

| Artifact            | Amplification                      | Isolates                                                                                                 |
| ------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `*_diff.png`        | ×5                                 | everything, at pixel scale                                                                               |
| `*_local_diff.png`  | ×20                                | spatially coherent error (grain averaged away) — the picture of `local_rmse`                             |
| `*_motion_diff.png` | ×5 (on 5×-amplified residual flow) | sub-pixel misregistration and non-rigid residuals (line-shift class) — intensity noise does not light up |
| `*_ssim.png`        | ×10                                | local structural-similarity defects                                                                      |
| `*_overlay.png`     | —                                  | red/green channel overlay; yellow = well-aligned identical content                                       |

The `motion_diff` instrument deserves its caveat repeated: dense optical flow (DIS,
medium preset) estimates the residual displacement field between the aligned rank
images, the field is amplified ×5, and the moving image is re-warped by it before
differencing. It is **strictly a diagnostic** — the amplified image never feeds any
metric — because aggressive non-rigid warping could erase the very information-loss
signal the tool exists to detect. For the same reason the evaluation mask is eroded an
extra 16 px for this artifact (the amplified warp samples beyond the nominal overlap).

## Threshold summary

| Metric                                                                 | Default bar       | Enforced?       |
| ---------------------------------------------------------------------- | ----------------- | --------------- |
| `rmse`                                                                 | ≤ 0.05            | yes (exit code) |
| `ssim`                                                                 | ≥ 0.95            | yes             |
| `grad_corr`                                                            | ≥ 0.95            | yes             |
| `grad_energy_ratio`                                                    | within ±0.15 of 1 | yes             |
| `mae`, `psnr`, `local_*`, `blur_sigma`, `frc_*`, coverage, diagnostics | —                 | reported only   |

The defaults describe *excellent* (near-lossless) agreement: a lossless synthetic pair
passes with a wide margin. Between two physically different digitizers, `grad_corr` in
particular is *expected* to fail — that is the true finding for grain-level texture, and
the reason same-scanner rescan calibration is the recommended way to set deployment
thresholds.
