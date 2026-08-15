# scanner-comparison — technical documentation

This directory documents **how the analysis works**: the problem the tool solves, the
design decisions behind each pipeline stage, what every metric measures and why, and how
the device-calibration constants are produced. It is the method reference that
accompanies the code; the top-level [README](../README.md) covers installation and
usage.

## The problem

Two digitizers scan the same physical radiograph films. The question the tool answers
is: **does the second scan preserve all the information in the first?** — and if not,
where, and at what spatial scale.

Naive pixel comparison cannot answer this. Two scanners differ in brightness, gamma,
contrast response, sensor gain per element, sampling pitch, sharpness (MTF), and they
auto-crop each film slightly differently. Every one of those device differences would
show up as "error" in a raw difference image while saying nothing about information
loss. The pipeline is therefore built around one organizing principle:

> **Remove or equalize every known device systematic first, then measure what remains.**

What remains after alignment, rank normalization, and blur/banding equalization is the
candidate information-loss signal.

## The pipeline at a glance

For each pair of scans (matched by identical file name):

1. **Registration** — the second scan is aligned to the first with masked, Euclidean
   (rotation + translation) ECC in a *log-DoG band-pass feature space*, coarse-to-fine,
   with a known inter-scanner scale difference applied as a pre-correction constant. See
   [registration.md](registration.md) for what DoG is and why we take its logarithm
   first.
1. **Metric mask construction** — only pixels that carry comparable anatomical content
   may enter the comparison: geometric overlap of the two frames, unsaturated pixels,
   the film region (borders and beveled corners excluded), the dark non-film surround
   excluded on *both* images, and stationary column-defect lines excluded last.
1. **Percentile-rank transform** — each masked pixel is replaced by the fractional rank
   of its intensity among the masked pixels. This removes *any strictly monotonic*
   response difference (gain, offset, gamma) exactly, so the remaining differences are
   structural.
1. **Device corrections** (rank domain, reference side only where applicable) — the
   measured sharpness gap is equalized by blurring the sharper side, and the stationary
   per-column banding profile is subtracted, so those known device differences cannot
   masquerade as information loss. See [calibration.md](calibration.md).
1. **Measurement** — value fidelity (MAE/RMSE/PSNR), structure (SSIM), fine detail
   (gradient correlation and energy ratio), the residual signed blur gap, the
   coherent-vs- incoherent split (local RMSE), and the effective shared resolution
   (Fourier ring correlation). See [metrics.md](metrics.md).
1. **Diagnostics and report** — signed difference heatmaps (raw, locally averaged,
   motion-amplified), an alignment overlay, the SSIM defect map, all intermediate
   images, and run-level JSON/CSV summaries. With `--both-directions`, the
   reverse-direction run and a cross-direction consistency report (including the warp
   round-trip closure in pixels) are produced as well.

## Documents

| Document                                     | Contents                                                                                                                                         |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| [analysis-pipeline.md](analysis-pipeline.md) | End-to-end walkthrough of one comparison run, stage by stage, with the load-bearing orderings explained                                          |
| [registration.md](registration.md)           | What the difference of Gaussians (DoG) is, why we apply it in log intensity, masked ECC, the pyramid, ORB fallback, and the scale pre-correction |
| [metrics.md](metrics.md)                     | Every reported metric: definition, purpose, how to read it, and which thresholds are enforced                                                    |
| [calibration.md](calibration.md)             | The four calibration modes (`find_scale`, `find_defect_mask`, `find_blur`, `find_column_gain`) and the measurement models behind them            |
| [report-outline.md](report-outline.md)       | Outline for the accompanying written report (to be reviewed and filled in)                                                                       |

## Code map

The package is layered; each subpackage may only depend on the ones before it:

```
core         typed image arrays, the OpenCV boundary, image IO
imaging      normalize, align, metrics, motion, frc — pure functions on arrays
calibration  scale, blur, defects, colgain — calibration domains + versioned JSON
records      run configuration (Thresholds, CompareConfig) and result records
report       per-pair artifacts and run-level summaries
pipeline     the preparation chain, the comparison run, prepare-dependent solves
cli          the scanner-compare entry point
```

## The development dataset

Reference numbers quoted in these documents come from the development scanner pair:
**BBGSC108 NDT Dosimetry Pro** (reference) vs **BBGSC113 Dental Film Digitizer** — 15
paired 16-bit grayscale PA cephalometric film scans, ~3000×3400 px each, nominally 600
ppi. Measured calibration constants for this pair (defect-masked, signed common-support
convention):

- scale correction **s = 0.9985** (SC113's sampling pitch is ~0.15 % finer)
- device blur **σ_dev = +0.5332 px** (SC113 intrinsically blurrier)
- resampling penalty **r̄ = 0.2832 px** (Lanczos4 warp kernel, phase-averaged)
- stationary column-gain profile rms ≈ **0.003** rank units per scanner
- effective shared resolution (FRC, 0.5 criterion): **3.5–9.2 px** across the 15 pairs
  (mean ≈ 5.6 px)
