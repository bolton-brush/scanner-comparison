# Registration: log-DoG features and masked Euclidean ECC

The registration stage answers: *which pixel of scan B corresponds to this pixel of scan
A?* Everything downstream depends on it — a 1-pixel misregistration would render as
structure in the difference images and masquerade as information loss. This document
explains the feature space (what DoG is, and why we take the log first), the estimator,
the pyramid, the masks, the fallback, and the scale pre-correction.

## The feature space: log-DoG

### What is a DoG?

The **difference of Gaussians (DoG)** is a band-pass filter: blur the image with a
Gaussian of small sigma, blur a copy with a Gaussian of large sigma, and subtract:

```
DoG(x) = G(σ_fine) ∗ I(x) − G(σ_coarse) ∗ I(x)      σ_fine = 2, σ_coarse = 16
```

What survives the subtraction is content whose spatial scale lies *between* the two
sigmas: fine detail smaller than σ_fine is smoothed away by both (and cancels), smooth
variation slower than σ_coarse (DC level, exposure falloff, vignetting, broad shading)
is captured by both (and cancels). What remains is **mid-frequency structural content**
— edges, trabecular texture, fiducial markers. It is the classic approximation of the
Laplacian-of-Gaussian blob detector, used here purely as an alignment feature space.

### Why the log first?

The two scanners digitize the same film with **different response curves**: different
gain, offset, and a gamma-like (roughly multiplicative/exponential) intensity mapping.
If we computed the DoG on raw intensities, the same anatomical edge would produce
different feature magnitudes on the two machines, and the alignment criterion would
partly be matching each scanner's response curve instead of the anatomy.

Taking the logarithm first converts multiplicative response differences into **additive
offsets**:

```
I_B ≈ γ(I_A)   ⇒   log I_B ≈ log I_A + c      (for multiplicative/exponential γ)
```

and the DoG — being a *difference* of two blurred versions — cancels any additive offset
exactly, including the smooth spatially-varying ones (vignetting, shading), which land
in the σ_coarse term of both images alike. So **log-DoG isolates the anatomy's
mid-frequency structure independent of each scanner's brightness/gamma response**: the
ideal common currency for cross-device registration.

(Implementation detail: the input is scaled to `img/255 + 1e-5` before `log` so the
function is defined at the black end of the scan; the absolute intensity scale is
immaterial because log turns it into an additive constant that the DoG cancels.)

After the subtraction, the feature image is min–max normalized to [0, 1] **restricted to
the film ROI** (so scanner-specific border apparatus cannot compress the feature range),
and pixels outside the ROI are zeroed.

These feature images are written per pair as `*_logdog_ref.png`, `*_logdog_mov.png`, and
`*_logdog_aligned.png` — comparing the first against the third shows the alignment
quality in feature space.

## The estimator: masked Euclidean ECC

**ECC (enhanced correlation coefficient) maximization** (`cv2.findTransformECCWithMask`)
directly maximizes the correlation between the reference feature image and the warped
moving feature image, iterating a Gauss–Newton-like scheme on the warp parameters. Three
properties make it the right tool here:

- It uses **all** masked pixels, not a sparse keypoint set — sub-pixel precision on
  3000-pixel frames (final correlations 0.97–0.99 on the development dataset).
- The **motion model is Euclidean** (rotation + translation only — a user mandate):
  films are rigid; any non-rigid freedom in the estimator could *absorb* real
  differences (warping one scan's content onto the other) instead of revealing them. The
  scale difference between the scanners is handled separately (below), so the ECC
  criterion itself never has a scale degree of freedom.
- It accepts **per-image criterion masks**: pixels excluded on either side do not
  contribute to the objective.

### Coarse-to-fine pyramid

ECC is a local optimizer, so it runs over a 3-level Gaussian pyramid with increasing
precision demands:

| Level  | Target max dimension | Termination ε | Iteration cap |
| ------ | -------------------- | ------------- | ------------- |
| coarse | 1200 px              | 1e-6          | 200           |
| mid    | 2400 px              | 1e-7          | 200           |
| fine   | full resolution      | 1e-8          | 400           |

Both images are downscaled by a **single shared factor** per level (the scans are at the
same physical scale; per-image factors would bake in a spurious scale difference a
Euclidean model cannot absorb — this was a real bug found and fixed during development).
The warp estimated at one level initializes the next, with the translation rescaled to
the new resolution. A finer level that fails to converge or regresses is skipped (the
coarser result stands); a final correlation below 0.3 fails the pair.

### ORB fallback initialization

ECC from identity can slide into a local optimum when scanner line-noise banding
dominates the band-pass features (observed on one banding-heavy pair: correlation 0.31
from identity). When the identity run lands below 0.6, a second run is initialized from
**ORB keypoint matching**: 5000 features per image, ratio-test filtered (0.75), RANSAC
(3 px) to flag inliers, then a closed-form **orthogonal Procrustes** fit of the
Euclidean transform on the inliers. The better of the two runs wins. On the problem pair
this lifted correlation from 0.31 to 0.97.

## The alignment masks

The ECC criterion masks (both sides, every pyramid level) exclude:

- the **border band** (3 %) and **beveled corners** (8 %) — the film edge is the
  strongest gradient in the frame and sits at a crop-dependent position; letting it into
  the criterion drags the warp (an early version had a visible ~1 % scale error from
  exactly this);
- the **dark surround** (same capped-Otsu + border-connected flood fill as the metric
  mask);
- a thin extra **edge band** (0.5 % of the frame) eroded inside the detected content;
- **stationary defect columns** (when a defect mask is configured): excluded so
  known-bad sensor columns cannot bias the warp estimate. At coarse pyramid levels the
  defect mask is area-resized with an any-residue threshold — a defect column's
  influence spreads over the level pixels it touches.

Note the deliberate split from the metric path: registration **always** uses these
default ROI margins, regardless of the configurable metric margins — edge apparatus
carries no anatomical information, so letting it into the alignment criterion can only
bias or break the estimate.

## The scale pre-correction

The two development scanners are nominally both 600 ppi but differ by ~0.15 % in true
sampling pitch (measured: **s = 0.9985** — see `find_scale` in
[calibration.md](calibration.md)). Because scale couples with rotation/translation at
first order in the ECC objective (a post-hoc scale sweep on a converged warp does
**not** work — the optimizer has already absorbed the scale error into its
rotation/translation), the correction is applied **before** estimation:

1. The moving scan is resampled by the uniform factor `s` about its center (bilinear —
   this is the feature path, not the metric path).
1. Euclidean ECC runs on the pre-scaled pair.
1. The scale map is **folded into the returned warp** (composed matrix `S·W`), so
   downstream code resamples the moving scan exactly once, with the high-quality
   Lanczos4 kernel.

The constant is a scanner-*pair* calibration, intentionally not estimated per film: real
per-pair scale variation exists (~0.1 %, film placement/transport) but sits at the
estimator noise floor, so one global constant is kept. `--both-directions` inverts it
automatically (1/s) for the reverse run.

## Consistency evidence

- Final ECC correlation is recorded per pair (`reg_correlation`).
- With `--both-directions`, `cross_direction.json` reports `roundtrip_max_px`: the
  largest frame-corner displacement of the composed forward∘reverse warp. On the
  development dataset this is 0.02–0.22 px across ~3000 px frames — the two directions
  invert each other to a fifth of a pixel. This validates the global (similarity)
  transforms only; it cannot see local/non-rigid residuals — those are the domain of the
  `motion_diff` diagnostic (see [metrics.md](metrics.md) and
  [analysis-pipeline.md](analysis-pipeline.md), stage 7).
