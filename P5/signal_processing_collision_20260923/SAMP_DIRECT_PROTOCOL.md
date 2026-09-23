# Frozen SAMP direct-competition protocol

Date frozen: 2026-09-11, before outcome simulation.

## Source fidelity

Official repository: `https://github.com/YehonatanSeg/--SAMP--Stracture-Aware_Matrix_Pencil_Method.git`, commit `3f8cf58a32d7c7849da89ef911827fbfd57a11bd`.

The repository implementation is MATLAB. A Python port will reproduce: Hankel construction; effective-rank SVD truncation; reduced pencil eigendecomposition; left MP modes; the FFT peak feature; the published squared-frequency-separation normalization; and a deterministic globally optimal one-dimensional two-means partition replacing MATLAB's randomized k-means initialization. No outcome-dependent parameter will be fitted.

## Compatibility distinction

P5 observes nonoscillatory real exponentials, so every pole has frequency zero. The official repository function `compute_SAMP_signal_detection_features.m` divides each feature by the sum of squared frequency separations. That denominator is identically zero on P5's one-vs-two decay task. This is recorded as `official_SAMP_applicable=false`, not converted into an artificial rejection or failure count.

For a numerical stress comparison we additionally report `SAMP-decay`, a declared adaptation that omits only this undefined frequency-normalization factor and otherwise retains the official effective-rank, MP-mode, FFT-peak and two-means steps. It is an ablation, not the published SAMP algorithm.

## Common task and data

- One scalar channel, 24 equally spaced samples on `[0,8]`, zero intercept.
- One real decay under the null; two positive-amplitude real decays under the alternative.
- `m=16384` independent replicates are supplied to both methods through the same sample mean.
- P5 uses its C=1 phase score with known variance and nominal alpha 0.05.
- SAMP-decay reports order two when its selected high-feature group contains at least two modes.
- Null trials: 2048. Alternative trials: 1024 for each target noncentrality 4 and 8 on the proportional collision path.
- Seeds start at `6091120000000` and are disjoint by cell.

## Interpretation gates

1. The port must reproduce the official formula graph and repository commit hash.
2. Official SAMP must be flagged inapplicable because of a verified zero denominator.
3. P5 null rejection must lie in `[0.03,0.07]`.
4. SAMP-decay null order elevation and alternative power must both be reported without post-hoc calibration.
5. The paper may compare operating characteristics but may not call SAMP-decay the published SAMP.
