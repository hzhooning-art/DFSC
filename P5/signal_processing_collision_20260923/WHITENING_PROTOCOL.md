# Frozen independent-training whitening protocol

Date frozen: 2026-09-11, before outcome simulation.

- Dimension: C=4 channels, n=12 times on `[0,8]`, d=48.
- Noise covariance: separable channel equicorrelation 0.25 and time AR(1) correlation 0.60, rescaled to unit marginal variance.
- Signal replicates: m=4096. Independent centered Gaussian noise-training vectors: K in `{64,128,512}`.
- For every trial, estimate the full covariance from a new independent training sample; fit the one-rate null by generalized least squares; construct the C-dimensional whitened collision space from projected first-derivative channel directions and the curvature direction. Multiply the inverse-covariance quadratic form by the prespecified Gaussian inverse-Wishart correction `(K-d-2)/(K-1)`; this tends to one and only improves finite-K centering.
- Null trials: 512 per K. Proportional-collision alternative trials: 512 per K, with oracle population Mahalanobis noncentrality 8.
- Also report the unwhitened phase score on the same signal draw to expose covariance misspecification.
- Seeds start at `6091130000000`, disjoint by K and null/alternative.

Frozen gates:

1. all fits, covariance factorizations and score ranks finite;
2. at K=512, whitened null rejection lies in `[0.025,0.075]`;
3. absolute error from the chi-square power prediction is at most 0.08 at K=512;
4. the null-size error at K=512 is smaller for whitened than unwhitened scores;
5. the population collision space has rank C in every audited design.
