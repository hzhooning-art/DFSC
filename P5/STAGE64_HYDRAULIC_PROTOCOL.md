# Stage 64 frozen protocol: hydraulic cooling transients

- Public source: UCI Machine Learning Repository dataset 447, DOI `10.24432/C5CW21`.
- Independent unit: one 60-s hydraulic load cycle.
- Fixed conditions: valve 100%, pump leakage 0, accumulator 130 bar, stable flag 0.
- Held groups: cooler condition 3%, 20%, and 100%; ten cycles per group.
- Channels: CE, CP, TS1, and TS2. Channels remain clustered within a cycle.
- Declared window: seconds 20--55 of each cycle.
- Preprocessing: no smoothing; subtract the first window value and divide by the range of the first 22 window samples. No late-window value enters normalization.
- Candidate ranks: 1, 2, and 3 shared positive rates with curve-specific signed amplitudes and offsets.
- Validation: leave-one-cooler-condition-out; calibrate amplitudes on the first 22 samples and predict the remaining 14.
- Decision gates: the package defaults frozen before fitting.
- Interpretation boundary: the result concerns a shared empirical cooling-transient realization, not a unique hydraulic-component mechanism.
