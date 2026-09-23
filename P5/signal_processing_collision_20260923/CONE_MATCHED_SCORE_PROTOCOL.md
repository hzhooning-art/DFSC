# Frozen cone-score comparison

Date: 2026-09-10. Freeze before simulation.

Use m=16384 and the twelve cells formed by target noncentralities 4, 8, 12, 18
and the diversity, crossover, proportional paths in
`SCORE_POWER_CURVE_PROTOCOL.md`. Reuse the same 1024 deterministic trials per
cell for paired comparison with the two-sided phase score. Add 4096 null trials
at m=4096 and m=16384 with seeds `6091060000000 + 100000*j + trial`.

Calculate the raw cone statistic as the diversity squared score plus the square
of the positive part of the oriented curvature score. Determine its .95
critical value by numerical inversion of the exact independent-scale null
mixture in `CONE_MATCHED_SCORE_THEORY.md`. Do not tune it from null simulations.

Predeclared gates:

1. cone null rejection lies in [.03,.07] at both m values;
2. on proportional paths with target 4 or 8, cone power is no lower than the
   two-sided score power and their mean gain is at least .015;
3. no alternative cell loses more than .03 power relative to the two-sided
   score;
4. all critical values and powers are finite.

This is a local cone-efficiency check, not a finite-sample theorem for plug-in
nuisance estimates.
