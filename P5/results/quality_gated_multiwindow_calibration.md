# Quality-gated multi-window calibration

- Calibration fits: 120
- Invalid calibration fits: 0
- Frozen threshold: 0.205141
- Threshold bootstrap 95% interval: [0.18731135074685368, 0.21744045957260447]
- Route pass: **False**

| Case | Trials | Terminal refusal | Multi-window refusal | Elevated rank |
|---|---:|---:|---:|---:|
| oscillation_decay_016 | 6 | 0.000 | 1.000 | 0.000 |
| oscillation_decay_036 | 6 | 0.000 | 0.333 | 0.000 |
| oscillation_zero | 12 | 0.000 | 0.167 | 0.000 |
| shifted_transient_020 | 6 | 0.000 | 1.000 | 0.000 |
| shifted_transient_055 | 6 | 0.000 | 1.000 | 0.000 |
| signed_zero | 12 | 0.000 | 0.000 | 0.000 |

## Prespecified checks

- at_least_100_independent_calibration_fits: **True**
- jittered_zero_false_refusal_at_most_1_of_12: **False**
- each_unseen_stress_refused_at_least_5_of_6: **False**
- no_systematic_rank_absorption: **True**
- threshold_bootstrap_width_at_most_0.15: **True**
- invalid_calibration_fraction_at_most_0.02: **True**
