# Mechanism model versus trajectory baselines

- Route pass: **False**
- The Prony-like modal rank and MLP capacity are not interpreted as memory rank.
- All metrics use the same sparse early-time training observations.

| Case | True rank | Memory rank recovery | Memory extrap. | Modal extrap. | MLP extrap. |
|---|---:|---:|---:|---:|---:|
| rank1_separated | 1 | 3/3 | 1.1843e-04 | 1.9037e-01 | 2.4999e-01 |
| rank2_separated | 2 | 3/3 | 1.5215e-04 | 4.3392e-02 | 1.1988e-01 |
| rank3_separated | 3 | 3/3 | 6.6199e-03 | 6.2136e-02 | 1.3770e-01 |

## Prespecified checks

- memory_rank_recovered_in_at_least_7_of_9_trials: **True**
- mechanism_median_extrapolation_at_least_25_percent_better_than_mlp: **True**
- mechanism_not_more_than_25_percent_worse_than_modal_on_median: **True**
- each_mlp_median_training_rmse_within_2p5_noise_std: **False**
- no_mechanism_quality_failures: **True**

This is a feasibility comparison, not a calibrated claim of universal
superiority. The direct modal baseline is a trajectory model, while the
positive-real model carries a memory-mechanism contract and can estimate
memory rank on its declared domain.
