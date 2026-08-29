# Memory-rank feasibility probe

Device: `cuda`

| Case | True rank | Channels | Decision | BIC gap | Jacobian condition | Validation RMSE |
|---|---:|---:|---|---:|---:|---:|
| scalar_rank1 | 1 | 1 | RANK_1 | 7.48 | 2.62 | 0.000586 |
| scalar_rank2_separated | 2 | 1 | RANK_2 | 7.68 | 74.6 | 0.000497 |
| scalar_rank2_near_coincident | 2 | 1 | INSUFFICIENT_EVIDENCE | 3.66 | 3.03e+15 | 0.000705 |
| field_shared_rank2 | 2 | 12 | RANK_2 | 172 | 27.4 | 0.000956 |

This probe is a route-selection test. It does not establish statistical coverage,
global identifiability, or scientific validity on real data.
