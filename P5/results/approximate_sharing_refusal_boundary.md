# Approximate-sharing refusal boundary

Device: `cuda`; route pass: **False**.

| Log drift | Noise corr. | Accepted | Old gate pass | Observed dispersion | True dispersion | Mechanism distortion | Val. ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.00 | 0.00 | 1.00 | 0.621 | 0 | 0.0416 | 0.165 |
| 0.00 | 0.60 | 0.67 | 1.00 | 0.0807 | 0 | 0.0512 | 0.145 |
| 0.05 | 0.00 | 0.33 | 1.00 | 0.12 | 0.0559 | 0.0707 | 0.354 |
| 0.05 | 0.60 | 0.00 | 0.67 | 0.239 | 0.0559 | 0.0743 | 0.462 |
| 0.15 | 0.00 | 0.00 | 0.00 | 0.189 | 0.168 | 0.199 | 1.12 |
| 0.15 | 0.60 | 0.00 | 0.00 | 0.373 | 0.168 | 0.217 | 1.25 |

The refusal rule uses only subgroup estimates and is frozen before the run.
True dispersion and mechanism distortion are diagnostic evaluation metrics.
