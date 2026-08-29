# Public UCI gas-recovery audit

Decision: **INDETERMINATE**.

Route pass: **True**.

## External baseline audit

| Method | Median experiment NRMSE | IQR |
|---|---:|---:|
| shared_rank3 | 0.05392 | [0.03420, 0.07168] |
| independent_nls_rank3 | 0.05910 | [0.05189, 0.07471] |
| fixed_grid_nnls | 0.25676 | [0.20728, 0.31036] |
| prony_rank3 | 0.05961 | [0.03894, 0.08373] |

Sensor channels were clustered within each independent gas exposure before inference.
