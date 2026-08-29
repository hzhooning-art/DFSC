# Quality-triggered retry replication

- Route pass: **True**
- Total runtime: 1373.4 s
- Retry triggers: 0/640

| Case | Role | Trials | Contract refusals | Calibration failures | Retry triggers | Retry recoveries | Median seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| signed_zero | null | 10 | 0 | 0 | 0 | 0 | 43.32 |
| oscillation_zero | null | 10 | 0 | 0 | 0 | 0 | 42.39 |
| oscillation_decay_016 | primary_alternative | 4 | 4 | 0 | 0 | 0 | 43.22 |
| shifted_transient_020 | primary_alternative | 4 | 4 | 0 | 0 | 0 | 43.64 |
| shifted_transient_055 | primary_alternative | 4 | 4 | 0 | 0 | 0 | 42.75 |
