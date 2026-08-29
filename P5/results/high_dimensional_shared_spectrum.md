# High-dimensional shared-spectrum audit

Device: `cuda`; route pass: **True**.

| Channels | Shared resolved | Shared rate error | Independent resolved | Shared time (s) | Independent time (s) | Peak memory (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.476 | 0.00 | 3.59 | 3.47 | 16.59 |
| 16 | 1.00 | 0.0512 | 0.19 | 4.4 | 5.24 | 21.12 |
| 64 | 1.00 | 0.0499 | 0.33 | 5 | 6.1 | 32.07 |
| 256 | 1.00 | 0.0333 | 0.27 | 6.39 | 7.8 | 78.81 |

The experiment fixes the generator, horizon, noise level, optimizer budget,
and decision rule before inspecting outcomes.  It is a synthetic scaling
audit; the true-rate error is an evaluation metric unavailable in deployment.
