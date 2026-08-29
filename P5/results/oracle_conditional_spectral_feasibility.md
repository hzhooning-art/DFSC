# Oracle conditional spectral feasibility map

This is an upper-bound experiment: each null is conditioned on the known clean deterministic trajectory.

| Prefix length | Control adequacy | Strong-memory detection | Route pass |
|---:|---:|---:|:---:|
| 78 | 1.000 | 0.167 | FAIL |
| 256 | 0.833 | 0.833 | PASS |
| 512 | 1.000 | 0.750 | PASS |

- Overall route: PASS
- Minimal passing prefix length: 256
- Passing this oracle experiment is necessary but not sufficient for an operational unknown-trend gate.
