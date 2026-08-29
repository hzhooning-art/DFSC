# Independent-proxy cross-generator transfer map

The threshold uses a role-separated iid proxy trace and a nonlinear-feedback calibration bank.

| Length | External mechanism | Strength | Control adequacy | Strong detection | Proxy in scope | Cell pass |
|---:|---|---:|---:|---:|:---:|:---:|
| 256 | rate_drift | 1.5 | 1.000 | 1.000 | yes | PASS |
| 256 | rate_drift | 3.0 | 0.500 | 1.000 | yes | FAIL |
| 256 | rate_drift | 8.0 | 0.000 | 1.000 | yes | FAIL |
| 256 | stretched_exponential | 0.69 | 0.000 | 1.000 | yes | FAIL |
| 256 | stretched_exponential | 0.7 | 0.000 | 1.000 | yes | FAIL |
| 256 | stretched_exponential | 0.71 | 0.000 | 1.000 | yes | FAIL |
| 512 | rate_drift | 1.5 | 1.000 | 1.000 | yes | PASS |
| 512 | rate_drift | 3.0 | 1.000 | 1.000 | yes | PASS |
| 512 | rate_drift | 8.0 | 0.500 | 1.000 | yes | FAIL |
| 512 | stretched_exponential | 0.69 | 0.000 | 1.000 | no | FAIL |
| 512 | stretched_exponential | 0.7 | 0.000 | 1.000 | yes | FAIL |
| 512 | stretched_exponential | 0.71 | 0.000 | 1.000 | yes | FAIL |

- Overall route: FAIL
- Minimal passing prefix length: none
- The proxy trace is independent of every evaluated project trace.
- Calibration and project trends come from different generator families.
- The known iid noise scale remains a declared dependency.
