# Increment-domain leave-one-strength-out transfer map

Each evaluated strength is calibrated only from independent controls at the other declared strengths.

| Length | Held-out strength | Control adequacy | Strong detection | Cell pass |
|---:|---:|---:|---:|:---:|
| 256 | 0.05 | 0.000 | 1.000 | FAIL |
| 256 | 0.085 | 1.000 | 1.000 | PASS |
| 256 | 0.2 | 1.000 | 0.500 | FAIL |
| 512 | 0.05 | 0.250 | 1.000 | FAIL |
| 512 | 0.085 | 1.000 | 1.000 | PASS |
| 512 | 0.2 | 1.000 | 0.500 | FAIL |

- Overall route: FAIL
- Minimal passing prefix length: none
- Every held-out strength must pass both frozen 75% criteria.
- The experiment still assumes the declared strength set and known iid noise scale.
