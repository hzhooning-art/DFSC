# Initial identifiability boundary

Device: `cuda`; noise standard deviation: `0.0008`.

Each cell reports the fraction of two repeated trials in which rank two was
resolved with adequate BIC support, conditioning, extrapolation error, and pole recovery.

## Channels = 1

| Horizon | Rate ratio 1.10 | 1.35 | 2.00 | 4.00 | 8.00 |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 |
| 8 | 0.00 | 0.00 | 0.00 | 0.50 | 1.00 |
| 12 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |

## Channels = 8

| Horizon | Rate ratio 1.10 | 1.35 | 2.00 | 4.00 | 8.00 |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.00 | 0.00 | 0.00 | 0.50 | 1.00 |
| 8 | 0.00 | 0.00 | 0.00 | 0.50 | 0.50 |
| 12 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |

The map is a low-budget feasibility screen, not a calibrated statistical
coverage result. A publishable boundary requires many more seeds and explicit
false-discovery control.
