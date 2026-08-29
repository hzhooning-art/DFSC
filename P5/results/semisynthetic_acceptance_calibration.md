# Semi-synthetic acceptance and refusal calibration

## separated_signed

```json
{
  "n": 8,
  "ordinary_decisions": {
    "SUPPORTED_RANK_2": 8
  },
  "ar1_decisions": {
    "SUPPORTED_RANK_2": 8
  },
  "median_rho_ar1": 0.4523253655994809,
  "median_normalized_boundary_index": 2.8068445454396533
}
```

## separated_nonnegative

```json
{
  "n": 8,
  "ordinary_decisions": {
    "SUPPORTED_RANK_2": 8
  },
  "ar1_decisions": {
    "SUPPORTED_RANK_2": 8
  },
  "median_rho_ar1": 0.4523253655359334,
  "median_normalized_boundary_index": 2.806844496914497
}
```

## coalesced_signed

```json
{
  "n": 8,
  "ordinary_decisions": {
    "SUPPORTED_RANK_1": 5,
    "INDETERMINATE": 3
  },
  "ar1_decisions": {
    "SUPPORTED_RANK_1": 8
  },
  "median_rho_ar1": 0.4542484354214082,
  "median_normalized_boundary_index": 0.04222045733491615
}
```

## coalesced_nonnegative

```json
{
  "n": 8,
  "ordinary_decisions": {
    "SUPPORTED_RANK_1": 5,
    "INDETERMINATE": 3
  },
  "ar1_decisions": {
    "SUPPORTED_RANK_1": 8
  },
  "median_rho_ar1": 0.4576135425354918,
  "median_normalized_boundary_index": 0.027269883350630554
}
```

## boundary_calibration

```json
{
  "rule": "geometric midpoint between calibration maxima for coalesced rates and minima for separated rates",
  "threshold": 0.3470538949815616,
  "calibration_seeds": [
    6501,
    6502,
    6503,
    6504
  ],
  "held_evaluation_seeds": [
    6505,
    6506,
    6507,
    6508
  ],
  "coalesced_false_accept_rate": 0.0,
  "coalesced_refusal_rate": 1.0,
  "separated_detection_rate": 1.0,
  "scope": "this declared semi-synthetic design only"
}
```
