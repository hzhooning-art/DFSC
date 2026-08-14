# P4 Active-Design Feasibility Gate

## Experiment

The first controlled test compared three eight-observation strategies for the
family `E_alpha(-lambda t^alpha)`:

1. uniform sampling from 64 candidate times;
2. standard D-optimal greedy selection using high-accuracy sensitivities;
3. an error-aware D-optimal rule using low-order sensitivities weighted by
   local value and gradient discrepancy against a high-accuracy evaluator.

The design was fixed at a nominal parameter pair and evaluated on 40 new noise
seeds at off-grid truth (`alpha=0.773`, `lambda=0.637`).

## Results

| strategy | alpha error | rate error | joint error | profile coverage | profile area |
|---|---:|---:|---:|---:|---:|
| uniform | 0.01573 | 0.00585 | 0.01805 | 1.00 | 0.0546 |
| standard D-optimal | **0.01058** | 0.00708 | **0.01447** | 1.00 | **0.0406** |
| error-aware D-optimal | 0.01460 | 0.00739 | 0.01769 | 1.00 | 0.0495 |

The raw result is stored in
`P4/results/p4_active_design_feasibility.json`.

## Gate decision

The current error-aware weighting rule **fails the first performance gate**:
it does not improve recovery error or profile area over standard D-optimal
selection. Coverage is uninformative here because all methods cover the true
grid cell in every replicate.

The weighting formula is therefore frozen as a failed candidate and should not
be used as the P4 paper's algorithmic contribution.

## Is P4 still worth pursuing?

The broad active-experiment-design question remains potentially useful, but the
current evidence does not justify an independent paper. One bounded redesign
is permissible: replace the pointwise inverse-error weight with a robust
minimax or Pareto objective evaluated over a parameter prior, and compare it
against a fair non-oracle baseline using the same evaluator budget. That test
must be pre-registered with a small number of objectives and a clear stop rule.

If the redesigned rule again fails to improve held-out recovery or information
per observation, P4 should be stopped as an independent paper and retained as
an optional DFSC design utility. This is a genuine route-risk, not a reason to
expand the experiment suite indefinitely.
