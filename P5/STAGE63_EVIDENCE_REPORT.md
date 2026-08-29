# Stage 63 evidence report: completing items 1--6

## 1. Independent public task

The second public task is UCI dataset 308, *Gas sensor array under flow
modulation* (`10.24432/C5BG7G`). It contains 58 independent five-minute
measurements from 16 metal-oxide sensors. Stage 63 uses the 50 non-air exposure
experiments and the registered 180--300 s recovery interval. Sensor channels
are clustered within exposure, and complete acquisition batches are left out.

The result is `INDETERMINATE`, not a failed analysis. Shared rank three reduced
held-batch median NRMSE to 0.05392, but two fitted rates coalesced. Predictive
adequacy therefore did not justify a three-timescale mechanism claim.

## 2. External baselines

All methods use the same 60/40 early/late split and the same 50 exposure units.

| Method | Median experiment NRMSE | IQR |
|---|---:|---:|
| Shared rank 3 | 0.05392 | [0.03420, 0.07168] |
| Independent nonlinear rank 3 | 0.05910 | [0.05189, 0.07471] |
| Fixed-grid NNLS | 0.25676 | [0.20728, 0.31036] |
| Prony recurrence | 0.05961 | [0.03894, 0.08373] |

The shared fit improved the median experiment error by 26.6% relative to the
independent nonlinear fit (`p=0.00110`), 80.6% relative to fixed-grid NNLS
(`p<1e-12`), and 8.25% relative to the recurrence baseline (`p=0.00928`). These
paired tests use exposure experiments, not sensor channels, as independent
units. The comparison supports predictive utility in this task; it does not
turn the coalesced spectrum into an identifiable physical mechanism.

## 3. Statistical robustness

- UCI rank 3 versus rank 1: median experiment-level improvement 65.9%, cluster
  bootstrap 95% interval [58.5%, 68.2%], one-sided paired Wilcoxon
  `p<1e-12` over 50 exposure experiments.
- PVA rank 3 versus rank 1: median curve improvement 48.4%, descriptive
  bootstrap interval [39.5%, 79.8%]. The PVA source has only three independent
  specimens, so no population-level claim is based on its nine cycles.
- Leave-one-specimen-out and leave-one-acquisition-batch-out are the primary
  transfer structures.
- UCI multi-start fits with 2, 4, and 8 starts return the same objective and
  rates. PVA returns rank 3 with 1, 2, 4, and 8 starts at the full protocol.
- Every one of 27 UCI threshold combinations remains `INDETERMINATE`. PVA is
  rank 3 in 18/27 combinations and indeterminate in the nine combinations with
  the strictest fold-stability threshold. Threshold scans are sensitivity
  audits, not post-hoc threshold selection.

## 4. Nonmonotone boundary explanation

The controlled PVA factor audit separates tail coverage, sample density, and
optimizer starts. At fixed spacing, the decisions progress from rank 2 at 4 s,
to indeterminate at 8 and 15 s, and rank 3 at 28 s. The median lag-one residual
correlation rises from -0.075 to 0.400 and the rate-sensitivity condition number
rises from 45.9 to 156.6. At the full 28 s horizon, changing the number of
optimizer starts from 1 to 8 does not change the rank-3 result.

Thus the observed nonmonotonicity is consistent with changing tail coverage,
correlated residual information, and sensitivity conditioning, rather than a
simple optimizer-start failure. These diagnostics are explanatory evidence,
not a causal theorem.

## 5. Minimal theory

`MINIMAL_THEORY.md` records four bounded results: the Jacobian/Fisher condition
for local identifiability, a mean-value bound showing why nearby exponential
rates coalesce, the distinction between raw sample count and effective
information under correlated observations, and the statistical meaning of
`INDETERMINATE`. The theory supports the gate design but does not claim a
universal Type-I/Type-II error bound.

## 6. Frozen software surface

The package `p5_memory_protocol` exposes `fit`, `evaluate`, `decide`, and
`report`, a versioned JSON schema, and a one-command reproduction CLI. The
contract and limitations are frozen in `API_CONTRACT.md`. Verification on
2026-08-25 completed 111 unit tests, parsed all 66 result JSON files, and
successfully replayed the UCI task through the public CLI.

## Current conclusion

The six additions raise P5 from a single-domain fitting study to a scoped,
auditable identification-and-refusal method. The strongest result is not that
one rank always wins. It is that the same contract supports rank 3 on the full
PVA observation, exposes sensitivity to specimen count and observation design,
and refuses a chemically distinct task despite favorable predictive baselines
when the fitted rates are not separable.
