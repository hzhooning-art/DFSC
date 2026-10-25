# Local DFSC revision candidate

This is **not a public release**. It starts from the archived original 32 Python
files and includes an opt-in fixed-SPD homogeneous admission path. Numerical
admission uses exact-arithmetic scalar tail envelopes plus a heuristic roundoff
proxy, not a rigorous floating-point or full-solver certificate.

```python
import torch
import dfsc
A = torch.diag(torch.tensor([0.2, 0.8], dtype=torch.float64))
p = dfsc.OperatorSpectralProblem(
    A, torch.ones((1, 2), dtype=torch.float64),
    torch.tensor([0., .1, .4], dtype=torch.float64), .75, beta=1.4)
s = dfsc.solve(p, "guarded-spd", require_admission=True)
values = s.require_accepted()
assert s.accepted and not s.reliability.trusted
assert not s.reliability.rigorous_error_bound
```

Any modal refusal aborts the entire call by exception, without fallback, values,
backward or optimizer update. This route supports only a fixed dense SPD matrix,
full retained spectrum, batch initial states, fixed nonnegative query times,
real float64, 0<alpha<=1 and 0<spatial beta<=2. The scalar kernel has second ML
parameter one; spatial beta is not that parameter. It recomputes an eigensystem
per solve and actually computes 258 scalar terms; no acceleration is claimed.
The already-tested composed output/gradient scope remains finite, not a guarantee.

Legacy paths preserve numerical values for diagnosis, but finite/success/converged
does not set accepted. `require_admission=True` rejects every other algorithm
before running it. `Solution.require_accepted()` rejects unadmitted results.
`trusted` is now reserved for rigorous certificates and is false for all current
paths. Legacy domain predicates are explicitly metadata predicates, not validation
or gradient accuracy. These are intentional candidate API semantic changes.

The scalar adaptive evaluator's `strict=True` now requires validated trust, not
successive-work agreement, and can therefore refuse all legacy scalar adaptive
calls. The separate Krylov algorithm's legacy strict flag still only checks its
work budget: use `solve(..., require_admission=True)` to enforce admission there.
Choose the explicit guarded path only if its limited scope applies. Raw legacy
value kernels are not automatically protected; automatic selection is unchanged.

Do not install alongside the public `dfsc` distribution: both use the `dfsc`
namespace. Use a fresh environment. Installing this candidate does not update
GitHub, PyPI, Zenodo or the published manuscript artifact.
