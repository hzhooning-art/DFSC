# P4 JSS Stage 3: Historical-Defect Intake and Fixed-Version Replay

Updated: 2026-09-03

## Objective

Replace an informal request for “real bugs” with an auditable admission rule.
A completed historical-defect case must have (1) an upstream issue or pull
request, (2) an executable failure on the reported buggy revision, and (3) an
executable pass on a fixed revision. Issue provenance or a current-version pass
alone is partial evidence and does not count toward the JSS project gate.

## Intake inventory

| Case | Project | Upstream report | Reported version/environment | Current replay | Buggy revision replay |
|---|---|---|---|---|---|
| Zero gradient for `xlogy(0,2)` | PyTorch | [#80770](https://github.com/pytorch/pytorch/issues/80770) | 1.11.0, CPU float64 | Fixed expectation passes | Not installed |
| Wrong `einsum` gradient for non-contiguous input | PyTorch | [#30303](https://github.com/pytorch/pytorch/issues/30303) | 1.3.1, CUDA | Fixed expectation passes | Not installed |
| Complex conjugate JIT/VJP exception | JAX | [#15400](https://github.com/jax-ml/jax/issues/15400) | 0.4.1, CPU complex128 | JAX unavailable | Not installed |
| Missing species-condition gradient route | PyBNF | [#538](https://github.com/lanl/PyBNF/issues/538) | Upstream 2026 issue snapshot | Dependencies unavailable | Not checked out |

The installed PyTorch 2.11.0+cu128 returns the expected `log(2)` derivative at
zero and gives exact agreement between matmul and non-contiguous `einsum`
gradients on the available CUDA device. These are fixed-version replays, not
executions of the historical failures.

## Admission result

- Candidate cases: 4.
- Candidate projects: 3.
- Passing current fixed-version replays: 2, both from PyTorch.
- Complete buggy/fixed pairs: 0.
- Projects with complete pairs: 0/3 required.

Accordingly, Stage 3 improves provenance and experiment readiness but does not
increase the number of empirically verified external projects. The JSS gate
remains failed.

## Next execution gate

The next stage must run the reported old PyTorch versions in isolated compatible
environments, then obtain fixed and buggy revisions for at least two additional
projects. Cases that require unavailable hardware or cannot identify a fixed
revision remain excluded. Reimplementing the reported bug in P4 code is not an
acceptable substitute for executing the upstream defect.

## Reproduction

```console
python P4/experiments/p4_historical_defect_intake.py --stdout-summary
python -m unittest P4.tests.test_historical_defect_intake -v
```

The case-level artifact is `P4/results/p4_historical_defect_intake.json`.
