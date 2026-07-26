# Pre-release Readiness

`dfsc` uses an executable 20-check gate for internal readiness before public
release. Each check contributes five percentage points. The gate covers
numerical identities, alpha/beta gradients, strict validated-domain handling,
the single `dfsc` distribution, solution reliability metadata, unit tests,
runtime smoke tests, application evidence, CI, documentation, and release
metadata.

Run the gate with:

```bash
python -B tools/pre_release_audit.py
```

The current report is written to `results/pre_release_readiness.json`. A score
of at least 95% is required. Passing this gate means that the private artifact
is internally ready for release preparation; it does not claim public ecosystem
maturity.

The following axes are deliberately excluded because they require public or
third-party evidence:

- a versioned public source and package release;
- availability of the hosted documentation site;
- independent downstream adopters;
- public issue and pull-request history;
- explicit redistribution permission for the current SPT raw data.

## Reliability Contract

`evaluate_mittag_leffler` and `evaluate_complex_mittag_leffler` return a
`ReliabilityReport`. It records whether the input is within the documented
validation domain, convergence of the embedded comparison, gradient
reliability, the type of error indicator, and whether that indicator is a
rigorous bound. Current embedded disagreements are empirical indicators and are
never labeled as rigorous bounds.

`strict=True` rejects evaluations that do not satisfy the contract. Every
solution returned by `dfsc.solve` also carries conservative reliability metadata
and exposes it through `solution.summary()`.
