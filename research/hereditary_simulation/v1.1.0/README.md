# Positive hereditary prediction: complete reproducibility archive

Sharp Predictability Criteria and Waiting-Time Recovery Rates for Positive Hereditary Models - Data, Code and Reproducibility Artifacts

Version **1.1.0**, snapshot 24 October 2026. GitHub release: https://github.com/hzhooning-art/DFSC/releases/tag/hereditary-simulation-v1.1.0
No Zenodo DOI is embedded until one is provided. This ZIP is the complete
research asset; GitHub's automatically generated source archive is not.

## Layout and integrity

- baseline_v1.0.0/: all 115 original public archive entries, byte-identical,
  including the large saved optimization witnesses omitted from compact copies.
- review_evidence/: the full subsequent research evidence, complete witnesses,
  immutable per-study source snapshots, observations and source attribution.
- metadata.json, CITATION.cff, LICENSE and THIRD_PARTY_NOTICES.md describe this
  version and distinguish authored code from original third-party material.

Extract to a new directory and run `python verify_archive.py` before use.
This checks all hashes; it is not a new numerical or mathematical audit.
All earlier numerical payloads survive unchanged. Internal historical paths and
old completion receipts are retained for hash/execution compatibility; their
"not public yet" statements describe their original snapshot, not this release.
No journal submission or Zenodo publication is implied by this preparation.

## Reproduce in fresh destinations

Use Python with numpy, scipy, mpmath and openpyxl. Recorded numerical dependency
versions are in the frozen protocols; timing depends on the machine. Never
overwrite the preserved result directories. The following destinations must
not already exist:

```text
python baseline_v1.0.0/verify_archive.py
python review_evidence/reproduce_pvc_studies_20261024.py review_evidence NEW_PVC_DESTINATION
python review_evidence/reproduce_new_benefit_studies_20261021.py review_evidence NEW_MATCHED_DESTINATION
```

The first entry point re-extracts the original PVC measured columns and runs
both the 90-request sensitivity study and 315-request scheduling comparison,
their complete audits, and outcome/array checks. It uses no network and includes
the original source workbook. Figure 10 can be regenerated with:

```text
python review_evidence/pvc_time_search_benefit_20261023/plot_pvc_benefit_20261024.py
```

The second entry point replays the 584-request matched-oracle comparison and
the FAILED orbital-fat pilot. Failure of physical qualification is expected;
do not retune bands or select favorable traces. Other complete new studies
have their original reproduce.py and README under prediction_time_search_20261020/,
prediction_time_transport_20261020/ and prediction_time_dual_20261020/ inside
review_evidence/. The complete baseline source, data and README are adjacent,
so earlier compacted evidence no longer needs a separate asset download.

## Supported use and limitations

The mathematical results concern the stated uncapped positive-spectrum class,
qualified bands, known loading and strict physical feasibility. Supplied mass
or rate caps must not be discarded. Waiting-time rates describe fixed-band
compatible ambiguity, not a statistical small-noise rate or fitted point error.
Generic positive-measure optimization, continuum-residual correction and
convexity are attributed classical tools; the loading-guided use and matched
computational benefits do not establish an exclusive capability.

All negative outcomes remain: slower synthetic tasks, failed material pilot,
calibration-underestimation cases and Prony nonconvergence. PVC observations
support conditional task-cost evidence only: the error bands are assumed,
physical loading/noise qualification is incomplete, and the same-profile
example does not validate the cross-loading boundary or a positive recovery
exponent. Requests and temperature conditions are correlated, not independent
replicate specimens. No universal speedup, point-accuracy superiority or
calibrated material reliability guarantee is claimed.

## Rights and citation

Authors: Haitao Duan, Ning Hu, Shuqun Li and Chuyang Hu, in that order.
Use CITATION.cff and cite original data authors as described in
THIRD_PARTY_NOTICES.md. MIT covers the authors' material, not every source file.
For Zenodo, declare the mixed licence scopes rather than marking all content
MIT. Editorial and author-only submission documents are not distributed.
