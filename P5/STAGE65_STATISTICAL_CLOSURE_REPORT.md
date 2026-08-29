# Stage 65: Statistical and evidential closure

## Objective

Close the remaining internal evidence gaps before manuscript assembly, excluding public release, external reproduction, and archival DOI work.

## Implemented

- AR(1) residual diagnostics, effective sample-size reporting, and Gaussian AR(1) profile BIC.
- Optional nonnegative amplitude fitting while retaining an unconstrained offset.
- Observation-count-normalized local boundary diagnostic with an explicit task-internal comparability contract.
- Disjoint semi-synthetic calibration/evaluation seeds for separated and coalesced true rank-two controls.
- Hydraulic model-family controls: stretched exponential, AR(2), Prony recurrence, and fixed-grid NNLS.
- Split theorem and conditional corollary: impossibility is proved under the independent Gaussian model; calibrated refusal requires separate assumptions.
- Verified literature additions for BIC, correlated time-series model selection, and close-node conditioning.
- Funding statements synchronized with P1--P4: GKZD010089 and SKLA202406 are reported in both manuscripts and submission declarations.

## Main evidence

- Separated true rank-two controls: 8/8 supported by ordinary and AR(1)-profile BIC under signed and nonnegative amplitudes.
- Held evaluation seeds: 4/4 separated controls accepted and 4/4 coalesced controls refused by the independently calibrated normalized boundary gate.
- Hydraulic sensitivity: ordinary BIC favors rank two, while AR(1)-profile BIC favors rank one; the disagreement is reported rather than hidden.
- Hydraulic median held-cycle NRMSE: shared rank one 0.244, stretched exponential 0.329, AR(2) 0.316, Prony 0.491, fixed-grid NNLS 0.625. The AR(2) bootstrap comparison is mixed.

## Verification

- 117 unit tests passed.
- 68 JSON result files parsed successfully.

## Remaining declared limitations

- The calibrated normalized boundary threshold is design conditional, not universal.
- The Gaussian theorem does not cover colored noise or model misspecification; correlation is handled as a sensitivity audit.
- The hydraulic task is an external multichannel stress test, not component-level mechanism validation.
- Public release, external reproduction, and archival DOI remain intentionally deferred.

