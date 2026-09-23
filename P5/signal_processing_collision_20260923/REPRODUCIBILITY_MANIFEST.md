# P5 minimal reproducibility manifest

## Environment used

- Python 3.12.14
- NumPy 2.5.0
- SciPy 1.18.0
- Matplotlib 3.11.0 (`Agg` backend for figures)

Run commands from the repository root with `P1/.venv/Scripts/python.exe -B`.

## Core theory validation

1. `uniform_crossover.py` -> `uniform_crossover_results.json`; audit with `audit_uniform_crossover.py`.
2. `local_limit_power.py` -> `local_limit_power_results.json`; audit with `audit_local_limit_power.py`.
3. `finite_local_limit.py` -> `finite_local_limit_results.json`; audit with `audit_finite_local_limit.py`.
4. `GENERAL_ATOM_PROTOCOL.md` freezes two non-exponential families; `general_atom_audit.py` writes `general_atom_results.json`, verified by `audit_general_atom.py`.

## Proposed scores

1. `phase_matched_score.py` -> `phase_matched_score_results.json`; audit with `audit_phase_matched_score.py`.
2. `cone_matched_score.py` -> `cone_matched_score_results.json`; included in `audit_score_innovations.py`.
3. `score_power_curve.py` -> `score_power_curve_results.json`; included in `audit_score_innovations.py`.
4. `cone_direction_power.py` -> `cone_direction_power_results.json`; deterministic diagnostic only.

## Matched and cross-design comparisons

1. `MATCHED_GLRT_PROTOCOL.md` freezes the comparison.
2. `matched_glrt_benchmark.py` -> `matched_glrt_results.json`; audit with `audit_matched_glrt.py`.
3. `SCORE_DESIGN_STRESS_PROTOCOL.md` freezes the twelve designs and gates.
4. `score_design_stress.py` -> `score_design_stress_results.json`; audit with `audit_score_design_stress.py`.
5. `SAMP_DIRECT_PROTOCOL.md` freezes the public-source compatibility comparison; `samp_direct_benchmark.py` writes `samp_direct_results.json`, audited by `audit_samp_direct.py`.
6. `WHITENING_PROTOCOL.md` freezes the independent-training covariance experiment; `whitening_benchmark.py` writes `whitening_results.json`, audited by `audit_whitening.py`.
7. `FINITE_TRAINING_CALIBRATION_THEORY.tex` gives the finite-training nested-space pivot; `finite_training_calibration.py` writes `finite_training_calibration_results.json`, audited by `audit_finite_training_calibration.py`.

## Manuscript artifacts

- `MANUSCRIPT_THEORY_DRAFT.tex`
- `PAPER_LEVEL_PROOFS.tex`
- `LOCAL_MINIMAX_THEORY.tex`
- `GENERAL_ATOM_EXTENSION.tex`
- `COVARIANCE_WHITENING_THEORY.tex`
- `build_ieee_submission.py` -> `MANUSCRIPT_IEEE_DRAFT.tex` and `SUPPLEMENT_IEEE_DRAFT.tex`; audit with `audit_ieee_split.py`.
- `make_manuscript_figures.py` -> `figures/phase_boundary.*`, `figures/matched_glrt_power.*`
- `audit_submission_bundle.py` checks labels, references, citations, inputs, figures, delimiters, and required sections.

Large `stage*.json` and early feasibility outputs are development history and should not be placed in the submission supplement. The files above are sufficient for every quantitative statement in the current manuscript. Result files used in the matched and stress audits retain their frozen settings, seeds, rejection counts, and source-integrity fields.

The local `vendor_samp` checkout is retained only to audit the authors' public
source at the recorded commit. It is not part of the proposed supplement and
should not be redistributed without a repository license. The supplement
needs only the recorded commit, source hashes, Python compatibility port, and
the resulting JSON.
