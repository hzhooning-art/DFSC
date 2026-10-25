# Positive hereditary simulation: complete reproducibility artifacts

Loading-Dependent Predictability and Precision Assessment for Positive Hereditary Simulations — Data, Code and Reproducibility Artifacts

Archive version: 1.0.0. Snapshot date: 14 October 2026. No archive DOI is embedded before a DOI has been supplied and verified.

All research payloads from the frozen submission supplement are included byte-for-byte. The one removed file is a manuscript-attachment layout QA receipt, not numerical evidence. All synthetic observations, complete saved optimization witnesses, cases, failures, protocols, sources and audit outcomes remain. No scientific calculation was rerun during packaging. No manuscript PDF, editorial correspondence, author-only disclosure, dependency environment or third-party source tree is included.

## Integrity and use

Extract into a fresh folder. With Python 3, run `python verify_archive.py` first. This verifies the manifest; it is not a new numerical audit. Then use the commands below. Keep the downloaded ZIP untouched and run reproduction only in a separate extraction. Saved result directories must be moved aside in that separate extraction before refitting, as the generating scripts intentionally refuse to overwrite frozen outcomes. Saved audit reports document developer checks, not independent human review.

Frozen internal folder labels are preserved for executable and hash compatibility; they are not part of the public research title.

## Rights and limitations

The included LICENSE copies the project's existing MIT License for the authors' code and accompanying artifacts. Dependencies retain their own licenses and are not bundled. The authors must confirm publication rights before releasing this archive. Synthetic records do not establish physical material validity, globally converged Prony optimization or a universal prediction-accuracy advantage.

## Existing numerical reproduction instructions

# Computational evidence for positive hereditary target prediction

This archive supplies the synthetic simulation specifications, computation
and audit code, saved numerical witness archives and result records underlying
the accompanying research article. It does not contain editorial correspondence,
old manuscript PDFs, reviewer reports or submission history.

Check the file hashes in `manifest.json` before running anything. Execution may
write newly computed outputs into the extracted copy. Preserve the downloaded
ZIP as the frozen record and use a separate extraction for each reproduction.
The recorded Python/NumPy/SciPy/Matplotlib dependencies are in
`P4/requirements_mcs_reproduction.txt` (including mpmath used by the high-
precision endpoint audit); no virtual environment or TeX engine is bundled.

## Recalculate the saved-case resource analysis

From the extracted archive root in an environment with those dependencies:

```text
python -B P4/tools/build_mcs_simulation_analysis.py
```

This checks the saved witness hashes and audits, reads the original 256 paired
cases, and reclassifies them at the unchanged diameter thresholds. It produces
the resource plot, editable count table and per-case summary. Expected result:
244 valid calibrations, with 238 short-window and 244 long-window sufficient
cases; conditional benefit 2.46 percentage points; response-duration sums 24
and 40. This command is retrospective recomputation without new random trials
or numerical solver calls. The calibration-valid subset uses ground truth
offline; it is not a subset an instrument can identify during deployment.

## Reproduce the revised publication figures

```text
python -B P4/tools/build_mcs_publication_figures.py
python -B P4/tools/build_mcs_simulation_analysis.py
```

The first command draws the explanatory workflow (Figure 1) and reads frozen
saved results for Figures 5 and 7. The second draws Figure 6 and its count table.
The bar-chart legends are outside the data axes. The workflow uses schematic
observation bands and interval geometry; these drawings are not measured data.
Vector PDF, editable SVG and PNG versions of Figures 1, 5 and 7 are included.
These commands change presentation only and do not generate new experiments.

## Recalculate the saved grid-refinement summary

```text
python -B P4/tools/build_mcs_computational_details.py
```

This verifies the frozen specification, code and witness archive hashes, then
summarizes 80 saved target/grid records and 320 earlier endpoint programs.
The maximum inner/outer endpoint gap decreases from 0.334896 at 16 cells to
0.003596 at 4096 cells, with the same 16 spectrum/target combinations at each
level. No new solver call or experimental comparison is performed. The table
concerns the uncapped observation-weighted study, not a convergence certificate
for every calibrated capped program.

## Audit or rerun numerical studies

The `P4/experiments` directory contains frozen protocols, generating programs
and alternative audits for observation-weighted prediction, loading endpoint
classification, matched loading designs, delayed rectangles, dead time/input
precision, capped calibration and matched classical continuous bounds.
For example:

```text
python -B P4/experiments/audit_loading_endpoint_phase.py
python -B P4/experiments/audit_dead_time_input_precision.py
python -B P4/experiments/audit_ultraslow_calibrated_validation.py
python -B P4/experiments/audit_matched_convex_baseline.py
```

Audits may invoke numerical computation and can take substantially longer than
the saved-case resource summary. We did not rerun all solver-based experiments
when assembling the submission files. Their saved audits document the earlier
executions. Do not change thresholds, priors or data to replace a failed result.

Failed 3% input precision, twelve underestimated calibrations, finite-grid
support failures and the tighter classical comparator are intentionally kept.
All results concern the specified synthetic models and information classes;
they do not establish physical material validity or new optimizer superiority.


## Direct fitting update (14 October 2026)

Three direct point-fit baselines and two midpoint references reuse all 256 saved cases; no new random trials. Their protocol, complete outcomes and convergence failures are included. See P4/docs/P4_DIRECT_FITTING_RESULTS_20261014.md.
Audit without refitting: python -B P4/experiments/audit_direct_fitting_baselines.py --phase full
Test arithmetic: python -B -m unittest discover -s P4/tests -p test_direct_fitting_baselines.py
To rerun fits, first make a separate extraction and move only the existing P4/results/direct_fitting_20261014 result directory aside; the scripts refuse to overwrite saved outcomes. Run the pilot, its audit, then the full runner and audit as described in the results README. Limit OpenBLAS/OMP to one thread per worker for the recorded scheduling.
