# P5: collision-aware multichannel order testing

This directory contains the frozen code, simulation results and protocols supporting the manuscript *Detecting Colliding Exponential Components in Multichannel Signals: Diversity-Dependent Resolution and Calibrated Score Tests*. This is a theory-and-controlled-simulation study; the files do not constitute an independent experimental dataset. The manuscript's previous hydraulic/PVA repeat archives were excluded from its inferential claims.

The paper's Figure 1 is generated analytically from the normalized distance boundary; Figure 2 uses `matched_glrt_results.json`. The primary reported power, calibration and stress ranges are recorded in the JSON files here. Each result's `hashes` map identifies the exact frozen source/protocol inputs. The scripts preserve their original source bytes so those maps remain auditable.

## Environment

Python 3.12; NumPy 2.5; SciPy 1.18; Matplotlib 3.11; mpmath (used by the crossover audit). Execute commands from this directory. Some full Monte Carlo generators are computationally expensive; the `audit_*.py` scripts provide fast checks of the frozen outputs.

## Key files

- `uniform_crossover_*`, `local_limit_power_*`, `finite_local_limit_*`: collision transition and local-power calibration.
- `phase_matched_score_*`, `cone_matched_score_*`, `score_power_curve_*`: proposed score tests and power curves.
- `matched_glrt_*`, `score_design_stress_*`: held-out profiled GLRT comparison and 12-design stress study.
- `general_atom_*`: smooth-atom extension.
- `samp_direct_*`: published SAMP compatibility check and clearly labeled real-decay ablation. The third-party SAMP source is **not** redistributed; see `SAMP_DIRECT_PROTOCOL.md` for its upstream commit and source hashes. The bundled fast audit checks frozen SAMP result structure, but full source-fidelity verification requires fetching the upstream commit into `vendor_samp` and running `audit_samp_direct.py` separately.
- `whitening_*`, `finite_training_calibration_*`: independent-training covariance and fixed-space F-pivot audits.

Run `python audit_public_bundle.py` to check every frozen source hash and run the audit scripts. The bundle contains no raw external experimental data or private credentials. The repository's MIT license applies to original project code; third-party source is excluded.
