# AMM Data and Claim Traceability Audit

Audit date: 2026-08-29

## Conclusion

No source-less experimental number was found in the current AMM manuscript. Every reported quantitative result is traceable to a machine-readable result or an experiment summary in `P5/results`. The abstract's calibration count was corrected to the actual design: four residual generators with 40 separated and 40 coalesced trials each, for 160 supported and 160 unresolved decisions.

## Claim-to-artifact map

| Manuscript claim block | Primary local evidence |
|---|---|
| 160/160 semi-synthetic support and refusal counts; Wilson intervals | `results/submission_calibration_transfer.json`, `results/submission_calibration_transfer.md` |
| PVA rank decisions, 28-s/96-sample boundary, held-specimen NRMSE, bootstrap interval | `results/public_pva_relaxation.json`, `results/public_pva_relaxation.md` |
| Copper-alloy ranks, nine-group transfer, BIC and AR(1)-profile sensitivity | `results/public_kupferdigital_relaxation.json`, `results/public_kupferdigital_relaxation.md` |
| Gas-sensor held-experiment prediction, rate coalescence, threshold scan | `results/public_uci_gas_recovery.json`, `results/public_uci_gas_recovery.md` |
| Hydraulic held-cycle results, boundary index, AR(1)-profile sensitivity | `results/public_uci_hydraulic_transients.json`, `results/public_uci_hydraulic_transients.md`, `results/submission_robustness_audit.json` |
| Oscillatory, partially shared, dense-spectrum, and group-conformal stress tests | `results/stage66_extension_combination.json` |
| Dataset record, license, sample/unit count, and file hashes | `results/public_task_provenance_audit.json` |
| Figure-level AMM summaries | `submission_amm/AMM_evidence_supplement/amm_evidence_summary.json` |

## Provenance boundary

- PVA and copper-alloy records are public material-relaxation datasets cited by DOI.
- The gas-sensor and hydraulic datasets are official UCI records cited by DOI and are used as scope/falsification tasks, not as direct material-mechanism validation.
- Semi-synthetic controls have known generating rank and are used only to calibrate and test decision behavior.
- Extension experiments are controlled stress tests of assumptions; they are not represented as real-data validation.

## Statistical interpretation boundary

- Bootstrap intervals reported for small material datasets are descriptive and clustered at the independent experimental-unit level.
- Holm adjustment is applied within each declared public-task comparison family.
- Threshold grids audit sensitivity and are not used to tune thresholds after observing outcomes.
- A supported finite realization is an empirical shared state model, not proof of a unique microscopic mechanism.
- `UNRESOLVED` records insufficient resolution under the declared design; it does not prove absence of a multi-timescale mechanism.

No invented industrial measurement, inaccessible proprietary record, or synthetic result presented as an observed public result remains in the manuscript.
