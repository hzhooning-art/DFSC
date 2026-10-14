# Direct fitting baseline results (14 October 2026)

New fits use saved observations; no new random trials, material measurements or independent blind cohort.
The protocol was fixed before fitting, but existing target results were already known. Pilot cases are included in the full set and do not add independent trials.
The same 80 responses, known unit mass, positive spectrum, rate cap 4 and exact loadings are used. Fit API accepts only observations, loading geometry and fixed protocol.
Prony order and ridge penalty are selected on the trailing quarter of each loading record, then refitted. Fixed numerical grids are not continuum truth assumptions.
Absolute point-error limits are 0.10/0.05, half the original diameter limits; this does not alter the old prospective width decisions.
Point-baseline intervals are NOT supplied or asserted calibrated. Mean point error does not establish worst-case uncertainty superiority.

## All-case results

| Window | Method | Successful fits | Mean absolute G error | Mean absolute stress error | Joint tolerance exceedances among successes |
|---|---|---:|---:|---:|---:|
| 8 | Grid LS | 256/256 | 0.015825 | 0.006767 | 0/256 |
| 8 | Grid ridge | 256/256 | 0.018426 | 0.007982 | 0/256 |
| 8 | Free-rate Prony | 194/256 | 0.017747 | 0.007639 | 1/194 |
| 8 | Symmetric midpoint | 256/256 | 0.017491 | 0.008643 | 0/256 |
| 8 | Classical midpoint | 256/256 | 0.017345 | 0.008535 | 0/256 |
| 16 | Grid LS | 256/256 | 0.003493 | 0.002184 | 0/256 |
| 16 | Grid ridge | 256/256 | 0.003703 | 0.002449 | 0/256 |
| 16 | Free-rate Prony | 204/256 | 0.003757 | 0.002204 | 0/204 |
| 16 | Symmetric midpoint | 256/256 | 0.002843 | 0.001933 | 0/256 |
| 16 | Classical midpoint | 256/256 | 0.002791 | 0.001894 | 0/256 |

## Family-stratified results

| Family | Window | Method | Success | Mean G error | Mean stress error | Joint exceedances |
|---|---|---|---:|---:|---:|---:|
| logwide_continuous | 8 | Grid LS | 64/64 | 0.016404 | 0.006975 | 0/64 |
| logwide_continuous | 8 | Grid ridge | 64/64 | 0.016942 | 0.007115 | 0/64 |
| logwide_continuous | 8 | Free-rate Prony | 51/64 | 0.027228 | 0.011923 | 0/51 |
| logwide_continuous | 8 | Symmetric midpoint | 64/64 | 0.012309 | 0.006347 | 0/64 |
| logwide_continuous | 8 | Classical midpoint | 64/64 | 0.012216 | 0.006238 | 0/64 |
| logwide_continuous | 16 | Grid LS | 64/64 | 0.004232 | 0.002942 | 0/64 |
| logwide_continuous | 16 | Grid ridge | 64/64 | 0.004288 | 0.003072 | 0/64 |
| logwide_continuous | 16 | Free-rate Prony | 54/64 | 0.006615 | 0.003575 | 0/54 |
| logwide_continuous | 16 | Symmetric midpoint | 64/64 | 0.002455 | 0.001640 | 0/64 |
| logwide_continuous | 16 | Classical midpoint | 64/64 | 0.002465 | 0.001641 | 0/64 |
| mixed_four | 8 | Grid LS | 64/64 | 0.020612 | 0.009330 | 0/64 |
| mixed_four | 8 | Grid ridge | 64/64 | 0.020857 | 0.009286 | 0/64 |
| mixed_four | 8 | Free-rate Prony | 50/64 | 0.021612 | 0.009689 | 0/50 |
| mixed_four | 8 | Symmetric midpoint | 64/64 | 0.015622 | 0.008331 | 0/64 |
| mixed_four | 8 | Classical midpoint | 64/64 | 0.015654 | 0.008279 | 0/64 |
| mixed_four | 16 | Grid LS | 64/64 | 0.003948 | 0.002729 | 0/64 |
| mixed_four | 16 | Grid ridge | 64/64 | 0.003740 | 0.002804 | 0/64 |
| mixed_four | 16 | Free-rate Prony | 55/64 | 0.003469 | 0.002579 | 0/55 |
| mixed_four | 16 | Symmetric midpoint | 64/64 | 0.003974 | 0.002485 | 0/64 |
| mixed_four | 16 | Classical midpoint | 64/64 | 0.003892 | 0.002434 | 0/64 |
| ultraslow_two | 8 | Grid LS | 64/64 | 0.015169 | 0.006458 | 0/64 |
| ultraslow_two | 8 | Grid ridge | 64/64 | 0.023429 | 0.010688 | 0/64 |
| ultraslow_two | 8 | Free-rate Prony | 43/64 | 0.012973 | 0.005306 | 1/43 |
| ultraslow_two | 8 | Symmetric midpoint | 64/64 | 0.032193 | 0.015885 | 0/64 |
| ultraslow_two | 8 | Classical midpoint | 64/64 | 0.031688 | 0.015618 | 0/64 |
| ultraslow_two | 16 | Grid LS | 64/64 | 0.002966 | 0.002000 | 0/64 |
| ultraslow_two | 16 | Grid ridge | 64/64 | 0.003556 | 0.002649 | 0/64 |
| ultraslow_two | 16 | Free-rate Prony | 45/64 | 0.002141 | 0.001371 | 0/45 |
| ultraslow_two | 16 | Symmetric midpoint | 64/64 | 0.002992 | 0.002846 | 0/64 |
| ultraslow_two | 16 | Classical midpoint | 64/64 | 0.002891 | 0.002763 | 0/64 |
| uniform_slow | 8 | Grid LS | 64/64 | 0.011115 | 0.004307 | 0/64 |
| uniform_slow | 8 | Grid ridge | 64/64 | 0.012476 | 0.004837 | 0/64 |
| uniform_slow | 8 | Free-rate Prony | 50/64 | 0.008317 | 0.003227 | 0/50 |
| uniform_slow | 8 | Symmetric midpoint | 64/64 | 0.009842 | 0.004007 | 0/64 |
| uniform_slow | 8 | Classical midpoint | 64/64 | 0.009824 | 0.004004 | 0/64 |
| uniform_slow | 16 | Grid LS | 64/64 | 0.002827 | 0.001065 | 0/64 |
| uniform_slow | 16 | Grid ridge | 64/64 | 0.003228 | 0.001269 | 0/64 |
| uniform_slow | 16 | Free-rate Prony | 50/64 | 0.002442 | 0.001063 | 0/50 |
| uniform_slow | 16 | Symmetric midpoint | 64/64 | 0.001950 | 0.000761 | 0/64 |
| uniform_slow | 16 | Classical midpoint | 64/64 | 0.001917 | 0.000739 | 0/64 |

## Calibration and failures

All 12 calibration-underestimation cases remain. The full JSON reports all/valid/invalid strata separately; the true validity flag never enters fitting.
Primary fitting wall time: 461.31 s with 4 workers. CPU occupancy is not converted into a fair LP-vs-fitting timing claim.
Independent-primitive replay: maximum prediction discrepancy 8.43769498715119e-15.
Final refit failures: {"grid_ls": 0, "grid_ridge": 0, "prony_cv": 114}.
Order/penalty candidate failures: {"grid_ls": 0, "grid_ridge": 0, "prony_cv": 465}.
Failed nonlinear starts: 2301.
These failures are retained; they are not evidence of inherent Prony failure or global optimality of successful fits.
Selected order distribution: {"2": 282, "4": 105, "8": 110}.
Selected penalty distribution: {"0.0": 164, "1e-07": 35, "1e-05": 172, "0.001": 141}.

## Decision implications

A small response residual alone is not a target certificate. Nevertheless, this dataset is often already information-sufficient, so accurate fitting results must be retained. No general fitting-error, confidence-coverage, runtime or physical-validity advantage is certified.
The direct point-prediction gap is addressed within this synthetic information class. Calibrated fitting intervals, robust global nonlinear optimization and real-material validation remain separate gaps.

Reproduce in a fresh extraction: run the five tests, then direct_fitting_baselines.py --phase pilot, audit_direct_fitting_baselines.py --phase pilot, run_direct_fitting_full.py, audit_direct_fitting_baselines.py --phase full and report_direct_fitting_baselines.py. Saved-result audits can be run without rerunning fits. Outputs are guarded against overwriting.
