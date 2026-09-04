# P5 Retargeting Plan

Updated: 2026-09-02

## Decision history

The Applied Mathematical Modelling submission was desk rejected because the manuscript did not fit the journal's emphasis on novel mathematical modelling of real-world engineering, industrial, or environmental systems. The rejected AMM sources and PDF are frozen under `archive/rejected_AMM_20260902/`. The active manuscript is the journal-neutral version in `paper/`.

## Scientific identity

P5 is a statistical signal-processing and system-identification study. It asks when sparse grouped transient observations support an additional shared decay mode and when model order must remain unresolved. The central results concern adjacent-rate resolution, correlated residuals, held-unit transfer, and evidence-gated model-order decisions. The software package is supporting infrastructure.

## Target sequence

### 1. Primary: Signal Processing

Fit: the journal explicitly covers statistical signal processing, detection and estimation, spectral analysis and filtering, optimization, signal-processing software, and new applications. This framing does not require one engineering application to dominate the paper. The journal offers a subscription publication route; paid open access is optional.

Required revision:

- replace relaxation-first framing with multichannel transient and spectral-resolution framing;
- state the adjacent-rate result as a finite-window distinguishability result for grouped exponential signals;
- compare against BIC/AIC, matrix-pencil or Prony recovery, vector fitting, and a stability-selection baseline under identical observation budgets;
- report detection probability, false order elevation, refusal rate, calibration, and runtime;
- explain why held-channel prediction and mechanism resolution are distinct statistical tasks;
- retain PVA and copper-alloy data as physical validation, with gas and hydraulic data as transfer or refusal tests.

### 2. Backup: Circuits, Systems, and Signal Processing

Fit: the journal ranges from mathematical foundations to practical system and signal-processing design and uses a hybrid publishing model. This is the lower-risk backup if the paper remains methodologically broad but does not reach the novelty threshold of Signal Processing.

### 3. Stretch: Mechanical Systems and Signal Processing

Fit exists under structural/system identification, parameter estimation, time-series methods, and uncertainty quantification. The journal expects a demonstrable advance in engineering knowledge and favors theory combined with experimental evidence. This route therefore requires a dominant mechanical dataset, a mechanical interpretation of the modes, and comparison with established system-identification methods. It is not the immediate submission target.

## Stop rule

Do not resubmit P5 as a generic relaxation-modelling paper. A new package is permitted only after the signal-processing baselines and common-budget detection study are complete. If those additions do not show a clear advantage, use Circuits, Systems, and Signal Processing rather than escalating the application rhetoric.
