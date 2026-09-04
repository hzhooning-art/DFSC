# P5: Identifiable Memory-Rank Discovery

[![P5 memory protocol](https://github.com/hzhooning-art/DFSC/actions/workflows/p5-memory-protocol.yml/badge.svg)](https://github.com/hzhooning-art/DFSC/actions/workflows/p5-memory-protocol.yml)

Archived software and reproducibility release: [Zenodo DOI 10.5281/zenodo.22168774](https://doi.org/10.5281/zenodo.22168774).

## Reproducible setup

From the repository root, install the package and test dependencies, verify the
software, and retrieve the public datasets with:

```console
python -m pip install -e "P5[test]"
python -m unittest discover -s P5/tests -p "test_*.py"
python P5/scripts/fetch_public_data.py --dataset all
```

The data command downloads each source from its authoritative repository and
checks its frozen SHA-256 digest. Raw public datasets are deliberately excluded
from Git; identifiers, provenance, and checksums are recorded in
`P5/data/README.md`.

## Stage 62: independent public-data validation

P5 now includes a direct public-data audit using raw PVA gel polymer
electrolyte stress-relaxation observations from Zenodo record 21333840. The
analysis combines shared positive-rate realizations, leave-one-specimen-out
late-time prediction, information gain, rate separation, fold stability, and
an explicit indeterminate outcome. See `STAGE62_PUBLIC_TASK_PROTOCOL.md`,
`STAGE62_PUBLIC_DATA_REPORT.md`, and `RELATED_WORK_MATRIX.md`.

P5 studies a question that is not answered by fitting a named relaxation curve:

> How many latent memory states are actually supported by finite, noisy observations,
> and when should a non-Markovian mechanism claim be refused?

The project learns a positive pole-residue memory realization, selects the smallest
supported memory rank, audits local identifiability, and emits an explicit refusal
when the observation protocol cannot separate competing realizations.

## Why this is a new P5

The previous P5 workspace, which compared exponential, KWW, and Mittag-Leffler
curves, was deleted. That route mainly performed family selection. The new route
instead targets a dynamical object: a minimal executable Markovian lifting of a
non-Markovian closure. It also tests whether multiple observed field channels can
share a low-dimensional memory spectrum.

## First probe

Run:

```powershell
..\P1\.venv\Scripts\python.exe experiments\probe_memory_rank.py
```

Outputs are written to `results/memory_rank_probe.json` and
`results/memory_rank_probe.md`.

## Status

This is a feasibility workspace, not a finished solver or a validated scientific
discovery package. The pass/fail criteria are defined in `RESEARCH_DIRECTION.md`.

The initial GPU probe passed its four route-selection checks: scalar rank one and
separated rank two were recovered, nearly coincident poles triggered refusal, and a
12-channel shared-spectrum system recovered rank two. These outcomes establish only
local synthetic feasibility. Repeated-seed phase diagrams, out-of-class tests,
high-dimensional scaling, theory, and real data remain open.

The second-stage boundary scan found a nontrivial observation limit. With a short
horizon and a pole-rate ratio of 4, rank-two recovery increased from 0/3 for one
channel to 3/3 for twelve shared-spectrum channels. At a ratio of 8 it increased from
2/3 to 3/3. A ratio of 2 remained unresolved even with twelve channels. This supports
the shared-spectrum hypothesis while preserving a clear refusal region.

A later high-dimensional audit extends that shared-spectrum result to 16, 64,
and 256 channels. Under the frozen short-horizon rank-two protocol, the shared
model recovered the spectrum in all three repeats at every high-dimensional
channel count, whereas only 19%--33% of channels were independently resolved.
At 256 channels the median shared log-rate error was 0.0333. Median shared-fit
time increased from 4.40 s at 16 channels to 6.39 s at 256 channels on the
recorded GPU, with peak allocated memory increasing from 21.1 to 78.8 MiB.
This is evidence for the statistical and computational value of an exactly
shared spectrum on the controlled generator; robustness to approximate sharing,
channel correlation, and real fields remains untested.

The third probe tested the positive-real model contract. Across three noise repeats
per case, positive rank-one and rank-two controls were accepted and assigned the
correct rank, while signed and oscillatory memory kernels were refused in every
trial.

The fourth-stage coarse and refined scans then moved the violations toward the class
boundary. Under the declared noise, horizon, and sampling protocol, small signed
residues (up to 0.005) and low oscillation frequencies (up to 0.03) remained
indistinguishable and were accepted. The next tested levels, 0.0075 and 0.04,
respectively, were refused in all three repeats because validation residuals became
strongly correlated. This supplies an empirical refusal transition rather than the
unrealistic claim that every arbitrarily small class violation is detectable.

A fifth calibration probe varied noise and horizon over 216 conditional rank-one fits.
Neither positive-real zero control produced a false refusal in 36 trials per family.
Refusal rates increased from 0.417 to 0.611 for signed residues and from 0.306 to
0.611 for oscillatory kernels between the lower and upper boundary levels. Detection
was strongest at low noise and long horizons. The confidence intervals still overlap,
so these results define the next concentrated sampling region rather than a final
statistical guarantee.

The sixth probe jointly fitted candidate memory ranks 1--3 at a representative
transition operating point instead of fixing rank one. Across ten repeats per setting,
both in-class zero controls had zero false refusals and zero false rank elevations.
Refusal rose from 4/10 to 10/10 across the signed-residue transition and from 1/10 to
10/10 across the oscillatory transition. One above-boundary signed-residue trial chose
rank two, but it remained refused by independent diagnostics. These results support
the local compatibility of minimal-rank selection and out-of-class refusal; ten trials
per cell are still insufficient for a uniform statistical guarantee.

The seventh probe extended that joint competition along an information gradient:
short-horizon/high-noise, medium, and long-horizon/low-noise observations. In both
out-of-class families, above-boundary refusal increased from 0/6 in the low-information
regime to 6/6 in the medium- and high-information regimes, while every zero control
remained accepted at rank one. No violating case was accepted by hiding the mismatch
in a higher memory rank. This establishes an observation-dependent detectability
effect, but horizon and noise were changed together and only six trials were run per
cell. In particular, passing the model check at low information means "no detectable
mismatch," not "the physical mechanism has been identified."

The eighth probe separated horizon and noise in 120 additional joint fits. Lower noise
monotonically improved refusal for both mismatch families without harming zero controls.
Longer horizons helped signed-residue detection, but not oscillatory detection: the
oscillatory refusal rate dropped from 6/6 at horizons 10 and 14 to 1/6 at horizon 18.
The late-window mismatch-to-noise ratio simultaneously fell from 5.82 to 4.38 and 3.85,
showing that a terminal validation window can miss a decayed transient. The associated
prespecified route check failed. This negative result redirects the next experiment
toward time-local or multi-window residual diagnostics rather than supporting a blanket
"longer is better" claim.

The ninth probe implemented that redesign with prespecified early, middle, and late
diagnostic windows. A separate zero-control calibration split fixed a multiplicity-aware
maximum-lag threshold before any evaluation cases were scored. At the previously failed
H=18 operating point, held-out oscillatory refusal increased from 0/8 with the terminal
gate to 8/8 with the multi-window gate; signed-residue refusal increased from 6/8 to
8/8, and both zero controls retained 0/8 false refusals. This is a successful local
repair, not yet a population-level error guarantee: the calibration set contains only
36 window statistics and the held-out confidence intervals remain wide.

The corresponding artifacts are
`results/calibrated_multiwindow_refusal.json` and
`results/calibrated_multiwindow_refusal.md`.

A tenth probe tested whether that threshold generalized beyond its original design.
The initial 120-fit calibration failed because two poorly optimized null fits polluted
the extreme empirical quantile; that failure is preserved in
`results/multiwindow_external_calibration.json`. A quality-gated, fixed-two-start
replication on fresh seeds produced 120/120 valid fits and a stable threshold of
0.20514 (bootstrap 95% interval 0.18731--0.21744). It detected slow-decay oscillations
and shifted early/middle transients in every trial, but a threshold calibrated on a
regular grid falsely refused 2/12 oscillatory zero controls under jittered sampling and
detected a faster-decaying oscillation in only 2/6 trials. The overall route therefore
failed. The next design must calibrate residual correlation for the observation-time
process itself; a universal lag-index threshold is not supported.

The quality-gated artifacts are
`results/quality_gated_multiwindow_calibration.json` and
`results/quality_gated_multiwindow_calibration.md`.

An eleventh probe calibrated the diagnostic directly on independently jittered
observation grids and compared the simple index-lag statistic with an
actual-time-gap-weighted alternative. All 120 calibration fits passed the numerical
quality gate. The frozen thresholds were 0.20669 and 0.22105, with bootstrap 95%
intervals of 0.19466--0.22658 and 0.19722--0.24225. On 48 fresh cases, both methods
refused all slow-decay oscillatory violations and both shifted-transient families,
while producing 0/12 false refusals for the oscillatory zero control and 1/12 for the
signed zero control. The known fast-decay boundary was refused in 3/6 trials. The two
statistics made identical decisions in every case, so the experiment supports
sampling-stratified calibration but does not support time-gap weighting as an added
methodological contribution. The primary route passed with the fast-decay family
declared in advance as a secondary boundary analysis; missingness and clustered or
strongly gapped sampling remain untested.

The sampling-aware artifacts are
`results/sampling_aware_residual_calibration.json` and
`results/sampling_aware_residual_calibration.md`.

A twelfth probe froze the preferred index-lag threshold at 0.20669 and tested it
without recalibration under 30% random missingness, strongly clustered observations,
and a continuous middle-horizon gap. All 84 fresh fits passed the numerical quality
gate and selected rank one. Each primary mismatch family was refused in 4/4 trials
under every sampling process. The zero controls produced three false refusals across
36 trials: two under random missingness and one under clustered sampling; the long-gap
controls produced none. These small cells satisfy the prespecified feasibility rule
but have wide Wilson intervals and do not establish population-level error control.
The fast-decay boundary remained weakly detectable at 1/4, 2/4, and 1/4. A first
attempt terminated before output because the clustered design left too few independent
training points; the corrected design retained clustered density while adding support
outside the diagnostic windows.

The stress-test artifacts are
`results/sampling_process_stress.json` and
`results/sampling_process_stress.md`.

A thirteenth probe replicated the null law with 60 calibration and 40 disjoint
evaluation fits per sampling process. It calibrated the 95th percentile of independent
per-fit three-window maxima, preserving window dependence. All 300 fits passed the
numerical quality gate. Sampling stratification reduced held-out false refusals from
6/40 to 1/40 for random missingness and from 5/40 to 1/40 for long gaps. The clustered
stratum failed the prespecified route: false refusals decreased only from 5/40 to 4/40,
with a Wilson 95% upper bound near 0.231. Three clustered refusals came from the
oscillatory zero family, and all four fits were accurate and well conditioned. The
result rejects the idea that a single categorical `clustered` threshold is sufficient;
the next test must condition on measurable observation geometry without tuning on the
failed evaluation cases.

The null-law artifacts are
`results/sampling_stratified_null_law.json` and
`results/sampling_stratified_null_law.md`.

A fourteenth probe tested the next proposed repair without reusing the failed
evaluation cases. It generated continuously varying clustered observation designs,
used 180 independent zero-model fits to define a residual-blind geometry score and
three equal-calibration-mass bins, and evaluated the frozen bins on 120 fresh fits.
All fits passed the numerical quality gate, but the geometry score had only 0.115 rank
correlation with the residual tail. The legacy clustered threshold, a newly calibrated
global threshold, and the geometry-conditional thresholds each falsely refused the
same 8/120 controls. Six failures came from the oscillatory zero family and seven were
localized to the late window. Coarse summaries of gap size, point concentration, and
minimum window support therefore do not justify geometry-conditioned acceptance. The
next falsification test must preserve the exact observation-time design, for example
through a frozen nested parametric bootstrap, rather than adding post-hoc bins.

The geometry-conditional artifacts are
`results/cluster_geometry_conditional_null.json` and
`results/cluster_geometry_conditional_null.md`.

A fifteenth probe replaced coarse geometry strata with exact-design conditional
parametric bootstrapping. For each of 30 fresh clustered zero-model cases, it preserved
the actual observation times, split, diagnostic windows, and noise level, generated 19
conditional copies from the fitted null, and fully refitted every copy. All 600 outer
and bootstrap fits passed the numerical quality gate. The two global thresholds each
falsely refused 2/30 controls, whereas the finite-sample conditional Monte Carlo rule
refused 0/30 (Wilson 95% upper bound about 0.114). Conditional thresholds varied from
0.15172 to 0.36430, showing that the repair was genuinely design dependent. This is a
successful null-control feasibility result, not yet a complete refusal method: the
bootstrap p-value grid is coarse, the additional refits are expensive, and power on
out-of-class alternatives remains unmeasured.

The exact-design bootstrap artifacts are
`results/exact_design_conditional_bootstrap.json` and
`results/exact_design_conditional_bootstrap.md`.

A sixteenth probe froze the 19-copy rule and measured power on fresh clustered
designs. Exact-design calibration refused 5/6 slow oscillatory violations and 6/6 of
both shifted-transient families, while the known fast-decay boundary remained weak at
2/6. The statistical power checks passed, but one of 456 conditional refits failed the
numerical quality gate, so the overall route failed as prespecified. A deterministic
replay showed that the failed fit had RMSE 0.194; four additional starts all recovered
RMSE near 6.05e-4. This supports a quality-triggered retry policy as the next hypothesis,
not a retroactive correction. The 24 conditional decisions took about 1019 seconds,
which also establishes a material deployment-cost limitation.

The power and diagnostic artifacts are
`results/exact_design_conditional_power.json`,
`results/exact_design_conditional_power.md`, and
`results/exact_design_bootstrap_failure_diagnostic.json`.

A seventeenth probe froze a quality-triggered retry policy before evaluation and
replicated exact-design conditional inference on disjoint seeds. Every fit began with
two starts; up to four additional starts were permitted only if the selected fit
failed the prespecified RMSE or conditioning gate. Across 20 zero controls and 12
primary alternatives, all 640 outer and bootstrap fits passed without triggering a
retry. The conditional rule produced 0/20 false refusals (Wilson 95% upper bound about
0.161) and refused 4/4 cases in each of the three primary mismatch families. Thus the
statistical null-control and primary-power pattern replicated and the frozen retry
policy did not alter valid fits. Because no fresh fit triggered the policy, this run
does not independently validate retry recovery; recovery evidence remains limited to
the deterministic replay of the earlier optimizer failure. The 32 decisions required
about 1373 seconds, confirming that exact-design bootstrapping is an offline audit.

The replication artifacts are
`results/quality_triggered_retry_replication.json` and
`results/quality_triggered_retry_replication.md`.

An eighteenth probe deliberately stressed the frozen retry policy using correctly
specified rank-one zero models, continuously varying clustered designs, only 20
training points, and a constrained 75-Adam/20-L-BFGS budget per start. Six starts were
evaluated for audit, while projected deployment cost charged starts 3--6 only after
the first two failed the unchanged quality gate. Among 120 fresh cases, the initial
two-start fit failed once; the retry recovered that case from validation RMSE 0.1865
to 6.45e-4, leaving 0/120 final calibration failures. Projected extra-start overhead
was 1.67%. Forty-three valid initial fits had a lower-RMSE exhaustive winner, but the
median RMSE ratio was 0.999984 and even the largest improvement was only about 0.0134%,
so the gate did not conceal material errors. The prespecified route nevertheless
failed because it required at least three initial failures. The experiment supplies
one fresh recovery event, not a statistically stable recovery-rate estimate.

The numerical-stress artifacts are
`results/retry_numerical_stress.json` and
`results/retry_numerical_stress.md`.

A nineteenth probe compared the positive-real memory model with direct trajectory
baselines on three correctly specified, separated-rank systems. Each method received
the same 48 sparse observations from the first 60% of a 16-unit horizon; the remaining
40% was reserved for genuine temporal extrapolation. Across nine independent noisy
instances, the mechanism model recovered memory rank in 9/9 cases and had no selected-
fit quality failure. A regularized damped-modal (Prony-like) fit reached the noise scale
on the observed interval but selected three trajectory modes in every case and had
median case-wise extrapolation errors between 0.043 and 0.190, compared with 1.18e-4,
1.52e-4, and 6.62e-3 for the mechanism model. Modal rank is not interpreted as memory
rank. The initial tanh MLP baseline failed its prespecified optimization-quality gate:
even after deterministic L-BFGS refinement its median training RMSE remained 2.25e-3
to 2.92e-3 against noise standard deviation 8e-4. Its extrapolation result is therefore
retained as a negative baseline-implementation result, not evidence of superiority.

A twentieth, optimizer-free baseline used cubic smoothing splines with a smoothing
budget fixed from the known noise variance. It achieved median training RMSE about
5.5e-4 and interpolation RMSE 5.8e-4 to 7.5e-4, but extrapolation RMSE rose to 0.565--
1.439. This confirms the expected benefit of a correct mechanism constraint over
unconstrained interpolation in a matched synthetic setting. It does not establish
universal superiority under model misspecification or on real data. Of 54 mechanism
candidate starts, 28 nonselected starts or ranks crossed the quality threshold, while
all nine final two-start/BIC selections were valid; this supports the existing quality
gate but contributes no new retry-recovery event.

The baseline artifacts are
`results/mechanism_vs_trajectory_baselines.json`,
`results/mechanism_vs_trajectory_baselines.md`,
`results/unconstrained_spline_baseline.json`, and
`results/unconstrained_spline_baseline.md`.

A twenty-first probe replaced exact model matching with two controlled departures
from a two-pole positive-real system. The first family increased both pole rates
linearly over the horizon; the second added cubic state damping. Each family used
control, mild, and strong perturbations with two independent noisy repeats per cell.
Rate drift up to a 75% end-of-horizon increase retained rank-two selections and
mechanism-model extrapolation RMSE below 6.5e-3. Mild nonlinear feedback selected an
effective rank-three model and retained extrapolation RMSE below 3.6e-3. Strong
nonlinear feedback selected rank two but produced Jacobian condition numbers near
2.24e15, so both repeats were refused by the frozen numerical quality gate. There
were no silent failures under the prespecified late-audit rule, and every strong case
was either more accurate than the trajectory baselines or refused. This is a
12-instance pilot: the refusal is retrospective because the audit uses observations
beyond the training horizon, and the transition probability is not calibrated.

The misspecification artifacts are
`results/controlled_misspecification.json` and
`results/controlled_misspecification.md`.

A twenty-second probe densified the nonlinear-feedback transition with seven
strengths from 0.05 to 0.20 and six independent noisy repeats per strength. The
frozen gate accepted all 12 fits at strengths 0.05 and 0.075, then refused all 30
fits at strengths 0.10 and above. The observed refusal bracket was therefore
0.075--0.10. Rank three absorbed the mismatch in every fit through strength 0.10;
the rank-three selection rate then decreased from 0.833 at 0.125 to zero at 0.20.
All refusals were available before extrapolation: training-fit quality caused the
first refusals, while ill-conditioned Jacobians appeared in one of six fits at 0.175
and four of six at 0.20. No case crossed the late-error limit without refusal. With
only six repeats per strength, the Wilson 95% interval is [0, 0.390] for 0/6 and
[0.610, 1] for 6/6. The bracket is consequently a fixed-design protocol boundary,
not a universal physical critical point or calibrated refusal probability.

The transition artifacts are
`results/nonlinear_transition_boundary.json` and
`results/nonlinear_transition_boundary.md`.

A twenty-third probe refined the majority-refusal bracket with six strengths from
0.075 to 0.10 and eight fresh repeats per strength. Refusal rates were 1/8, 3/8,
7/8, 7/8, 8/8, and 8/8, narrowing the observed majority transition to
0.080--0.085. Every fit selected effective rank three. All 34 refusals were caused
by the selected training RMSE crossing the frozen 4-noise-standard-deviation limit;
none was caused by ill-conditioning or the retrospective late-error audit. Across
the six strength-group medians, strength and normalized training RMSE had exploratory
Spearman rho 0.943. This six-group association and the Wilson intervals are too small
for calibration. The result identifies the operational driver of the current gate,
but does not show that the same strength bracket transfers across noise, horizon, or
channel geometry.

The refinement artifacts are
`results/nonlinear_boundary_refinement.json` and
`results/nonlinear_boundary_refinement.md`.

A twenty-fourth probe tested whether the immediate gate driver transfers across
noise levels. Three paired noise levels (4e-4, 8e-4, and 1.6e-3) shared the same
clean trajectories, standard-normal perturbations, and sampling indices at four
nonlinear strengths with four repeats per cell. The noise-normalized gate accepted
15 of 48 fits and produced no accepted case above the prespecified operational
extrapolation target of 10 noise standard deviations; its largest accepted value
was 6.02. A fixed absolute training-RMSE gate accepted 11 fits, admitted two cases
above that target, and reached a maximum accepted extrapolation error of 12.11 noise
standard deviations. The two gates disagreed on 12 cases. The fixed gate also made
22 refusals without a corresponding relative extrapolation failure, compared with
16 for the normalized gate.

This paired experiment supports normalization by the declared observation-noise
scale when transferring a residual gate across noise regimes. It does not calibrate
a universal threshold: the 10-sigma extrapolation level is an operational audit
target rather than a theoretical error bound, and the normalized gate still made
16 conservative refusals under that target. Transfer across horizon, channel
geometry, unknown or heteroscedastic noise, and real observations remains open.

The transfer artifacts are
`results/noise_normalized_gate_transfer.json` and
`results/noise_normalized_gate_transfer.md`.

A twenty-fifth probe removed the assumption that the gate receives the true noise
scale. Two independent measurements were generated at every observation point under
an increasing heteroscedastic noise profile. Their mean was used for fitting, and
their paired difference estimated the effective training-noise RMS without using
the mechanism-model residual. Across 27 fits, the maximum noise-scale estimation
error was 5.42%. The estimated gate agreed with the oracle-noise gate in 26/27 cases;
the only disagreement was a conservative refusal at an estimated ratio of 4.03
versus an oracle ratio of 3.81.

The estimated gate accepted 4 fits and had no accepted case above the operational
10-noise-standard-deviation extrapolation target; its maximum accepted value was
3.26. The fixed absolute gate accepted 11 fits, admitted one case at 10.58 effective
extrapolation-noise standard deviations, and therefore failed the silent-failure
check. The estimated gate also made 14 refusals without a corresponding relative
extrapolation failure. The result supports replicate-based noise normalization as a
safe controlled extension, but not as an efficient or universally calibrated gate.
It assumes repeated Gaussian measurements and does not address a single noisy
trajectory, temporal correlation, or non-Gaussian tails.

The replicate-noise artifacts are
`results/replicate_noise_gate_transfer.json` and
`results/replicate_noise_gate_transfer.md`.

A twenty-sixth probe removed the repeated-measurement requirement. Each fit used a
single noisy trajectory: a dense training-prefix trace estimated the observation
noise, while the mechanism model still used only 48 sparse observations. The
estimator combined robust first- and second-lag second differences to separate a
locally smooth signal contribution from an iid noise floor. Gaussian noise and a
contaminated Gaussian design with 2% ten-standard-deviation outliers were evaluated
at three base noise levels and three nonlinear strengths, with two repeats per cell
for 36 fits.

The robust noise estimate had median and 90th-percentile relative errors of 10.9%
and 26.6%. Its gate agreed with the oracle-noise gate in 32/36 cases (88.9%), accepted
18 fits, and produced no accepted extrapolation error above the prespecified
10-oracle-noise operational limit; the maximum accepted value was 8.73. In contrast,
the naive second-difference standard deviation accepted 24 fits and produced four
silent relative-extrapolation failures, reaching 14.18 oracle noise standard
deviations. Under contamination, the median robust scale estimate was 1.22 times the
core noise, compared with 2.19 for the naive estimate. The fixed absolute gate made
one silent failure.

All prespecified feasibility checks passed, but the result is not a general
single-series noise solution. The robust gate made seven refusals without a
corresponding relative extrapolation failure, and its four disagreements with the
oracle gate were safe but less selective acceptances caused by mild scale
overestimation. The estimator assumes a densely sampled calibration prefix, smooth
local signal curvature, iid core noise, and sparse gross contamination. Temporal
correlation and an arbitrarily sparse series remain outside the validated domain.

The single-series artifacts are
`results/single_series_robust_noise_gate.json` and
`results/single_series_robust_noise_gate.md`.

A twenty-seventh probe challenged the single-series gate with stationary AR(1)
observation noise. The design crossed correlations 0, 0.3, 0.6, and 0.85 with
three nonlinear strengths and two repeats, producing 24 fits. A conditional
AR(1) profile was evaluated over Chebyshev mean models of degrees 8, 9, and 10.
The gate used the resulting sensitivity envelope to refuse normalization when
correlation was not identifiable, rather than silently treating the series as
iid.

Twenty of 24 cases were identifiable and all four unidentifiable cases were
refused. On identifiable cases, the correlation-aware marginal scale had median
and 90th-percentile relative errors of 9.25% and 31.60%; the median absolute
correlation error was 0.108. Its decisions agreed with the oracle-scale gate in
17/20 identifiable cases (85%). The aware gate accepted 13 cases, produced no
accepted extrapolation error above the prespecified 10-oracle-noise operational
limit, and had a maximum accepted ratio of 7.38. The iid comparator also had no
silent failure, but refused 17 cases and made 11 refusals below the late-error
limit, compared with 5 for the correlation-aware gate.

All prespecified feasibility checks passed. The evidence supports an active
refusal contract for one stationary AR(1) model class; it does not establish a
general correlated-noise estimator. At correlation 0.85 the aware estimator
still underestimated marginal scale, although much less severely than the iid
estimator. Long-memory, nonstationary, cross-channel-correlated, and irregularly
sampled observations remain outside the validated scope.

The correlated-noise artifacts are
`results/correlated_noise_gate.json` and
`results/correlated_noise_gate.md`.

A twenty-eighth probe tested whether the AR(1) adequacy audit could safely detect
long-memory model mismatch before an expensive mechanism-fit matrix. The frozen
prescreen crossed finite-burn-in ARFIMA(0,d,0) orders 0, 0.15, 0.30, and 0.45,
three nonlinear strengths, and two repeats, giving 24 complete diagnostic cases.
A pooled eight-lag Ljung-Box test at alpha=0.01 was applied after conditional
AR(1) whitening. Entry to the full fitting matrix required at least 75% adequacy
in the d=0 controls and at least 75% mismatch detection for d>=0.30.

The route failed both frozen criteria. Only 2/6 iid controls were declared
adequate (33.3%), while only 7/12 strong-memory cases were detected (58.3%). The
experiment therefore stopped before mechanism optimization, as specified in the
design. No threshold was changed after observing the outcomes. This rules out the
present short-prefix pooled Ljung-Box construction as a reliable long-memory
refusal gate; it does not rule out frequency-domain or explicitly fractional
diagnostics.

The failed-route artifacts are
`results/long_memory_mismatch_prescreen.json` and
`results/long_memory_mismatch_prescreen.md`.

A twenty-ninth probe tested the independently frozen frequency-domain route.
It used a sensitivity-median local Whittle statistic over three detrending
degrees and three low-frequency bandwidths. Separate 99% null thresholds were
calibrated from 128 iid Gaussian Monte Carlo draws at prefix lengths 78, 256,
and 512. Each length then crossed four memory orders, three nonlinear signal
strengths, and two repeats, for 72 complete project cases. A length could pass
only with at least 75% adequacy among its six iid controls and at least 75%
detection among its twelve cases with d>=0.30.

No length passed. At length 78, all six controls were accepted but only 2/12
strong-memory cases were detected (16.7%). At lengths 256 and 512, all strong-
memory cases were detected, but control adequacy fell to 1/6 (16.7%) and 0/6.
The calibrated threshold was zero at every length because the nonnegative local
Whittle estimate placed almost the entire pure-iid null distribution on its
boundary. On project traces, residual deterministic curvature then produced
positive estimates even when d=0. The route was therefore rejected before any
mechanism fitting; neither the threshold nor the detrending family was changed
after observing the result.

This experiment rules out the present unconditional iid Monte Carlo calibration
for mixed deterministic-signal traces. It also separates two regimes: 78 samples
lack strong-memory power, while longer traces require a conditional null that
accounts for trend-estimation uncertainty. Any such conditional calibration is a
new route and must be frozen and tested independently.

The frequency-route artifacts are
`results/spectral_long_memory_feasibility.json` and
`results/spectral_long_memory_feasibility.md`.

A thirtieth probe isolated whether the failed frequency route was limited by
the statistic itself or by its unconditional null. It retained the same 72
project traces, local Whittle sensitivity statistic, 99% quantile, 128 draws,
and dual 75% acceptance criteria. The only change was to calibrate a separate
null for each prefix length and nonlinear strength around the known clean
deterministic trajectory. This is an oracle upper-bound experiment, not an
operational unknown-trend method.

The oracle route passed at lengths 256 and 512. At 256 points, 5/6 iid controls
were accepted and 10/12 strong-memory cases were detected, both 83.3%. At 512
points the corresponding rates were 6/6 and 9/12, or 100% and 75%. The 78-point
route remained underpowered, detecting only 2/12 strong-memory cases despite
accepting all controls. The minimal passing prefix length was therefore 256.

The success is conditional and heterogeneous. At strength 0.20, d=0.30 was
detected in only 1/2 cases at 256 points and 0/2 at 512 points. Conditioning on
the true trajectory establishes that the spectral statistic has usable power
in part of the declared domain, but it does not solve trend estimation or
justify a deployable refusal gate. The next required experiment must replace
the oracle trajectory with an independently estimated mean while preserving
the frozen decision rule.

The oracle artifacts are
`results/oracle_conditional_spectral_feasibility.json` and
`results/oracle_conditional_spectral_feasibility.md`.

A thirty-first probe replaced the oracle clean trajectory with a mean estimated
from an independent iid calibration trace. The frozen experiment retained only
the previously eligible prefix lengths 256 and 512, the same local Whittle
sensitivity statistic, 99% conditional-null quantile, 128 Monte Carlo draws,
four memory orders, three nonlinear strengths, two repeats, and the dual 75%
acceptance criteria. A cubic smoothing spline estimated the calibration mean;
every conditional-null draw regenerated a calibration trace, refitted the mean,
and then generated an independent evaluation trace. The design therefore
propagated trend-estimation uncertainty without reusing the evaluated residual.

Control calibration was reliable: all six iid controls were accepted at both
lengths. Mean-estimation RMSE across the three nonlinear strengths ranged from
2.44e-4 to 2.85e-4 at length 256 and from 2.01e-4 to 2.30e-4 at length 512.
However, strong-memory detection reached only 7/12 (58.3%) and 8/12 (66.7%),
below the frozen 75% requirement. All d=0.45 cases were detected, but only 1/6
and 2/6 d=0.30 cases were detected at lengths 256 and 512, respectively; d=0.15
was not part of the strong-memory acceptance count and was never detected.

The estimated-mean route therefore failed and no minimum passing length exists.
This closes the present spline-conditioned local Whittle construction before
mechanism fitting. It establishes that independent trend estimation can control
false positives and retain power for very strong memory, but not the declared
d>=0.30 domain. The experiment still assumes a known iid noise scale, so it is
not a deployable long-memory gate. Any follow-up based on a joint semiparametric
mean-memory model, increments, or a different statistic is a new route and must
receive a separately frozen design rather than a tuned continuation.

The estimated-mean artifacts are
`results/estimated_conditional_spectral_feasibility.json` and
`results/estimated_conditional_spectral_feasibility.md`.

A thirty-second probe opened an independently frozen increment-domain route.
Instead of estimating low-frequency power directly, it computed second
differences at lags 1, 4, 8, and 16 and summarized the log variance growth at
lags 4/8/16 relative to lag 1. Second differences exactly remove a linear
trend; uncertainty from the remaining smooth curvature was represented by the
same independent conditional-null construction, including mean refitting in
each of 128 Monte Carlo draws. The eligible lengths, memory orders, nonlinear
strengths, repeats, 99% quantile, and dual 75% criteria were unchanged.

Both lengths passed. At 256 points all 6 iid controls were accepted and all 12
d>=0.30 cases were detected. At 512 points 5/6 controls were accepted and all
12 strong-memory cases were detected. Every d=0.30 case had a positive margin;
the minimum margins were 0.078 at length 256 and 0.129 at length 512. Detection
also extended to 4/6 d=0.15 cases at 256 and 6/6 at 512, although those cases
were not part of the frozen power criterion. The minimal passing length was 256.

This establishes feasibility for a calibrated increment-domain refusal gate,
not a deployable universal test. Thresholds were conditioned on an independent
calibration trace from the same synthetic nonlinear-strength regime and still
used the known iid noise scale. Their large variation across strengths shows
that cross-regime transfer is the next falsification target. The route may
proceed only if a pooled or conservative calibration preserves control adequacy
and d>=0.30 power without access to the generating strength.

The increment-route artifacts are
`results/increment_variogram_feasibility.json` and
`results/increment_variogram_feasibility.md`.

A thirty-third probe tested whether the increment gate transfers without a
strength-matched calibration trace. For each evaluated nonlinear strength, the
conditional null pooled 128 draws from each of the other two strengths. The
project matrix increased to four repeats, producing 96 records. Every held-out
strength had to achieve at least 75% control adequacy and at least 75% detection
for d>=0.30; aggregate success could not compensate for a failed regime.

No length passed all held-out regimes. The middle strength 0.085 passed at both
lengths with 4/4 controls accepted and 8/8 strong-memory cases detected. At the
low strength 0.05, however, control adequacy was 0/4 at length 256 and 1/4 at
length 512, despite 8/8 strong-memory detections. At the high strength 0.20,
all controls were accepted but strong-memory detection was only 4/8 at both
lengths: every d=0.30 case was missed while every d=0.45 case was detected.

The failure exposes a structural transfer problem rather than Monte Carlo
noise. Null thresholds learned from stronger-curvature donors were too high for
moderate memory at strength 0.20, while thresholds learned without the weakest
curvature regime were too low for its d=0 controls. The present gate therefore
cannot use a pooled categorical calibration as a general replacement for
strength matching. A defensible new route must condition on an observable,
continuous curvature or trend-leakage diagnostic and test that normalization
without access to the synthetic strength label.

The transfer artifacts are
`results/increment_transfer_feasibility.json` and
`results/increment_transfer_feasibility.md`.

A thirty-fourth probe replaced the categorical strength label with a continuous,
observable curvature proxy. Five calibration strengths (0.035, 0.065, 0.11,
0.16, and 0.24), all disjoint from the three project strengths, formed a frozen
calibration bank. At each eligible length an affine map converted the proxy into
an increment-null threshold. Project strength labels were not passed to the
gate, and a project record outside the calibration proxy range would have forced
refusal. The full matrix retained four repeats and therefore contained 96
records.

The conditional route passed at both lengths. All 24 iid controls were accepted
and all 48 d>=0.30 records were detected; every one of the six length-strength
cells achieved 4/4 control adequacy and 8/8 strong-memory detection. No project
record fell outside the proxy range. The affine calibration fit was strong at
both lengths (R^2=0.9988 at 256 and R^2=0.9962 at 512), and every d=0.30 margin
was positive. The minimal passing prefix length remained 256.

This result repairs the categorical transfer failure within the declared
synthetic family, but it is not yet a general mechanism-identification result.
The calibration bank uses the same nonlinear-feedback generator family, the
noise scale remains known, and the curvature proxy is computed from the
evaluated observation. The next falsification gate should obtain the nuisance
proxy from an independent trace or calibration segment and then test transfer to
a different trend generator. Failure there would restrict the contribution to a
same-family normalization result rather than a deployable refusal protocol.

The observable-normalization artifacts are
`results/curvature_normalized_increment_feasibility.json` and
`results/curvature_normalized_increment_feasibility.md`.

A thirty-fifth probe applied the same frozen scalar normalization outside its
training generator and removed target contamination from the proxy. The affine
proxy-to-threshold map was still calibrated only on nonlinear-feedback controls.
Project trends instead came from rate-drift dynamics at strengths 1.5, 3.0, and
8.0 and stretched-exponential decays at beta 0.69, 0.70, and 0.71. For every
length, mechanism, strength, and repeat, a role-separated iid control supplied
the curvature proxy; a different seed generated each evaluated project trace.
The resulting matrix contained 192 records. External strengths were frozen from
d=0 proxy-scope checks before any memory-detection outcome was inspected.

No length passed. Strong-memory power remained perfect: all 48 d>=0.30 records
were detected at each length. Control calibration collapsed under generator
shift, however. At length 256 only 6/24 controls were accepted, and at length
512 only 10/24 were accepted. Rate drift at strength 1.5 passed both lengths and
strength 3.0 passed only length 512. Strength 8.0 failed both lengths, and every
stretched-exponential cell failed with 0/4 controls accepted. One of four
independent proxy repeats for beta=0.69 at length 512 was also outside the
calibration range.

The diagnostic clean-trend statistic confirms that the external trends have
multiscale second-difference geometry not represented by a scalar curvature
level, although that diagnostic was never used by the decision rule. The scalar
normalization route is therefore closed for general transfer. Its previous pass
remains valid only within the nonlinear-feedback generator family. A new route,
if pursued, must use a predeclared multiscale nuisance signature and calibrate it
across generator families; it cannot be presented as a tuned continuation of
the scalar proxy.

The cross-generator artifacts are
`results/independent_proxy_cross_generator_feasibility.json` and
`results/independent_proxy_cross_generator_feasibility.md`.

The high-dimensional shared-spectrum route was subsequently falsified under
blockwise spectral drift and correlated noise. Independent subgroup dispersion
was rejected because it falsely refused exact sharing. A nested global-versus-
grouped audit retained all exact/mild cases and refused all severe cases in the
initial 18-record matrix. A 30-record boundary map then localized the operational
transition to the interval between log drifts 0.05 and 0.075, but did not pass
its original 0.075--0.125 bracketing criterion. The observed decision boundary
is primarily controlled by the frozen held-out error limit; it must not be
reported as universally consistent heterogeneity identification.

The sharing-audit artifacts are
`results/approximate_sharing_refusal_boundary.json`,
`results/nested_group_sharing_gate.json`, and
`results/nested_group_boundary_map.json`, with matching Markdown summaries.

A subsequent 60-record dense map used new seeds and resolved the pooled 50%
refusal crossing to log drift 0.06875--0.075 (descriptive interpolation
0.071875). Refusal remained monotone after pooling, but the correlation-regime
difference reached 0.50 near the boundary. This is therefore an operational,
generator-specific error-gate boundary rather than a universal heterogeneity
threshold. See `results/dense_nested_group_boundary.json` and its Markdown
summary.

A subsequent correlation-aware calibration used an observable second-difference
proxy, 20 exact-sharing calibration records, and 32 independent project records.
It removed the boundary refusal-rate gap between independent and correlated
noise (0.25 to 0.00) and retained all exact-sharing controls, but it refused all
four mild-drift records under independent noise and placed three project proxies
outside its frozen calibration range. The route is therefore rejected. The
negative result shows that noise-only calibration cannot also supply the model-
approximation allowance required by an approximate-sharing contract. Future work
must separate declared model tolerance from observable noise tolerance. See
`results/noise_aware_sharing_gate.json` and its Markdown summary.

The follow-up experiment implemented that separation. Twenty-five exact-sharing
records calibrated observable-noise error, ten independent records at the
declared acceptable drift 0.05 calibrated a model allowance, and 32 new records
tested the frozen sum of both budgets. The decomposed gate retained all 16 exact
or mild-sharing records, refused all eight severe-drift records, produced equal
1/4 refusal rates at the two drift-0.075 boundary cells, and kept every project
proxy in scope. This repairs the preceding single-threshold failure within the
declared synthetic family. Transfer across channel count, noise scale, and
heterogeneity construction remains required before claiming a reusable
approximate-sharing contract. See
`results/decomposed_tolerance_sharing_gate.json` and its Markdown summary.

That transfer test has now been completed without recalibration. The frozen
Stage 45 rule was applied to 72 records spanning 32/128 channels, two noise
scales, two heterogeneous-spectrum constructions, three drifts, and three new
repeats per cell. All severe-drift cells were refused and all proxies stayed in
scope, but the exact-sharing 128-channel, high-noise, curved-construction cell
was falsely refused in two of three repeats. The frozen route therefore fails
its transfer criterion. The failure is attributable to shared validation error
exceeding the fixed decomposed budget while grouped-model BIC support remains
negative; it is not evidence of a heterogeneous mechanism. This bounds the
Stage 45 result to its calibrated family and motivates a separately frozen
noise-scale and optimizer-adequacy audit rather than post-hoc threshold
inflation. See `results/decomposed_tolerance_transfer.json` and its Markdown
summary.

Stage 47 then retained the Stage 45 budget as a floor and independently
calibrated a mixed time/channel difference noise-scale proxy plus a four-start
fitting-adequacy rule. The fresh 72-record transfer matrix removed the previous
high-noise exact-sharing failure: all three 128-channel, high-noise, curved
exact controls were accepted, and every diagnostic stayed in scope. The full
route still failed because the proposed adequacy statistic used absolute
train-RMSE relative to noise. It labelled consistent low-noise model-mismatch
fits as optimization-indeterminate, giving 0/3 explicit refusals in every
low-noise severe cell and 2/3 indeterminate outcomes in one mild cell. The
noise-scale proxy remains useful, but residual-based optimizer adequacy is
closed. Future adequacy checks must measure cross-start disagreement or
stationarity without using the residual magnitude that the mechanism gate is
supposed to test. See `results/noise_scale_optimizer_transfer.json` and its
Markdown summary.

Stage 48 replaced the invalid residual-based adequacy rule with a frozen
cross-start functional-consistency audit. Sixteen exact-sharing controls fixed
maximum objective-gap and prediction-gap limits, and 72 fresh records tested
the rule without recalibration. All exact and mild-drift cells met the
retention criterion and every diagnostic stayed in scope. The route still
failed because only three of eight severe-drift cells reached the required
explicit-refusal rate; the remaining difficult cells produced genuine
cross-start ambiguity, particularly for curved spectral drift. Cross-start
agreement is retained as a valid optimizer-reproducibility diagnostic, but an
indeterminate fit cannot be relabelled as evidence against a mechanism. See
`results/cross_start_consistency_transfer.json` and its Markdown summary.

Stage 49 increased only the symmetric deterministic-refinement budget while
keeping every Stage 48 calibration and decision threshold frozen. Across 72
fresh records, all eight severe-drift cells were explicitly refused in every
repeat and the severe indeterminate fraction fell from 0.542 to zero. The route
still failed: four low-noise mild-drift cells exceeded the permitted adverse
rate, including complete refusal of both curved cells. More refinement therefore
changes the effective mechanism boundary instead of monotonically improving
the decision. The optimizer and its budget are now treated as declared contract
metadata, and the tri-state outcome is retained. See
`results/extended_refinement_transfer.json` and its Markdown summary.

Stage 50 then evaluated optimizer-budget sensitivity on paired data instead of
comparing independent experiment rounds. Twenty-four datasets were each run at
L-BFGS budgets 80, 160, and 240 with identical initializations and frozen
calibrations. Only 50.0% retained the same tri-state decision, while 45.8%
showed a direct `RETAIN`--`REFUSE` reversal. All diagnostics remained in scope.
The reversals concentrated in the low-noise cells; most high-noise cells were
stable. This falsifies the assumption that the refinement budget is merely a
computational detail. Budget and stopping policy are now required contract
metadata, and budget-sensitive observations must be exposed rather than
resolved post hoc. See `results/optimizer_budget_stability.json` and its
Markdown summary.

Stage 51 tested a preregistered cross-budget consensus rule on 72 independent
datasets and 216 total fits. Conflicting `RETAIN` and `REFUSE` votes forced an
explicit budget-sensitive abstention; otherwise two matching votes were needed
for a determinate result. The rule refused 23/24 severe-drift cases and exposed
seven budget-sensitive cases, but it falsely refused 4/24 exact controls and
5/24 declared mild-drift cases. All nine errors occurred at the higher noise
scale and were caused by shared validation error exceeding the frozen tolerance,
despite negative grouped-model BIC support. Consensus therefore detects budget
conflicts but cannot repair common-mode validation-budget error. Voting-only
repair is closed; any further route requires disjoint high-noise calibration
and a fresh evaluation split. See `results/budget_consensus_abstention.json`
and its Markdown summary.

Stage 52 performed that disjoint calibration without using Stage 51 failures
or severe-drift examples. Forty-eight high-noise exact/mild calibration fits
supported no increase over the existing validation allowance: the frozen
one-sided multiplier was 1.0 (maximum calibration ratio 0.951). On a separate
24-dataset, 72-fit validation matrix, exact/mild false refusal was 0.0625,
slightly above the preregistered 0.05 limit. Exact/mild retention was 0.875,
severe-drift refusal was 1.0, budget sensitivity was 0.0417, and all
diagnostics remained in scope. The route therefore fails rather than receiving
a post-hoc tolerance increase. See
`results/high_noise_conditional_calibration.json` and its Markdown summary.

Stage 53 froze the Stage 52 multiplier and transferred it to 48/192 channels,
two new core noise levels, a rotated heterogeneity construction, and a declared
out-of-range stress noise level. Across 48 core paired datasets, false refusal
was 0.0625 and determinate retention was 0.78125, narrowly missing the frozen
0.05 and 0.80 criteria. Severe-drift refusal remained 1.0 and budget
sensitivity was 0.1042. At the 0.002 stress noise level, false refusal rose to
0.375 while severe refusal remained 1.0. This confirms a bounded operating
region rather than transferable universal calibration. See
`results/conditional_contract_transfer.json` and its Markdown summary.

Stage 54 replaced idealized noise morphology with standardized residual
segments from three public elastomer relaxation traces while retaining known
semisynthetic mechanism labels. Conditional consensus achieved full coverage
and perfect severe-drift refusal, but its false-refusal fraction was 0.2917 and
selective accuracy was 0.8056. A BIC-only consensus control obtained 0.9444
coverage, 0.1667 false refusal, 0.9167 severe refusal, and 0.8824 selective
accuracy. The conditional route therefore fails and is empirically worse than
the simpler evidence control on this audit. A separate eight-curve blind audit
returned `INDETERMINATE`; it has no truth label and is not counted as mechanism
recovery. See `results/real_background_mechanism_audit.json` and its Markdown
summary.

Taken together, Stages 52--54 close conditional validation-tolerance inflation
as a general repair for the budget-consensus mechanism gate. Validation error,
optimizer reproducibility, and nested-model evidence must remain separate
contract dimensions. The negative result prevents a high-noise calibration
that looks adequate on Gaussian controls from being promoted to a real-
background decision rule.

Stage 55 implemented that separation as a frozen three-axis hierarchy without
refitting any model. Numerical eligibility required in-scope diagnostics and
at least two optimizer budgets with two adequate starts; predictive validation
and structural BIC evidence then had to agree before a binary decision was
issued. On the Gaussian core matrix, the hierarchy achieved 0.750 coverage,
1.000 selective accuracy, zero false refusal, and 0.750 severe-drift refusal,
passing every frozen criterion. On real residual backgrounds, false refusal
fell from 0.2917 to 0.0417 and selective accuracy rose from 0.8056 to 0.9565,
while coverage remained 0.6389. However, severe-drift refusal fell to 0.5833,
below the frozen 0.75 requirement. The symmetric agreement rule is therefore
too conservative for rejection even though it repairs most false refusal.
The next architecture must use asymmetric evidence: mechanism retention should
require cross-axis agreement, while refusal needs an independently calibrated
strong-evidence route rather than another scalar tolerance. See
`results/multiaxis_hierarchical_contract.json` and its Markdown summary.

Stage 56 replaced the symmetric binary rule with a preregistered asymmetric
hierarchy. The strong-validation threshold was calibrated only on the Stage 53
Gaussian core records and then frozen before the Stage 54 real-residual labels
were evaluated. Retention still required validation/BIC agreement. Refusal was
allowed by BIC consensus or by repeated validation exceedance above 1.147753
with at least one structural refusal vote. The Gaussian development matrix had
0.8333 coverage, 1.000 selective accuracy, zero false refusal, and 1.000 severe
refusal. On the locked real-background matrix, coverage was 0.8333, selective
accuracy 0.9000, and severe refusal 1.000, but false refusal was 3/24 = 0.125,
narrowly exceeding the frozen 0.10 limit. All three errors were declared mild
drift cases from Dragon Skin or Mold Star residual backgrounds. The route is
therefore not repaired by moving to asymmetric logic alone. Further progress
requires new disjoint real-morphology calibration data or a representation of
background uncertainty; the Stage 54 labels must not be used to retune the
threshold. See `results/asymmetric_evidence_hierarchy.json` and its Markdown
summary.

Stage 57 supplied that missing calibration domain without reopening the Stage
54 labels. Seven additional public elastomer residual morphologies calibrated
separate repeated-validation and repeated-structural evidence envelopes. The
three Stage 54 truth-labelled background files were excluded from calibration,
and their 36 paired records were evaluated only after both thresholds and the
asymmetric rule were frozen. The calibration matrix contained 42 paired cases
and achieved 0.6905 coverage, 1.000 selective accuracy, zero false refusal, and
0.7857 severe-drift refusal.

On the locked Stage 54 matrix, the morphology-calibrated hierarchy achieved
0.750 coverage, 0.9630 selective accuracy, 0.0417 false refusal, and 0.9167
severe-drift refusal, passing all four preregistered criteria. The sole false
refusal was a Mold Star 30 rotated mild-drift case whose repeated structural
score exceeded the independently calibrated morphology envelope. The result
supports a scoped asymmetric reliability contract: real-morphology calibration
can control false refusal without removing most severe-drift detection, but a
quarter of cases still require an explicit `INDETERMINATE` outcome. See
`results/morphology_calibrated_asymmetric_hierarchy.json` and its Markdown
summary.

Stages 58--61 complete the current confirmatory branch. The Stage 57 contract
transferred without retuning to three independent physical data sources
(coverage 0.6667, selective accuracy 1.0000, false refusal 0, severe-drift
refusal 0.8571). Independent SciPy and finite-difference references confirmed
the forward values and parameter gradients, and a standalone implementation
replayed all 78 frozen decisions exactly. A 162-fit scope audit then classified
the normal external regime as supported while doubled noise, sparse training,
and a shortened observation window were scope-limited. The frozen evidence and
claim boundary are recorded in `STAGE61_FREEZE_REPORT.md` and
`results/stage61_evidence_manifest.json`.

## Stage 63: cross-domain identification and refusal

The public-data branch now includes UCI gas-sensor recovery as a second
independent task. It adds same-split nonlinear, fixed-grid NNLS, and Prony
baselines, experiment-cluster uncertainty, threshold and multi-start audits,
and a controlled explanation of the nonmonotone PVA boundary. The gas task
improves prediction but returns `INDETERMINATE` because fitted rates coalesce.
This directly tests the central claim that predictive fit alone is insufficient
for mechanism-level memory-rank identification.

The stable software surface is `fit/evaluate/decide/report`, documented in
`API_CONTRACT.md`. Evidence, limitations, and verification are summarized in
`STAGE63_EVIDENCE_REPORT.md`.

## Stage 67: finite-budget spectral resolution

A multichannel block matrix-pencil comparator now maps when a two-rate
interpretation is supportable under finite horizon, sampling, noise, and rate
separation. Across 54 design cells and 1,080 trials, BIC selected rank two 157
times, including 60 inaccurate rate recoveries. Requiring local information,
rate separation, and agreement across three Hankel aspect ratios retained 74
supports with no inaccurate recovery in the declared atlas. The result defines
a conditional resolution region rather than a universal threshold. See
`STAGE67_FINITE_BUDGET_RESOLUTION_REPORT.md` and
`results/finite_budget_resolution_atlas.json`.

## Stage 68: common-budget order detection

A common-budget comparison now evaluates AIC, AICc, BIC, strong BIC, cross-
pencil stability, and the complete selective detector on 864 rank-one/rank-two
trials under white and AR(1) noise.  The frozen selective route eliminates
observed false order elevation but fails to improve the risk--coverage tradeoff:
coverage is 0.8356, selective accuracy 0.5720, and rank-two detection 0.2188.
The route is closed as a positive result.  A rank-one mechanism claim requires
a separate design-power certificate; see
`STAGE68_COMMON_BUDGET_ORDER_DETECTION_REPORT.md`.

## Stage 69: power-certified rank-one decisions

Stage 69 used disjoint calibration and evaluation seeds to require an
observation-design power certificate before reporting rank one. Eight of 24
designs qualified. On 1,152 confirmatory trials, the power-certified detector
had 0.2943 coverage, 0.8614 selective accuracy, 0.1386 selective risk, zero
false order elevation, and 0.0612 false order reduction. It passed the frozen
risk--coverage checks separately under white and AR(1) noise, while explicitly
abstaining on 70.57% of cases. The result is relative to a declared 0.32 rate
gap and is not unconditional order-classification superiority. See
`STAGE69_POWER_CERTIFIED_ORDER_DETECTION_REPORT.md` and
`results/power_certified_order_detection.json`.

## Stage 70: risk--coverage sensitivity

The frozen Stage 69 evaluation set now supplies a six-point power-confidence
sensitivity curve. The 0.70 operating point is on the non-dominated frontier,
and thresholds from 0.70 through 0.90 give the same 0.2943 coverage and 0.1386
selective risk. Relaxing the lower bound to 0.30 raises coverage to 0.3472 but
also raises risk to 0.1900; removing the power condition returns to the Stage
68 high-risk regime. The audit supports a stable reliability--coverage tradeoff
but is not post hoc threshold selection. See
`STAGE70_POWER_CERTIFICATE_RISK_COVERAGE_REPORT.md`.

## Stage 71: same-budget order-selection baselines

Four explicit order comparators now join matrix-pencil AICc on all 1,152 frozen
evaluation records: block-Hankel AIC/MDL and shared-Prony AICc/BIC. Block-Hankel
AIC has the best full-coverage accuracy (0.6658) but a 0.6771 false-elevation
rate; matrix-pencil AICc has zero false elevation but 0.5339 false reduction.
The power-certified method occupies a different regime with 0.2943 coverage,
0.1386 selective risk, zero false elevation, and 0.0612 false reduction. The
result supports selective reliability, not unconditional accuracy superiority.
See `STAGE71_COMMON_BUDGET_SUBSPACE_BASELINES_REPORT.md`.

## Stage 72: retrospective public-data transfer

The frozen Stage 69 rule is now scope-tested on 56 groups from four public
tasks after a fixed six-curve, 24-point, dimensionless adapter. Only one PVA
group enters the calibrated scope and supplies evidence against rank-one
sufficiency. Fifty gas groups are refused because ten observed samples cannot
be interpolated into a 24-sample evidence budget; two KupferDigital and three
hydraulic groups are refused as outside the calibrated monotone-decay
morphology. The 1/56 eligibility rate rules out a broad transfer claim. See
`STAGE72_EXTERNAL_POWER_CERTIFICATE_TRANSFER_REPORT.md`.

## Stage 73: PVA group-composition sensitivity

All 84 six-of-nine subsets of the public PVA curves were replayed through the
unchanged Stage 72 adapter and frozen Stage 69 certificate. All 84 entered the
same calibrated noise cell, passed all five rank-two checks, and supplied
evidence against rank-one sufficiency; the criterion improvement ranged from
491.2603 to 596.2218. This removes dependence on the original first-six choice,
but the overlapping subsets remain one retrospective dataset rather than 84
independent confirmations. See
`STAGE73_PVA_GROUP_COMPOSITION_SENSITIVITY_REPORT.md`.

## Stage 75: preregistered cable-ageing transfer

A pre-outcome P5 contract freezes six cable-ageing curves and the unchanged
Stage 69/72 adapter and thresholds. All six curves enter scope; criterion
improvement is 647.951, all five rank-two checks pass, and the rule supplies
evidence against rank-one sufficiency. Prior use in P3 prevents describing the
result as investigator-blind prospective confirmation. See
`STAGE75_PREREGISTERED_CABLE_AGEING_TRANSFER_REPORT.md`.

## Stage 76: cable-window sensitivity

Twelve post-result start/end-window settings retain at least 2,177 raw points
per curve. All remain eligible and return the Stage 75 decision, with criterion
improvement from 399.989 to 714.271. The overlapping windows are sensitivity
checks rather than independent replications. See
`STAGE76_CABLE_WINDOW_SENSITIVITY_REPORT.md`.
