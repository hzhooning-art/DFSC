# Current P5 Status

## Stage 62 update: independent public-data task

Stage 62 adds the first direct analysis of an independently published public
dataset. The frozen shared-rate protocol was applied to raw PVA gel polymer
electrolyte stress-relaxation curves (Zenodo 21333840; 3 specimens x 3 cycles).
The full 28 s/96-point task supports rank 3: held-specimen median NRMSE improves
from 0.01757 (rank 1) to 0.00932 (rank 3), with a paired bootstrap 95% interval
of [0.395, 0.798] for relative improvement over rank 1 and one-sided Wilcoxon
p=0.00195. The horizon-by-budget audit also contains nonmonotone indeterminate
cells, supporting the refusal mechanism rather than a claim that more samples
always reveal more memory modes. Formal statistics, workflow, and boundary
figures are frozen under `figures/`; the verified literature boundary is in
`RELATED_WORK_MATRIX.md`.

## Decision

**Continue the route.** The current evidence supports local synthetic feasibility of
minimal shared-memory-rank discovery. It does not yet support a paper-level claim.

## Evidence obtained

1. Rank-one and well-separated rank-two scalar systems can be recovered.
2. Nearly coincident poles produce insufficient evidence and extreme Jacobian
   conditioning instead of a confident higher-rank claim.
3. A first phase slice shows that longer horizons and larger pole separation improve
   identifiability.
4. Under a short observation horizon, twelve channels sharing pole locations recover
   moderate rank-two separation more reliably than one channel.
5. Positive rank-one and rank-two controls were accepted in all six trials, whereas
   signed and oscillatory memory kernels were refused in all six out-of-class trials.
   The refusal was supported by long-horizon error and strongly correlated residuals,
   not by access to the generating labels.
6. A coarse and refined near-boundary scan found nontrivial empirical detectability
   transitions. At the declared noise and horizon, signed-residue violations up to
   0.005 were accepted and violations from 0.0075 were refused; oscillation frequencies
   up to 0.03 were accepted and frequencies from 0.04 were refused. All transitions
   were monotone across three repeats per setting and were triggered by residual
   structure rather than by generator labels or raw error alone.
7. A 216-fit calibration screen varied three noise levels, three observation horizons,
   two violation levels, and zero controls. Both zero-control families had 0/36 false
   refusals (Wilson 95% upper bound 0.096). Aggregate refusal increased from 15/36 to
   22/36 for signed residues and from 11/36 to 22/36 for oscillatory kernels when moving
   from the previously accepted to the previously refused violation level. Low-noise,
   long-horizon cells refused the above-boundary violations in at least 3/4 trials.
8. A joint rank-selection/refusal experiment then fitted candidate ranks 1--3 rather
   than conditioning on rank one. At horizon 14 and noise standard deviation
   0.0006, both zero controls produced 0/10 false refusals and 0/10 false rank
   elevations. Signed-residue refusal increased from 4/10 below the transition to
   10/10 above it; oscillatory refusal increased from 1/10 to 10/10. Only one of 60
   cases selected rank two, an above-boundary signed-residue case that was still
   refused by the condition and residual gates. Thus, extra rank capacity did not
   systematically absorb out-of-class dynamics in this local experiment.
9. A 108-case joint information-gradient experiment repeated the rank-1--3 competition
   under low-, medium-, and high-information observation regimes. Both model-mismatch
   families had zero false refusals and zero false rank elevations on all zero controls.
   Above-boundary refusal increased monotonically from 0/6 at short-horizon/high-noise
   observations to 6/6 at both medium and high information. Below-boundary refusal
   remained lower at medium information (2/6 signed; 0/6 oscillatory) and increased
   when information was highest (6/6 signed; 3/6 oscillatory). Across all 108 cases,
   34 refusals came from residual correlation and one from an ill-conditioned rank-two
   fit; no out-of-class case was accepted after elevating its memory rank.
10. A 120-case one-factor experiment separated horizon and noise effects around the
    reference operating point. The noise sweep behaved monotonically for both mismatch
    families: above-boundary refusal fell from 6/6 at noise 0.0004 to 2/6 (signed) and
    1/6 (oscillatory) at noise 0.0009, while all zero controls remained accepted at
    rank one. The horizon hypothesis was only partly supported. Signed-residue refusal
    rose from 3/6 at H=10 to 6/6 at H=14 and H=18, but oscillatory refusal fell from
    6/6 at H=10 and H=14 to 1/6 at H=18. A deterministic mismatch-to-noise audit
    explains the reversal: the oscillatory mismatch in the late validation window
    decreased from 5.82 at H=10 to 4.38 at H=14 and 3.85 at H=18 as the transient
    decayed. The prespecified route check therefore failed, disproving the simplistic
    claim that a longer terminal horizon always improves detection.
11. A calibrated multi-window experiment addressed that failure without tuning on the
    failing evaluation cases. Twelve independent zero-control calibration fits supplied
    36 early/middle/late residual statistics; a Bonferroni empirical quantile fixed the
    joint lag-1 threshold at 0.20595. On a disjoint 32-case evaluation split at H=18,
    both zero-control families had 0/8 false refusals. Oscillatory above-boundary
    refusal improved from 0/8 under the old terminal-window gate to 8/8 under the
    calibrated multi-window gate, while signed-residue refusal improved from 6/8 to
    8/8. No case selected an elevated rank. Six oscillatory trials were strongest in
    the late window and two in the early window; in one trial the late statistic was
    only 0.10 while the early statistic was 0.44. The prespecified route checks passed,
    showing that time-local coverage repairs this specific transient-blindness failure.
12. A larger external-calibration probe then stress-tested threshold portability. Its
    first version used 120 one-start zero-control fits (360 window statistics), but two
    obvious optimization failures had window RMSE values of 0.25--0.30 and contaminated
    all three tail statistics. The resulting threshold rose to 0.9745, its bootstrap
    interval spanned 0.2057--0.9789, and every stress case was accepted. This failed
    route demonstrates that numerical fit quality must be gated before a statistical
    null distribution is estimated; simply increasing the calibration sample does not
    make a contaminated calibration valid.
13. A fresh-seed quality-gated replication used two fixed starts for every one of 120
    calibration fits. All 120 passed the numerical gate. The frozen threshold was
    0.20514 with a much narrower bootstrap 95% interval of 0.18731--0.21744. On jittered
    sampling, slow-decay oscillations and early/middle shifted transients were refused
    in 6/6 trials each, and no case elevated its selected rank. The route nevertheless
    failed for two substantive reasons: the oscillatory zero control had 2/12 false
    refusals, and the fast-decay oscillation was refused in only 2/6 trials. The false
    refusals came from early/middle windows despite accurate, well-conditioned fits,
    showing that regular-grid residual-correlation thresholds do not automatically
    transfer to jittered sampling. Fast transient decay also remains an information
    boundary rather than a threshold-tuning problem.
14. A sampling-aware calibration probe then calibrated and evaluated the residual
    diagnostic on independently jittered observation grids. All 120 calibration fits
    passed the same numerical quality gate. The frozen thresholds were 0.20669 for the
    simple index-lag statistic and 0.22105 for an actual-time-gap-weighted variant; their
    bootstrap 95% intervals were 0.19466--0.22658 and 0.19722--0.24225, respectively.
    On 48 fresh evaluation cases, both statistics refused all 6/6 slow-decay oscillatory
    violations and all 6/6 early and 6/6 middle shifted transients. They produced 0/12
    false refusals for the oscillatory zero control and 1/12 for the signed zero control,
    with no selected-rank elevation. The previously identified fast-decay boundary was
    refused in only 3/6 trials and was explicitly treated as a secondary boundary case,
    not as evidence of complete stress coverage. The two statistics made identical
    decisions in all 48 cases. The primary route therefore passed, but actual-time-gap
    weighting did not add decision value under this independent-jitter design; the
    simpler statistic with sampling-stratified calibration remains the preferred rule.
15. A frozen-rule sampling-process stress test extended the observation mechanism to
    30% random missingness, strongly clustered sampling, and a continuous gap spanning
    36%--54% of the horizon. The index-lag threshold remained fixed at 0.20669 from the
    preceding jittered calibration; no threshold was re-estimated. Across 84 fresh
    cases, all fits passed the numerical quality gate, all selected rank one, and the
    largest Jacobian condition estimate was about 3.33. Every primary violation was
    refused in 4/4 trials under each sampling process. Zero-control false refusals were
    1/6 signed and 1/6 oscillatory under random missingness, 1/6 signed and 0/6
    oscillatory under clustered sampling, and 0/6 for both controls under the long-gap
    design. The three processes passed the prespecified feasibility checks, but the
    zero-control Wilson intervals remain very wide. Fast-decay refusal remained only
    1/4, 2/4, and 1/4, respectively. One attempted run terminated before producing
    results because the first clustered design left too few independent training points;
    the corrected design added prespecified support outside diagnostic windows while
    retaining nonuniform density.
16. A larger null-law replication used 60 independent calibration fits and 40 disjoint
    evaluation fits for each sampling process. Thresholds were estimated from the 95th
    empirical quantile of independent per-fit three-window maxima, rather than treating
    the three windows as independent observations. All 300 rank-one fits passed the
    numerical quality gate. The random-missing, clustered, and long-gap thresholds were
    0.27013, 0.22364, and 0.23208; their bootstrap widths were 0.04106, 0.13958, and
    0.09989. Relative to the frozen jitter threshold, stratification reduced held-out
    false refusals from 6/40 to 1/40 under random missingness and from 5/40 to 1/40 under
    long gaps. The clustered route failed: false refusals decreased only from 5/40 to
    4/40, whose Wilson 95% upper bound was about 0.231 and exceeded the prespecified
    0.20 limit. Three of the four clustered refusals came from the oscillatory zero
    family. Their fits were accurate and well conditioned, so the failure reflects a
    sampling-geometry-dependent null tail rather than optimizer contamination. The
    overall route was therefore not passed.
17. A geometry-conditional clustered-null experiment then replaced the single fixed
    cluster design with a continuum of cluster locations, widths, and point counts.
    It used 180 independent zero-model fits for calibration and 120 disjoint fits for
    evaluation. A geometry risk score was defined before residual inspection as the
    equal-weighted standardized sum of maximum normalized time gap, histogram
    concentration, and inverse minimum diagnostic-window count; calibration-score
    tertiles fixed three bins without using residual labels. All 300 fits passed the
    numerical quality gate. The new global threshold was 0.22349, and the three bin
    thresholds were 0.22962, 0.23226, and 0.22005. The geometry score had only 0.115
    rank correlation with the null statistic. Legacy, newly calibrated global, and
    geometry-conditional rules each falsely refused the same 8/120 held-out controls;
    the Wilson 95% upper bound was about 0.126. Five false refusals occurred in the
    highest geometry-risk bin, whose conditional Wilson upper bound was about 0.255.
    Six of the eight failures came from the oscillatory zero family and seven were
    triggered by the late window, despite accurate, well-conditioned fits. The route
    therefore failed and rules out coarse geometry binning as the next repair.
18. An exact-design conditional parametric-bootstrap probe then preserved each outer
    case's actual observation times, training split, diagnostic windows, and noise
    level. Thirty independent zero-model cases were evaluated, with 19 newly simulated
    and fully refitted conditional copies per case. The finite-sample decision used
    `(1 + # bootstrap statistics >= observed statistic) / 20`, so refusal at level
    0.05 required the observed statistic to exceed every conditional copy. All 30 outer
    fits and all 570 bootstrap fits passed the numerical quality gate. The legacy and
    newly calibrated global clustered thresholds each falsely refused 2/30 controls,
    both oscillatory and late-window cases. Exact-design calibration refused 0/30,
    with a Wilson 95% upper bound of about 0.114. The conditional thresholds ranged
    from 0.15172 to 0.36430, compared with a median of 0.24350, demonstrating material
    design dependence rather than a uniform threshold increase. The two global-rule
    failures had conditional Monte Carlo p-values 0.15 and 0.30. The prespecified
    feasibility route passed, but 19 bootstrap copies provide only a coarse 0.05 p-value
    grid and the computational cost is substantial.
19. A frozen-rule power probe evaluated three primary mismatch families and one
    prespecified fast-decay boundary on fresh clustered designs. Each family contained
    six outer cases and each case used the unchanged 19-copy conditional rule. The
    exact-design method refused 5/6 slow-decay oscillations and 6/6 of both early- and
    middle-shifted transients, losing at most one detection relative to the global
    thresholds. It refused only 2/6 fast-decay boundary cases, compared with 3/6 for
    the global rules, preserving that case as an explicit low-information boundary.
    However, one slow-decay outer case produced only 18/19 valid bootstrap fits, so the
    route failed its requirement that all 456 conditional refits pass the numerical
    quality gate. The complete experiment required about 1019 seconds, with a median
    of roughly 42 seconds per outer decision. A deterministic replay localized the
    failure to one optimizer solution with validation RMSE 0.194 and condition estimate
    3.25. Four additional independent starts all recovered RMSE about 6.05e-4 and
    condition estimate 2.946. This diagnoses insufficient fixed-start robustness, but
    the post-hoc retries do not change the failed power-probe verdict.
20. A fresh-seed replication then froze the proposed numerical safeguard: every outer
    and bootstrap fit used two initial starts, and up to four additional starts were
    allowed only after failure of the existing RMSE or conditioning gate. The test
    included 20 zero controls and four repetitions of each of the three primary
    mismatch families, with 19 exact-design bootstrap copies per outer case. All 640
    fits passed the gate without invoking a retry. Exact-design inference produced
    0/20 false refusals, whose Wilson 95% upper bound is about 0.161, and refused 4/4
    cases in every primary family. All prespecified replication checks passed and the
    32 decisions took about 1373 seconds. This replicates the null-control and primary-
    power pattern and shows that the frozen safeguard is non-disruptive on valid fits.
    It does not yet establish fresh-sample recovery performance because no retry was
    triggered; that claim remains supported only by deterministic replay of the prior
    isolated optimizer failure.
21. A prespecified numerical-stress audit evaluated the same retry policy on 120 fresh,
    correctly specified rank-one zero controls. Observation designs remained strongly
    clustered, training support was reduced to 20 points, and each start used a fixed
    75-Adam/20-L-BFGS budget. All six starts were run for counterfactual audit, but the
    projected deployment policy charged starts 3--6 only when starts 1--2 failed the
    unchanged RMSE or conditioning gate. One oscillatory-zero case triggered the retry:
    its initial validation RMSE was 0.1865 and the retry reduced it to 6.45e-4. Final
    calibration failures were therefore 0/120 (Wilson 95% upper bound about 0.031), and
    projected extra-start overhead was 1.67%. In 43 other cases, exhaustive search found
    a nominally lower RMSE despite an initially valid fit, but the median exhaustive-to-
    initial RMSE ratio was 0.999984 and the maximum improvement was only about 0.0134%.
    The route still failed because only one initial failure was exposed, below the three
    required for a recovery-rate claim. Together with the earlier deterministic replay,
    the evidence supports retry as a low-cost rare-failure safeguard, not a calibrated
    guarantee of recovery probability.

## Important negative result

A rate ratio of 2 was not reliably resolved, even with twelve channels. More channels
increase likelihood evidence, but they do not remove the dynamical non-identifiability
of nearby time scales. This is a scientific boundary of the proposed problem, not a
software defect to conceal.

## Remaining blockers

- The evidence uses synthetic linear positive-real systems only.
- Repetition counts are too small for calibrated error rates.
- Near-boundary signed and oscillatory cases have been tested, and an independently
  jittered calibration/evaluation split now supports sampling-stratified thresholding.
  However, only one jitter process and amplitude have been studied; missing,
  clustered, and strongly gapped observations remain untested. Time-varying rates
  and cubic feedback now have a 12-instance pilot, but not a calibrated detection or
  refusal boundary.
- The broad calibration screen remains conditional on a rank-one fit. The new joint
  experiment removes that restriction at one horizon/noise operating point, but its
  ten repeats per setting leave wide Wilson intervals: even 0/10 events have a 95%
  upper bound of about 0.278. It therefore does not establish a publication-level,
  uniformly calibrated detection boundary.
- The information-gradient experiment changes horizon and noise together. It shows
  that detectability improves with aggregate information, but it does not identify the
  separate causal contribution of horizon and noise. With six trials per cell, a 0/6
  rate still has a Wilson 95% upper bound near 0.39.
- `ACCEPT_CONTRACT` currently means that the fitted positive-real model was not rejected
  by the implemented diagnostics. Especially in the low-information regime, it must not
  be interpreted as proof that the generating mechanism is positive-real or uniquely
  identified. A future public API should distinguish model-check passage from mechanism
  identification.
- The calibrated early/middle/late diagnostic repaired the observed terminal-window
  failure at H=18, but its threshold is based on only 36 calibration statistics and
  each held-out cell contains eight trials. The 0/8 false-refusal result therefore has
  a Wilson 95% upper bound near 0.324 and is feasibility evidence, not a calibrated
  familywise guarantee. Larger calibration and evaluation sets, additional horizons,
  and alternative transient shapes are still required.
- The 120-fit calibration stabilized only after a numerical quality gate and two fixed
  starts were placed before null estimation. Even then, a regular-grid threshold caused
  2/12 false refusals on an oscillatory zero control under jittered sampling. Thresholds
  therefore need sampling-design-aware strata or a residual statistic whose null law
  accounts for irregular time gaps.
- Sampling-aware recalibration removed the oscillatory-control false refusals on a fresh
  jittered split, but the signed control still had 1/12 false refusals. Its Wilson 95%
  interval is wide (about 0.015--0.354), so the result is feasibility evidence rather
  than a calibrated population guarantee. Actual-time-gap weighting produced no decision
  changes relative to the simpler index-lag statistic in 48 cases and is not retained as
  a claimed improvement.
- The frozen jitter-calibrated threshold passed feasibility checks under random missing,
  clustered, and long-gap sampling, but three of 36 zero-control trials were refused.
  A 1/6 rate has a Wilson 95% interval of roughly 0.030--0.564, so these small cells do
  not establish sampling-process-wide false-refusal control. In particular, one
  random-missing oscillatory control produced a residual statistic of 0.389 despite an
  accurate and well-conditioned fit, indicating a remaining null-tail shift.
- Sampling-stratified calibration substantially improved random-missing and long-gap
  error control on disjoint seeds, but the clustered stratum still refused 4/40 held-out
  zero controls. A single threshold indexed only by the label `clustered` is therefore
  too coarse: cluster locations, local point counts, and time-gap geometry can alter the
  residual-correlation null tail even when numerical fits remain accurate.
- Prespecified geometry binning did not repair the clustered null. Maximum normalized
  gap, coarse histogram concentration, and minimum window count were weak predictors
  of the residual-statistic tail, and all three compared thresholds made exactly the
  same 120 held-out decisions. Further post-hoc bin refinement would risk overfitting;
  the next diagnostic must condition on the actual observation-time design or replace
  the lag-index statistic.
- Exact-design conditional bootstrapping repaired the observed clustered false
  refusals in a 30-case pilot, but it required 570 additional two-start refits and has
  not yet been tested on out-of-class alternatives. The result establishes null-control
  feasibility, not useful power, broad calibration, or production-level cost.
- The first conditional-power probe retained strong primary mismatch detection, but
  failed because one of 456 bootstrap refits fell into a clearly poor optimizer basin.
  A frozen retry-on-quality-failure policy was non-disruptive on 640 fresh standard fits,
  and a separate 120-case constrained-budget stress test exposed and recovered one fresh
  catastrophic fit at only 1.67% projected start overhead. That single recovery, plus
  deterministic replay of the original failure, supports a rare-event safeguard but is
  insufficient to estimate recovery probability. Runtime remains about 43 seconds per
  exact-design decision, so conditional bootstrapping is an offline audit rather than an
  interactive default.
- A faster-decaying oscillatory violation was detected in only 2/6 jittered trials at
  the current signal and noise levels. This case should remain an explicit
  low-information refusal/indeterminacy region; lowering a global threshold would trade
  that miss rate for unacceptable zero-control false refusals.
- No theorem yet relates horizon, noise, channel diversity, and pole separation to
  recoverable memory rank.
- Scaling beyond twelve channels and comparison with Prony, subspace realization,
  and unconstrained neural kernel baselines remain open.
- No public scientific dataset has been evaluated.

## Next falsification tests

1. Scaling: test 16--256 channels and report runtime, memory, and recovery probability.
2. Baselines: the first regularized Prony and unconstrained-spline comparison is
   complete on matched synthetic systems. The attempted neural trajectory baseline
   failed its own optimization-quality gate and is retained only as a negative
   implementation result. A stronger neural baseline remains necessary before any
   broad comparative claim.
3. Contract breadth: a 12-instance pilot now covers time-varying rates and weak cubic
   feedback. A six-level, eight-repeat refinement brackets the observed majority
   nonlinear transition at 0.080--0.085. A paired three-noise-level experiment then
   found no silent relative-extrapolation failure under the normalized residual gate,
   whereas a fixed absolute gate had two. A follow-up heteroscedastic experiment used
   paired measurement differences to estimate the unknown training-noise scale and
   matched the oracle gate in 26/27 cases. Horizon, channel geometry, single-series
   noise estimation, correlated noise, and non-Gaussian noise remain necessary before
   a broad transfer claim.
4. Retry uncertainty: retain the frozen safeguard, but do not spend the next stage
   tuning stress severity merely to manufacture failures. Accumulate recovery events
   during later baseline and breadth experiments and report a recovery interval only
   when enough natural quality-gate failures have occurred.

## 22. Mechanism model versus trajectory baselines

The first baseline probe used identical sparse early-time observations for a
positive-real memory fit, a regularized damped-modal fit, and a fixed-capacity tanh
trajectory MLP. On separated rank-one, rank-two, and rank-three generators, the memory
model recovered the true rank in all 9/9 noisy instances and every selected fit passed
the quality gate. Its median case-wise extrapolation RMSE was 1.18e-4, 1.52e-4, and
6.62e-3. The modal baseline fit the observed interval near the noise scale but selected
three trajectory modes in every case and extrapolated at RMSE 0.190, 0.0434, and 0.0621.
This is a prediction comparison only: damped trajectory-mode rank is not memory-kernel
rank. The MLP comparison failed its own quality gate because training RMSE remained
above 2.5 noise standard deviations after Adam and L-BFGS; its poor extrapolation is
not used as affirmative evidence.

## 23. Optimizer-free unconstrained trajectory baseline

A cubic smoothing spline removed the neural-optimization confound. Its smoothing
budget was fixed as `n_train * noise_variance` per channel and did not use held-out
values. Median training and interpolation RMSE stayed at the noise scale, while
extrapolation RMSE increased to 0.565--1.439. The mechanism-to-spline median
extrapolation-error ratio was 2.60e-4, and all prespecified feasibility checks passed.
The interpretation is deliberately narrow: correct mechanism constraints improve
long-horizon prediction when the generator belongs to the declared positive-real
family. That matched-baseline experiment alone does not establish real-data benefit
or robustness to misspecification.

## 24. Controlled model misspecification

The next probe replaced exact model matching with two controlled departures from a
two-pole positive-real system. A time-varying family increased both pole rates
linearly over the horizon, while a weakly nonlinear family added cubic state damping.
Each family used control, mild, and strong levels with two fresh noisy repeats. Rate
drift at strengths 0.25 and 0.75 retained rank-two selections, well-conditioned fits,
and median extrapolation RMSE 2.82e-3 and 6.42e-3. Mild nonlinear feedback at strength
0.05 was absorbed as an effective rank-three fit with median extrapolation RMSE
2.29e-3. At strength 0.20, both nonlinear repeats selected rank two but had Jacobian
condition numbers near 2.24e15 and were refused by the unchanged quality gate. The
mechanism model remained more accurate than the fitted Prony and spline baselines in
all cells, so no observed failure was hidden by favorable baseline selection.

All prespecified feasibility checks passed: controls stayed below 5e-3 extrapolation
RMSE, mild cases were not more than 25% worse than the best trajectory baseline, every
strong case was either useful or refused, and there were no silent late-audit failures.
The result supports a useful distinction between mild misspecification that can be
absorbed by an effective higher rank and strong misspecification that destroys local
identifiability. It remains a 12-instance pilot. The refusal is retrospective because
the late audit requires post-training observations, and the transition strength has
not been statistically calibrated.

## 25. Nonlinear transition-boundary scan

A denser follow-up evaluated seven cubic-feedback strengths from 0.05 to 0.20 with
six independent noisy repeats at each strength, for 42 fits under the unchanged gate.
All fits at strengths 0.05 and 0.075 were accepted, while all fits at 0.10 and above
were refused. The observed fixed-design transition was therefore bracketed between
0.075 and 0.10. Rank three absorbed the mismatch in all fits through strength 0.10;
the rank-three rate then fell to 0.833, 0.333, 0.167, and zero at strengths 0.125,
0.15, 0.175, and 0.20.

The refusal mechanism changed across the scan. At strengths 0.10--0.15, all refusals
were caused by the frozen training-fit quality criterion despite moderate Jacobian
condition numbers. At 0.175, one of six fits was ill-conditioned, and at 0.20 four of
six were ill-conditioned. No fit exceeded the late-audit error limit, so every
observed refusal in this experiment was available from training-region fit quality or
conditioning before extrapolation. The refusal rate was monotone and there were no
silent failures.

These counts do not calibrate a population transition: 0/6 has a Wilson 95% upper
bound of 0.390, and 6/6 has a lower bound of 0.610. The bracket is also tied to the
current noise level, horizon, channel geometry, optimizer, and frozen threshold. It
should be reported as a protocol-specific boundary and not as a physical critical
nonlinearity.

## 26. Refined nonlinear boundary and normalized gate driver

The transition region was refined at strengths 0.075, 0.080, 0.085, 0.090, 0.095,
and 0.100 with eight fresh noisy repeats per strength. Refusal counts were 1/8, 3/8,
7/8, 7/8, 8/8, and 8/8, respectively. The majority-refusal bracket therefore
narrowed from 0.075--0.10 to 0.080--0.085. Wilson intervals remain broad: 3/8 gives
[0.137, 0.694] and 7/8 gives [0.529, 0.978].

All 48 fits selected effective rank three. The median selected training RMSE divided
by the known noise standard deviation increased from 3.56 at strength 0.075 to 5.21
at strength 0.10. Every one of the 34 refusals was caused by crossing the frozen
4-sigma training-fit threshold; none required a conditioning rejection or the
retrospective late-error audit. Across the six group medians, the exploratory
Spearman association between strength and normalized training RMSE was rho=0.943
(nominal p=0.0048). With only six ordered groups this is descriptive, not a calibrated
inferential result.

This experiment clarifies that the current transition is an operational protocol
boundary, not a physical phase transition. Its next falsification test is whether a
noise-normalized residual criterion transfers when the noise level, observation
horizon, or channel geometry changes.

## 27. Paired cross-noise transfer of the normalized gate

The next probe evaluated noise standard deviations 4e-4, 8e-4, and 1.6e-3 at
nonlinear strengths 0.075, 0.085, 0.125, and 0.20 with four repeats per cell. For a
given strength and repeat, all noise levels used the same clean trajectory,
standard-normal perturbation, and training indices. This paired construction
separates a change in noise scale from a change in the underlying random draw.

The noise-normalized training gate, selected RMSE/noise <= 4, accepted 15 of 48 fits.
None exceeded the prespecified operational extrapolation target of 10 noise standard
deviations; the maximum accepted ratio was 6.02 and the median was 3.16. The fixed
absolute gate, selected RMSE <= 3.2e-3, accepted 11 fits, had two accepted cases above
the relative extrapolation target, and reached a maximum accepted ratio of 12.11.
The gates disagreed on 12 cases. The fixed gate produced 22 refusals without a
corresponding relative extrapolation failure, compared with 16 for the normalized
gate.

These results support the narrower claim that a declared-noise normalization gives
the residual gate a more stable risk meaning across the three paired noise regimes
tested here. They do not establish a universal 4-sigma cutoff or a probabilistic
guarantee. In particular, the normalized gate remains conservative in 16 cases, the
relative extrapolation target is an audit convention rather than a theorem, and the
experiment assumes known homoscedastic Gaussian noise. The next transfer tests should
vary observation horizon and channel geometry, then replace the known noise scale
with an estimator under heteroscedastic or correlated noise.

## 28. Replicate-estimated noise gate under heteroscedasticity

The next experiment replaced the known homoscedastic scale with an estimate from two
independent measurements at each observation point. Noise amplitude increased
linearly over time and was normalized to a declared full-grid RMS. The two replicates
were averaged for fitting. Their training-region difference estimated the effective
noise RMS of that average, so the estimator did not reuse the mechanism-model residual
whose misspecification the gate is intended to detect. Three base noise levels, three
nonlinear strengths, and three repeats produced 27 fits.

The maximum relative error of the replicate noise estimate was 5.42%. The estimated
gate agreed with the oracle-noise gate in 26/27 cases, or 96.3%. The sole disagreement
was a conservative borderline refusal: the oracle-normalized training residual was
3.81, while the estimated ratio was 4.03. The estimated gate accepted four fits, had
no silent relative-extrapolation failures, and its maximum accepted extrapolation
error was 3.26 oracle effective noise standard deviations. The oracle gate accepted
five fits and also had no silent failure.

The fixed absolute gate accepted eleven fits and had one silent relative-
extrapolation failure at 10.58 oracle effective noise standard deviations. This
supports the narrower claim that repeated measurements can recover a useful
normalization scale under the controlled heteroscedastic Gaussian design. The result
also exposes conservatism: the estimated gate made fourteen refusals without a
corresponding 10-sigma extrapolation failure. It therefore supports risk control, not
optimal acceptance efficiency or a universal 4-sigma threshold. A harder next test
must estimate noise from a single series, preferably under temporal correlation or
heavy-tailed contamination, without allowing model misspecification to inflate the
noise estimate and conceal itself.

## 29. Single-series robust noise gate under sparse contamination

The repeated-measurement assumption was removed in a controlled single-trajectory
experiment. A dense calibration prefix was observed once, while the model fit used
48 sparse observations as before. A robust multiscale estimator combined first- and
second-lag second differences; the corresponding naive comparator used the ordinary
standard deviation of first-lag second differences. The design crossed Gaussian and
2% sparse gross-contamination noise, three core noise levels, three nonlinear
strengths, and two repeats, yielding 36 fits in 18 complete cells.

The robust estimator recovered the core noise with median relative error 0.109 and
90th-percentile error 0.266. Under the contaminated design its median scale ratio was
1.216, whereas the naive estimator's ratio was 2.188. The resulting robust gate
agreed with the oracle-noise gate in 32/36 cases. It accepted 18 cases, had zero
silent relative-extrapolation failures, and limited the largest accepted
extrapolation RMSE to 8.73 oracle core-noise standard deviations. The naive gate
accepted 24 cases but admitted four failures above the 10-sigma operational audit
limit, including a maximum of 14.18. The fixed absolute gate admitted one such
failure.

The safety result comes with a measurable efficiency cost. The robust gate made
seven refusals below the late-error audit limit. Its four oracle disagreements were
acceptances where the oracle residual gate refused but the observed extrapolation
error remained below 10 sigma; thus they reduce decision agreement without creating
an observed silent failure. These 36 controlled fits establish route feasibility,
not a calibrated population guarantee.

The validated claim is now narrower and more useful: a single dense calibration
prefix can supply a robust normalization scale under smooth signals, iid core noise,
and sparse gross contamination, and this scale preserves the tested operational risk
meaning substantially better than a naive variance estimate. The next falsification
target is temporal correlation. A correlation-aware estimator must either recover a
usable innovation scale or explicitly refuse to normalize when correlation is not
identifiable; silently treating correlated residuals as iid would invalidate the
present contract.

## 30. Correlation-aware refusal under stationary AR(1) noise

The temporal-correlation falsification test crossed four stationary AR(1)
coefficients (0, 0.3, 0.6, and 0.85), three nonlinear strengths, and two repeats,
for 24 complete fits. A conditional AR(1) profile estimated a smooth Chebyshev
mean and innovation process jointly. Degrees 8, 9, and 10 formed a prespecified
sensitivity envelope; excessive variation across that envelope, a boundary
estimate, or residual innovation correlation made the case unidentifiable and
therefore forced refusal.

Twenty cases were identifiable and four were refused as unidentifiable. Among
the identifiable cases, median marginal-scale relative error was 0.0925, its
90th percentile was 0.3160, and median absolute correlation error was 0.1075.
The correlation-aware gate agreed with the oracle-noise gate in 17/20 cases. It
accepted 13 fits with no silent relative-extrapolation failure and a maximum
accepted error of 7.38 oracle noise standard deviations. The iid gate also made
no silent failure, but accepted only seven fits and made eleven conservative
refusals below the 10-sigma audit limit; the aware gate made five.

All frozen route checks passed without post-result threshold changes. The result
therefore supports the practical value of estimating correlation before applying
a noise-normalized residual gate: in this controlled design it retained observed
safety while reducing avoidable refusals. It does not prove optimality or a
probabilistic guarantee. The estimator assumes a stationary AR(1) class, a dense
calibration prefix, smooth channel means, and no cross-channel noise coupling.
Performance at rho=0.85 remained biased low, so long-memory, nonstationary,
irregularly sampled, and multivariate correlated noise require separate tests.

## 31. Long-memory mismatch route rejected at frozen prescreen

The next route asked whether an AR(1)-normalized gate could detect that its own
noise model was inadequate under finite-sample long memory. A finite-burn-in
ARFIMA(0,d,0) generator was verified for marginal scale and long-lag dependence.
The prescreen then crossed d in {0, 0.15, 0.30, 0.45}, three nonlinear strengths,
and two repeats. Conditional AR(1) residuals were audited with a pooled eight-lag
Ljung-Box statistic at alpha=0.01.

The frozen go/no-go rule required at least 75% adequacy among six d=0 controls
and at least 75% mismatch detection among twelve d>=0.30 cases. The observed
rates were 2/6 (0.333) and 7/12 (0.583), respectively. Both checks failed despite
all 24 cells completing. The code therefore skipped the expensive mechanism-fit
matrix instead of tuning the significance threshold after seeing the data.

This is a negative route result, not a failure of the broader refusal principle.
It shows that short-prefix polynomial detrending plus a pooled time-domain
portmanteau statistic confounds deterministic curvature with serial dependence
and still misses some strong finite-sample ARFIMA realizations. The defensible
next route is an independently specified frequency-domain diagnostic, such as a
local Whittle or low-frequency slope estimator with Monte Carlo-calibrated
acceptance regions. That route must receive a new frozen design; the present
thresholds must not be recycled or retrofitted.

## 32. Unconditional frequency-domain calibration also rejected

The independently frozen follow-up evaluated prefix lengths 78, 256, and 512.
For each length, a local Whittle statistic was aggregated across Chebyshev
detrending degrees 8, 9, and 10 and three low-frequency bandwidths. A 99% null
threshold was calibrated with 128 iid Gaussian draws before evaluating four
ARFIMA memory orders, three nonlinear strengths, and two repeats. The resulting
matrix contained all 72 expected project records.

The route required both 75% iid-control adequacy and 75% detection for d>=0.30
at a given prefix length. Length 78 achieved 6/6 control adequacy but detected
only 2/12 strong-memory cases. Length 256 detected 12/12 strong-memory cases but
accepted only 1/6 controls. Length 512 also detected 12/12 strong-memory cases
and accepted none of its six controls. Consequently no prefix length qualified
for a later mechanism-fit experiment.

The failure is diagnostic rather than ambiguous. The constrained local Whittle
estimate concentrated the pure-iid Monte Carlo null at its lower boundary, so
all three calibrated thresholds were zero. At 256 and 512 samples, deterministic
curvature remaining after the fixed polynomial detrending envelope produced
positive estimates in d=0 project traces. At 78 samples, the same statistic had
insufficient power. No threshold, bandwidth, detrending degree, or acceptance
rate was changed after observing these outcomes.

The supported conclusion is therefore limited: unconditional iid calibration
cannot validate this frequency-domain refusal gate on deterministic mechanism
traces. A future conditional-null route would need to propagate uncertainty from
signal estimation into the spectral statistic and must be treated as a new,
separately frozen experiment. The current frequency route is closed.

## 33. Oracle conditional null establishes a frequency-domain upper bound

The next experiment changed only the null construction. For every prefix length
and nonlinear strength, 128 iid noise draws were added to the known clean
trajectory and passed through the unchanged detrending and local Whittle
pipeline. The project evaluation retained all 72 ARFIMA traces, the 99% null
quantile, and the two 75% acceptance thresholds. This oracle construction asks
whether a correctly conditioned frequency-domain test could work before adding
the uncertainty of estimating the deterministic mean.

Lengths 256 and 512 passed the frozen aggregate criteria. The 256-point design
accepted 5/6 iid controls and detected 10/12 d>=0.30 cases. The 512-point design
accepted 6/6 controls and detected 9/12 strong-memory cases. Length 78 accepted
all six controls but detected only 2/12 strong-memory cases, so the minimum
passing prefix length was 256.

The result identifies a feasible upper bound rather than an operational method.
Detection was not uniform over nonlinear strength: at strength 0.20, d=0.30 was
detected in 1/2 cases at length 256 and 0/2 at length 512. Furthermore, the null
uses the exact clean trajectory, which is unavailable in the intended setting.
The defensible conclusion is that conditioning resolves the gross false-positive
failure and that the unchanged spectral statistic can separate strong memory in
an aggregate declared domain once at least 256 samples are available. It does
not establish robustness to unknown trend, nor does it validate deployment.

The next gate is therefore specific: replace the oracle mean with an independently
estimated trajectory and include estimation uncertainty in the conditional null.
That experiment must retain 256 and 512 as the only eligible lengths and keep the
same aggregate decision thresholds. If control adequacy or strong-memory power
falls below 75%, this frequency route should not proceed to mechanism fitting.

## 34. Estimated conditional null controls size but misses moderate memory

The operational follow-up used an independent iid calibration trajectory at each
eligible length and nonlinear strength. A cubic smoothing spline estimated its
deterministic mean. Each of 128 conditional-null draws regenerated a pseudo-
calibration trace, refitted the spline, generated an independent pseudo-evaluation
trace, and applied the unchanged spectral sensitivity statistic. This nested
construction propagated mean-estimation uncertainty while keeping calibration
and evaluation noise independent. Lengths 256 and 512, the 99% quantile, the
four memory orders, three nonlinear strengths, two repeats, and the two 75%
go/no-go criteria were fixed before the result was inspected.

The conditional null solved the false-positive problem: both lengths accepted
6/6 iid controls. The fitted-mean RMSE remained between 2.01e-4 and 2.85e-4 over
all six length-strength calibration cells. Power nevertheless fell below the
prespecified requirement. Length 256 detected 7/12 d>=0.30 cases (58.3%), and
length 512 detected 8/12 (66.7%). The distinction is concentrated at the
moderate boundary: d=0.45 was detected in all 12 evaluated cases across both
lengths, whereas d=0.30 was detected in only 1/6 cases at length 256 and 2/6 at
length 512.

No eligible length therefore passed, and the route stopped before mechanism
fitting. This negative result narrows the achievable claim: independent smooth-
mean estimation plus a conditional local Whittle null is well calibrated and
detects very strong finite-sample memory in this design, but it is not a reliable
refusal gate over the declared d>=0.30 range. The known iid noise scale also
remains an oracle input. The current frequency-domain route is closed rather
than retuned. A joint semiparametric mean-memory construction or an increment-
based statistic would constitute a new research route with new frozen criteria.

## 35. Increment-domain conditional route passes the first feasibility gate

A new route used multiscale second-difference variance slopes rather than the
failed local Whittle statistic. The statistic compares second-difference
variances at lags 4, 8, and 16 with lag 1 and takes the median log-scale slope
over lags and channels. This construction exactly annihilates a linear trend.
The remaining smooth curvature entered a separately frozen conditional null:
an independent iid calibration trace supplied a spline mean, and every one of
128 Monte Carlo draws regenerated and refitted that calibration mean before
evaluating an independent pseudo-trace. No statistic lag, quantile, or route
criterion was changed after the project records were inspected.

The full 48-record matrix passed at both eligible lengths. At 256 points,
control adequacy and d>=0.30 detection were 6/6 and 12/12, respectively. At
512 points, the corresponding counts were 5/6 and 12/12. All twelve d=0.30
records across both lengths were detected, with minimum positive margins of
0.078 at length 256 and 0.129 at length 512. Detection at d=0.15 was 4/6 and
6/6, but this exploratory observation does not alter the frozen strong-memory
definition. The minimum passing prefix length is 256.

The result supports continuing the increment-domain route, but its current
claim remains conditional. Each threshold used a strength-matched independent
calibration trajectory, and the known iid noise scale remained available.
Thresholds varied from 1.184 to 1.623 at length 256 and from 0.353 to 0.587 at
length 512, so conditioning on the synthetic regime materially affects the
decision boundary. The next required experiment is therefore a transfer test
with pooled or leave-one-strength-out calibration. Failure of either 75%
criterion under that transfer would prevent promotion to a general refusal gate.

## 36. Leave-one-strength-out transfer fails in opposite directions

The transfer experiment removed access to a strength-matched calibration trace.
For each held-out nonlinear strength, its conditional null pooled 128 draws from
each of the other two strength regimes. Four project repeats yielded 16 records
per held-out strength and 96 records overall. Each regime independently had to
accept at least 3/4 controls and detect at least 6/8 d>=0.30 cases. This stronger
per-regime rule was frozen before the project matrix was run.

Neither length passed. The held-out middle strength 0.085 satisfied both gates
at lengths 256 and 512, with 4/4 controls accepted and 8/8 strong-memory cases
detected. The outer regimes failed in opposite ways. For held-out strength 0.05,
control adequacy was 0/4 at length 256 and 1/4 at length 512, while strong-memory
detection remained 8/8. For held-out strength 0.20, control adequacy was 4/4 at
both lengths but strong-memory detection fell to 4/8: all four d=0.30 cases were
missed and all four d=0.45 cases were detected at each length.

The pattern identifies deterministic curvature leakage as the transfer-limiting
quantity. At length 256 the leave-one-out thresholds ranged only from 1.555 to
1.626, but low-strength d=0 controls lay above the donor threshold whereas high-
strength d=0.30 cases lay below it. The corresponding length-512 thresholds
ranged from 0.541 to 0.587 and produced the same directional failure. Post-hoc
selection of donors, lags, or quantiles would therefore conceal rather than
solve the regime dependence.

The strength-matched increment result remains a conditional feasibility bound;
it is not a general gate. A subsequent route must replace the categorical
strength label with a predeclared observable normalization, such as a curvature-
leakage proxy estimated from the independent calibration trace. That route must
be frozen and evaluated separately, including controls at both ends of the
curvature range.

## 37. Observable curvature normalization restores same-family transfer

The next frozen experiment replaced the synthetic strength label with a
dimensionless curvature-leakage proxy computed from observed values. A disjoint
five-strength calibration bank (0.035, 0.065, 0.11, 0.16, and 0.24) was used to
fit one affine proxy-to-threshold map at each eligible length. None of the three
project strengths (0.05, 0.085, and 0.20) appeared in that bank or entered the
gate. Records outside the fitted proxy interval were predeclared refusals. Four
repeats produced 96 project records, and the per-strength 75% control and power
requirements were unchanged.

Both eligible lengths passed every regime. At each length all 12 iid controls
were accepted and all 24 d>=0.30 records were detected. Consequently, each of
the six length-strength cells achieved 4/4 control adequacy and 8/8 strong-
memory detection. All project proxies remained within scope. The calibration
models had R^2=0.9988 at length 256 and R^2=0.9962 at length 512; all d=0.30
records had positive margins. The minimum passing prefix length was 256.

The evidence supports a continuous nuisance-normalization route and removes the
need for the project strength label inside this synthetic generator family. It
does not yet establish a universal long-memory refusal gate. The calibration
bank shares the nonlinear-feedback generator with the project cases, the iid
noise scale is known, and the proxy is estimated from the evaluated trace rather
than from an independent nuisance-only measurement. The next mandatory test is
therefore independent-proxy and cross-generator transfer. Only if both control
adequacy and d>=0.30 power survive that test should mechanism fitting or real-
data claims resume.

## 38. Independent-proxy cross-generator transfer rejects the scalar route

The mandatory transfer test separated nuisance measurement from target
evaluation and changed the trend generator. The nonlinear-feedback calibration
bank from the preceding probe was retained without modification. Independent
iid proxy traces were generated for rate-drift dynamics at strengths 1.5, 3.0,
and 8.0 and for stretched-exponential decays at beta 0.69, 0.70, and 0.71. The
project traces used different seeds. These six external cells were selected only
from d=0 proxy-scope checks before the 192-record memory matrix was run. Each
cell still had to accept at least 3/4 controls and detect at least 6/8 d>=0.30
records.

Neither eligible length passed. Detection remained 48/48 for strong-memory
records at both lengths, but control adequacy was only 6/24 (25.0%) at length
256 and 10/24 (41.7%) at length 512. Rate drift at strength 1.5 passed at both
lengths, and strength 3.0 improved from 2/4 accepted controls at length 256 to
4/4 at length 512. Strength 8.0 accepted 0/4 and 2/4 controls, respectively.
All six stretched-exponential length-strength cells rejected every control. One
of four independent proxies for beta=0.69 at length 512 was outside the fitted
proxy interval; the remaining stretched-exponential failures occurred in scope.

A diagnostic-only statistic computed on each noise-free external trend was
stored after the frozen decision had been made. It shows that matching a single
curvature magnitude does not match the trend's complete multiscale second-
difference geometry. This explains why proxy values can be in range while the
null statistic is badly miscalibrated. The diagnostic does not alter any pass or
fail label.

This falsifies scalar observable-curvature normalization as a general transfer
solution. The same-family result in Section 37 remains a conditional finding,
not a deployable refusal gate. Any continuation must start a new, explicitly
frozen route based on a multiscale nuisance signature or direct null-statistic
prediction across generator families. Post-hoc adjustment of the current affine
map, strength grid, or quantile is prohibited.

## 39. High-dimensional shared-spectrum scaling passes the frozen gate

The experiment returned to the main minimal-memory-rank question after the
scalar long-memory proxy route was closed. A rank-two short-horizon generator
with rate ratio 4, horizon 4, and noise standard deviation 8e-4 was fixed before
evaluation. Channel counts were 1, 16, 64, and 256, with three independent
repeats and two optimizer starts for each rank candidate. The shared model used
one pair of pole locations and channel-specific positive weights. A vectorized
control instead fitted channel-specific pole locations and weights. Independent
recovery was scored with genuinely per-channel BIC, validation error, and pole
error rather than copying one aggregate failure to every channel.

The shared model failed in all three scalar trials, as expected at this
short-horizon boundary, but recovered rank two in 3/3 trials at each of 16, 64,
and 256 channels. Median shared log-rate error was 0.0512, 0.0499, and 0.0333,
respectively. Median independently resolved channel fractions were 0.19, 0.33,
and 0.27. All three frozen checks passed: at least 2/3 shared recovery at every
high-dimensional channel count, median 256-channel pole error no greater than
0.10, and at least a 0.25 recovery advantage over independent fitting at 64 and
256 channels.

The implementation avoided the non-scalable full parameter Jacobian used in
the early low-dimensional probe. Median shared-fit time increased from 4.40 s
at 16 channels to 6.39 s at 256 channels, while peak allocated GPU memory rose
from 21.1 to 78.8 MiB. This supports synthetic computational feasibility up to
256 channels and completes the planned channel-count range. It does not prove
that arbitrary fields share an exact spectrum: the generator satisfies that
assumption by construction, channel noise is independent and homoscedastic,
and true pole error is available only for evaluation. The next falsification
test must perturb pole locations across channels and introduce correlated noise
to determine when pooling changes from beneficial regularization to biased
mechanism inference.

## 40. Independent subgroup dispersion is rejected as a sharing gate

The first heterogeneity probe estimated four subgroup spectra independently and
thresholded their observed log-rate dispersion. Eighteen GPU records covered
exact sharing, mild log-spectrum drift 0.05, severe drift 0.15, independent
noise, and within-block noise correlation 0.60. The route failed its frozen
retention criterion. Even under exact sharing, short-horizon subgroup fits were
unstable: the median observed dispersion reached 0.621 under independent noise,
and the resulting gate rejected all three exact-sharing trials. The severe
heterogeneity cases were refused, but that sensitivity did not compensate for
the false refusals.

This negative result closes independent subgroup dispersion as a primary gate.
It demonstrates that a diagnostic assembled from separately ill-conditioned
inverse problems can be less reliable than the shared model it is intended to
audit. The corresponding artifacts are
`results/approximate_sharing_refusal_boundary.json` and
`results/approximate_sharing_refusal_boundary.md`.

## 41. A nested global-versus-grouped gate separates mild and severe drift

A second frozen probe compared the global rank-two model with a joint
four-group rank-two alternative. BIC supplied evidence for the more complex
model, while a separate held-out error requirement prevented complexity alone
from forcing refusal. The same 18-record matrix retained all exact and mild
sharing cases and refused all severe-drift cases under both noise-correlation
conditions. Thus the two prespecified checks passed: mild-sharing refusal was
0/12 and severe-heterogeneity refusal was 6/6.

This result supports the nested construction as an operational sharing audit,
but not as a universal heterogeneity detector. In several cells the grouped
model had worse held-out error or negative BIC support; refusal can therefore be
triggered by the independently frozen absolute validation-error limit even when
the grouped alternative is not selected. The artifacts are
`results/nested_group_sharing_gate.json` and
`results/nested_group_sharing_gate.md`.

## 42. The first boundary map is complete but misses the transition bracket

The next experiment preserved the gate and all thresholds, increased each cell
to five random seeds, and evaluated log-spectrum drifts 0.075, 0.100, and 0.125
under noise correlations 0.0 and 0.60. All 30 GPU records were finite. Refusal
fractions were identical across the two noise regimes: 3/5 at drift 0.075 and
5/5 at both 0.100 and 0.125. The sequence was monotone, and the maximum observed
noise-regime difference was zero.

The frozen boundary-map criterion nevertheless failed because the lowest tested
drift already had a 60% refusal rate rather than the required maximum of 40%.
Combined with the preceding 0/6 refusal result at drift 0.05, the empirical
transition lies inside the narrower interval 0.05--0.075 for this generator and
sample design. Five trials per cell yield wide Wilson intervals (0.23--0.88 for
3/5 and 0.57--1.00 for 5/5), so no precise critical drift is claimed. Moreover,
median grouped-model BIC support remained negative in five of six cells; the
current boundary chiefly characterizes the fixed validation-error gate, not
consistent direct detection of spectral heterogeneity. The route remains useful
as a conservative operational refusal rule, while precise mechanism-boundary
estimation requires a separately frozen denser study below drift 0.075.

The artifacts are `results/nested_group_boundary_map.json` and
`results/nested_group_boundary_map.md`.

## 43. A dense map resolves a noise-sensitive operational boundary

The unchanged nested gate was evaluated on five log-spectrum drifts from 0.05
to 0.075, two noise-correlation regimes, and six new seeds per cell. The repeat
offset excluded all seeds used by the preceding maps. All 60 GPU records were
finite. Pooled refusal increased monotonically from 1/12 at drifts 0.05 and
0.05625, to 2/12 at 0.0625, 3/12 at 0.06875, and 9/12 at 0.075. The first
empirical 50% crossing was therefore bracketed by 0.06875 and 0.075, with a
piecewise-linear descriptive estimate of 0.071875.

All frozen dense-boundary checks passed, but the result is not a universal
critical constant. Wilson intervals remain broad: the pooled refusal interval
is 0.09--0.53 at drift 0.06875 and 0.47--0.91 at drift 0.075. Noise correlation
also matters near the boundary. At drift 0.06875 the independent-noise regime
refused 0/6 records, whereas correlation 0.60 refused 3/6; at drift 0.075 the
corresponding counts were 3/6 and 6/6. The maximum regime difference was 0.50,
exactly the prespecified admissible limit.

Median grouped-model BIC support was negative at every pooled drift. Hence this
experiment localizes the operational failure boundary of the fixed held-out
error contract; it does not establish consistent statistical identification of
blockwise spectral heterogeneity. A stronger continuation should model the
drift-by-noise interaction or normalize the validation-error gate for correlated
noise before treating the bracket as transferable beyond this generator.

The artifacts are `results/dense_nested_group_boundary.json` and
`results/dense_nested_group_boundary.md`.

## 44. Correlation-aware calibration reduces noise disparity but over-refuses mild drift

The next frozen experiment tested whether the sharing gate could account for
correlated observation noise without using the oracle correlation coefficient.
An observed second-difference cross-channel correlation proxy was calibrated on
20 exact-sharing records spanning five noise-correlation conditions. A monotone
affine centre and a one-sided 90% residual allowance then defined the validation
limit. The independent project matrix contained 32 records: four spectral
drifts, two noise regimes, and four new repeats per cell. True noise correlation
was retained only as an evaluation diagnostic.

All 32 project records were completed. The calibrated rule retained all exact
sharing controls and refused all eight severe-drift records. At the boundary
drift 0.075, it reduced the absolute refusal-rate difference between the two
noise regimes from 0.25 to 0.00. It nevertheless failed the frozen route. At
drift 0.05 under independent noise, the calibrated limit refused 4/4 records,
whereas the original fixed gate refused 0/4; under correlation 0.60 it refused
1/4. In addition, three project proxies fell just outside the calibration range,
which was defined in advance as a route failure rather than silently clipped.

The observed proxy had a nonzero baseline near 0.16 even when the simulated
noise correlation was zero, because second differences retain some smooth-signal
curvature. It is therefore an operational correlation proxy, not an unbiased
estimator of the noise parameter. More importantly, exact-sharing calibration
learned stochastic fitting error only. It did not include the model discrepancy
that the contract intentionally permits under approximate sharing, so the new
limit conflated acceptable mild drift with numerical failure. This closes a
single noise-calibrated threshold as the next gate. A defensible continuation
must predeclare separate allowances for model approximation and observable noise,
for example a total contract of the form `tau_model + tau_noise(proxy)`, and
validate those components independently rather than retuning this experiment.

The artifacts are `results/noise_aware_sharing_gate.json` and
`results/noise_aware_sharing_gate.md`.

## 45. Separate model and noise budgets repair the mild-drift refusal failure

A new frozen experiment separated the held-out error contract into an
observable-noise budget and a declared model-approximation budget. The noise
component was calibrated from 25 exact-sharing records across five correlation
conditions. A disjoint 10-record bank at the predeclared maximum acceptable log
spectral drift 0.05 supplied a one-sided model allowance. The project matrix
then used 32 new records with four drifts, two noise regimes, and four repeats.
Neither project decisions nor proxy-scope checks used the true project noise
correlation.

The calibrated model allowance was 0.001336. Combined validation limits were
approximately 0.00357 under independent noise and 0.00363 under correlation
0.60. All five frozen checks passed. Exact and mild sharing were retained in
all 16 records, whereas the preceding noise-only decision refused 1/4 exact
correlated-noise records and 4/8 mild-drift records. At drift 0.075, the
decomposed gate refused 1/4 records in each noise regime, giving a boundary
noise gap of zero. All eight drift-0.15 records were refused, and every project
proxy remained inside the calibration-only guarded scope.

This is positive evidence that model discrepancy and stochastic fitting error
should not share one learned threshold. It does not establish a universal
tolerance formula. The model budget was calibrated at a declared synthetic
scope boundary, the generator family and noise scale were unchanged, and four
project repeats per cell provide only coarse rates. The next falsification step
must freeze these budgets and test transfer across channel count, noise scale,
and a different heterogeneous spectrum construction. Failure there would limit
the result to an internally calibrated contract rather than a transferable
sharing protocol.

The artifacts are `results/decomposed_tolerance_sharing_gate.json` and
`results/decomposed_tolerance_sharing_gate.md`.

## 46. The decomposed budget does not fully transfer across noise scale and construction

The Stage 45 coefficients and decision rule were frozen before evaluating a
72-record transfer matrix. The matrix crossed channel counts 32 and 128, noise
standard deviations 0.0004 and 0.0016, antisymmetric and curved heterogeneous
spectrum constructions, log-spectrum drifts 0, 0.05, and 0.15, and three new
repeats per cell. No transfer record was used to recalibrate either budget.

All 72 records were finite and all observable proxies remained inside the
Stage 45 calibration scope. Every severe-drift cell was refused in all three
repeats. Most exact and mild cells also met the frozen retention criterion. The
route nevertheless failed in the exact-sharing cell with 128 channels, noise
standard deviation 0.0016, and the curved construction: two of three repeats
were refused. Their shared-model validation RMSE values were 0.006824 and
0.005769, above frozen total tolerances of 0.003589 and 0.003596. Grouped-model
BIC support remained strongly negative (-679 and -1082), so these refusals do
not constitute evidence for heterogeneous mechanisms. They expose a transfer
failure of the frozen validation-error budget.

This negative result limits the decomposed tolerance to an internally
calibrated contract. The correlation proxy captures dependence and smooth-trend
geometry but does not explicitly identify observation-noise scale or fitting
adequacy. A defensible continuation must predeclare a noise-scale observable
and a convergence/optimizer adequacy check, then evaluate them on a fresh full
matrix. Selectively retrying the two failed records or enlarging the tolerance
after inspection would invalidate the falsification test and is not done here.

The artifacts are `results/decomposed_tolerance_transfer.json` and
`results/decomposed_tolerance_transfer.md`.

## 47. Observable noise scale repairs the high-noise control, but residual-based adequacy is invalid

Stage 47 retained the Stage 45 tolerance as a floor and added two independently
calibrated diagnostics. Sixteen exact-sharing records at 64 channels and four
noise scales calibrated a robust mixed time/channel second-difference MAD proxy,
a one-sided noise-scale correction, and a four-start fitting-adequacy threshold.
The project split then evaluated 72 fresh records over the Stage 46 transfer
matrix. Calibration and project seed ranges were disjoint.

The noise-scale proxy behaved as intended and all 72 project diagnostics stayed
inside the calibration scope. The key Stage 46 failure cell--128 channels, high
noise, curved construction, and exact sharing--was accepted in all three fresh
repeats. All high-noise exact and mild cells were retained, and both
antisymmetric severe cells were refused in all repeats. This supports retaining
the observable noise-scale component.

The combined route nevertheless failed. The fitting-adequacy rule normalized
training RMSE by the noise proxy and required at least two of four starts below
an exact-control threshold. At low noise, genuine model mismatch raises this
ratio even when all starts converge consistently. Consequently, every low-noise
severe cell produced 0/3 explicit refusals and 3/3 optimization-indeterminate
outcomes. The 128-channel, low-noise, antisymmetric mild cell also had 2/3
indeterminate outcomes. One high-noise curved severe cell had only 1/3 explicit
refusals. Under the frozen scoring rule, indeterminate outcomes count against
retention and cannot substitute for severe-mechanism refusal.

The start diagnostics confirm the conceptual confounding. Among low-noise
severe indeterminate records, the median minimum train-to-noise ratio was about
1.70 while the median within-record range across four starts was only about
0.11. Thus the starts often agreed with one another but jointly exceeded an
absolute residual threshold. The implemented adequacy metric measures model
discrepancy as well as optimizer failure and is rejected. A next-stage adequacy
audit must use start-to-start parameter or objective disagreement, gradient
stationarity, or deterministic replay checks without conditioning on the
absolute residual level.

The artifacts are `results/noise_scale_optimizer_transfer.json` and
`results/noise_scale_optimizer_transfer.md`.

## 48. Cross-start functional consistency removes residual confounding but exposes genuine multi-solution ambiguity

Stage 48 replaced the rejected train-RMSE-to-noise adequacy statistic with two
diagnostics that do not use the absolute residual level. Sixteen disjoint
exact-sharing controls calibrated the maximum second-start objective gap and
the maximum second-start prediction-function gap. A project fit was considered
optimization-adequate only when at least two of four initializations agreed
with the best fit under both frozen limits. The observable noise proxy and the
Stage 45 validation budget remained separate from this adequacy decision.

The full 72-record transfer matrix completed on the GPU and every diagnostic
remained inside its calibration scope. All 16 exact or mild-drift cells met the
predeclared retention rule. In particular, all 128-channel exact and mild cells
were retained within the allowed one-of-three adverse limit. This removes the
systematic low-noise residual confounding observed in Stage 47 and shows that
cross-start functional agreement is a more defensible optimizer audit.

The overall route still failed its severe-case requirement. Only three of eight
severe-drift cells achieved at least two explicit refusals in three repeats.
Five severe cells instead contained multiple optimization-indeterminate
outcomes, especially under the curved construction. Descriptively, the mean
explicit-refusal fraction over severe cells increased from 0.375 in Stage 47 to
0.458, while the mean indeterminate fraction decreased from 0.583 to 0.542;
these are independent-seed comparisons rather than paired effect estimates.

The result distinguishes two questions that earlier stages conflated. The
cross-start gate can assess whether the fitted shared model is reproducible
across initializations. It cannot force a mechanism decision when a
misspecified latent model has several comparably plausible fitted functions.
`INDETERMINATE_OPTIMIZATION` must therefore remain a first-class outcome. A
future route should test whether additional deterministic refinement or a
model-ensemble evidence calculation can resolve these cases, with an explicit
exit condition if ambiguity persists. Reclassifying indeterminate records as
mechanism refusals would not be valid.

The artifacts are `results/cross_start_consistency_transfer.json` and
`results/cross_start_consistency_transfer.md`.

## 49. Extended deterministic refinement resolves severe ambiguity but over-rejects mild drift

Stage 49 tested the most direct continuation from Stage 48 without changing any
calibrated threshold. The observable-noise correction, cross-start objective
and prediction-gap limits, minimum number of consistent starts, and the full
72-record project matrix were frozen. Only the deterministic L-BFGS refinement
budget was increased symmetrically for the shared and grouped candidates, from
80 to 240 iterations; the Adam budget remained 280 iterations. Project seeds
were disjoint from all preceding stages.

The longer refinement removed the severe-case ambiguity. All eight severe-
drift cells achieved explicit refusal in all three repeats, increasing the mean
severe refusal fraction from 0.458 in Stage 48 to 1.000 and reducing the mean
indeterminate fraction from 0.542 to 0.000. The complete project matrix and all
diagnostic-scope checks passed.

The frozen route nevertheless failed its retention requirement. Four low-noise
mild-drift cells exceeded the allowed one-of-three adverse outcomes: the
antisymmetric cells at 32 and 128 channels each produced two refusals, while
the corresponding curved cells produced three refusals. One medium-noise exact
record was optimization-indeterminate, but its cell remained within the
predeclared retention allowance. Thus additional refinement sharpened the
shared-versus-grouped distinction but moved the effective mechanism boundary;
it did not produce a uniformly better binary decision rule.

This result closes the route of repairing Stage 48 solely by increasing the
optimizer budget. Optimizer family, initialization policy, and refinement
budget must be part of the declared reliability contract because they can
change the scientific decision even when diagnostic thresholds are fixed.
The protocol retains `INDETERMINATE_OPTIMIZATION` as a first-class outcome and
does not replace it with an optimizer-dependent forced decision. A defensible
next stage should quantify decision stability across predeclared optimizer
budgets or use model-ensemble evidence, rather than selecting one budget after
observing which decision it produces.

The artifacts are `results/extended_refinement_transfer.json` and
`results/extended_refinement_transfer.md`.

## 50. Paired optimizer-budget audit confirms decision instability

Stage 50 directly tested the methodological implication of Stage 49. The same
24 datasets, initializations, calibration constants, and scientific-decision
thresholds were evaluated at deterministic L-BFGS refinement budgets of 80,
160, and 240 iterations. The matrix crossed 32/128 channels, low/high noise,
antisymmetric/curved heterogeneity constructions, and three paired repeats per
cell. Adam remained fixed at 280 iterations. No budget was selected or omitted
after observing its result.

The preregistered stability route failed decisively. Exact tri-state agreement
across all three budgets was 0.500, below the required 0.750. Eleven of 24
paired datasets (0.458) exhibited a direct `RETAIN`--`REFUSE` reversal, far
above the permitted one-of-24 rate. One additional pair contained an
`INDETERMINATE_OPTIMIZATION` outcome. All diagnostics remained inside the
frozen calibration scope, so the instability cannot be attributed to an
out-of-scope proxy value.

The instability was strongly localized. At noise standard deviation 0.0004,
both 32-channel constructions had direct reversals in all three repeats; the
128-channel antisymmetric and curved cells had reversal fractions 2/3 and 3/3,
respectively. At noise standard deviation 0.0016, three of four cells had exact
agreement in every repeat. The remaining 128-channel curved cell had two exact
agreements and one indeterminate-to-retain transition, but no direct binary
reversal. Thus a larger refinement budget is not uniformly better: it sharpens
the low-noise sharing decision enough to move observations across the frozen
scientific boundary.

This result closes the use of any single, unexamined refinement budget as a
budget-independent mechanism decision. Optimizer family, initialization
policy, stopping rule, and refinement budget are required reliability-contract
metadata. A result that changes between preregistered adequate budgets must be
reported as budget-sensitive rather than forced to `RETAIN` or `REFUSE` by
choosing a favorable run. The next defensible route is a frozen consensus or
abstention rule over multiple budgets, with an explicit requirement that it
preserve exact/mild controls and severe-drift refusals on independent data.

The artifacts are `results/optimizer_budget_stability.json` and
`results/optimizer_budget_stability.md`.

## 51. Budget consensus exposes conflicts but cannot repair common-mode false refusals

Stage 51 froze a conservative budget-ensemble rule before evaluating new seeds.
Each of 72 independent datasets was fitted at L-BFGS budgets 80, 160, and 240,
for 216 total evaluations across 32/128 channels, two noise scales, two drift
constructions, and exact, mild, and severe spectral drift. A determinate result
required at least two matching binary votes and no opposite binary vote. Any
`RETAIN`--`REFUSE` conflict was reported as a budget-sensitive abstention rather
than resolved by majority vote or budget selection.

The rule preserved strong severe-drift sensitivity: 23/24 severe datasets were
refused and the remaining case was indeterminate. Every severe cell met the
predeclared two-of-three refusal requirement. Seven of 72 datasets were
explicitly marked budget-sensitive, and all 216 diagnostics stayed inside the
frozen calibration scope. Thus the abstention mechanism successfully exposes
binary optimizer-budget conflicts.

The complete route nevertheless failed because consensus cannot correct an
error shared by all budgets. Four of 24 exact controls and five of 24 declared
mild-drift cases were refused, giving a combined false-refusal fraction of
9/48 = 0.1875 versus the frozen maximum 1/48. Exact/mild determinate retention
was 32/48 = 0.667, while seven mild datasets were budget-sensitive rather than
forced into a binary decision.

All nine false refusals occurred at noise standard deviation 0.0016. Their
grouped-model BIC support was negative at every refusing budget, so the records
did not supply positive evidence for a heterogeneous spectrum. Instead, the
shared validation RMSE exceeded the frozen augmented tolerance by factors from
about 1.025 to 1.912. The failure is therefore a common-mode validation-budget
error, not a defect in the conflict-abstention logic and not evidence that exact
sharing was false.

This result closes voting or consensus over fixed optimizer budgets as a
standalone repair. Budget consensus remains useful metadata because it prevents
post-hoc optimizer selection and identifies seven unstable observations. A
subsequent route must address conditional calibration of the validation-error
budget under high-noise exact and acceptable-drift controls, while keeping BIC
mechanism evidence separate. It must not enlarge the tolerance using these
failed project records. Any new calibration requires disjoint controls and a
fresh evaluation split.

The artifacts are `results/budget_consensus_abstention.json` and
`results/budget_consensus_abstention.md`.

## 52. Disjoint high-noise calibration does not justify tolerance inflation

Stage 52 used calibration seeds disjoint from both Stage 51 and the locked
validation matrix. Only exact-sharing and declared mild-drift controls were
eligible for calibration; severe drift and every Stage 51 failure were
excluded. The preregistered 95% one-sided ratio calibration selected a
validation-tolerance multiplier of exactly 1.0, because all 48 calibration
ratios were below one (maximum 0.9506). Thus the data did not justify increasing
the existing allowance.

On 24 new paired datasets evaluated at three optimizer budgets, exact/mild
false refusal was 1/16 = 0.0625, just above the frozen 0.05 maximum. The other
criteria passed: exact/mild retention was 0.875, severe-drift refusal was 1.0,
budget sensitivity was 1/24 = 0.0417, and all diagnostics were in calibration
scope. The route is recorded as failed; the threshold was not rounded or
relaxed after observing the result.

Artifacts: `results/high_noise_conditional_calibration.json` and
`results/high_noise_conditional_calibration.md`.

## 53. Cross-scale transfer confirms a narrow operating boundary

The frozen multiplier was next applied without recalibration to 48 and 192
channels, noise levels 0.0008 and 0.0014, antisymmetric and rotated drift
constructions, and independent repeats. The 0.002 noise condition was declared
in advance as a stress test and excluded from the core pass criterion.

The 48 core pairs produced 0.0625 false refusal, 0.78125 determinate retention,
1.0 severe-drift refusal, and 0.1042 budget sensitivity. The route failed the
first two frozen limits (0.05 and 0.80) while passing severe detection,
budget-sensitivity, matrix-completeness, and diagnostic-scope checks. Under
stress noise, false refusal increased to 0.375 and retention fell to 0.625.
This is evidence for a bounded contract, not a transferable universal gate.

Artifacts: `results/conditional_contract_transfer.json` and
`results/conditional_contract_transfer.md`.

## 54. Real residual morphology exposes common-mode false refusal

Stage 54 used standardized residual morphology from Ecoflex 00-30, Dragon Skin
20, and Mold Star 30 public relaxation curves. Known exact, mild, and severe
mechanisms were injected for a truth-labelled semisynthetic audit. A separate
blind audit over eight real curves was explicitly excluded from recovery
accuracy because no mechanism truth label is available.

Conditional consensus obtained coverage 1.0, selective accuracy 0.8056,
false-refusal fraction 0.2917, and severe-refusal fraction 1.0. The uncalibrated
and single-budget controls were identical because Stage 52 selected multiplier
1.0. BIC-only consensus performed better on this matrix: coverage 0.9444,
selective accuracy 0.8824, false refusal 0.1667, and severe refusal 0.9167. The
blind real-data consensus was `INDETERMINATE` due to insufficient determinate
votes.

The route fails three preregistered criteria and is closed as a general repair.
The defensible conclusion is not that the underlying shared-spectrum question
is unanswerable, but that validation tolerance, optimizer stability, and
nested-model evidence cannot be collapsed into one calibrated scalar gate.
Future work must preserve them as separate reliability-contract dimensions.

Artifacts: `results/real_background_mechanism_audit.json` and
`results/real_background_mechanism_audit.md`.

## 55. A frozen multi-axis hierarchy repairs false refusal but loses rejection power

Stage 55 consumed the locked Stage 53 and Stage 54 records without refitting or
recalibrating. It kept numerical eligibility, predictive validation, and BIC
structure evidence as separate axes. A binary result was allowed only when the
numerical axis was eligible and the validation and structural axes agreed;
otherwise the protocol returned `INDETERMINATE`. The Stage 54 success limits
were reused unchanged.

The Gaussian core matrix passed all four criteria: coverage was 0.750,
selective accuracy 1.000, false refusal 0.000, and severe-drift refusal 0.750.
The out-of-scope Gaussian stress matrix correctly abstained completely rather
than extrapolating the calibration. On real residual backgrounds, the hierarchy
reduced false refusal from 0.2917 to 0.0417 and increased selective accuracy
from 0.8056 to 0.9565 at coverage 0.6389. Severe-drift refusal was only 0.5833,
so the frozen route did not pass.

This is a useful architectural result rather than another failed threshold.
Separating the axes solves most common-mode false refusal, but requiring
symmetric agreement for both outcomes discards too much valid rejection
evidence. A subsequent stage should preregister an asymmetric hierarchy:
retention requires concordant evidence, whereas refusal is permitted only by a
separately calibrated strong-evidence route. It must not tune that route on the
Stage 54 labels.

Artifacts: `results/multiaxis_hierarchical_contract.json` and
`results/multiaxis_hierarchical_contract.md`.

## 56. Asymmetric evidence restores rejection power but misses false-refusal control

Stage 56 used the Stage 53 Gaussian core matrix as a development set and kept
the Stage 54 real residual matrix locked for evaluation. The repeated strong-
validation score was defined as the second-largest validation-to-tolerance
ratio across the three optimizer budgets. The frozen threshold, 1.147753, was
the next representable floating-point value above the maximum acceptable
development score. The minimum severe development score was 1.243974, giving a
strict development separation before real-background evaluation.

The asymmetric hierarchy retained the conservative Stage 55 rule for
retention. Refusal was permitted by structural BIC consensus, or when the
validation axis refused, at least one structural refusal vote was present, and
the repeated exceedance score crossed the frozen threshold. The development
matrix achieved coverage 0.8333, selective accuracy 1.000, false refusal 0.000,
and severe refusal 1.000.

On the locked real-background evaluation, coverage was 0.8333, selective
accuracy 0.9000, and severe refusal 1.000. False refusal was 3/24 = 0.125,
exceeding the frozen 0.10 maximum by one case. The three errors were all mild
drift examples: Dragon Skin 20 under the antisymmetric construction and two
Mold Star 30 examples. Two were triggered by repeated validation exceedance
with partial structural support, and one by structural consensus itself.

The route is recorded as failed without threshold adjustment. The remaining
error is a calibration-domain shift from Gaussian controls to real residual
morphology, not a lack of rejection power. Stage 54 labels are now closed to
calibration. Any further confirmatory route requires new disjoint
real-morphology controls or an explicit uncertainty model for residual
backgrounds.

Artifacts: `results/asymmetric_evidence_hierarchy.json` and
`results/asymmetric_evidence_hierarchy.md`.

## 57. Disjoint real-morphology calibration closes the frozen route

Stage 57 used seven public elastomer residual backgrounds that do not overlap
the three truth-labelled Stage 54 evaluation backgrounds. For each calibration
background, two drift constructions and three declared drift levels were fit
under all three optimizer budgets. This produced 126 fits and 42 paired
calibration cases. The repeated validation score and repeated structural BIC
score were each defined by their second-largest budget value. Their strong-
evidence thresholds were fixed as the next representable floating-point values
above the maximum acceptable calibration scores: 2.042843 and 43.158548.

Calibration achieved coverage 0.6905, selective accuracy 1.000, false refusal
0.000, and severe-drift refusal 0.7857. The frozen rule was then applied to the
36 Stage 54 pairs without refitting. Locked evaluation achieved coverage 0.750,
selective accuracy 0.9630, false refusal 1/24 = 0.0417, and severe-drift refusal
11/12 = 0.9167. All preregistered success criteria passed. The one false
refusal was `Mold Star 30 / rotated / drift 0.05`, caused by repeated structural
evidence above the morphology-calibrated envelope.

This stage changes the project conclusion. A universal scalar gate remains
unsupported, but a scoped, multi-axis, asymmetric contract calibrated on real
residual morphology is empirically viable. Its cost is deliberate abstention:
9/36 locked cases were `INDETERMINATE`. The next confirmatory step should test
this frozen contract on newly acquired backgrounds or a genuinely external
implementation, not recalibrate it again on Stage 54.

Artifacts: `results/morphology_calibrated_asymmetric_hierarchy.json` and
`results/morphology_calibrated_asymmetric_hierarchy.md`.

## 58. Frozen external transfer succeeds with explicit abstention

The Stage 57 thresholds and asymmetric rule were applied without adjustment to
42 paired cases built from seven residual backgrounds spanning brain tissue,
copper-alloy relaxation, and martensitic-steel relaxation. Coverage was 0.6667,
selective accuracy was 1.0000, false refusal was 0, and severe-drift refusal
was 0.8571. All frozen transfer checks passed. Fourteen cases were deliberately
`INDETERMINATE`; they are evidence of bounded identifiability, not errors hidden
from the denominator.

## 59. Independent numerical and decision replay succeeds

A SciPy matrix-exponential implementation agreed with the PyTorch lifted
propagator to a maximum relative value error of 2.665e-15. Central finite
differences agreed with PyTorch parameter gradients to 4.067e-9 relative
error. A standalone rule implementation replayed all 78 Stage 57/58 decisions
and reasons with 100% concordance. This is a local cross-implementation check,
not an external-team reproduction claim.

## 60. Observation degradation defines the operating boundary

The frozen rule was evaluated in 162 additional fits covering three physical
sources, two drift constructions, three drift levels, and three optimizer
budgets. The normal external baseline remained `SUPPORTED`. Doubled residual
noise, an 18-point training subset, and a 45% observation window were each
`SCOPE_LIMITED`. The thresholds were not retuned. Reliable decisions therefore
require the declared observation-quality contract.

## 61. Confirmatory evidence is frozen

The Stage 57--60 evidence chain, scripts, thresholds, headline metrics, claim
boundaries, and SHA-256 hashes are recorded in
`results/stage61_evidence_manifest.json` and `STAGE61_FREEZE_REPORT.md`.
The freeze explicitly does not claim an external reproducer, a universal
identification guarantee, or a software DOI/public archival release.

## 62. A public PVA task establishes a specimen-level rank boundary

Raw PVA gel stress-relaxation observations support rank 3 only at the full
28-second, 96-sample protocol. Shorter and intermediate observations produce
rank 2 or `INDETERMINATE` rather than a forced mechanism label. Inference keeps
the three specimens as the independent units; the nine cycles are repeated
curves and do not inflate the statistical sample size.

## 63. A chemically distinct public task validates refusal and the software contract

The UCI gas-flow dataset adds 50 independent non-air exposure experiments, 16
channels, five acquisition batches, and a registered recovery interval. Shared
rank 3 has the lowest median held-batch NRMSE (0.05392) among four audited
methods, but two decay rates coalesce and all 27 threshold variants remain
`INDETERMINATE`. This provides direct evidence that predictive advantage and
mechanistic identifiability are separate claims.

External baselines, experiment-cluster statistics, multi-start stability,
leave-group-out transfer, a controlled PVA boundary-factor audit, and minimal
identifiability theory are complete. The versioned package surface exposes
`fit`, `evaluate`, `decide`, and `report`, plus schema `1.0.0` and a one-command
reproduction CLI. Verification completed 111 tests and parsed 66 result JSON
files. See `STAGE63_EVIDENCE_REPORT.md`, `MINIMAL_THEORY.md`, and
`API_CONTRACT.md`.
