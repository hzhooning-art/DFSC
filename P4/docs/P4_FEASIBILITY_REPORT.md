# P4 route-feasibility report

## Decision

**Partial go.** The constrained learnable-memory-kernel route is numerically
feasible and sufficiently distinct from P3 to justify a dedicated development
track. The current test is not yet evidence for a complete P4 paper.

## Tested route

The prototype learns a causal non-negative normalized mixture of 12 exponential
modes to approximate a tempered power-law kernel on a finite time window. The
parameterization uses a softmax for non-negative normalized weights and a
softplus transform for positive rates. Causal convolution uses midpoint
quadrature, so the reference and learned operators share the same discrete
convention.

## Current result

The experiment was run with PyTorch 2.11.0+cu128 on CUDA.

| Diagnostic | Result |
|---|---:|
| Full-window kernel relative L2 error | 0.02145 |
| Held-out kernel relative L2 error | 0.25599 |
| Full-window convolution relative L2 error | 0.01058 |
| Held-out convolution relative L2 error | 0.01351 |
| Minimum learned kernel value | 0.00161 |
| Learned finite-window kernel mass | 0.97359 |
| Endpoint parameter gradients finite | Yes |

The convolution output is substantially more accurate than the pointwise
kernel tail because the input-weighted operator is less sensitive to small tail
errors on this test. The positive minimum and finite gradients establish only
the local implementation contract, not a general stability theorem.

## Why this is independent of P1--P3

- P1 standardizes and differentiates a fixed Mittag--Leffler propagation
  primitive.
- P2 controls numerical and gradient reliability of that propagation.
- P3 decides when a learned correction should be composed with a fixed
  structured operator.
- P4 learns the memory kernel or spectral representation itself, while keeping
  causal and dissipative constraints explicit.

## Required next tests before calling P4 mature

1. Compare fixed fractional, tempered, Prabhakar, and learned-kernel models on
   the same observations and compute cost.
2. Repeat across alpha, tempering, noise level, sparsity, and observation-window
   length with multiple seeds.
3. Test independent reference discretizations and convergence under grid
   refinement.
4. Measure parameter and model-class identifiability, not only prediction
   error.
5. Add a nonlinear or forcing-driven memory task and an external public dataset.
6. Connect the learned kernel to the dfsc operator/kernel registry without
   moving the experimental route into the stable API prematurely.

## Practicality probe

The second test uses 50% sparse observations in the first 20 time units,
2.5% relative observation noise, two kernel families, and three random seeds.
It compares the constrained 12-mode mixture with a fitted single-exponential
baseline. The following values are mean plus/minus one sample standard
deviation over seeds.

| Target family | Method | Full convolution error | Held-out convolution error |
|---|---|---:|---:|
| Tempered power law | Constrained mixture | 0.0340 +/- 0.0089 | 0.0320 +/- 0.0025 |
| Tempered power law | Single exponential | 0.6765 +/- 0.0070 | 0.6713 +/- 0.0079 |
| Two time scales | Constrained mixture | 0.0325 +/- 0.0142 | 0.0339 +/- 0.0168 |
| Two time scales | Single exponential | 0.6558 +/- 0.0074 | 0.6248 +/- 0.0078 |

The constrained mixture remained positive in every run and all endpoint
gradients were finite. This is useful evidence that the representation can
recover more than one time scale under sparse noisy observations. It is not yet
a fair comparison with all fractional solvers: the target kernels are synthetic,
the rate grid is finite, and the single-exponential baseline is deliberately
simple. The next benchmark must include fixed fractional and tempered models,
an unconstrained signed mixture, and an independently discretized reference.

## Strong-baseline probe

The strong-baseline experiment adds a fitted fixed tempered-fractional family
and an unconstrained signed exponential mixture. All results use the same
sparse/noisy observations and three seeds.

| Target family | Method | Full convolution error | Held-out convolution error | Negative-kernel fraction |
|---|---|---:|---:|---:|
| Tempered power law | Constrained mixture | 0.0340 +/- 0.0089 | 0.0320 +/- 0.0025 | 0.000 |
| Tempered power law | Fractional-tempered | 0.0374 +/- 0.0252 | 0.0417 +/- 0.0284 | 0.000 |
| Tempered power law | Signed mixture | 13.751 +/- 10.474 | 24.672 +/- 18.794 | 0.304 +/- 0.124 |
| Two time scales | Constrained mixture | 0.0325 +/- 0.0142 | 0.0339 +/- 0.0168 | 0.000 |
| Two time scales | Fractional-tempered | 0.0552 +/- 0.0027 | 0.0653 +/- 0.0042 | 0.000 |
| Two time scales | Signed mixture | 12.135 +/- 8.383 | 21.666 +/- 14.969 | 0.292 +/- 0.145 |

The constrained mixture is competitive with the correctly specified
fractional-tempered family on the power-law target and is better on the
two-timescale target. The signed mixture is not a valid physical baseline in
this form: its unconstrained least-squares fit is ill-conditioned, creates
negative kernels, and extrapolates poorly. This supports a conditional claim
that physical constraints improve practical learnability; it does not establish
universal superiority.

## Mechanism-selection probe

The next experiment tests the stronger scientific question: can a prefix-only
validation rule identify the correct memory-kernel family before the long-time
region is observed? The candidates are the constrained exponential spectrum,
the distributed-order fractional family, and the fixed tempered-fractional
family. Selection uses only the observed prefix; the test score is computed on
the later unobserved horizon.

| Target family | Prefix-selection accuracy | Selected long-time error | Oracle long-time error | Mean selection regret |
|---|---:|---:|---:|---:|
| Tempered power law | 33.3% | 0.0404 | 0.0258 | 0.0147 |
| Two time scales | 0.0% | 0.0666 | 0.0339 | 0.0327 |

This is a deliberate negative result. Current prefix validation does not
reliably identify the correct mechanism, even though the candidate families
can each produce accurate convolutions in their favorable regions. The result
changes the P4 research question from simply learning a kernel to learning a
kernel together with an identifiability or uncertainty diagnostic. A future
P4 method must report when the data do not support a unique mechanism, rather
than silently selecting a plausible-looking model.

The regularized signed mixture is a stronger control. Its ridge parameter is
selected on an internal observation-prefix validation split. It improves over
the unregularized signed fit, but remains worse than the constrained mixture:

| Target family | Method | Full convolution error | Held-out convolution error |
|---|---|---:|---:|
| Tempered power law | Ridge signed mixture | 0.0512 +/- 0.0220 | 0.0603 +/- 0.0234 |
| Tempered power law | Constrained mixture | 0.0340 +/- 0.0089 | 0.0320 +/- 0.0025 |
| Two time scales | Ridge signed mixture | 0.0951 +/- 0.0258 | 0.1028 +/- 0.0232 |
| Two time scales | Constrained mixture | 0.0325 +/- 0.0142 | 0.0339 +/- 0.0168 |

The ridge signed fits happened to remain non-negative on these finite windows,
so the present evidence should be interpreted as a combined accuracy and
conditioning result, not as proof that positivity is always the only source of
the gain. A future ablation should vary the ridge strength and include a
constraint-free neural kernel parameterization.

## Distributed-order and independent reference probe

An additional baseline learns a non-negative mixture of six fractional orders
with a shared learned tempering parameter. A four-times finer quadrature grid
provides an independent reference for the coarse-grid convolution.

| Target family | Method | Full convolution error | Held-out convolution error |
|---|---|---:|---:|
| Tempered power law | Distributed-order fractional | 0.0366 +/- 0.0237 | 0.0400 +/- 0.0263 |
| Tempered power law | Exponential mixture | 0.0346 +/- 0.0088 | 0.0325 +/- 0.0025 |
| Two time scales | Distributed-order fractional | 0.0556 +/- 0.0020 | 0.0666 +/- 0.0056 |
| Two time scales | Exponential mixture | 0.0326 +/- 0.0141 | 0.0340 +/- 0.0167 |

The distributed-order baseline is competitive on the long-tailed target but
cannot represent the sharp two-timescale relaxation as efficiently as the
exponential spectrum. This is evidence for a model-selection problem, not a
universal winner. The coarse-to-fine reference discrepancy is 0.0308 for the
tempered power-law family and 0.0484 for the two-timescale family. These values
are large enough that future P4 reporting must separate quadrature error from
learned-kernel error.

## Selective mechanism-discovery application probe

The mechanism margin can be used as an abstention score. On six noise
realizations per target family, accepting only cases with a sufficiently large
validation margin produced the following risk--coverage behavior:

| Target family | Margin threshold | Coverage | Accepted convolution error | Accepted regret |
|---|---:|---:|---:|---:|
| Tempered power law | 0.0005 | 83.3% | 0.0266 | 0.0069 |
| Tempered power law | 0.0010 | 66.7% | 0.0281 | 0.0087 |
| Two time scales | 0.0010 | 50.0% | 0.0531 | 0.0324 |
| Two time scales | 0.0020 | 33.3% | 0.0414 | 0.0194 |
| Two time scales | 0.0030 | 16.7% | 0.0223 | 0.0000 |

The result is asymmetric. Abstention improves the accepted tempered-power-law
error at moderate coverage, while the clearest benefit occurs for the
two-timescale family at low coverage, where the accepted routes match the
retrospective oracle in this small test. This supports a selective-use case,
but the sample is too small for a general guarantee. A mature P4 method should
calibrate this risk--coverage curve on an independent validation bank and report
confidence intervals before deployment.

## Frozen-threshold robustness sweep

The threshold `0.005` was frozen from the formal validation protocol and then
tested without retuning under two noise levels and two observation-window
lengths. Each cell contains three random seeds.

| Target | Noise | Observed end | Coverage | Accepted regret |
|---|---:|---:|---:|---:|
| Tempered power law | 0% | 10 | 100% | 0.0000 |
| Tempered power law | 0% | 14 | 100% | 0.0000 |
| Tempered power law | 5% | 10 | 100% | 0.0376 |
| Tempered power law | 5% | 14 | 100% | 0.0297 |
| Two time scales | 0% | 10 | 100% | 0.0000 |
| Two time scales | 0% | 14 | 100% | 0.0000 |
| Two time scales | 5% | 10 | 66.7% | 0.0000 |
| Two time scales | 5% | 14 | 100% | 0.0020 |

The frozen threshold is effective in the noiseless settings but is not yet
calibrated for 5% noise: the tempered-power-law cases remain overconfident.
This negative result prevents a universal deployment claim. It motivates the
next P4 contribution: noise-aware uncertainty calibration or a conformal/
bootstrap mechanism margin, with calibration performed independently of the
final test bank.

## Formal independent validation

The first formal protocol separates six calibration realizations from eight
independent test realizations for each target family. The 50% target coverage
threshold is calibrated only from the calibration margins. Bootstrap intervals
use 2,000 resamples of the accepted test cases.

| Target family | Calibrated threshold | Test coverage | Accepted regret mean | 95% bootstrap interval |
|---|---:|---:|---:|---:|
| Tempered power law | 0.00498 | 62.5% | 0.00910 | [0.0000, 0.0218] |
| Two time scales | 0.00556 | 37.5% | 0.00033 | [0.0000, 0.0010] |

The unconditional regret means were 0.00733 for the tempered-power-law test
bank and 0.02385 for the two-timescale test bank. Selective mechanism discovery
therefore reduces the accepted regret most clearly for the two-timescale case,
at the cost of rejecting 62.5% of test cases. The result is evidence for a
useful abstention workflow, not a safety guarantee: the test bank is still small
and the threshold calibration must be repeated across observation lengths,
noise levels, and independent physical datasets.

## Noise-aware calibration probe

As a next-stage test, the selection threshold was calibrated independently for
each noise level and observation window using three calibration realizations,
then evaluated on four held-out realizations. The fixed `0.005` threshold was
reported alongside the condition-specific threshold.

| Target | Noise | Observed end | Calibrated threshold | Fixed coverage | Calibrated coverage | Fixed regret | Calibrated regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tempered power law | 0% | 10 | 1.0000 | 100% | 100% | 0.0000 | 0.0000 |
| Tempered power law | 0% | 14 | 1.0000 | 100% | 100% | 0.0000 | 0.0000 |
| Tempered power law | 5% | 10 | 0.0022 | 100% | 100% | 0.0326 | 0.0326 |
| Tempered power law | 5% | 14 | 0.0095 | 75% | 75% | 0.0112 | 0.0112 |
| Two time scales | 0% | 10 | 0.9388 | 100% | 0% | 0.0000 | -- |
| Two time scales | 0% | 14 | 0.8477 | 100% | 100% | 0.0000 | 0.0000 |
| Two time scales | 5% | 10 | 0.0006 | 75% | 100% | 0.0039 | 0.0149 |
| Two time scales | 5% | 14 | 0.0304 | 25% | 25% | 0.0000 | 0.0000 |

The probe does not validate a simple noise-aware threshold rule. With only
three calibration realizations per condition, the empirical quantile is itself
unstable; in one noiseless two-timescale condition it rejected every test
case, while in a noisy condition it increased coverage and regret. This is a
useful failure diagnosis rather than evidence of a solved uncertainty problem.
The next defensible implementation is a hierarchical or conformal-style
calibration protocol with substantially more independent realizations, an
explicit coverage target, and calibration strata that include noise level and
observation length. Until then, P4 should claim selective mechanism discovery
and auditable abstention, but not distribution-free risk control.

## Bootstrap selection-confidence probe

To distinguish a stable mechanism decision from a large one-shot validation
margin, each validation interval was resampled 256 times. The bootstrap winner
frequency was used as a confidence score. This test used two independent seeds
per condition, so it is diagnostic rather than a final statistical claim.

| Target | Noise | Observed end | Confidence threshold | Coverage | Accepted regret |
|---|---:|---:|---:|---:|---:|
| Tempered power law | 0% | 10 | 0.75 | 100% | 0.0000 |
| Tempered power law | 5% | 10 | 0.75 | 50% | 0.1121 |
| Tempered power law | 5% | 14 | 0.75 | 100% | 0.0512 |
| Tempered power law | 5% | 14 | 0.90 | 0% | -- |
| Two time scales | 0% | 10 | 0.75 | 100% | 0.0000 |
| Two time scales | 5% | 10 | 0.90 | 50% | 0.0099 |
| Two time scales | 5% | 14 | 0.75 | 50% | 0.0187 |

Bootstrap confidence is informative: noiseless decisions have essentially
zero entropy, whereas noisy cases exhibit lower winner concentration and higher
entropy. However, confidence alone does not guarantee low future regret. The
tempered-power-law case at 5% noise is a counterexample in which a selected
mechanism can remain bootstrap-stable while extrapolation is poor. Therefore,
the next P4 design should combine selection stability with a calibrated
out-of-window error proxy, rather than present bootstrap confidence as a
stand-alone correctness certificate.

## Expanded joint validation

The joint criterion was rerun on 60 independent synthetic realizations: five
seeds for each of two target families, three noise levels (0%, 2.5%, and 5%),
and two observation windows. At 5% noise, requiring temporal agreement reduced
mean regret from 0.01285 at 100% confidence-only coverage to 0.00812 at 50%
coverage (threshold 0.5), and reduced the maximum accepted regret from 0.0543
to 0.0304. At 2.5% noise, the lowest threshold did not improve mean regret
(0.01744 for confidence-only versus 0.01762 with agreement), whereas higher
thresholds showed a more favorable reduction, for example 0.01613 versus
0.01157 at the respective 50% and 30% coverages.

The expanded result supports the intended interpretation of the joint score:
it provides a tunable conservative risk--coverage mechanism. It does not
produce a monotone or universally optimal curve, and the coverage penalty is
substantial. P4 should therefore present the criterion as an auditable
selection policy for abstention, with the threshold chosen for the application,
rather than as a universal model-selection theorem.

## External elastomer stress-relaxation probe

The first external-data probe used four independent curves from the public
elastomer stress-relaxation dataset already archived in P3 under the frozen
response-independent extraction rule. The source archive is distributed under
CC BY 4.0 and is identified in the project contract by DOI
`10.5281/zenodo.14983287`. Each curve was normalized at the post-loading peak,
split into 55% fitting, 20% validation, and 25% terminal evaluation. The three
P4 candidates were fitted directly to the normalized relaxation response.

| Curve | Validation-selected mechanism | Terminal oracle | Selected error | Oracle error | Regret |
|---|---|---|---:|---:|---:|
| Cheetah | Fractional-tempered | Fractional-tempered | 0.0661 | 0.0661 | 0.0000 |
| Dragon Skin 20 | Fractional-tempered | Distributed-order | 0.5102 | 0.4839 | 0.0264 |
| Ecoflex 00-20 | Fractional-tempered | Distributed-order | 0.5232 | 0.4974 | 0.0257 |
| Filaflex 60A | Fractional-tempered | Fractional-tempered | 0.3092 | 0.3092 | 0.0000 |

The terminal-oracle agreement was 50%, with mean selected error 0.3522, mean
oracle error 0.3392, and mean regret 0.0130. This is an external feasibility
result, not evidence that one P4 mechanism is universally best. In particular,
the two disagreement cases show that validation-only mechanism identification
can fail on real relaxation data even when the candidate fits are numerically
stable. The result supports the need for the joint stability/abstention policy,
but also makes clear that the next external-data experiment must include more
specimens, uncertainty calibration, and a comparison with standard stretched-
exponential and Prony-series relaxation models.
rather than as a universal model-selection theorem.

## Risk--coverage curve analysis

Using the joint-stability samples, confidence thresholds from 0.5 to 0.95
were swept without changing any fitted model. Under 5% noise, confidence-only
acceptance reduced mean regret from 0.0148 at 87.5% coverage to 0 at 37.5%
coverage. Requiring temporal agreement was more conservative: it achieved 0
mean regret at 37.5% coverage, while retaining 50% coverage at threshold 0.5
with mean regret 0.00525.

These curves provide the first direct evidence that P4 can expose an
application-level risk--coverage trade-off instead of returning an opaque
mechanism choice. The curves are diagnostic because they inherit only eight
rows per noise level from the preceding probe; they must be recomputed with a
larger independent seed bank before being used as a paper-level performance
claim.

## Joint stability and temporal-agreement probe

The next diagnostic combined bootstrap confidence with agreement between the
early and late halves of the validation interval. Two independent seeds were
used per condition, so the numbers are intended to test the design direction,
not to establish a final confidence guarantee.

| Target | Noise | Observed end | Confidence threshold | Agreement required | Coverage | Accepted regret |
|---|---:|---:|---:|:---:|---:|---:|
| Tempered power law | 0% | 10 | 0.75 | yes | 100% | 0.0000 |
| Tempered power law | 5% | 10 | 0.75 | yes | 0% | -- |
| Tempered power law | 5% | 14 | 0.75 | yes | 50% | 0.0000 |
| Two time scales | 0% | 10 | 0.75 | yes | 100% | 0.0000 |
| Two time scales | 5% | 10 | 0.75 | yes | 50% | 0.0000 |
| Two time scales | 5% | 14 | 0.75 | yes | 50% | 0.0000 |

Temporal agreement is a useful conservative filter in this probe. Under noisy
conditions it rejects cases whose validation ranking changes across the
interval, and all retained cases in the small accepted subsets have zero
observed regret. The result is not yet sufficient for a statistical guarantee:
coverage is low, several cells contain only two realizations, and agreement can
still occur for a poorly extrapolating mechanism. The P4 implementation should
therefore expose the joint score as an auditable abstention diagnostic and
report its risk--coverage curve, rather than silently treating it as a solver
certificate.
## Strong-baseline comparison on the external elastomer curves

The same four curves were compared with two conventional relaxation baselines:
a stretched-exponential model and a nonnegative three-mode Prony series. All
models used the first 75% of each extracted curve for fitting and the last 25%
for terminal evaluation.

| Method | Mean terminal relative error | Best on curves |
|---|---:|---:|
| Exponential mixture | 0.7060 | 0/4 |
| Distributed-order fractional | 0.3578 | 0/4 |
| Fractional-tempered | 0.3522 | 0/4 |
| Stretched exponential | 0.00370 | 4/4 |
| Nonnegative 3-mode Prony | 0.0293 | 0/4 |

This is an important negative boundary for P4. On these particular elastomer
records, the stretched-exponential relaxation law is far better than the
current P4 candidate family, so these curves do not support a P4 application
advantage. The experiment prevents an overbroad claim and redirects the next
external validation toward datasets with independently reported anomalous or
fractional memory evidence, where mechanism selection is scientifically
motivated rather than imposed by the candidate library.
## Final targeted anomalous-diffusion validation and route decision

The final targeted test used the locally archived public anomalous-diffusion
trajectories. For each dataset, the observable was the normalized ensemble
MSD divided by lag, which provides a compact relaxation-like diagnostic for
long-memory motion. Two hundred eligible trajectories were used per dataset;
the first 80% of the 50-point observable was used for fitting and the last 20%
for terminal evaluation.

| Dataset | Best P4 error | Best traditional error | P4/traditional ratio | P4 win |
|---|---:|---:|---:|:---:|
| H-actin | 0.9977 | 0.0061 | 164.2x | No |
| Brownian | 0.9976 | 0.0050 | 200.0x | No |

The preregistered exit rule required P4 to win on at least one dataset and to
remain within 2x of the traditional best on the other. It failed decisively.
This result is not evidence that differentiable fractional mechanisms are
useless; it shows that the current P4 candidate parameterization is not a
competitive direct model for these trajectory-derived observables. The main
P4 route is therefore frozen as an independent research line.

The retained contribution is narrower and useful as infrastructure: candidate
mechanism diagnostics, bootstrap stability, temporal agreement, and explicit
abstention can serve DFSC/P3 as an optional audit layer. Future work should
only reopen an independent P4 paper after a new observable and a new candidate
family pass the same external-data exit rule.
