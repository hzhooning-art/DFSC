# Related Work and Novelty Boundary

## Search scope

Searches were run on 2026-08-25 across publisher pages and DOI records using
combinations of: `memory-kernel learning`, `Prony identifiability`, `shared
relaxation spectrum`, `stress-relaxation spectrum identification`, `model
order selection`, `held-out prediction`, and `evidence refusal`. Only records
whose title, authors, venue, and DOI could be checked on a publisher or
repository page are retained below.

## Verified literature matrix

| Work | Established contribution | Direct overlap with P5 | Boundary relative to P5 |
|---|---|---|---|
| Lang and Lu (2026), *Learning Memory Kernels in Generalized Langevin Equations*, SIAM J. Math. Data Sci. 8, 141-166, [doi:10.1137/24M1651101](https://doi.org/10.1137/24M1651101) | Learns GLE memory kernels using regularized Prony correlation estimates and RKHS/Sobolev regression; supplies an error-control argument. | Memory-kernel learning, Prony structure, identifiability. | Estimates a kernel from trajectories. P5 instead asks for the minimum shared finite realization supported across independent specimens and permits an `INDETERMINATE` output when predictive and stability gates disagree. |
| Ge, Zhang, and Lei (2024), *Data-Driven Learning of the Generalized Langevin Equation with State-Dependent Memory*, Phys. Rev. Lett. 133, 077301, [doi:10.1103/PhysRevLett.133.077301](https://doi.org/10.1103/PhysRevLett.133.077301) | Jointly learns state features and state-dependent non-Markovian coupling for reduced molecular dynamics. | Data-driven non-Markovian reduced models. | Targets expressive state-dependent memory and kinetic prediction, not finite-window rank support, cross-specimen shared spectra, or refusal boundaries. |
| Lei, Baker, and Li (2016), *Data-driven parameterization of the generalized Langevin equation*, PNAS 113, 14183-14188, [doi:10.1073/pnas.1609587113](https://doi.org/10.1073/pnas.1609587113) | Builds a hierarchy of GLE models through rational approximation and auxiliary variables. | Finite-dimensional realizations of memory. | Provides parameterization and simulation machinery; it does not define a joint evidence rule for selecting the smallest supported realization from sparse independent experiments. |
| Stampfer and Plonka (2020), *The Generalized Operator Based Prony Method*, Constructive Approximation 52, 247-282, [doi:10.1007/s00365-020-09501-6](https://doi.org/10.1007/s00365-020-09501-6) | Generalizes Prony recovery to sparse eigenfunction expansions and alternative sampling operators/functionals. | Exponential realization recovery and identifiability. | Gives exact-model reconstruction theory and discusses numerical instability. P5 concerns noisy finite-window evidence, held-specimen transfer, and operational refusal rather than exact recovery alone. |
| Gustavsen and Semlyen (1999), *Rational approximation of frequency domain responses by vector fitting*, IEEE Trans. Power Delivery 14, 1052-1061, [doi:10.1109/61.772353](https://doi.org/10.1109/61.772353) | Introduces vector fitting for stable rational approximation of frequency responses. | Shared poles and rational realizations. | Primarily an approximation algorithm. P5 does not compete with vector fitting as an optimizer; it evaluates whether an inferred shared realization is statistically and predictively supported. |
| Stankiewicz (2023), *Two-Level Scheme for Identification of the Relaxation Time Spectrum Using Stress Relaxation Test Data with the Optimal Choice of the Time-Scale Factor*, Materials 16, 3565, [doi:10.3390/ma16093565](https://doi.org/10.3390/ma16093565) | Identifies a regularized relaxation spectrum from stress-relaxation measurements and optimizes a time-scale factor. | Stress-relaxation data, spectrum recovery, regularization. | Focuses on accurate spectrum reconstruction. P5 adds cross-specimen shared-rate validation, early/late prediction, explicit rate-separation and fold-stability gates, and refusal. |
| Schelkow et al. (2026), *Stress Relaxation Test Dataset of Cylindrical PVA Gel Polymer Electrolyte Samples*, Zenodo, [doi:10.5281/zenodo.21333840](https://doi.org/10.5281/zenodo.21333840) | Publishes raw time, force, and displacement for three specimens and three cycles per specimen under displacement-controlled compression relaxation. | Independent public observations used in Stage 62. | This is the data source, not a competing inference method. Stage 62 preserves its raw workbook and checksum and performs no smoothing. |
| Ziyatdinov et al. (2015), *Data set from gas sensor array under flow modulation*, Data in Brief 3, 131-136, [doi:10.1016/j.dib.2015.02.016](https://doi.org/10.1016/j.dib.2015.02.016) | Publishes 58 independent five-minute measurements from 16 metal-oxide sensors under controlled respiratory flow modulation. | Independent public recovery task used in Stage 63. | The original work studies signal features for chemical sensing. P5 uses only the registered room-air recovery interval to test cross-batch shared-relaxation support and refusal. |
| Ziyatdinov et al. (2015), *Bioinspired early detection through gas flow modulation in chemo-sensory systems*, Sensors and Actuators B 206, 538-547, [doi:10.1016/j.snb.2014.09.001](https://doi.org/10.1016/j.snb.2014.09.001) | Establishes the 16-sensor ventilated acquisition system and low-/high-frequency cycle features for early gas detection. | Supplies the acquisition physics and registered 5-breath/min cycle. | It does not perform shared recovery-rank inference or define predictive/stability/separation refusal gates. |
| Bosner (2021), *Parallel Prony's method with multivariate matrix pencil approach and its numerical aspects*, SIAM J. Matrix Anal. Appl. 42(2), 635-658, [doi:10.1137/20M1343658](https://doi.org/10.1137/20M1343658) | Develops multivariate matrix-pencil/Prony reconstruction and analyzes numerical aspects. | Supplies a recognized exponential-identification baseline family. | Stage 63 uses a conservative Prony recurrence diagnostic under the same early/late split; P5's contribution is the joint evidence/refusal contract rather than a new pencil algorithm. |

## Search-based novelty assessment

The components of the Stage 62 procedure are individually established: finite
exponential/Prony representations, rational realizations, relaxation spectrum
identification, information criteria, cross-validation, and non-Markovian
kernel learning. The search did **not** identify a verified work that combines:

1. positive shared rates across independent specimens with specimen-specific amplitudes;
2. minimal-rank comparison using both information and held-specimen prediction;
3. explicit fold-stability and rate-separation gates;
4. an `INDETERMINATE` outcome for unsupported mechanism-level rank claims; and
5. an executable horizon-by-budget boundary map on unmodified public raw data.

This scoped search is not proof that no adjacent implementation exists. The
defensible claim is a **joint evidence-and-refusal protocol for shared finite
memory realizations**, not a new Prony method, a new constitutive law, or a
universal memory-kernel learner.

## Related Work draft

Finite sums of exponentials connect hereditary dynamics to finite-dimensional
state-space realizations. Classical and generalized Prony methods recover
sparse exponential or eigenfunction expansions, while vector fitting builds
rational approximations with shared poles. These methods supply the
representation and recovery machinery used here and document the sensitivity
of weak or closely spaced modes. Our objective is complementary: to determine
whether a finite noisy experiment supports a particular minimal shared
realization.

Data-driven non-Markovian modeling has progressed from rationally
parameterized generalized Langevin equations to state-dependent memory and
regularized kernel estimators. Recent work combines regularized Prony estimates
with function-space regression and error control. Those studies mainly optimize
kernel estimation or reduced-model prediction. P5 instead asks what realization
order is jointly supported by information gain, external prediction, and
cross-fold stability, and when no rank claim should be made.

Stress-relaxation spectrum identification is an established inverse problem in
viscoelasticity. Stage 62 uses this domain as an external audit rather than as
evidence for a new material model. Rates are shared across specimens,
amplitudes remain specimen specific, the decision thresholds are frozen in
advance, and late-time observations from a held-out specimen provide the
external prediction test.

Stage 63 adds a chemically distinct public task. The gas-sensor source provides
58 independent measurements, 16 sensor channels, five acquisition batches, and
a registered ventilator cycle. P5 excludes air controls from mechanism fitting,
aggregates the recovery signal only within physical cycles, and leaves complete
acquisition batches out. This task therefore tests whether the refusal protocol
survives a different instrument, noise process, and grouping structure rather
than merely repeating the PVA example.
