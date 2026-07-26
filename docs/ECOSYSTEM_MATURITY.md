# dfsc Ecosystem Maturity

`dfsc` is developed as a specialized tool-algorithm library for differentiable
Mittag-Leffler spectral learning. Its design borrows the problem/algorithm
separation common in mature scientific-computing ecosystems, while keeping a
strict scope: known or user-supplied spectral propagators with differentiable
fractional parameters.

## Current Maturity Layer

Passing the repository audit means the research artifact is internally
consistent; it does not mean that dfsc has reached the release maturity or
equation coverage of long-lived public solver ecosystems.

| Layer | Current state | Remaining gap |
|---|---|---|
| Package identity | Single public namespace: `dfsc` | Public release and hosted docs |
| Internal release gate | 20/20 pre-release checks pass | External adoption and public history |
| Core primitive | Batched PyTorch real-negative MLSL plus controlled moderate-radius complex evaluation | Large complex sectors and certified complex error control |
| Algorithms | Direct, stable, forced, Lanczos, controlled Arnoldi, FFT Caputo-L1 history, operator, graph, generalized operator, semilinear Picard, and L1 fallback | Contour/rational methods and online fast-memory algorithms |
| Problem interface | Spectral, graph, assembled operator, forced, semilinear, linear Caputo L1, and four scoped application templates | Standardized inverse and variable-order problem templates |
| Solution object | Values, time axis, retcode, warnings, stats, and selection diagnostics | Serialization and richer error estimates |
| Numerical routing | `AutoDFSC` and diagnostic Mittag-Leffler evaluation | Calibrated rules over wider verified regimes |
| Autograd/GPU | Tested for alpha/beta, matrix-free, nonlinear, and FFT-history gradients on CUDA | Larger multi-GPU and mixed-precision profiling |
| Reliability contract | Validated-domain, convergence, gradient, and empirical error metadata | Rigorous global error bounds |
| Data evidence | One experimental H-actin SPT condition with provenance, checksum, five trajectory splits, and model comparisons | Explicit redistribution permission and independent physical modalities |
| Experimental orders | Variable-order and distributed-order wrappers | Systematic theory and large-scale validation |

## Intended Differentiation

`dfsc` is not yet a general fractional solver library. Its primary path remains
Mittag-Leffler spectral propagation, while the constant-order linear Caputo L1
solver establishes the first explicit history-aware fallback path.

The main differentiator is not solver breadth. It is the ability to expose that
propagator as a differentiable layer, algorithm object, and solution-producing
workflow inside PyTorch.

The current center of application coverage is anomalous transport and linear
fractional relaxation. Around that center, the same primitive now supports
assembled finite-element systems, graph diffusion, and controlled
non-self-adjoint advection--diffusion. Each template carries machine-readable
assumptions and limitations rather than implying general domain coverage.

## Next Maturity Milestones

1. Resolve redistribution permission for the integrated SPT data and add a
   second benchmark with an explicit open-data license.
2. Add an ecosystem comparison script against at least one PINN/fPINN workflow
   and one fractional numerical solver on the same task.
3. Add block Krylov/Arnoldi and preconditioning strategies for larger batches and stiff spectra.
4. Add contour/rational complex actions and an online fast-memory CQ/SOE time stepper before promoting broader solver claims.
5. Promote experimental order wrappers only after accuracy and stability tests.
6. Publish a versioned source archive, wheel, and hosted API documentation.
