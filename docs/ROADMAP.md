# dfsc Roadmap

This roadmap separates implemented functionality from research extensions. It
is intended to keep the software claim precise while making the ecosystem
direction reproducible.

## Implemented Core

- PyTorch `dfsc` package as the only public library namespace.
- Differentiable Mittag-Leffler spectral layer for retained real spectra.
- Trainable `alpha` and `beta` through PyTorch autograd.
- 1D/2D tensor-product Dirichlet, Neumann, periodic, and mixed constructors.
- Symmetric PSD operator and graph-Laplacian adapters for user-supplied discretizations.
- Problem--algorithm--solution interface through `dfsc.solve`.
- Automatic direct/stable routing with numerical diagnostics and solution retcodes.
- Generalized stiffness/mass operator adapter with mass-aware projection.
- First history-aware implicit L1 fallback for constant-order linear Caputo systems.
- Fully reorthogonalized Lanczos matrix-function action with an embedded
  subspace-disagreement diagnostic and automatic size-based routing.
- Sparse tensor and matrix-free self-adjoint operator contracts with
  differentiable `matvec` parameters and CUDA validation.
- Direct and FFT Caputo-L1 full-trajectory derivative operators with automatic
  length-based routing, analytic convergence checks, and CUDA gradients.
- Controlled complex Mittag-Leffler series and Arnoldi actions for moderate
  reduced radii, including non-normal diagnostics and complex CUDA validation.
- Semilinear mild-form Picard problem with explicit convergence retcodes.
- Forced-dynamics wrapper, inverse-order workflow, hybrid residual workflow.
- Scoped application cases for anomalous diffusion, assembled linear
  relaxation, network memory diffusion, and periodic fractional
  advection--diffusion.
- CPU/GPU smoke tests, unit tests, and synthetic experiment scripts.
- Single distributable `dfsc` namespace and a 95% pre-release internal gate.
- Numerical reliability reports for real and controlled complex evaluation and solver outputs.

## Experimental Extensions

- `dfsc.VariableOrderMLSL`: direct-query wrapper using one alpha per query time.
- `dfsc.DistributedOrderMLSL`: differentiable quadrature mixture over alpha nodes.
- `dfsc.mlsl_applicability_report`: machine-readable check for the current
  spectral assumptions.
- `dfsc.validate_dataset_manifest`: provenance-first contract for future public
  physical benchmark datasets.

These extensions are useful for early SciML experiments. They do not yet make
dfsc a general variable-order or distributed-order fractional solver library.

## Near-Term Milestones

1. Publish a versioned source release and wheel.
2. Deploy the existing API documentation as a hosted site.
3. Add a second public physical benchmark with an explicit redistribution license.
4. Promote experimental wrappers after accuracy and stability validation.
5. Add compatibility examples with a neural-operator workflow.
6. Add block/preconditioned Krylov/Arnoldi, contour actions, and online fast-memory
   CQ/SOE time-stepping algorithms.

## Longer-Term Research Milestones

- Large-radius complex-spectrum operators with contour/rational error control.
- Variable-order formulations with history-aware validation.
- Distributed-order kernels with calibrated quadrature rules.
- Mesh/discretization adapters for sparse and matrix-free irregular-domain workflows.
- Optional JAX or Julia backend interoperation.
