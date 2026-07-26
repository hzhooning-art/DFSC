# Changelog

All notable changes to `dfsc` are recorded here. The project follows semantic
versioning once public releases begin.

## 0.1.0rc1 - Unreleased

### Added

- Differentiable Mittag-Leffler spectral propagation with trainable orders.
- Dense, generalized, graph, Krylov, Arnoldi, forced, semilinear, and history-aware paths.
- Reliability reports, algorithm selection, application templates, and a SciML-style solve interface.
- CPU/GPU checks, 1D/2D boundary constructors, real-data protocols, and external solver benchmarks.
- CC BY 4.0 GeoTES cross-cycle forced-response benchmark.
- CC BY 4.0 heated-steam condition-OOD benchmark.
- Strict series remainder certificates, identifiability diagnostics, and reusable prepared Krylov bases.

### Known limitations

- The validated direct-propagator domain is narrower than a general fractional solver.
- Variable-order and distributed-order wrappers remain experimental.
- Public package, source, documentation, and archival releases are pending.
