# P4 Evidence Audit

## Supported by current results

1. A backend-independent contract for a batched differentiable map
   `y = P(x; theta)`.
2. Dimension-level reporting for value fidelity, gradients, calibration,
   module reuse, OOD behavior, and long-horizon behavior.
3. A substantive cross-backend audit covering MLSL, matrix-exponential action,
   linear RK4, and Logistic RK4.
4. Five-seed OOD/long-horizon checks for the matrix and linear RK4 backends.
5. Three-seed common neural OOD tests at two target times and two noise levels.
6. Batch/GPU profiles for three generic backends on an RTX 5070 Laptop GPU.
7. A public package API, registry schema, JSON reports, and passing local tests.

## Must be stated as limitations

- MLSL entries marked `reported_prior_validation` are inherited case-study
  evidence, not a newly rerun generic benchmark in this paper.
- The common neural task gives a positive Logistic result and a negative matrix
  result; primitive composition is therefore task-dependent.
- The benchmarks are controlled and low-dimensional. They do not establish
  performance on real physical datasets, complex geometries, unstructured
  meshes, variable-order/distributed-order operators, JAX, or Julia.
- The current local test suite is verified; hosted CI and independent external
  reproduction are pending.
- The protocol is not a universal fractional solver library.

## Additional experiments needed only for a stronger claim

The current evidence is sufficient for a protocol-focused first draft. More
experiments are needed if the paper is repositioned as a broad SciML ecosystem
or application paper:

1. one public physical benchmark with a declared external reference;
2. an independent reproduction on a second machine or GPU;
3. a hosted CI run from a clean environment;
4. a higher-dimensional or non-structured backend;
5. a verified JAX or Julia adapter.

These are scope-expansion tasks, not grounds for inventing stronger claims in
the current manuscript.
