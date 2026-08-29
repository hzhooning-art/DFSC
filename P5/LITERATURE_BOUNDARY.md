# Literature Boundary for the New P5

This note records the novelty boundary used to choose the new direction. It is not a
systematic review and must be expanded before a manuscript is drafted.

## Work that already exists

1. Data-driven Mori--Zwanzig methods already learn Markov and memory operators from
   reduced observations. Lin et al., *SIAM Journal on Applied Dynamical Systems*
   (2021), DOI: https://doi.org/10.1137/21M1401759.
2. Regression-based projections already support linear, polynomial, spline, and neural
   approximations of Mori--Zwanzig operators. Lin et al., *SIAM Journal on Applied
   Dynamical Systems* (2023), DOI: https://doi.org/10.1137/22M1506146.
3. Memory kernels in generalized Langevin equations can be learned with regularized
   Prony estimation and RKHS regression, with an identifiability analysis. Lang and
   Lu, *SIAM Journal on Mathematics of Data Science* (2026), DOI:
   https://doi.org/10.1137/24M1651101.
4. Constrained neural kernel discovery is already demonstrated for sparse/noisy 1D
   and 2D integro-differential equations. Tleubek and Faroughi, arXiv:2607.11110,
   https://arxiv.org/abs/2607.11110.
5. Latent non-Markovian feature embeddings and extended Markovian dynamics are also
   established ideas; therefore, a generic latent-state learner is not a sufficient
   novelty claim.

## Proposed claim boundary

The defensible P5 question is narrower and more auditable:

> Discover the smallest observation-supported latent memory realization shared by a
> high-dimensional resolved state, and refuse a physical memory-rank claim when
> finite noisy observations do not identify that realization.

The intended increment is the combination of:

- minimal latent memory-rank selection rather than unrestricted kernel fitting;
- shared pole locations across many observed channels;
- an explicit evidence/refusal decision tied to identifiability diagnostics; and
- compilation of the accepted realization into differentiable auxiliary states.

Each ingredient has precedents in neighboring fields. A publishable contribution
requires showing that their joint formulation solves a scientific inference problem
that existing kernel-learning or latent-dynamics methods do not address. Claims of
being the first must not be made until a full database search is completed.

