# P5 Research Direction

## Working title

**Evidence-Controlled Discovery of Minimal Latent Memory Realizations for
Non-Markovian Scientific Dynamics**

## Core question

Given finite, noisy observations of one or more resolved variables, can we recover
the smallest latent Markovian realization that explains their memory, quantify when
that realization is identifiable, and refuse a mechanistic conclusion when the data
cannot distinguish memory orders?

## Mathematical object

For a completely monotone memory kernel,

\[
K(t)=\sum_{j=1}^{m} w_j e^{-r_j t},\qquad w_j>0,\;r_j>0,
\]

the nonlocal dynamics can be lifted to auxiliary states

\[
\dot{x}=-\sum_{j=1}^{m}w_j z_j,\qquad
\dot{z}_j=x-r_jz_j.
\]

The integer `m` is the latent memory rank. P5 treats `m` as a scientific quantity
that must be earned by evidence, rather than a hyperparameter chosen for fit alone.
For field data, channels may have channel-specific residues but share pole locations,
which represents a low-dimensional material memory spectrum acting on a
high-dimensional state.

## Intended contributions

1. **Minimal-memory discovery:** select the smallest supported pole-residue
   realization rather than merely fitting a flexible kernel curve.
2. **Identifiability-aware refusal:** combine held-out prediction, information
   criteria, Jacobian conditioning, and resampling stability; return
   `INSUFFICIENT_EVIDENCE` when nearby poles or short horizons make the memory rank
   non-identifiable.
3. **Shared-spectrum field inference:** infer common memory time scales jointly from
   many spatial or modal channels while allowing channel-dependent coupling
   strengths.
4. **Executable closure:** compile the discovered kernel into differentiable
   auxiliary ODE states, avoiding explicit history storage during deployment.
5. **Numerical-error-aware evidence:** later integrate P2 error budgets and P4
   reliability records so optimizer or evaluator error cannot masquerade as a new
   memory mode.

## Boundary from prior work

P5 does not claim that memory-kernel learning, Mori--Zwanzig closure, rational
approximation, Prony fitting, or constrained kernel networks are new individually.
The proposed research claim is the joint problem of minimal latent memory rank,
multi-channel shared realization, and evidence-controlled refusal under finite noisy
observation. This boundary must be re-audited before paper submission.

## Relationship to P1--P4

- **P1:** supplies differentiable fractional propagation primitives.
- **P2:** supplies value-gradient error diagnostics for adaptive evaluation.
- **P3:** supplies selective correction and abstention logic at task level.
- **P4:** supplies a backend-independent reliability protocol and audit record.
- **P5:** uses those ideas for scientific mechanism discovery: infer a minimal memory
  realization only when its order and parameters are identifiable.

## Feasibility milestones

### M1: Scalar rank recovery

Recover one- and two-mode positive memory systems from sparse noisy trajectories.

### M2: Refusal boundary

Map observation horizon, noise, sampling density, and pole separation to regions of
correct recovery, over-selection, and refusal.

### M3: High-dimensional shared memory

Show that joint inference across 16--256 channels recovers shared pole locations more
reliably than fitting each channel independently, while scaling with memory rank
rather than retained history length.

### M4: Out-of-class detection

Reject oscillatory, signed, or nonstationary kernels when the positive-real model
class is inadequate; do not reinterpret model mismatch as additional memory rank.

### M5: Scientific benchmark

Validate on at least one public dataset with a defensible memory mechanism and an
independent held-out protocol. Candidate domains include viscoelastic relaxation,
single-particle anomalous diffusion, or reduced molecular dynamics.

## Exit criteria

The route should be stopped or redesigned if, after controlled tuning:

1. separated rank-1/rank-2 systems are correctly classified in less than 80% of
   repeated trials;
2. near-coincident poles are confidently over-interpreted instead of refused in more
   than 20% of trials;
3. joint multi-channel inference does not materially improve rank or pole recovery
   over independent fits;
4. the inferred realization fails held-out long-horizon prediction despite good
   in-window fit; or
5. literature review finds a prior method that already combines the same minimal-rank,
   shared-spectrum, and refusal contract with comparable evidence.

