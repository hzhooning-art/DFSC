# P4 Forcing-as-Input Gate

This gate tested whether P4 could directly accept a sampled forcing function
and learn a parameter-conditioned operator on top of the P1 forced MLSL
primitive.

## Outcome

The gate is currently **blocked by numerical stability**, before any learning
metric is reported. On the RTX 5070 run, the forced MLSL backbone produced a
maximum absolute value of approximately `1.58e68`. Values were finite, but are
outside any credible numerical regime for this experiment. The earlier draft
of the script produced apparently excellent hybrid scores only because both
the target and hybrid reused this same unstable backbone; those scores have
been discarded.

## Interpretation

This is not evidence against the revised P4 hypothesis. It identifies a
necessary dependency: forcing-input P4 experiments require a validated forced
Mittag-Leffler evaluator regime, error control, and a bounded forcing protocol
before operator-learning comparisons are meaningful.

The finding strengthens the boundary between P1/P2 and P4:

- P1 supplies the differentiable spectral propagation primitive;
- P2 must certify forced propagation stability and reject unsafe regimes;
- P4 can only claim forcing-conditioned family learning inside that certified
  regime.

## Required next test

Before adding a forcing-input neural baseline, calibrate the forced primitive
over quadrature size, fractional order, query time, and modal rate. The test
must compare against a high-precision scalar/modal reference and implement an
abstention rule when the forced backbone exceeds a numerical envelope.

Result file: `P4/results/p4_forcing_function_operator.json`.
