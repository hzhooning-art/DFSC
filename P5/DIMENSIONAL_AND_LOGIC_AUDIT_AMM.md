# AMM Dimensional and Logical Consistency Audit

Audit date: 2026-08-29

## Dimensional checks

1. In the finite-memory model, the response, offset, and amplitudes have response units; `exp(-lambda t)` is dimensionless because `lambda` has inverse-time units.
2. Ordered rates are parameterized through `log(lambda/lambda_ref)`, so every logarithm has a dimensionless argument.
3. Adjacent-rate separation uses a ratio of rates and is dimensionless.
4. NRMSE divides a response-norm error by a response-norm scale. Its stabilizing epsilon is declared in the same units before division; registered tasks are dimensionless.
5. BIC is written as `n log(SSE/(n s_y^2)) + q log n`, with the fixed response scale `s_y` giving the logarithm a dimensionless argument. All registered responses use `s_y=1` after normalization.
6. The two-point bound uses a response difference divided by a noise standard deviation, so its testing distance is dimensionless. Rate separation times observation time is likewise dimensionless.
7. The local sensitivity proxy differentiates with respect to normalized log rates and divides by residual noise; sample-count normalization removes leading design scaling. The text explicitly treats this quantity as a calibrated task-internal proxy, not a universal estimator of the theoretical distance.
8. Partial sharing is expressed as `log(lambda_gk/lambda_ref)=mu_k+delta_gk`; both random effects are dimensionless.
9. Conformal scores, ranks, false-alarm rates, information-criterion differences, and boundary indices are dimensionless.

## Logical chain

1. **Reality gap:** fitted extra poles are often interpreted as physical timescales although finite windows, correlated residuals, and repeated channels may not resolve them.
2. **Mechanism:** adjacent rate sensitivities become nearly collinear as rates coalesce, causing local information loss.
3. **Method:** support requires agreement among information gain, held-unit transfer, foldwise rate stability, and local separation.
4. **Theory:** a two-point testing bound and a local sensitivity-rank condition establish design-conditional resolution limits.
5. **Evidence:** known-rank controls test the decision rule; material data test its intended use; chemically and mechanically distinct public tasks test refusal behavior; controlled extensions probe individual model assumptions.
6. **Output:** the method returns the smallest supported finite realization or an explicit unresolved state with reasons.

The theoretical contribution is local and design conditional. The manuscript does not claim distribution-free identifiability, global nonlinear-kernel recovery, or microscopic mechanism proof.
