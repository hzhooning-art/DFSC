# P5 Submission Hardening Design

## Objective

Raise the P5 manuscript from an internally reproducible feasibility paper to a submission-ready statistical-computational methods paper without broadening its claims beyond shared finite sums of positive real decays on grouped observations.

## Evidence design

The existing four-seed held evaluation is replaced as the primary calibration claim by a larger, disjoint experiment. Calibration and evaluation seeds remain non-overlapping. The experiment reports Wilson intervals for separated-support, coalesced-refusal, false-acceptance, and false-refusal rates. It additionally freezes the calibration threshold on one declared noise generator and evaluates transfer under independent Gaussian, AR(1), AR(2), and heteroscedastic noise.

The threshold remains design conditional. A transfer failure is retained as a scope result and cannot be repaired by tuning on evaluation seeds. Correlated-noise sensitivity is evaluated with ordinary BIC, AR(1)-profile BIC, and a block-resampling decision audit; these are sensitivity analyses rather than universal corrections.

## Public-data design

The three existing public tasks retain their current roles: PVA is the positive physical support task, gas recovery is a coalescence refusal, and hydraulic cooling is a transfer/refusal stress test. Each task must state its independent experimental unit, grouping rule, split unit, and leakage controls. A fourth public task may be added only if its source, license, independent units, preprocessing, and predeclared gates can be verified. Failure to find a defensible task is reported rather than hidden.

## Manuscript design

The paper will add an explicit prior-work boundary. Public P1 work is cited when it supplies concepts, code, or software lineage; unpublished P2--P4 material is described only when disclosure is necessary to distinguish reused assets. The theorem remains an independent-Gaussian impossibility statement. The empirical normalized boundary index is described as a diagnostic proxy, and the conditional refusal corollary is not presented as a verified global guarantee.

Related work will be expanded only with verified primary sources covering exponential-sum identifiability, shared-pole estimation, correlated-noise model selection, abstention/refusal, and memory-kernel learning. Every bibliographic record used in the manuscript must have a verification trail.

## Reproducibility design

Generated JSON records include the experiment version, seed partitions, noise generator, thresholds, uncertainty intervals, and claim boundary. A freeze manifest records file hashes and the Git commit when available. Repository release and archival DOI remain separate release operations and must not be represented as complete before external verification.

## Acceptance criteria

- At least 40 held evaluation seeds per principal separated/coalesced class.
- Wilson intervals and error rates are present in machine-readable output.
- Calibration/evaluation seed sets are disjoint and tested.
- At least four declared noise generators are evaluated without threshold retuning.
- Public-task independence and leakage statements are explicit.
- Every cited reference is verified or removed.
- English and Chinese manuscripts remain claim-equivalent and compile without unresolved references.
- Full unit tests pass and every JSON result parses.

