# Positive hereditary simulation reproducibility artifacts — 1.0.0

This release supplies the complete numerical evidence and reproduction code for
"Loading-Dependent Predictability and Precision Assessment for Positive Hereditary Simulations".
It is a dedicated synthetic simulation release, not an upgrade of the earlier
contract-testing software and not a claim of journal acceptance.

Download **Hereditary_Simulation_Reproducibility_v1.0.0_20261014.zip** from the
release assets. The automatically generated GitHub "Source code" archives are
not the full evidence package: large saved solver witnesses are distributed in
the explicit release asset, not in Git history.

ZIP size: **169,492,920 bytes**.

SHA-256:

```text
b61b71fffcd469b735e7c4e2b5b9a38a1e19beaeaab21fb7ef2b2ced93b2433a
```

The archive retains all 110 research payload files from the complete submission
supplement, including synthetic observations, frozen protocols, complete saved
primal/dual witnesses, per-case outcomes, audit reports, code, figure assets,
and direct-fitting failure records. A non-research manuscript-attachment layout
QA receipt is omitted. Packaging added no new random cases or scientific results.

Extract into a fresh directory and run:

```text
python verify_archive.py
```

Then follow the archive README. Its requirements describe the recorded
Python/NumPy/SciPy environment; dependency environments are not bundled.
Existing result directories are overwrite-guarded. Full audit/refitting should
be performed in a separate extraction; preserve the original ZIP.

Preparation checks passed on an isolated extraction: all 114 manifest-bound
files, five arithmetic/interface tests and replay of 1422 successful fits.
The 256 saved synthetic cases form 512 paired-window records. Grid and
regularized fits meet the fixed point-error tolerances in every case; all
114 Prony final-refit nonconvergence records remain. No universal accuracy,
runtime, calibrated fitting-interval or material-validity advantage is claimed.
These are developer-run checks, not independent human validation or a
global-optimum certificate.

The source snapshot is provided on `codex/hereditary-simulation-v1.0.0`.
Download the explicit ZIP for all numerical witnesses. Internal frozen paths
remain unchanged for executable compatibility; they are not the public title.
The authors' code and accompanying artifacts retain the existing MIT License.
Third-party dependencies retain their own licenses. This release has no newly
claimed DOI; the earlier software archive DOI is not its identifier.
