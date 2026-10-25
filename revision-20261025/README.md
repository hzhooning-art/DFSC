# DFSC numerical-admission and CPU-repair research snapshot

This directory contains the software/adapter source for the 25 October 2026 revision.
It does not replace the public dfsc 0.1.0 package or change historical releases.
Optional distributions retain their documented development versions and scopes.

Download the exact complete artifact from:
https://github.com/hzhooning-art/DFSC/releases/tag/dfsc-revision-20261025

Archive: DFSC_Revision_Reproducibility_20261025.zip
SHA-256: 3b755fce55d5b44e0a0489eda323b8524a1ca177eebce0e50b18e29fb1d72b79

The archive supplies the exact current paper/source, frozen protocols, all 583
recovery terminal records, original adverse history, explicit same-task I/O retry,
18 complete paired 100-update trajectories, and recorded 127-test final regression.
Source files here are a browsable subset; numerical reproduction uses the full
archive layout and recorded dependencies. After extraction:

    python -B P1/tools/verify_targeted_evidence.py

That standard-library command verifies integrity and paired saved evidence without
training or claiming a new dependency installation. See the included README and
REPRODUCE for installed-backend and original-experiment instructions. Shared-host
timings are not uncontended performance evidence. No global certificate, GPU
repair, general nonnormal/forced solver, new material replicates or universal
predictive superiority is claimed. Third-party licenses are retained in the archive.
