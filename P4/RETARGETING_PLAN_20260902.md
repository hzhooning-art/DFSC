# P4 Retargeting Plan

Updated: 2026-09-02

## Decision history

The Computer Standards & Interfaces submission was desk rejected because the editor did not find the level of scientific significance required by that journal. The submitted English and Chinese sources and PDF are frozen under `archive/rejected_CSI_20260902/`. The active manuscript in `paper/` has been returned to a journal-neutral form.

This decision is treated as a positioning failure, not as evidence that the implementation, numerical results, or mutation records are invalid.

## Scientific identity

P4 is a software-quality and testing study for differentiable numerical components. Its strongest evidence is the executable oracle structure, mutation campaign, interface-equivalence checks, scope freezing, and machine-readable qualification record. Standards mapping is supporting context rather than the central contribution.

## Target sequence

### 1. Primary: Journal of Systems and Software

The active route is now JSS. The paper must be framed as an empirical software-
testing study for differentiable numerical components: testing-strategy
increments, executable oracles, interface-equivalence properties, historical-
defect relevance, and cross-project validity are primary. Standards mapping is
supporting context only. Submission remains gated on broader independent SUT
coverage and stronger historical-defect evidence; Stage 6 supplies one real-
defect-derived strategy comparison but not an executed buggy release.

### 2. Backup: Software Quality Journal

Fit: the journal explicitly covers software measurement and metrics, software testing, quality-assurance techniques, technical aspects of quality, and internal or external quality standards. Its hybrid publishing model permits the standard subscription route without selecting paid open access.

Required revision:

- formulate explicit software-testing research questions;
- make mutation score, false rejection, oracle independence, coverage adequacy, and interface equivalence the main outcomes;
- compare the method with ordinary unit tests, tolerance-only numerical tests, and at least one metamorphic or property-based testing baseline;
- separate the reusable testing method from the four scientific-computing subjects used as cases;
- retain the ISO/IEEE mapping as one related-work subsection rather than the abstract-level motivation.

### 3. Stretch: Software Testing, Verification and Reliability

This journal directly covers new testing and verification criteria, measurement, non-functional testing, reliability, and evaluations of open-source testing tools. It is a stronger thematic match but currently reports a very selective acceptance rate. Submission is justified only after adding a stronger comparative evaluation and, preferably, defects or implementations produced independently of the authors.

## Stop rule

Do not submit P4 again after only changing the title, abstract, and cover letter. A new package is permitted only when the baseline comparison and software-testing research questions have been added and the manuscript no longer relies on standards mapping as its novelty claim.
