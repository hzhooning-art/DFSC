# P5 Submission Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen P5 statistical evidence, public-task provenance, literature integrity, bilingual manuscript claims, and reproducibility assets for journal submission.

**Architecture:** Extend the existing `p5_memory_protocol` interfaces through standalone, versioned experiment drivers and machine-readable result records. Keep calibration, held evaluation, public-task audits, citation verification, and manuscript generation separable so each claim can be independently rejected or reproduced.

**Tech Stack:** Python 3.10+, NumPy, SciPy, Matplotlib, unittest, LaTeX/Tectonic, JSON, PowerShell.

**Spec:** `P5/docs/superpowers/specs/2026-08-26-p5-submission-hardening-design.md`

## Global Constraints

- Preserve all negative and mixed results.
- Never tune thresholds on held evaluation seeds.
- Do not claim universal calibration or mechanism validation.
- Do not modify P1--P4 files.
- Keep English and Chinese manuscript claims synchronized.

---

### Task 1: Large-sample calibration and transfer audit

**Files:**
- Create: `P5/experiments/probe_submission_calibration_transfer.py`
- Create: `P5/tests/test_submission_calibration_transfer.py`
- Create: `P5/results/submission_calibration_transfer.json`
- Create: `P5/results/submission_calibration_transfer.md`

**Interfaces:**
- Consumes: `p5_memory_protocol.evaluate`, `fit`, `identifiability_certificate`, and `report`.
- Produces: versioned calibration/evaluation rows, Wilson intervals, transfer matrix, and fixed-threshold decisions.

- [ ] Write tests asserting disjoint seed sets, at least 40 evaluation seeds per class, four noise generators, and bounded confidence intervals.
- [ ] Run the new test and verify it fails because the artifact is absent.
- [ ] Implement simulation, fixed-threshold calibration, Wilson intervals, and transfer evaluation.
- [ ] Generate the JSON/Markdown artifacts and rerun the test.

### Task 2: Correlated-noise and grouped-unit robustness

**Files:**
- Create: `P5/experiments/build_submission_robustness_audit.py`
- Create: `P5/tests/test_submission_robustness_audit.py`
- Modify: `P5/results/statistical_robustness_audit.json`

**Interfaces:**
- Consumes: Task 1 output and the three public-task result records.
- Produces: ordinary/AR-profile decision disagreement, block-resampling sensitivity, independent-unit declarations, and leakage warnings.

- [ ] Write tests for public-task independent-unit declarations and correlated-noise sensitivity fields.
- [ ] Run tests and verify the expected missing-field failures.
- [ ] Implement the audit without altering frozen primary decisions.
- [ ] Run focused and full tests.

### Task 3: Public-task provenance and candidate positive task

**Files:**
- Create: `P5/PUBLIC_TASK_PROVENANCE_AUDIT.md`
- Create: `P5/results/public_task_provenance_audit.json`
- Optionally create a new experiment only if a fourth task passes provenance and design gates.

**Interfaces:**
- Consumes: source dataset metadata, preprocessing scripts, and public-task protocols.
- Produces: source/license/unit/split/leakage matrix and an include/refuse decision for a fourth task.

- [ ] Audit all current public tasks against source, license, unit, split, and leakage fields.
- [ ] Search primary dataset sources for a second defensible positive domain.
- [ ] Include it only after predeclaring the protocol; otherwise record the refusal rationale.

### Task 4: Citation and prior-work integrity

**Files:**
- Modify: `P5/paper/references.bib`
- Modify: `P5/RELATED_WORK_MATRIX.md`
- Create: `P5/references/CITATION_VERIFICATION_2026-08-26.md`

**Interfaces:**
- Consumes: DOI/Crossref/publisher/arXiv metadata and P1--P4 public status.
- Produces: verified bibliography and explicit contribution boundary.

- [ ] Verify every existing reference against a primary bibliographic source.
- [ ] Add only sources needed for the five declared related-work categories.
- [ ] Record P1--P4 overlap and disclosure decisions.
- [ ] Remove or quarantine unverifiable records.

### Task 5: Bilingual manuscript revision

**Files:**
- Modify: `P5/paper/manuscript_en.tex`
- Modify: `P5/paper/manuscript_zh.tex`
- Create: `P5/STAGE66_SUBMISSION_HARDENING_REPORT.md`

**Interfaces:**
- Consumes: Tasks 1--4 verified outputs.
- Produces: claim-equivalent English/Chinese manuscripts and a revision report.

- [ ] Update abstract, methods, results, discussion, prior-work boundary, limitations, and availability statements.
- [ ] Ensure all quantitative claims are generated from frozen JSON outputs.
- [ ] Keep target-journal formatting generic until a venue is selected.

### Task 6: Final verification

**Files:**
- Modify only generated PDFs/logs and the freeze manifest.
- Create: `P5/results/submission_freeze_manifest.json`

**Interfaces:**
- Consumes: all source, result, bibliography, and manuscript files.
- Produces: test, JSON, bibliography, compile, and layout verification evidence.

- [ ] Run the complete unit-test suite.
- [ ] Parse every JSON result.
- [ ] Compile both manuscripts with Tectonic.
- [ ] Scan logs for undefined references, missing citations, overfull boxes, and missing glyphs.
- [ ] Render both PDFs and visually inspect representative pages.
- [ ] Record hashes and exact verification commands in the freeze manifest.

