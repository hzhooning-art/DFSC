# Current P4 Status

Updated: 2026-09-04

## Submission route

Primary target: **Journal of Systems and Software**. The active manuscript is
being recast as an empirical software-testing study for differentiable
numerical components. Standards mapping is supporting context rather than the
novelty claim.

## Evidence completed

- Four cumulative strategies detect 3/10, 5/10, 8/10, and 10/10 classes on the
  paired synthetic fault catalogue, with 0/40 clean-record rejections.
- On 240 PyTorch external-interface behavior faults, detection is 25.00%,
  73.75%, 99.17%, and 100%; all 30 clean controls pass.
- The complete strategy detects all 24 subject--fault clusters and retains
  1.000 leave-one-interface-out detection.
- Four current fixed-side historical cases are confirmed across PyTorch and
  SciPy.
- A SciPy #8906-derived comparison shows 0/32 detections for a weak masked
  example and 32/32 for representation equivalence; current SciPy passes all
  32 controls.
- A matched seven-fault benchmark now covers NumPy, SciPy, and PyTorch. Across
  252 injected trials, cumulative detection is 28.57%, 57.14%, 85.71%, and
  100%; all 36 clean controls pass and all 21 clusters are finally detected.
- The English and Chinese manuscripts now include explicit RQ1--RQ3, external-
  subject results, the historical-defect-derived case, and validity limits.
- Stage 8 executes the unchanged frozen runner in the official PyTorch
  1.11.0+cpu wheel and current PyTorch 2.11.0+cu128. The buggy side reproduces
  the reported zero `xlogy(0, 2)` derivative, while the fixed side returns
  `log(2)` with zero stored error. Both roles pass, so the complete historical
  buggy/fixed-pair count became 1 rather than 0.
- Stage 9 executes a second unchanged runner in the official SciPy 1.14.1
  wheel and current SciPy 1.18.0. The old version raises the reported
  `IndexError` for the conventional 1x1 band representation while the padded
  form masks the fault; the current version solves both exactly. The complete
  historical-pair count is now 2 across PyTorch and SciPy.
- Stage 11 executes a third unchanged runner for SciPy #15620. In the official
  1.14.1 wheel, `int16` and `int32` `resample_poly` outputs are silently all
  zero while the float64 reference reaches 3.001552; current SciPy 1.18.0
  matches the reference with zero maximum error. The complete historical
  total is now 3 families across two projects and three failure modes.
- Tectonic 0.17.0 successfully builds the Stage-11 English manuscript (34
  pages), Chinese manuscript (28 pages), and supplement (3 pages). Contact-
  sheet QA found no blank pages, visibly clipped figures, or obvious
  pagination anomalies; extracted text confirms the new pair metrics.
- Stage 10 adds a JSS-specific submission-readiness pack with length-checked
  highlights, synchronized title-page metadata, a scope-focused cover letter,
  declarations, an evidence-to-claim matrix, and a final upload checklist.
  The remaining pre-upload condition is an immutable public release containing
  Stage 11 plus explicit approval and metadata confirmation from every author.

## Blocking limitations

1. Three reported buggy/fixed package pairs have been completed across
   PyTorch and SciPy, covering a wrong derivative, an indexing exception, and
   silent dtype-dependent output. This closes the two-family limitation, but
   three families remain insufficient for a strong field-effectiveness claim.
2. The controlled external benchmark now spans three projects, but each adds
   only one production interface and the NumPy/SciPy arms do not test autodiff.
   Fixed-side historical provenance still spans only two projects.
3. The SciPy strategy comparison still uses 32 source-derived surrogate
   variants, but Stage 9 now separately executes the official historical
   SciPy 1.14.1 wheel; repeated variants remain within-family evidence.
4. The predeclared controlled three-project gate and minimum one-pair
   historical gate now pass, but a strong JSS field-effectiveness claim still
   requires replication across more real defect families.
5. Stage-11 evidence is integrated in both manuscript languages and the JSS
   submission package has passed local consistency checks. The live portal,
   immutable archive, coauthor approval, and reviewer-conflict choices require
   author action at submission time.
