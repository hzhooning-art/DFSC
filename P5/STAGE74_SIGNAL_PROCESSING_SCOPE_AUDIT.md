# Stage 74: Signal Processing Scope and Submission-Format Audit

Updated: 2026-09-03

## Official scope match

The journal's current official description explicitly includes statistical
signal processing, detection and estimation, spectral analysis and filtering,
optimization methods for signal processing, multidimensional signal
processing, and new applications. P5 directly addresses the first three: it
performs grouped parametric estimation, tests adjacent model orders under
finite noisy observations, maps spectral resolution, and abstains when a
design lacks certified power.

The manuscript title and keywords now foreground model-order selection,
abstention, detection and estimation, spectral resolution, and multichannel
transient signals. Software-contract language remains only in reproducibility.

## Evidence-to-scope map

| Journal area | P5 evidence | Status |
|---|---|---|
| Statistical signal processing | Disjoint calibration/evaluation; white and AR(1) noise; risk--coverage analysis | Direct fit |
| Detection and estimation | Rank-one/rank-two decisions, directional errors, parameter recovery | Direct fit |
| Spectral analysis | Matrix pencil, block Hankel, Prony, adjacent-rate resolution | Direct fit |
| Multichannel processing | Six-channel shared-spectrum observations and grouped holdouts | Direct fit |
| New applications | Four public transient-data domains | Supporting fit |

## Submission-format gate

The official journal page states that original research articles must not
exceed 30 pages in single-column, double-spaced form, including figures,
tables, and references. The active English and Chinese sources are now set to
single-column double spacing. The page-count gate must be checked on every
compiled submission build.

The 2026-09-03 Tectonic 0.17.0 audit build produced a 29-page English PDF
and a 23-page Chinese PDF on A4 paper. The English submission manuscript
therefore passes the 30-page gate with one page of margin. Both builds
completed without TeX box, undefined-reference, or BibTeX warnings; the only
console message was the environment-level Fontconfig notice.

A full-document contact-sheet inspection found no blank pages, visibly clipped
figures, or obvious pagination anomalies. PDF text extraction also confirmed
that the retargeted title, frozen common-budget metrics, and references are
present in the rendered English artifact.

## Remaining scientific boundary

Scope fit is strong, but acceptance readiness is not equivalent to scope fit.
The power-certified method covers 0.2943 of the frozen evaluation trials and
abstains on 0.7057; its guarantee is conditional on the declared 0.32 rate
gap. Retrospective cross-task transfer admits only 1/56 groups. The 84 PVA
subsets establish composition stability within one dataset, not prospective
external confirmation. These limitations remain explicit in the abstract,
results, discussion, and conclusion.

Official source checked: Elsevier, *Signal Processing*, ISSN 0165-1684,
current journal description and contribution requirements.
