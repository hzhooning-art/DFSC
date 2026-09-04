# Current P5 Status

Updated: 2026-09-03

## Submission status

- The Applied Mathematical Modelling submission was desk rejected as outside the journal scope.
- The rejected AMM manuscript is frozen under `archive/rejected_AMM_20260902/`.
- `paper/manuscript_en.tex` and `paper/manuscript_zh.tex` are again the active, journal-neutral manuscripts.
- Experimental records, code, public-data analyses, GitHub materials, and Zenodo DOI remain valid and are retained.

## Scientific position

P5 studies evidence-gated model-order selection for shared finite-memory realizations from sparse multichannel transient data. Its main contribution is not a new relaxation solver or a domain-specific constitutive model. It is a detection-and-estimation rule that separates predictive improvement from support for a mechanistic order and permits an unresolved decision when the observation design cannot separate adjacent rates.

## Retargeting route

Primary target: **Signal Processing**. The manuscript will be revised around statistical signal processing, detection and estimation, spectral resolution, grouped model-order selection, and abstention under insufficient information.

Backup target: **Circuits, Systems, and Signal Processing**. This route retains the mathematical and system-identification content but requires less emphasis on a single physical application.

Stretch target: **Mechanical Systems and Signal Processing** only after a stronger mechanical-system experiment and a direct comparison with established system-identification order-selection methods are added.

## Next blocking work

1. Stage 68 exposed that unresolved rank-two signals were frequently labelled
   rank one. Stage 69 repaired this failure with a disjoint-seed design-power
   certificate and passed the prespecified white/AR(1) risk--coverage gate, but
   overall coverage is only 0.2943 and the guarantee is relative to a 0.32 gap.
2. Stage 70 confirmed that the frozen 0.70 certificate point lies on a stable
   three-plateau risk--coverage curve; 0.70--0.90 produce identical decisions.
   This is a sensitivity audit, not permission to retune the confirmatory rule.
3. Stage 71 added matrix-pencil AICc, block-Hankel AIC/MDL, and shared-Prony
   AICc/BIC to one 1,152-trial table with directional errors, abstention, and
   runtime. It supports a distinct selective-reliability regime, not higher
   unconditional accuracy or lower cost.
4. Stage 72 retrospectively transferred the frozen rule to 56 public-data
   groups. Only one PVA group entered scope and rejected rank-one sufficiency;
   the other 55 groups were scope-refused. A PVA group-composition audit and
   truly prospective external confirmation remain required.
   Stage 73 has now completed the composition audit: all 84 six-of-nine PVA
   subsets remain in scope and reject rank-one sufficiency under the unchanged
   rule. Because these subsets overlap within one dataset, truly prospective
   external confirmation remains required.
5. Recast the research questions as detection, estimation, and order-resolution
   questions; reduce software-protocol language to reproducibility support.
   Stage 68--73 results have now been integrated into both active manuscript
   languages, including the common-budget comparator table and claim limits;
   a final journal-specific editorial pass remains.
6. Use PVA and copper alloy as physical validation and gas/hydraulic tasks as
   transfer and refusal tests.
7. Freeze a new submission package only after prospective external validation,
   a Signal Processing scope audit, and final figure/reference QA pass.
   Stage 74 has completed the official scope audit and converted both active
   sources to single-column double spacing without an external line-spacing
   package. After Stage-76 integration and removal of redundant prose, the
   2026-09-03 Tectonic audit build is 29 pages in English and 24 pages in
   Chinese, leaving one page of margin under the 30-page gate. Full-document
   contact-sheet inspection found no blank pages, visibly clipped figures, or
   obvious pagination anomalies; extracted text confirms the frozen metrics.
8. Stage 75 froze a new-to-P5 cable-ageing transfer contract before revealing
   the P5 decision. All six curves entered scope; the unchanged rule mapped
   them to the 0.005 white-noise cell and returned evidence against rank-one
   sufficiency (criterion improvement 647.951; all five checks passed).
   Because P3 had previously analysed the raw dataset for another question,
   this adds a second eligible public dataset but does not close the stronger
   independently acquired prospective-validation requirement.
9. Stage 76 audited 12 post-result cable start/end windows. All 12 remained
   eligible, passed all five rank-two checks, and matched the Stage 75
   decision; criterion improvements ranged from 399.989 to 714.271. Because
   the windows overlap within the same six curves, this is robustness evidence
   rather than independent or prospective replication.
10. Stage 77 adds a Signal Processing-specific submission-readiness pack with
    length-checked highlights, current metadata (first author Haitao Duan;
    corresponding author Ning Hu), a
    signal-processing cover letter, declarations, an evidence-to-claim matrix,
    and an upload checklist. Before upload, the existing Zenodo DOI must be
    audited and refreshed if it does not contain Stages 67--76.
