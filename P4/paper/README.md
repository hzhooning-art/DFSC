# P4 Manuscript and Reproducibility Package

This folder contains the current bilingual manuscript, supplementary material,
and the reproducibility files used to validate the P4 results. Submission-only
packages, build trees, rendered QA pages, and previous-journal notes are
intentionally excluded from version control.

## Files

- `dfsc_primitive_protocol_en.tex`: current English source.
- `dfsc_primitive_protocol_zh.tex`: synchronized Chinese source.
- `dfsc_primitive_protocol_en_CSI.pdf`: current English manuscript PDF.
- `dfsc_primitive_protocol_zh_CSI.pdf`: synchronized Chinese manuscript PDF.
- `references.bib`: verified bibliography shared by both sources.
- `figures/`: publication figures included by both sources.
- `make_figures.py`: regenerates the publication figures.
- `compile_paper.ps1`: compiles both sources in temporary directories and
  replaces the current bilingual manuscript and supplementary PDFs.
- `build_paper_data.py`: rebuilds the data bundle from `P4/results/*.json`.
- `paper_data.json`: generated provenance bundle used to audit manuscript
  tables.
- `EVIDENCE_GAPS.md`: current evidence boundary and publication checklist.

## Compile

From this directory, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\compile_paper.ps1
```

The script uses the bundled Tectonic executable under `P1/tools/tectonic`,
keeps intermediate files outside the project, and leaves only the current
English and Chinese PDFs in this folder.
