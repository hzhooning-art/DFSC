# P5 bilingual manuscript

## Canonical files

- `manuscript_en.tex` / `manuscript_en.pdf`: English manuscript (30 pages in the current single-column, double-spaced author-guide build).
- `manuscript_zh.tex` / `manuscript_zh.pdf`: Chinese manuscript (24 pages in the corresponding build) with the same structure, evidence, figures, tables, declarations, and references.
- `references.bib`: shared verified bibliography.

## Author order

1. Haitao Duan / 段海涛 (first author)
2. Ning Hu / 胡宁 (corresponding author)
3. Shuqun Li / 李书群
4. Chuyang Hu / 胡初扬

Correspondence: `huning@hdu.edu.cn`.

Affiliations, e-mail addresses, and ORCID identifiers follow the project author record and the author-confirmed P5 roles above. This work acknowledges support from the State Key Laboratory of Ocean Engineering, Shanghai Jiao Tong University (Grant No. GKZD010089), and the State Key Laboratory of Acoustics, Chinese Academy of Sciences (Grant No. SKLA202406).

## Build

Both manuscripts were compiled with Tectonic 0.17.0 on 2026-09-05. The
sources implement double spacing with LaTeX's built-in baseline-stretch
setting, avoiding a non-cached package dependency. The audit builds complete
without TeX overfull-box, undefined-reference, or BibTeX warnings. The
remaining Fontconfig message is an environment-level notice and does not
affect PDF output.

