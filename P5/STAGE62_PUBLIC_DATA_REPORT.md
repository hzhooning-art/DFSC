# Stage 62: Independent Public-Data Audit

## Purpose and provenance

This stage directly analyzes public observations that were neither generated
by nor used to tune P5. The source is *Stress Relaxation Test Dataset of
Cylindrical PVA Gel Polymer Electrolyte (GPE) Samples*, Zenodo record 21333840,
The record is open access under CC BY 4.0, as declared by `metadata.license.id` in the retrieved Zenodo record. It contains three specimens and three cycles per specimen with raw
time, force, and displacement. The preserved workbook has MD5
`403b3254288b4ce8aa36cada05c0e1e4`.

## Frozen analysis

The preprocessing and gates are defined in `STAGE62_PUBLIC_TASK_PROTOCOL.md`.
Positive shared relaxation rates of ranks 1--3 are fitted with curve-specific
nonnegative amplitudes and offsets. Each fold holds out one specimen. The first
60% of each held curve calibrates amplitudes at rates learned from the other
specimens; the final 40% is predicted.

## Primary result

The complete 28 s, 96-point task returns `SUPPORTED_RANK_3`.

| Rank | Mean BIC | Median held NRMSE | Maximum fold log-rate SD |
|---:|---:|---:|---:|
| 1 | -6944.806 | 0.01757 | 0.0284 |
| 2 | -8350.562 | 0.01121 | 0.1540 |
| 3 | -8670.737 | 0.00932 | 0.3685 |

Rank 2 versus rank 1 yields median relative held-curve improvement 0.552,
bootstrap 95% interval [0.232, 0.617], and paired one-sided Wilcoxon p=0.0371
(9 curves). Rank 3 versus rank 1 yields median 0.484, interval [0.395, 0.798],
and p=0.00195. Adjacent rates exceed the frozen 1.25 separation threshold.

## Boundary result

The horizon-by-budget map is nonmonotone: all 4 s cells are indeterminate; one
8 s cell supports rank 2; three 15 s cells support rank 2; and only the 28 s,
96-point cell supports rank 3. Nominal sample count is therefore not a complete
measure of information. Correlation, tail coverage, and optimization variation
all enter the joint gates. This supports retaining a refusal outcome and rules
out a stronger claim that additional samples must monotonically increase rank.

## Formal figures

- `figures/fig_stage62_public_data_statistics.pdf`
- `figures/fig_stage62_method_workflow.pdf`
- `figures/fig_stage62_identifiability_boundary.pdf`

Each has a high-resolution PNG preview. The full machine-readable result is
`results/public_pva_relaxation.json`.

## Claim boundary

The result supports the protocol on one independently published material
relaxation dataset. It does not establish three unique microscopic mechanisms
or universal rank-three behavior. The conclusion is restricted to a supported
finite shared realization on the declared dataset and preprocessing domain.
