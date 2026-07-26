# Multi-Domain Real-Data Evidence

## Dataset and Provenance

The first experimental domain uses `750nm_mesh_size.mat` from AnomDiffDB.
It contains 3,112 fluorescent-bead trajectories and 426,620 two-dimensional
localizations acquired in an H-actin network with a nominal 750 nm mesh size.
The source publication describes the actin-network trajectories as fractional
Brownian motion (FBM).

- Data page: https://anomdiffdb.github.io/DB/
- Repository: https://github.com/AnomDiffDB/DB
- Publication: Granik et al., *Biophysical Journal* 117, 185-192 (2019)
- DOI: https://doi.org/10.1016/j.bpj.2019.06.015
- Local file SHA-256: `fd11f82f60df8235f311f34940ac9fc9226348221ca17cf80a02c3436401d4cd`

The source page does not state a redistribution license. The raw MAT file is
therefore excluded from version control and release artifacts. The fetch script
downloads it from the source and verifies its checksum.

## Leakage-Controlled Protocol

For each of five seeds, complete trajectories are split 70/30 before any
displacements are calculated. The training target is the directional ensemble
intermediate scattering estimate over lags 1-20 frames. Evaluation uses held-out
trajectories, with lags 21-40 reported separately as temporal extrapolation.

The wave number is fixed before fitting as 0.5 divided by the pooled median
one-frame displacement. Coordinates remain in source units and time in frames;
no undocumented physical calibration is introduced.

## Compared Models

| Model | Role | Parameters |
|---|---|---:|
| MSD power law | Traditional exponent estimate on a separate MSD target | 2 |
| Stretched exponential | Traditional Gaussian/FBM scattering baseline | 2 |
| Direct MLSL inverse | Trainable Mittag-Leffler order and rate | 2 |
| Pure MLP | Structure-free neural scattering fit | 1,153 |
| MLSL + residual MLP | Trainable MLSL backbone with gated neural correction | 323 |

## Five-Split Results

For the common scattering target, late-lag relative errors are:

| Model | Mean | Standard deviation |
|---|---:|---:|
| Stretched exponential | 0.0674 | 0.0279 |
| Direct MLSL inverse | **0.0366** | 0.0189 |
| Pure MLP | 0.1652 | 0.0555 |
| MLSL + residual MLP | 0.0429 | 0.0151 |

Direct MLSL reduces mean late-lag error by 45.7% relative to the traditional
stretched-exponential baseline. The hybrid reduces it by 74.0% relative to the
pure MLP, but does not improve on direct MLSL in this condition. This result
supports the value of the structured propagation prior and also shows that an
extra neural residual is not automatically beneficial.

The direct MLSL effective order is `0.9134` on average, close to the traditional
MSD exponent `0.9113`. Because the source describes this condition as FBM while
MLSL represents time-fractional Mittag-Leffler relaxation, this agreement must
not be interpreted as unique mechanism identification. The fitted MLSL order is
a model-conditional effective parameter.

## Reproduction

```bash
python tools/fetch_anomdiffdb.py
python experiments/exp41_real_spt_evidence_chain.py
```

Machine-readable results are written to
`results/real_spt_evidence_chain_summary.json` and the corresponding CSV tables
under `results/tables/`.

## Brownian-Bead Stress Test

The second AnomDiffDB file, `brownian_beads.mat`, contributes 660 trajectories
longer than 40 frames. A deterministic two-means split of log median one-frame
diffusivity yields a slow condition (104 trajectories) and a fast condition
(556 trajectories). These groups are empirical stress-test conditions rather
than ground-truth bead labels.

At late lags, the slow condition favors the stretched-exponential and pure-MLP
controls (errors 0.0205 and 0.0202); direct MLSL reaches 0.0281. In the fast
condition direct MLSL obtains the lowest mean error, 0.1524, compared with
0.1722 for the exponential and 0.1961 for the stretched exponential. The
hybrid residual model is worse in the fast condition. This mixed result defines
where the learned correction is useful and where it is not.

```bash
python experiments/exp43_real_brownian_two_population.py
```

## Geomembrane Stress Relaxation

The second physical domain uses the four stress-relaxation tests in the Zenodo
dataset *Effects of viscoelastic properties on geomembrane mechanics and
deformations* (DOI `10.5281/zenodo.14329961`, CC BY 4.0). The archive checksum
is MD5 `557d7f5b4996234ff0c37493fceb91fd`. Load-cell channels are summed, the
post-peak signal is reduced to 10-second median bins, and the first 60% of 120
uniform-time samples is used for fitting.

Across four stretch ratios and three seeds, mean extrapolation errors are
0.2614 for the exponential model, 0.3498 for the stretched exponential,
0.4090 for bare fractional-Zener MLSL, 0.3107 for the pure MLP, and 0.2481 for
MLSL plus a residual MLP. The hybrid has the lowest mean and median error but
larger run-to-run variation. Bare MLSL is not competitive here, so this result
supports composition rather than universal replacement.

```bash
python experiments/exp44_real_geomembrane_relaxation.py
```

Generated summaries for both additions are stored under `generated_results/`.

## Pilot-Scale GeoTES Cross-Cycle Transfer

The third physical domain uses the pilot-scale 3 x 3 modular geopolymer
thermal-energy-storage measurements deposited at Zenodo
(`10.5281/zenodo.18979098`, CC BY 4.0). The source workbook contains four
thermocouple histories over two consecutive charging--discharging cycles.
Because the dataset does not provide sensor coordinates in the workbook or
README, the protocol does not claim a reconstructed spatial field: T1 is the
observed thermal driver and ordered channels T2--T4 are joint responses.

The first cycle is used for identification and the second for transfer. Each
cycle is reduced to 56 uniformly indexed observations without smoothing or
interpolation. Across three seeds, second-cycle relative errors are 0.892 for
integer-order propagation, 0.667 for bare DFSC, 0.363 +/- 0.007 for a 1,251-
parameter pure MLP, and 0.362 +/- 0.009 for a 750-parameter DFSC--residual
model. The fractional primitive therefore improves the matched integer model;
the hybrid reaches error parity with a 40% smaller network, but does not show a
statistically decisive accuracy advantage over the pure MLP.

```bash
python tools/prepare_geotes.py data/external/geotes/thermocouple_measurements.xlsx
python experiments/exp48_real_geotes_cross_cycle.py
```

The raw source workbook, aligned CSV, SHA-256 checksums, license, split, and
preprocessing boundary are recorded in `data/external/geotes/manifest.json`.

## Heated-Steam Spatial Condition OOD

The fourth physical domain uses the CC BY 4.0 Zenodo dataset *Heated steam
injection experimental data and numerical outputs* (DOI
`10.5281/zenodo.15064388`). Its source MD5 is
`d14aa1ee2a77890e75817ef0e185a363`. The standardized table contains 32,370
measured temperatures from 16 experiments and ten depths from 0.03 to 0.30 m;
the conversion applies no interpolation or smoothing.

Experiments 5, 8, 12, and 16 are held out because they respectively contain the
largest flow, inlet temperature, column height, and water content. After a
fixed five-minute subsampling rule, training contains 5,090 rows and testing
contains 1,470 rows. Three seeds give:

| Model | Parameters | Held-out RMSE (K) |
|---|---:|---:|
| Integer forced spectral model | 46 | 43.64 +/- 0.38 |
| DFSC forced spectral model | 47 | 43.31 +/- 0.52 |
| Pure MLP | 4,673 | **13.49 +/- 1.56** |
| DFSC + residual MLP | 1,360 | 15.98 +/- 0.63 |

The fractional order reduces error by less than 1% relative to the matched
integer model, so the bare-propagator gain is practically small. The pure MLP
is most accurate. The hybrid uses about 29% of its parameters but has about 18%
higher RMSE. The experiment therefore supports a compact-model tradeoff and
documents a setting where the retained propagation model is incomplete.

```bash
python experiments/prepare_heated_steam.py
python experiments/exp51_real_heated_steam.py
```

The source workbook, long-form CSV, manifest, split rule, raw seed results, and
summary are retained in `data/external/heated_steam/` and
`revision_results/real_heated_steam.json`.
