# AnomDiffDB experimental trajectories

The local benchmarks use `750nm_mesh_size.mat` and `brownian_beads.mat` from
the public AnomDiffDB repository. The first contains two-dimensional
fluorescent-bead trajectories in an H-actin network; the second contains
water--glycerol bead trajectories used for a two-population stress test.

- Source: https://github.com/AnomDiffDB/DB/blob/master/750nm_mesh_size.mat
- Data page: https://anomdiffdb.github.io/DB/
- Citation: N. Granik et al., *Single-Particle Diffusion Characterization by
  Deep Learning*, Biophysical Journal 117 (2019) 185-192.
- DOI: https://doi.org/10.1016/j.bpj.2019.06.015
- SHA-256: `fd11f82f60df8235f311f34940ac9fc9226348221ca17cf80a02c3436401d4cd`
- Brownian SHA-256: `1bc05922899172053126df80b28e43a645411d8b6919b1575a9a70ef4aa9d8e6`

The source page does not state a dataset redistribution license. Consequently,
the MAT file is ignored by version control and must not be included in a public
dfsc release without permission from the data owners. Run
`python tools/fetch_anomdiffdb.py` to retrieve and verify it from the source.

Coordinates are analyzed in source units and time in frames. The benchmark
therefore reports dimensionless anomalous exponents and rates per frame rather
than assigning an unsupported physical calibration.

For `brownian_beads.mat`, two deterministic clusters are formed from the log
median one-frame diffusivity. They are empirical conditions, not ground-truth
bead labels.
