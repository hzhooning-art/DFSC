# Historical Defect Replay Matrix

The same case-specific runner file must be used on both sides of each pair.
Captured JSON, interpreter/package version output, artifact hashes, and
environment records form one replay record. A current pass alone is not a
completed pair.

| Case | Buggy command | Fixed command | Additional requirement |
|---|---|---|---|
| PyTorch #80770 | `python pytorch_historical_pair_runner.py --case pytorch_80770 --expected-role buggy` under PyTorch 1.11.0 | same case with `--expected-role fixed` under the frozen current version | **Complete:** both roles confirmed with runner SHA-256 `fdb828...014` |
| PyTorch #30303 | `python pytorch_historical_pair_runner.py --case pytorch_30303 --expected-role buggy` under PyTorch 1.3.1 | same case with `--expected-role fixed` under the frozen current version | CUDA device and compatible legacy CUDA runtime |
| SciPy #8906 | `python scipy_historical_pair_runner.py --expected-role buggy` under SciPy 1.14.1 | same runner with `--expected-role fixed` under SciPy 1.18.0 | **Complete:** both roles confirmed with runner SHA-256 `ce901...bc8` |
| SciPy #15620 | `python scipy_resample_poly_historical_pair_runner.py --expected-role buggy` under SciPy 1.14.1 | same runner with `--expected-role fixed` under SciPy 1.18.0 | **Complete:** both roles confirmed with runner SHA-256 `0a7a72...f79` |

Required captured metadata:

- exact Python, PyTorch, CUDA, driver, and platform versions;
- package lock or immutable container digest;
- runner SHA-256;
- stdout JSON and process exit code;
- upstream issue URL and expected role;
- no source change between buggy and fixed runs.

The Stage-8 capture completed the first row with the official PyTorch
1.11.0+cpu wheel and Python 3.10.11 embedded runtime. Full artifact hashes,
commands, stderr, observations, and exit codes are stored in
`P4/results/p4_complete_historical_pair.json`.

The Stage-9 capture completed the SciPy row with the official SciPy 1.14.1
wheel and the current 1.18.0 installation. Its complete record is
`P4/results/p4_scipy_complete_pair.json`. The historical-pair total is now two
across two projects.

The Stage-11 capture completed SciPy #15620 with the same official 1.14.1
wheel and current 1.18.0 installation. It records all-zero integer output on
the buggy side and exact agreement with the float64 reference on the fixed
side in `P4/results/p4_scipy_resample_poly_pair.json`. The total is now three
complete defect families across two projects.
