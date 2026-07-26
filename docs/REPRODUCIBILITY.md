# Reproducibility

`dfsc` separates installation verification from paper-experiment
reproduction.

## Installed-artifact verification

Create a new environment, install the released wheel from PyPI or the GitHub
Release, and run:

```bash
python reproduction/verify_install.py
```

The script records the package version, operating system, Python and PyTorch
versions, accelerator information, and CPU/GPU forward-backward smoke tests.
It exits with a nonzero status if the installed version is not the expected
release or if any value or gradient is non-finite.

For the `0.1.0` release, install from PyPI:

```bash
python -m pip install "dfsc==0.1.0"
```

The archived `0.1.0rc1` TestPyPI artifact and its verification hashes remain
recorded in `reproduction/release_manifest.json`.

## Source-tree verification

From a clean checkout of the matching tag:

```bash
python -m pip install ".[test,docs]"
python -m unittest discover -s tests -p "test_*.py"
python -m mkdocs build --strict
python -B tools/pre_release_audit.py
python -B experiments/exp35_software_artifact_audit.py
```

Core maturity experiments can then be run with:

```bash
python tools/reproduce_core.py
```

Some real-data experiments require the source datasets described in their
manifests. The AnomDiffDB raw files are not redistributed until their
redistribution terms are explicit.

## Independent report

An independent verifier should start from the public release rather than a
developer environment and complete
`reproduction/EXTERNAL_REPRODUCTION_REPORT.md`. A GitHub-hosted CI run
demonstrates clean infrastructure execution; it is not labeled an independent
human reproduction.
