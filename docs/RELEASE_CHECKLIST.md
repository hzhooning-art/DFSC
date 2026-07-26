# dfsc Release Checklist

Use this checklist before tagging a public release.

## Package

- [ ] `python -B tools/pre_release_audit.py` reports at least 95% internal readiness.
- [ ] `pyproject.toml` version matches the release tag.
- [ ] `pip install -e .[test]` succeeds in a clean environment.
- [ ] `python -m build` creates both wheel and source distribution.
- [ ] `python -m twine check dist/*` passes.
- [ ] `python -m unittest discover -s tests` passes.
- [ ] `python tools/mlsl_doctor.py` passes on CPU.
- [ ] CUDA smoke test is recorded when a GPU is available.

## Documentation

- [ ] `README.md` quick start runs without modification.
- [ ] `docs/API.md` lists all public dfsc entry points.
- [ ] `docs/ROADMAP.md` separates implemented, experimental, and planned work.
- [ ] The hosted documentation site builds from `mkdocs.yml`.
- [ ] `mkdocs build --strict` passes in CI.

## Reproducibility

- [ ] Experiment outputs are regenerated or explicitly marked as archived.
- [ ] Paper tables and figures are regenerated from current results.
- [ ] The software artifact audit reports all mandatory checks as passing.

## Publication Boundary

- [ ] Claims avoid presenting dfsc as a universal fractional solver.
- [ ] Experimental wrappers are labeled experimental in code, docs, and paper.
- [ ] Public benchmark data provenance is recorded when external data are used.

## Public Release

- [ ] Create the public Git repository and protect the release branch.
- [ ] Publish the signed version tag and attach CI-built distributions.
- [ ] Publish the package to PyPI from a trusted release workflow.
- [ ] Archive the tagged source on Zenodo and add the assigned DOI to `CITATION.cff`.
- [ ] Replace all pre-release availability wording only after the public URLs resolve.
