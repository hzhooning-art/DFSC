# P4 Testing 0.3.0

MIT-licensed contract compilation and auditable numerical-software test
execution. Prepared for an SCP Original Software Publication; this does not
claim journal acceptance or external adoption.

```powershell
python -m pip install ./P4
python P4/scripts/verify_software_install.py
python P4/scripts/compare_workflows.py --output-dir workflow-check
python -m pip install "./P4[ode-example]"
python P4/scripts/verify_ode_transfer.py --output-dir ode-check
```

Use fresh output directories. Core Python >=3.10; pinned ODE dependencies
require Python >=3.11. Legacy `dfsc_protocol` imports need the optional
`protocol` extra (PyTorch). Repository-side workers are not in the wheel;
the installed executor uses `--artifact-root` to locate reviewed workers.

Release: https://github.com/hzhooning-art/DFSC/releases/tag/p4-scp-osp-v0.3.0

The release ZIP contains a source/evidence manifest and installable wheel.
Only this P4 software snapshot and its dedicated CI workflow are added;
other DFSC research is unchanged. Read `docs/SCP_SOFTWARE_USER_GUIDE.md`,
`docs/SCP_SOFTWARE_DEVELOPER_GUIDE.md`, `docs/SCP_RELATED_TOOLS.md`, and
`docs/SCP_ODE_TRANSFER_GUIDE.md`.

Evidence boundaries: the strict scripted comparator agrees with P4 in nine
controlled cases. The exit-only control is not pytest. The ODE example is
developer-run transfer evidence; controlled bias is not a real historical
defect. The historical benchmark remains seven families. Historical upstream
runtimes are not bundled; unavailable-runtime tests skip. Read the frozen
metadata clarifications in `docs/SCP_HISTORICAL_REPRODUCTION_BOUNDARY.md`.

Worker registries contain trusted executable commands. Declared mechanism
coverage, worker call counts, and numerical oracles are not independently
certified by the executor. No external adoption or productivity claim is made.

The author-review manuscript is not a public release asset. A new Zenodo
version must be created in the existing P4 record series; its reserved DOI
will be synchronized into the manuscript after the author supplies it.
