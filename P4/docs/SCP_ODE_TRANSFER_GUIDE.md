# SciPy ODE transfer example

Use Python 3.11+ for the pinned NumPy 2.3.3 / SciPy 1.16.3 example. The base
P4 compiler/executor supports Python 3.10+ and no longer pulls PyTorch into
a normal installation. The legacy dfsc_protocol API requires the protocol
extra. Historical replay runtimes remain separate.

After extracting the software ZIP into a new directory:

```text
python -m venv ode-review-venv
# Activate the environment using the normal command for your platform.
python -m pip install ./P4[ode-example]
python P4/scripts/verify_software_install.py
python P4/scripts/compare_workflows.py --output-dir workflow-rerun
python P4/scripts/verify_ode_transfer.py --output-dir ode-rerun
```

For exact wheel review, install the included wheel instead of ./P4, then
install numpy==2.3.3 scipy==1.16.3. Do not add P4/src to PYTHONPATH. Check
summary.json: p4_import must point to the new environment's site-packages.
Use a new workflow output directory on every rerun to prevent stale reports
from being mistaken for newly produced evidence.

The component is scipy.integrate.solve_ivp with DOP853 on y'=-k*y. Two
rates (0.3, 1.2) and 17 grid points on [0,2] are used. The reference relation
compares numerical trajectories against the analytic exponential solution
for initial values 1 and 2. A property relation compares amplitude scaling;
an execution relation checks a terminal event at y=0.5 against log(2)/k.
All tolerances are read from the supplied contract. Each report contains
three observations and ten reported solve_ivp calls, not RHS evaluations.

The positive report must be CONFORMANT. The negative control adds 0.01 only
to the trajectory worker's observed value and must be NONCONFORMANT while
the other two relations remain PASS. This is a controlled example, not a
discovered SciPy defect. It does not cover stiffness, nonlinear systems,
parameter inference or sensitivity accuracy.

Extension consists of three JSON files and one worker. The compiler/executor
is unchanged. The worker consumes the contract explicitly, avoiding a
second, hidden tolerance definition. Its command arguments include that
file so the executor records its file hash. Numerical oracle correctness
and reported call counts still rely on worker trust.
