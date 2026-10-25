# Optional CPU repair development backend

Distribution `dfsc-cpu-repair-candidate==0.1.0.dev0`, namespace `dfsc_cpu_repair`.
This separate optional backend does not replace the historical `dfsc` distribution,
change `dfsc.solve`, register itself automatically, or update PyPI/GitHub/Zenodo.
Use explicit imports in a dedicated environment. No editable install is required.

```powershell
python -m pip install dfsc_cpu_repair_candidate-0.1.0.dev0-py3-none-any.whl
python -m pip check
```

Dependencies: torch >=2.11,<2.12 and mpmath==1.3.0. The verified fresh environment
uses CPU-only torch 2.11.0+cpu on Windows/Python 3.12. A CUDA build may run this
backend on CPU tensors, but GPU tensors are rejected, never silently transferred.

```python
import torch
from dfsc_cpu_repair import SelectiveSpectral, backend_info
A = torch.diag(torch.tensor([0.2,6.0], dtype=torch.float64))
model = SelectiveSpectral(A, mode='selective_repair')
u0 = torch.tensor([[1.0,0.7]], dtype=torch.float64, requires_grad=True)
t = torch.tensor([0.0,1.0], dtype=torch.float64)
a = torch.tensor(0.75, dtype=torch.float64, requires_grad=True)
b = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
result = model(u0,t,a,beta=b)
gradients = torch.autograd.grad(result.values.square().mean(), (a,b,u0))
print(result.fast_entries, result.repaired_entries, backend_info())
```

Modes: `guard_only` refuses unadmitted entries; `selective_repair` repairs only
such entries; `all_high_precision` evaluates every scalar at high precision.
The low-precision mask is inspected without a derivative graph, then the graph
is rebuilt for accepted entries only. Whole-call failure raises `ActionRefusal`
before output/backward/update. Do not catch refusal and update with stale values.

Fixed real SPD dense full eigenbasis, homogeneous evolution, CPU float64 tensors,
scalar alpha in [0.45,1], spatial beta in (0,2], nonnegative fixed times, modal
radii <=16; first-order alpha/beta/u0 gradients only. No operator/time/basis
gradients, modal truncation, nonnormal/Krylov/forcing, two-parameter ML, implicit
device transfer or higher-order derivative guarantee. Scalar `evaluate_scalar`
allows alpha through 1.95 within the same radius. Budgets may refuse calls.

Risk proxies and precision refinement are not rigorous floating-point certificates.
Finite reference checks and twelve-update inverse tasks do not prove global
accuracy, inverse convergence, production readiness or predictive superiority.
The old hybrid was faster and slightly more accurate on the finite hard-task
prediction metric despite inaccurate actions in some separate numerical probes.
