"""Propagate a complex non-self-adjoint operator with controlled Arnoldi."""

import torch

import dfsc


operator = torch.tensor(
    [[1.0 + 0.2j, 0.6 - 0.1j], [0.0 + 0.0j, 1.5 - 0.15j]],
    dtype=torch.complex128,
)
u0 = torch.tensor([1.0 + 0.1j, -0.25j], dtype=torch.complex128)
times = torch.linspace(0.0, 0.1, 4, dtype=torch.float64)
alpha = torch.tensor(0.85, dtype=torch.float64, requires_grad=True)

problem = dfsc.GeneralOperatorProblem(operator, u0, times, alpha)
solution = dfsc.solve(problem, dfsc.MLSLArnoldi(arnoldi_dimension=2))
solution.values.abs().square().mean().backward()

print(solution.algorithm)
print(solution.diagnostics)
print(alpha.grad)
