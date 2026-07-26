"""Differentiate a batched trajectory with the FFT Caputo-L1 operator."""

import torch

import dfsc


torch.set_default_dtype(torch.float64)
times = torch.linspace(0.0, 1.0, 2049)
trajectory = torch.stack((times.square(), torch.sin(times)), dim=-1).requires_grad_(True)
alpha = torch.tensor(0.7, requires_grad=True)

problem = dfsc.CaputoHistoryProblem(trajectory, alpha, final_time=1.0)
solution = dfsc.solve(problem)
solution.values.square().mean().backward()

print(solution.algorithm)
print(solution.diagnostics)
print(alpha.grad, torch.isfinite(trajectory.grad).all())
