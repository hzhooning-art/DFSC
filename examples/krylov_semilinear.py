"""Krylov and semilinear paths through the dfsc solve interface."""

import torch

import dfsc


torch.set_default_dtype(torch.float64)

# A dense symmetric operator can be propagated without a full eigendecomposition.
size = 64
diagonal = 2.0 * torch.ones(size)
off_diagonal = -torch.ones(size - 1)
operator = torch.diag(diagonal) + torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1)
u0 = torch.sin(torch.pi * torch.linspace(0.0, 1.0, size))
times = torch.linspace(0.0, 0.05, 8)
alpha = torch.tensor(0.85, requires_grad=True)

krylov_problem = dfsc.OperatorSpectralProblem(operator, u0, times, alpha)
krylov_solution = dfsc.solve(krylov_problem, dfsc.MLSLKrylov(krylov_dimension=24))
print(krylov_solution.algorithm, krylov_solution.diagnostics)

matrix_free_operator = dfsc.SelfAdjointLinearOperator(
    size,
    lambda vector: operator @ vector,
    torch.float64,
    "cpu",
    name="matrix-free-example",
)
matrix_free_solution = dfsc.solve(
    dfsc.LinearOperatorSpectralProblem(matrix_free_operator, u0, times, alpha),
    dfsc.MLSLKrylov(krylov_dimension=24),
)
print(matrix_free_solution.diagnostics["operator_representation"])

# The same solve contract exposes a nonlinear mild-form fixed-point iteration.
x, layer = dfsc.build_mlsl(
    dimension=1,
    boundary="dirichlet",
    num_points=64,
    num_modes=16,
    config=dfsc.MLSLConfig.stable(),
)
gamma = torch.tensor(0.02, requires_grad=True)
semilinear_problem = dfsc.SemilinearSpectralProblem(
    layer=layer,
    u0=torch.sin(torch.pi * x),
    times=times,
    alpha=alpha,
    nonlinearity=lambda state: -gamma * state.pow(3),
)
semilinear_solution = dfsc.solve(semilinear_problem, dfsc.MLSLPicard())
semilinear_solution.final.square().mean().backward()
print(semilinear_solution.retcode, alpha.grad, gamma.grad)
