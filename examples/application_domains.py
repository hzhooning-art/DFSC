"""Four domain-oriented entry points built from the same dfsc core."""

import torch

import dfsc


torch.set_default_dtype(torch.float64)

diffusion = dfsc.anomalous_diffusion_case(
    initial=lambda x: torch.sin(torch.pi * x),
    times=torch.linspace(0.0, 0.05, 4),
    alpha=torch.tensor(0.8, requires_grad=True),
    beta=torch.tensor(1.7, requires_grad=True),
    diffusivity=0.1,
    num_points=24,
    num_modes=10,
)

relaxation = dfsc.assembled_relaxation_case(
    stiffness=torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
    mass=torch.diag(torch.tensor([2.0, 1.0])),
    initial=torch.tensor([1.0, 0.0]),
    times=torch.tensor([0.0, 0.1]),
    alpha=0.9,
)

network = dfsc.network_diffusion_case(
    adjacency=torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    initial=torch.tensor([1.0, 0.0, 0.0]),
    times=torch.tensor([0.0, 0.2]),
    alpha=0.75,
)

transport = dfsc.advection_diffusion_case(
    initial=lambda x: torch.sin(2.0 * torch.pi * x),
    times=torch.tensor([0.0, 0.01]),
    alpha=0.9,
    diffusivity=0.02,
    velocity=0.15,
    num_points=8,
    arnoldi_dimension=8,
)

for case in (diffusion, relaxation, network, transport):
    solution = case.solve()
    print(case.name, solution.algorithm, tuple(solution.values.shape), solution.success)
