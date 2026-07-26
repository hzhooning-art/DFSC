"""Experimental dfsc extensions.

These wrappers expose natural next-step primitives without claiming to solve
general variable-order or distributed-order fractional equations. They compose
the validated MLSL backbone over order samples, remain differentiable, and are
useful for ablation studies and early ecosystem validation.
"""

from __future__ import annotations

import torch
from torch import nn


class VariableOrderMLSL(nn.Module):
    """Evaluate an MLSL backbone with query-dependent fractional orders.

    The wrapper assumes a callable ``order_fn(times)`` that returns one alpha
    value per query time. It performs independent direct-query MLSL evaluations
    and stacks the results. This is a differentiable surrogate primitive, not a
    full variable-order history solver.
    """

    def __init__(self, backbone: nn.Module, order_fn: nn.Module | callable) -> None:
        super().__init__()
        self.backbone = backbone
        if isinstance(order_fn, nn.Module):
            self.order_fn = order_fn
        else:
            self.order_fn = _CallableOrder(order_fn)

    def forward(
        self,
        u0: torch.Tensor,
        times: torch.Tensor,
        *,
        beta: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if times.ndim == 0:
            times = times.unsqueeze(0)
        alphas = self.order_fn(times).to(dtype=u0.dtype, device=u0.device)
        if alphas.ndim == 0:
            alphas = alphas.expand_as(times)
        if alphas.shape[0] != times.shape[0]:
            raise ValueError("order_fn(times) must return one alpha per query time")

        states = [
            self.backbone(u0, times[i], alphas[i], beta=beta)
            for i in range(times.shape[0])
        ]
        return torch.stack(states, dim=0)


class DistributedOrderMLSL(nn.Module):
    """Weighted mixture of MLSL evaluations over retained fractional orders."""

    def __init__(
        self,
        backbone: nn.Module,
        alpha_nodes: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        trainable_weights: bool = False,
    ) -> None:
        super().__init__()
        if alpha_nodes.ndim != 1:
            raise ValueError("alpha_nodes must be one-dimensional")
        self.register_buffer("alpha_nodes", alpha_nodes.detach().clone())
        if weights is None:
            weights = torch.ones_like(alpha_nodes) / alpha_nodes.numel()
        if weights.shape != alpha_nodes.shape:
            raise ValueError("weights must match alpha_nodes")
        logits = torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))
        if trainable_weights:
            self.logits = nn.Parameter(logits.detach().clone())
        else:
            self.register_buffer("logits", logits.detach().clone())
        self.backbone = backbone

    @property
    def normalized_weights(self) -> torch.Tensor:
        """Return non-negative quadrature weights that sum to one."""

        return torch.softmax(self.logits, dim=0)

    def forward(
        self,
        u0: torch.Tensor,
        times: torch.Tensor,
        *,
        beta: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        outputs = []
        for alpha in self.alpha_nodes.to(dtype=u0.dtype, device=u0.device):
            outputs.append(self.backbone(u0, times, alpha, beta=beta))
        stacked = torch.stack(outputs, dim=0)
        weights = self.normalized_weights.to(dtype=u0.dtype, device=u0.device)
        view_shape = (weights.shape[0],) + (1,) * (stacked.ndim - 1)
        return (weights.reshape(view_shape) * stacked).sum(dim=0)


class _CallableOrder(nn.Module):
    def __init__(self, fn: callable) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        return self.fn(times)
