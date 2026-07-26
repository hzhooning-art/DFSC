"""Small neural baselines for dfsc experiments."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import math


class MLPField(nn.Module):
    """A compact neural field baseline mapping ``(x, t) -> u``."""

    def __init__(self, hidden: int = 64, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(-1)
        t_flat = t.reshape(-1)
        xt = torch.stack([x_flat, t_flat], dim=-1)
        return self.net(xt).squeeze(-1).reshape(torch.broadcast_shapes(x.shape, t.shape))


class ConditionalMLPField(nn.Module):
    """Neural field baseline mapping ``(x, t, alpha) -> u``."""

    def __init__(self, hidden: int = 96, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 3
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(-1)
        t_flat = t.reshape(-1)
        alpha_full = alpha[:, None].expand_as(x).reshape(-1) if alpha.ndim == 1 else alpha.reshape(-1)
        xta = torch.stack([x_flat, t_flat, alpha_full], dim=-1)
        return self.net(xta).squeeze(-1).reshape(torch.broadcast_shapes(x.shape, t.shape))


class SpectralConv1d(nn.Module):
    """Minimal 1D Fourier layer used by the FNO baseline."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.width = width
        self.modes = modes
        scale = 1.0 / (width * width)
        self.weight_real = nn.Parameter(scale * torch.randn(width, width, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(width, width, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, width, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            batch,
            width,
            x_ft.shape[-1],
            dtype=torch.cfloat,
            device=x.device,
        )
        modes = min(self.modes, x_ft.shape[-1])
        weight = torch.complex(
            self.weight_real[:, :, :modes],
            self.weight_imag[:, :, :modes],
        ).to(dtype=x_ft.dtype, device=x.device)
        out_ft[:, :, :modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[:, :, :modes],
            weight,
        )
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNO1D(nn.Module):
    """Small FNO-style baseline mapping ``(u0(x), t) -> u(x,t)``."""

    def __init__(self, modes: int = 12, width: int = 32, layers: int = 4) -> None:
        super().__init__()
        self.lift = nn.Linear(2, width)
        self.spectral = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(layers)])
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.proj1 = nn.Conv1d(width, 64, 1)
        self.proj2 = nn.Conv1d(64, 1, 1)

    def forward(self, u0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the FNO baseline.

        Parameters
        ----------
        u0:
            Tensor with shape ``(batch, num_points)``.
        t:
            Tensor with shape ``(batch,)``.
        """

        t_channel = t[:, None].expand_as(u0)
        x = torch.stack([u0, t_channel], dim=-1)
        x = self.lift(x).permute(0, 2, 1)
        for spectral, pointwise in zip(self.spectral, self.pointwise, strict=True):
            x = F.gelu(spectral(x) + pointwise(x))
        x = F.gelu(self.proj1(x))
        return self.proj2(x).squeeze(1)


class DeepONet1D(nn.Module):
    """Minimal DeepONet baseline conditioned on ``u0`` and optional ``alpha``.

    The branch net encodes the input function and fractional order. The trunk
    net encodes query coordinates ``(x, t)``. Their inner product gives the
    predicted field value.
    """

    def __init__(
        self,
        num_points: int,
        latent: int = 64,
        hidden: int = 96,
        condition_alpha: bool = True,
    ) -> None:
        super().__init__()
        self.condition_alpha = condition_alpha
        branch_in = num_points + (1 if condition_alpha else 0)
        self.branch = nn.Sequential(
            nn.Linear(branch_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent),
        )
        self.bias = nn.Parameter(torch.zeros(()))
        self.latent = latent

    def forward(
        self,
        u0: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.condition_alpha:
            if alpha is None:
                raise ValueError("alpha is required when condition_alpha=True")
            branch_input = torch.cat([u0, alpha[:, None]], dim=-1)
        else:
            branch_input = u0

        branch = self.branch(branch_input)
        xt = torch.stack([x.reshape(-1), t.reshape(-1)], dim=-1)
        trunk = self.trunk(xt).reshape(*x.shape, -1)
        return torch.einsum("bl,bnl->bn", branch, trunk) / math.sqrt(self.latent) + self.bias
