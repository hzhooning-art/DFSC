"""Compare custom Mittag-Leffler backward against autograd and finite difference."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfsc import mittag_leffler_e


def loss_value(alpha_value: float, z: torch.Tensor, target: torch.Tensor, *, custom_backward: bool) -> torch.Tensor:
    alpha = torch.tensor(alpha_value, dtype=z.dtype)
    pred = mittag_leffler_e(alpha, z, terms=90, custom_backward=custom_backward)
    return torch.mean((pred - target) ** 2)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    z = -torch.linspace(0.0, 1.5, 32)
    alpha_true = torch.tensor(1.35)
    target = mittag_leffler_e(alpha_true, z, terms=120, custom_backward=False).detach()

    alpha_auto = torch.tensor(1.65, requires_grad=True)
    pred_auto = mittag_leffler_e(alpha_auto, z, terms=90, custom_backward=False)
    loss_auto = torch.mean((pred_auto - target) ** 2)
    loss_auto.backward()

    alpha_custom = torch.tensor(1.65, requires_grad=True)
    pred_custom = mittag_leffler_e(alpha_custom, z, terms=90, custom_backward=True)
    loss_custom = torch.mean((pred_custom - target) ** 2)
    loss_custom.backward()

    print("loss_auto:", loss_auto.item())
    print("loss_custom:", loss_custom.item())
    print("grad_auto:", alpha_auto.grad.item())
    print("grad_custom:", alpha_custom.grad.item())
    print(
        "custom_vs_auto_rel_error:",
        abs(alpha_custom.grad.item() - alpha_auto.grad.item()) / max(abs(alpha_auto.grad.item()), 1e-14),
    )

    for eps in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
        lp = loss_value(1.65 + eps, z, target, custom_backward=True)
        lm = loss_value(1.65 - eps, z, target, custom_backward=True)
        grad_fd = ((lp - lm) / (2.0 * eps)).item()
        rel = abs(alpha_custom.grad.item() - grad_fd) / max(abs(grad_fd), 1e-14)
        print(f"eps={eps:.1e} grad_fd={grad_fd:.8e} rel_error={rel:.3e}")


if __name__ == "__main__":
    main()
