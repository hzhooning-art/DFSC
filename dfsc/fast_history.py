"""Differentiable direct and FFT Caputo-L1 history operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CaputoHistoryDiagnostics:
    """Detached execution metadata for a Caputo-L1 history evaluation."""

    method: str
    num_steps: int
    fft_length: int | None
    time_dimension: int

    def to_dict(self) -> dict[str, object]:
        return {
            "history_method": self.method,
            "num_steps": self.num_steps,
            "fft_length": self.fft_length,
            "time_dimension": self.time_dimension,
            "stores_full_input_history": True,
            "online_time_stepper": False,
        }


def caputo_l1_weights(
    num_steps: int,
    alpha: torch.Tensor | float,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Return L1 increment weights without differentiating through ``0**p``."""

    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
    if alpha_t.numel() != 1:
        raise ValueError("Caputo-L1 history evaluation requires scalar alpha")
    if not bool(((alpha_t > 0.0) & (alpha_t < 1.0)).detach().item()):
        raise ValueError("Caputo-L1 requires 0 < alpha < 1")
    if num_steps == 1:
        return torch.ones(1, dtype=dtype, device=device)
    indices = torch.arange(1, num_steps, dtype=dtype, device=device)
    tail = (indices + 1.0).pow(1.0 - alpha_t) - indices.pow(1.0 - alpha_t)
    return torch.cat((torch.ones(1, dtype=dtype, device=device), tail))


def _prepare_values(values: torch.Tensor, time_dim: int | None) -> tuple[torch.Tensor, int]:
    if not torch.is_tensor(values) or not values.is_floating_point():
        raise TypeError("values must be a floating-point tensor")
    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")
    if time_dim is None:
        time_dim = 0 if values.ndim == 1 else values.ndim - 2
    normalized = int(time_dim) % values.ndim
    if values.shape[normalized] < 2:
        raise ValueError("the time dimension must contain at least two samples")
    if not bool(torch.isfinite(values.detach()).all().item()):
        raise ValueError("values must be finite")
    return values.movedim(normalized, -1), normalized


def _scale(alpha: torch.Tensor | float, dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_t = torch.as_tensor(alpha, dtype=dt.dtype, device=dt.device)
    scale = dt.pow(-alpha_t) / torch.exp(torch.lgamma(2.0 - alpha_t))
    return alpha_t, scale


def caputo_l1_derivative_direct(
    values: torch.Tensor,
    *,
    alpha: torch.Tensor | float,
    final_time: float,
    time_dim: int | None = None,
) -> tuple[torch.Tensor, CaputoHistoryDiagnostics]:
    """Evaluate the uniform-grid Caputo-L1 derivative by direct convolution."""

    moved, normalized_time_dim = _prepare_values(values, time_dim)
    num_steps = moved.shape[-1] - 1
    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    dt = torch.as_tensor(final_time / num_steps, dtype=values.dtype, device=values.device)
    alpha_t, scale = _scale(alpha, dt)
    weights = caputo_l1_weights(num_steps, alpha_t, dtype=values.dtype, device=values.device)
    increments = moved[..., 1:] - moved[..., :-1]
    derivatives = []
    for step in range(1, num_steps + 1):
        history = increments[..., :step].flip(-1) * weights[:step]
        derivatives.append(scale * history.sum(dim=-1))
    result = torch.stack(derivatives, dim=-1).movedim(-1, normalized_time_dim)
    return result, CaputoHistoryDiagnostics("direct", num_steps, None, normalized_time_dim)


def caputo_l1_derivative_fft(
    values: torch.Tensor,
    *,
    alpha: torch.Tensor | float,
    final_time: float,
    time_dim: int | None = None,
) -> tuple[torch.Tensor, CaputoHistoryDiagnostics]:
    """Evaluate all uniform-grid Caputo-L1 history terms with FFT convolution.

    This is an offline full-trajectory operator for residual evaluation and
    training. It is not an online implicit time stepper and still stores the
    supplied trajectory.
    """

    moved, normalized_time_dim = _prepare_values(values, time_dim)
    if values.dtype not in (torch.float32, torch.float64):
        raise TypeError("FFT history evaluation currently supports float32 and float64")
    num_steps = moved.shape[-1] - 1
    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    dt = torch.as_tensor(final_time / num_steps, dtype=values.dtype, device=values.device)
    alpha_t, scale = _scale(alpha, dt)
    weights = caputo_l1_weights(num_steps, alpha_t, dtype=values.dtype, device=values.device)
    increments = moved[..., 1:] - moved[..., :-1]
    fft_length = 1 << max(0, (2 * num_steps - 1).bit_length())
    increment_spectrum = torch.fft.rfft(increments, n=fft_length, dim=-1)
    weight_spectrum = torch.fft.rfft(weights, n=fft_length)
    convolution = torch.fft.irfft(increment_spectrum * weight_spectrum, n=fft_length, dim=-1)
    result = (scale * convolution[..., :num_steps]).movedim(-1, normalized_time_dim)
    return result, CaputoHistoryDiagnostics("fft", num_steps, fft_length, normalized_time_dim)


def caputo_l1_history(
    values: torch.Tensor,
    *,
    alpha: torch.Tensor | float,
    final_time: float,
    time_dim: int | None = None,
    method: str = "auto",
    direct_threshold: int = 128,
) -> tuple[torch.Tensor, CaputoHistoryDiagnostics]:
    """Dispatch a Caputo-L1 trajectory derivative to direct or FFT convolution."""

    if method not in {"auto", "direct", "fft"}:
        raise ValueError("method must be 'auto', 'direct', or 'fft'")
    moved, _ = _prepare_values(values, time_dim)
    selected = "direct" if method == "auto" and moved.shape[-1] - 1 <= direct_threshold else method
    if selected == "auto":
        selected = "fft"
    function = caputo_l1_derivative_direct if selected == "direct" else caputo_l1_derivative_fft
    return function(values, alpha=alpha, final_time=final_time, time_dim=time_dim)
