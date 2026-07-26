"""Autograd-friendly Mittag-Leffler functions for dfsc.

This file provides differentiable Mittag-Leffler evaluators for the real
non-positive spectral arguments used by MLSL. The hybrid path combines a
truncated series, a negative-real asymptotic branch, and a narrow smooth
transition band so gradients remain usable near the numerical switch.
"""

from __future__ import annotations

import torch


def hybrid_switch_region(
    alpha: torch.Tensor,
    z: torch.Tensor,
    *,
    threshold: float = 12.0,
) -> tuple[float, float, float]:
    """Return the active threshold and smooth-transition interval.

    The policy is shared by the one- and two-parameter evaluators and by the
    public dfsc diagnostics.  It is a numerical routing rule, not an error
    bound.  Lower precision and small fractional orders switch to the
    asymptotic branch earlier to avoid unstable high-order powers.
    """

    active_threshold = float(threshold)
    alpha_min = float(torch.min(alpha.detach()).cpu())
    if alpha_min < 1.0:
        active_threshold = min(active_threshold, 8.0)
    if alpha_min < 0.60:
        active_threshold = min(active_threshold, 4.0)
    if z.dtype in (torch.float16, torch.bfloat16, torch.float32):
        active_threshold = min(active_threshold, 1.0)
    width = max(0.15 * active_threshold, 0.25)
    return active_threshold, active_threshold - width, active_threshold + width


def reciprocal_gamma(x: torch.Tensor) -> torch.Tensor:
    """Autograd-friendly reciprocal gamma with sign support.

    ``torch.lgamma`` returns ``log(abs(Gamma(x)))`` and drops the sign for
    negative non-integers. The reflection formula gives

    ``1/Gamma(x) = sin(pi x) / (pi Gamma(1 - x))``.

    We use the direct positive branch where possible and the reflection branch
    otherwise. This is useful for asymptotic Mittag-Leffler terms such as
    ``1/Gamma(1 - alpha k)``.
    """

    positive = x > 0
    out = torch.empty_like(x)
    if torch.any(positive):
        out[positive] = torch.exp(-torch.lgamma(x[positive]))
    if torch.any(~positive):
        x_reflect = x[~positive]
        out[~positive] = (
            torch.sin(torch.pi * x_reflect)
            * torch.exp(-torch.lgamma(1.0 - x_reflect))
            / torch.pi
        )
    return out


def _sum_to_shape(value: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Reduce a broadcasted gradient back to the original tensor shape."""

    if shape == torch.Size([]):
        return value.sum()

    while value.ndim > len(shape):
        value = value.sum(dim=0)
    for dim, size in enumerate(shape):
        if size == 1 and value.shape[dim] != 1:
            value = value.sum(dim=dim, keepdim=True)
    return value


def _mittag_leffler_series_terms(
    alpha: torch.Tensor,
    z: torch.Tensor,
    *,
    terms: int,
) -> torch.Tensor:
    k = torch.arange(terms, dtype=z.dtype, device=z.device)
    log_gamma = torch.lgamma(alpha * k + 1.0)
    powers = z.unsqueeze(-1).pow(k)
    coeffs = torch.exp(-log_gamma)
    return torch.sum(powers * coeffs, dim=-1)


def mittag_leffler_e_ab(
    alpha: torch.Tensor | float,
    beta: torch.Tensor | float,
    z: torch.Tensor,
    *,
    terms: int = 80,
    method: str = "series",
    asymptotic_terms: int = 8,
    threshold: float = 12.0,
) -> torch.Tensor:
    """Compute the two-parameter Mittag-Leffler function ``E_{alpha,beta}(z)``.

    ``E_{alpha,beta}(z) = sum_{k=0}^{inf} z^k / Gamma(alpha k + beta)``.
    The ``series`` method is intended for moderate regimes. The ``hybrid``
    method switches large non-positive real arguments to an algebraic asymptotic
    expansion, which is important for forced multi-mode spectral dynamics.
    """

    if terms < 2:
        raise ValueError("terms must be >= 2")

    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha, dtype=z.dtype, device=z.device)
    else:
        alpha = alpha.to(dtype=z.dtype, device=z.device)
    if not torch.is_tensor(beta):
        beta = torch.as_tensor(beta, dtype=z.dtype, device=z.device)
    else:
        beta = beta.to(dtype=z.dtype, device=z.device)

    if method == "series":
        return _mittag_leffler_e_ab_series(alpha, beta, z, terms=terms)
    if method == "hybrid":
        return mittag_leffler_e_ab_hybrid(
            alpha,
            beta,
            z,
            series_terms=terms,
            asymptotic_terms=asymptotic_terms,
            threshold=threshold,
        )
    raise ValueError(f"unknown two-parameter Mittag-Leffler method: {method}")


def _mittag_leffler_e_ab_series(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    z: torch.Tensor,
    *,
    terms: int,
) -> torch.Tensor:
    k = torch.arange(terms, dtype=z.dtype, device=z.device)
    powers = z.unsqueeze(-1).pow(k)
    coeffs = torch.exp(-torch.lgamma(alpha * k + beta))
    return torch.sum(powers * coeffs, dim=-1)


def _mittag_leffler_e_ab_negative_asymptotic(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    z: torch.Tensor,
    *,
    terms: int,
) -> torch.Tensor:
    x = (-z).clamp_min(torch.finfo(z.dtype).tiny)
    k = torch.arange(1, terms + 1, dtype=z.dtype, device=z.device)
    rgamma = reciprocal_gamma(beta - alpha * k)
    signs = torch.where(
        (k.to(torch.int64) % 2) == 1,
        torch.ones_like(k),
        -torch.ones_like(k),
    )
    return torch.sum(signs * x.unsqueeze(-1).pow(-k) * rgamma, dim=-1)


def mittag_leffler_e_ab_hybrid(
    alpha: torch.Tensor | float,
    beta: torch.Tensor | float,
    z: torch.Tensor,
    *,
    series_terms: int = 80,
    asymptotic_terms: int = 8,
    threshold: float = 12.0,
) -> torch.Tensor:
    """Hybrid evaluator for ``E_{alpha,beta}(z)`` on non-positive real inputs."""

    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha, dtype=z.dtype, device=z.device)
    else:
        alpha = alpha.to(dtype=z.dtype, device=z.device)
    if not torch.is_tensor(beta):
        beta = torch.as_tensor(beta, dtype=z.dtype, device=z.device)
    else:
        beta = beta.to(dtype=z.dtype, device=z.device)

    if torch.any(z > 0):
        raise ValueError("hybrid two-parameter evaluator supports non-positive real z only")

    active_threshold, lo, hi = hybrid_switch_region(alpha, z, threshold=threshold)

    out = torch.empty_like(z)
    radius = torch.abs(z)
    small = radius <= lo
    large = radius >= hi
    blend = ~(small | large)
    if torch.any(small):
        out[small] = _mittag_leffler_e_ab_series(alpha, beta, z[small], terms=series_terms)
    if torch.any(large):
        out[large] = _mittag_leffler_e_ab_negative_asymptotic(
            alpha,
            beta,
            z[large],
            terms=asymptotic_terms,
        )
    if torch.any(blend):
        eta = ((radius[blend] - lo) / (hi - lo)).clamp(0.0, 1.0)
        weight = eta * eta * (3.0 - 2.0 * eta)
        series = _mittag_leffler_e_ab_series(alpha, beta, z[blend], terms=series_terms)
        asymptotic = _mittag_leffler_e_ab_negative_asymptotic(
            alpha,
            beta,
            z[blend],
            terms=asymptotic_terms,
        )
        out[blend] = (1.0 - weight) * series + weight * asymptotic
    return out


def _mittag_leffler_negative_asymptotic(
    alpha: torch.Tensor,
    z: torch.Tensor,
    *,
    terms: int,
) -> torch.Tensor:
    """Large-``x`` asymptotic for ``E_alpha(-x)``.

    For negative real arguments ``z=-x`` with ``x>0``:

    ``E_alpha(-x) ~ - sum_{k>=1} z^{-k} / Gamma(1 - alpha k)``.

    For ``1 < alpha < 2`` the principal exponential pair contributes an
    additional damped oscillatory term on the negative real axis:

    ``(2 / alpha) exp(rho cos(pi / alpha)) cos(rho sin(pi / alpha))``,
    where ``rho = x^(1 / alpha)``. This is the first paper-oriented correction
    beyond the purely algebraic prototype.
    """

    x = (-z).clamp_min(torch.finfo(z.dtype).tiny)
    k = torch.arange(1, terms + 1, dtype=z.dtype, device=z.device)
    rgamma = reciprocal_gamma(1.0 - alpha * k)
    signs = torch.where(
        (k.to(torch.int64) % 2) == 1,
        torch.ones_like(k),
        -torch.ones_like(k),
    )
    algebraic = torch.sum(signs * x.unsqueeze(-1).pow(-k) * rgamma, dim=-1)

    if bool(torch.all(alpha.detach() > 1.0).item()):
        theta = torch.pi / alpha
        rho = x.pow(1.0 / alpha)
        oscillatory = (2.0 / alpha) * torch.exp(rho * torch.cos(theta)) * torch.cos(
            rho * torch.sin(theta)
        )
        return algebraic + oscillatory
    return algebraic


def mittag_leffler_e_hybrid(
    alpha: torch.Tensor | float,
    z: torch.Tensor,
    *,
    series_terms: int = 80,
    asymptotic_terms: int = 8,
    threshold: float = 12.0,
) -> torch.Tensor:
    """Hybrid evaluator for real negative arguments.

    The small/moderate branch uses the differentiable series. The large negative
    branch uses an algebraic asymptotic expansion. Positive and complex
    arguments are outside this first-stage prototype.
    """

    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha, dtype=z.dtype, device=z.device)
    else:
        alpha = alpha.to(dtype=z.dtype, device=z.device)

    if torch.any(z > 0):
        raise ValueError("hybrid evaluator currently supports non-positive real z only")

    active_threshold, lo, hi = hybrid_switch_region(alpha, z, threshold=threshold)

    out = torch.empty_like(z)
    radius = torch.abs(z)
    small = radius <= lo
    large = radius >= hi
    blend = ~(small | large)
    if torch.any(small):
        out[small] = _mittag_leffler_series_terms(alpha, z[small], terms=series_terms)
    if torch.any(large):
        out[large] = _mittag_leffler_negative_asymptotic(
            alpha,
            z[large],
            terms=asymptotic_terms,
        )
    if torch.any(blend):
        eta = ((radius[blend] - lo) / (hi - lo)).clamp(0.0, 1.0)
        weight = eta * eta * (3.0 - 2.0 * eta)
        series = _mittag_leffler_series_terms(alpha, z[blend], terms=series_terms)
        asymptotic = _mittag_leffler_negative_asymptotic(
            alpha,
            z[blend],
            terms=asymptotic_terms,
        )
        out[blend] = (1.0 - weight) * series + weight * asymptotic
    return out


class MittagLefflerSeriesFunction(torch.autograd.Function):
    """Custom-autograd truncated series for ``E_alpha(z)``.

    This function is a research prototype: it makes the backward pass explicit
    and inspectable. It does not yet solve the large-``|z|`` stability problem;
    that belongs to the next stable evaluator stage.
    """

    @staticmethod
    def forward(ctx, alpha: torch.Tensor, z: torch.Tensor, terms: int) -> torch.Tensor:
        ctx.terms = int(terms)
        ctx.alpha_shape = alpha.shape
        ctx.save_for_backward(alpha, z)
        return _mittag_leffler_series_terms(alpha, z, terms=int(terms))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        alpha, z = ctx.saved_tensors
        terms = ctx.terms

        k = torch.arange(1, terms, dtype=z.dtype, device=z.device)
        log_gamma = torch.lgamma(alpha * k + 1.0)
        inv_gamma = torch.exp(-log_gamma)
        powers = z.unsqueeze(-1).pow(k)

        d_e_dz = torch.sum(k * z.unsqueeze(-1).pow(k - 1.0) * inv_gamma, dim=-1)
        digamma = torch.digamma(alpha * k + 1.0)
        d_e_dalpha = -torch.sum(powers * k * digamma * inv_gamma, dim=-1)

        grad_z = grad_output * d_e_dz
        grad_alpha = _sum_to_shape(grad_output * d_e_dalpha, ctx.alpha_shape)
        return grad_alpha, grad_z, None


def mittag_leffler_e(
    alpha: torch.Tensor | float,
    z: torch.Tensor,
    *,
    terms: int = 80,
    custom_backward: bool = False,
    method: str = "series",
) -> torch.Tensor:
    """Compute the one-parameter Mittag-Leffler function ``E_alpha(z)``.

    ``E_alpha(z) = sum_{k=0}^{inf} z^k / Gamma(alpha k + 1)``.

    Parameters
    ----------
    alpha:
        Fractional order. Can be a scalar tensor requiring gradients.
    z:
        Tensor of real arguments.
    terms:
        Number of series terms. Keep experiments in moderate ``|z|`` regimes.
    """

    if terms < 2:
        raise ValueError("terms must be >= 2")

    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha, dtype=z.dtype, device=z.device)
    else:
        alpha = alpha.to(dtype=z.dtype, device=z.device)

    if method == "hybrid":
        if custom_backward:
            raise ValueError("custom_backward is currently only available for method='series'")
        return mittag_leffler_e_hybrid(alpha, z, series_terms=terms)

    if method != "series":
        raise ValueError(f"unknown Mittag-Leffler method: {method}")

    if custom_backward:
        return MittagLefflerSeriesFunction.apply(alpha, z, int(terms))
    return _mittag_leffler_series_terms(alpha, z, terms=terms)


def mittag_leffler_e_prime_z(
    alpha: torch.Tensor | float,
    z: torch.Tensor,
    *,
    terms: int = 80,
) -> torch.Tensor:
    """Derivative ``d E_alpha(z) / dz`` from the differentiated series."""

    if terms < 2:
        raise ValueError("terms must be >= 2")

    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha, dtype=z.dtype, device=z.device)
    else:
        alpha = alpha.to(dtype=z.dtype, device=z.device)

    k = torch.arange(1, terms, dtype=z.dtype, device=z.device)
    log_gamma = torch.lgamma(alpha * k + 1.0)
    powers = z.unsqueeze(-1).pow(k - 1.0)
    coeffs = k * torch.exp(-log_gamma)
    return torch.sum(powers * coeffs, dim=-1)
