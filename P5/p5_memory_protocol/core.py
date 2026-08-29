"""Backend-neutral shared-relaxation fitting and auditable decision primitives."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares, lsq_linear, nnls


@dataclass(frozen=True)
class CurveRecord:
    unit: str
    group: str
    channel: str
    time: np.ndarray
    value: np.ndarray


@dataclass(frozen=True)
class GateConfig:
    delta_bic: float = 10.0
    predictive_gain: float = 0.05
    max_log_rate_std: float = 0.80
    min_rate_ratio: float = 1.20
    use_ar1_bic: bool = False


def _design(time: np.ndarray, rates: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones_like(time), np.exp(-np.outer(time, rates))))


def _stack(curves: Iterable[CurveRecord]) -> tuple[np.ndarray, np.ndarray]:
    rows = list(curves)
    if not rows:
        raise ValueError("at least one curve is required")
    time = np.asarray(rows[0].time, dtype=float)
    if any(len(row.time) != len(time) or not np.allclose(row.time, time) for row in rows):
        raise ValueError("the shared-rate backend requires a common registered time grid")
    values = np.stack([np.asarray(row.value, dtype=float) for row in rows])
    if not np.isfinite(values).all():
        raise ValueError("curves must be finite")
    return time, values


def _initial_rates(rank: int, start: int, bounds: tuple[float, float]) -> np.ndarray:
    shifts = (0.55, 0.78, 1.0, 1.32, 1.75, 2.25, 2.9, 3.6)
    base = np.geomspace(max(bounds[0] * 3.0, 0.003), min(bounds[1] / 3.0, 1.0), rank)
    return np.clip(base * shifts[start % len(shifts)], bounds[0] * 1.001, bounds[1] / 1.001)


def _conditional_fit(
    time: np.ndarray,
    values: np.ndarray,
    rates: np.ndarray,
    *,
    nonnegative_amplitudes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    design = _design(time, rates)
    if nonnegative_amplitudes:
        lower = np.concatenate(([-np.inf], np.zeros(len(rates))))
        upper = np.full(len(rates) + 1, np.inf)
        coefficients = np.column_stack([
            lsq_linear(design, row, bounds=(lower, upper)).x for row in values
        ])
    else:
        coefficients = np.linalg.lstsq(design, values.T, rcond=None)[0]
    return coefficients, (design @ coefficients).T


def residual_ar1_diagnostics(residuals: np.ndarray) -> dict:
    """Estimate pooled within-curve AR(1) dependence and effective sample size."""
    array = np.asarray(residuals, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    centered = array - np.mean(array, axis=1, keepdims=True)
    numerator = float(np.sum(centered[:, 1:] * centered[:, :-1]))
    denominator = float(np.sum(centered[:, :-1] ** 2))
    rho = float(np.clip(numerator / max(denominator, 1e-30), -0.95, 0.95))
    n = int(array.size)
    effective = float(np.clip(n * (1.0 - rho) / (1.0 + rho), array.shape[0], n))
    return {"rho_ar1": rho, "effective_sample_size": effective, "n_observations": n}


def ar1_profile_bic(residuals: np.ndarray, parameter_count: int) -> dict:
    """Profiled Gaussian AR(1) BIC using independent curve blocks."""
    array = np.asarray(residuals, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    diagnostics = residual_ar1_diagnostics(array)
    rho = diagnostics["rho_ar1"]
    innovations = np.concatenate([
        np.sqrt(max(1.0 - rho * rho, 1e-8)) * array[:, :1],
        array[:, 1:] - rho * array[:, :-1],
    ], axis=1)
    innovation_sse = float(np.sum(innovations**2) / max(1.0 - rho * rho, 1e-8))
    n = int(array.size)
    logdet = array.shape[0] * (array.shape[1] - 1) * np.log(max(1.0 - rho * rho, 1e-8))
    bic = n * np.log(max(innovation_sse / n, 1e-300)) + logdet + parameter_count * np.log(n)
    return {**diagnostics, "innovation_sse": innovation_sse, "ar1_bic": float(bic)}


def fit(
    curves: Iterable[CurveRecord],
    rank: int,
    *,
    starts: int = 6,
    rate_bounds: tuple[float, float] = (1.0 / 300.0, 2.0),
    nonnegative_amplitudes: bool = False,
) -> dict:
    """Fit shared positive decay rates and curve-specific signed amplitudes."""
    rows = list(curves)
    time, values = _stack(rows)
    low, high = np.log(rate_bounds)

    def residual(log_rates: np.ndarray) -> np.ndarray:
        rates = np.sort(np.exp(log_rates))
        _, prediction = _conditional_fit(
            time, values, rates, nonnegative_amplitudes=nonnegative_amplitudes
        )
        return (prediction - values).ravel()

    best = None
    for start in range(int(starts)):
        result = least_squares(
            residual,
            np.log(_initial_rates(rank, start, rate_bounds)),
            bounds=(np.full(rank, low), np.full(rank, high)),
            max_nfev=350,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        rates = np.sort(np.exp(result.x))
        error = residual(np.log(rates))
        sse = float(error @ error)
        candidate = {"rates": rates.tolist(), "sse": sse, "success": bool(result.success)}
        if best is None or sse < best["sse"]:
            best = candidate
    assert best is not None
    n = int(values.size)
    parameters = rank + len(rows) * (rank + 1)
    best["bic"] = float(n * np.log(max(best["sse"] / n, 1e-300)) + parameters * np.log(n))
    rates = np.asarray(best["rates"])
    _, prediction = _conditional_fit(
        time, values, rates, nonnegative_amplitudes=nonnegative_amplitudes
    )
    best.update(ar1_profile_bic(prediction - values, parameters))
    best["minimum_rate_ratio"] = float(np.min(rates[1:] / rates[:-1])) if rank > 1 else math.inf
    best["n_curves"] = len(rows)
    best["n_observations"] = n
    best["amplitude_constraint"] = "nonnegative" if nonnegative_amplitudes else "signed"
    return best


def identifiability_certificate(
    curves: Iterable[CurveRecord],
    rates: Iterable[float],
    *,
    noise_std: float | None = None,
) -> dict:
    """Quantify local rate information after eliminating linear nuisance terms.

    The reported index uses derivatives with respect to log-rates, so the
    adjacent-gap proxy is dimensionless.  It is a local diagnostic, not a
    universal confidence statement.
    """
    rows = list(curves)
    time, values = _stack(rows)
    rates_array = np.sort(np.asarray(list(rates), dtype=float))
    if rates_array.ndim != 1 or len(rates_array) == 0 or np.any(rates_array <= 0):
        raise ValueError("rates must be a non-empty vector of positive values")

    design = _design(time, rates_array)
    coefficients, prediction = _conditional_fit(time, values, rates_array)
    projected_blocks = []
    for curve_index in range(len(rows)):
        amplitudes = coefficients[1:, curve_index]
        sensitivities = np.column_stack([
            -amplitudes[k] * rates_array[k] * time * np.exp(-rates_array[k] * time)
            for k in range(len(rates_array))
        ])
        nuisance_projection = design @ np.linalg.lstsq(design, sensitivities, rcond=None)[0]
        projected_blocks.append(sensitivities - nuisance_projection)
    projected = np.vstack(projected_blocks)

    residual = values - prediction
    parameter_count = len(rates_array) + len(rows) * (len(rates_array) + 1)
    degrees_of_freedom = max(int(values.size) - parameter_count, 1)
    estimated_noise = float(np.sqrt(np.sum(residual**2) / degrees_of_freedom))
    sigma = estimated_noise if noise_std is None else float(noise_std)
    sigma = max(sigma, np.finfo(float).eps)

    singular_values = np.linalg.svd(projected, compute_uv=False)
    smallest = float(singular_values[-1]) if len(singular_values) else 0.0
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    minimum_log_gap = (
        float(np.min(np.diff(np.log(rates_array)))) if len(rates_array) > 1 else math.inf
    )
    information_eigenvalue = float((smallest / sigma) ** 2)
    boundary_index = (
        float(minimum_log_gap * smallest / sigma) if len(rates_array) > 1 else math.inf
    )
    normalized_boundary_index = (
        float(boundary_index / np.sqrt(values.size)) if len(rates_array) > 1 else math.inf
    )
    return {
        "rates": rates_array.tolist(),
        "noise_std": sigma,
        "noise_std_source": "residual" if noise_std is None else "declared",
        "projected_log_rate_singular_values": singular_values.tolist(),
        "minimum_projected_information_eigenvalue": information_eigenvalue,
        "projected_condition_number": float(largest / max(smallest, np.finfo(float).eps)),
        "minimum_log_rate_gap": minimum_log_gap,
        "local_boundary_index": boundary_index,
        "normalized_local_boundary_index": normalized_boundary_index,
        "interpretation": (
            "Local information after curve-specific offsets and amplitudes are projected out; "
            "small values indicate an unresolved rate direction."
        ),
        "comparability": (
            "The normalized index removes leading square-root sample-size scaling, but both "
            "indices remain task-internal diagnostics until calibrated by simulation."
        ),
    }


def _held_errors(curves: list[CurveRecord], rates: np.ndarray, calibration_fraction: float) -> list[dict]:
    time, values = _stack(curves)
    split = max(4, min(len(time) - 2, int(math.ceil(calibration_fraction * len(time)))))
    coefficients, _ = _conditional_fit(time[:split], values[:, :split], rates)
    prediction = (_design(time[split:], rates) @ coefficients).T
    scales = np.maximum(np.ptp(values[:, :split], axis=1), 0.10)
    errors = np.sqrt(np.mean((prediction - values[:, split:]) ** 2, axis=1)) / scales
    return [
        {"unit": row.unit, "group": row.group, "channel": row.channel, "nrmse": float(err)}
        for row, err in zip(curves, errors)
    ]


def evaluate(
    curves: Iterable[CurveRecord],
    ranks: tuple[int, ...] = (1, 2, 3),
    *,
    starts: int = 6,
    calibration_fraction: float = 0.60,
    rate_bounds: tuple[float, float] = (1.0 / 300.0, 2.0),
    nonnegative_amplitudes: bool = False,
) -> dict:
    """Leave-one-group-out evaluation with group-independent rate fitting."""
    rows = list(curves)
    groups = sorted({row.group for row in rows})
    records = {}
    for rank in ranks:
        folds = []
        for held in groups:
            training = [row for row in rows if row.group != held]
            testing = [row for row in rows if row.group == held]
            fitted = fit(
                training,
                rank,
                starts=starts,
                rate_bounds=rate_bounds,
                nonnegative_amplitudes=nonnegative_amplitudes,
            )
            folds.append({
                "held_group": held,
                **fitted,
                "errors": _held_errors(testing, np.asarray(fitted["rates"]), calibration_fraction),
            })
        rate_matrix = np.asarray([np.log(fold["rates"]) for fold in folds])
        errors = [item for fold in folds for item in fold["errors"]]
        records[str(rank)] = {
            "folds": folds,
            "mean_bic": float(np.mean([fold["bic"] for fold in folds])),
            "mean_ar1_bic": float(np.mean([fold["ar1_bic"] for fold in folds])),
            "mean_residual_rho_ar1": float(np.mean([fold["rho_ar1"] for fold in folds])),
            "mean_effective_sample_size": float(np.mean([fold["effective_sample_size"] for fold in folds])),
            "median_prediction_nrmse": float(np.median([item["nrmse"] for item in errors])),
            "max_log_rate_std": float(np.max(np.std(rate_matrix, axis=0, ddof=1))) if len(groups) > 1 else 0.0,
            "minimum_rate_ratio": float(min(fold["minimum_rate_ratio"] for fold in folds)),
            "curve_errors": errors,
            "all_finite": bool(np.isfinite(rate_matrix).all() and all(fold["success"] for fold in folds)),
        }
    return {"groups": groups, "ranks": list(ranks), "rank_records": records}


def decide(evaluation: dict, gates: GateConfig = GateConfig()) -> dict:
    """Apply predeclared nested-rank acceptance and explicit refusal rules."""
    records = evaluation["rank_records"]
    ranks = sorted(int(key) for key in records)
    selected = ranks[0]
    conflict = False
    transitions = []
    bic_key = "mean_ar1_bic" if gates.use_ar1_bic else "mean_bic"
    for lower_rank, rank in zip(ranks[:-1], ranks[1:]):
        lower, current = records[str(lower_rank)], records[str(rank)]
        delta_bic = lower[bic_key] - current[bic_key]
        gain = (lower["median_prediction_nrmse"] - current["median_prediction_nrmse"]) / max(
            lower["median_prediction_nrmse"], 1e-15
        )
        checks = {
            "bic": bool(delta_bic >= gates.delta_bic),
            "prediction": bool(gain >= gates.predictive_gain),
            "stability": bool(current["max_log_rate_std"] <= gates.max_log_rate_std),
            "separation": bool(current["minimum_rate_ratio"] >= gates.min_rate_ratio),
            "finite": bool(current["all_finite"]),
        }
        transitions.append({"from_rank": lower_rank, "to_rank": rank, "delta_bic": delta_bic, "information_criterion": bic_key, "prediction_gain": gain, "gates": checks})
        if checks["bic"]:
            if all(checks.values()) and selected == lower_rank:
                selected = rank
            else:
                conflict = True
    result = decide_transitions(transitions)
    return {**result, "transitions": transitions, "gates": asdict(gates)}


_GATE_KEYS = {
    "information": "bic",
    "transfer": "prediction",
    "stability": "stability",
    "separation": "separation",
}


def decide_transitions(
    transitions: Iterable[dict],
    *,
    active_gates: Iterable[str] | None = None,
) -> dict:
    """Reapply the nested-rank rule to frozen transition records.

    This helper supports diagnostic leave-one-gate-out audits. Finiteness is
    always mandatory and cannot be disabled. The returned decision measures
    dependence on a declared evidence gate, not a newly fitted model.
    """
    rows = list(transitions)
    active = set(_GATE_KEYS) if active_gates is None else set(active_gates)
    unknown = active.difference(_GATE_KEYS)
    if unknown:
        raise ValueError(f"unknown evidence gates: {sorted(unknown)}")
    if not rows:
        raise ValueError("at least one transition is required")

    selected = int(rows[0]["from_rank"])
    conflict = False
    for row in rows:
        checks = row["gates"]
        triggered = "information" not in active or bool(checks["bic"])
        required = [bool(checks[_GATE_KEYS[name]]) for name in sorted(active)]
        passed = bool(checks.get("finite", False)) and all(required)
        if triggered:
            if passed and selected == int(row["from_rank"]):
                selected = int(row["to_rank"])
            elif not passed:
                conflict = True
    return {
        "decision": "INDETERMINATE" if conflict else f"SUPPORTED_RANK_{selected}",
        "selected_rank": None if conflict else selected,
        "active_gates": sorted(active),
    }


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Return Holm family-wise-error adjusted p-values by hypothesis name."""
    if not pvalues:
        return {}
    checked = {name: float(value) for name, value in pvalues.items()}
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in checked.values()):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    ordered = sorted(checked.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def fixed_grid_nnls_error(curve: CurveRecord, calibration_fraction: float = 0.60, grid_size: int = 16) -> float:
    split = max(4, min(len(curve.time) - 2, int(math.ceil(calibration_fraction * len(curve.time)))))
    rates = np.geomspace(1.0 / 300.0, 2.0, grid_size)
    coefficients, _ = nnls(_design(curve.time[:split], rates), curve.value[:split])
    prediction = _design(curve.time[split:], rates) @ coefficients
    scale = max(float(np.ptp(curve.value[:split])), 0.10)
    return float(np.sqrt(np.mean((prediction - curve.value[split:]) ** 2)) / scale)


def independent_nls_error(curve: CurveRecord, rank: int = 3, calibration_fraction: float = 0.60) -> float:
    split = max(4, min(len(curve.time) - 2, int(math.ceil(calibration_fraction * len(curve.time)))))
    short = CurveRecord(curve.unit, curve.group, curve.channel, curve.time[:split], curve.value[:split])
    fitted = fit([short], rank, starts=4)
    rates = np.asarray(fitted["rates"])
    coefficients, _ = _conditional_fit(curve.time[:split], curve.value[None, :split], rates)
    prediction = (_design(curve.time[split:], rates) @ coefficients).ravel()
    scale = max(float(np.ptp(curve.value[:split])), 0.10)
    return float(np.sqrt(np.mean((prediction - curve.value[split:]) ** 2)) / scale)


def prony_error(curve: CurveRecord, rank: int = 3, calibration_fraction: float = 0.60) -> float:
    split = max(4, min(len(curve.time) - 2, int(math.ceil(calibration_fraction * len(curve.time)))))
    observed = curve.value[:split].astype(float).tolist()
    lag = min(rank + 1, max(2, split // 2))
    y = np.asarray(observed)
    matrix = np.asarray([[y[k - j - 1] for j in range(lag)] for k in range(lag, split)])
    target = y[lag:]
    ridge = 1e-6 * np.eye(lag)
    coefficients = np.linalg.solve(matrix.T @ matrix + ridge, matrix.T @ target)
    for _ in range(len(curve.time) - split):
        observed.append(float(np.dot(coefficients, observed[-lag:][::-1])))
    prediction = np.asarray(observed[split:])
    scale = max(float(np.ptp(curve.value[:split])), 0.10)
    return float(np.sqrt(np.mean((prediction - curve.value[split:]) ** 2)) / scale)


def report(payload: dict, output: str | Path) -> Path:
    """Write a canonical, deterministic JSON reliability record."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
