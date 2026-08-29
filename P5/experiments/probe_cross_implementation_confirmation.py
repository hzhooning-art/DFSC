"""Stage 59: independent numerical and decision-rule replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm

from probe_high_dimensional_shared_spectrum import DEVICE, DTYPE, independent_lifted_response


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STAGE57 = RESULTS / "morphology_calibrated_asymmetric_hierarchy.json"
STAGE58 = RESULTS / "external_multiphysics_confirmation.json"


def scipy_lifted_response(times: np.ndarray, weights: np.ndarray, rates: np.ndarray) -> np.ndarray:
    channels, rank = weights.shape
    output = np.empty((len(times), channels), dtype=float)
    for channel in range(channels):
        matrix = np.zeros((rank + 1, rank + 1), dtype=float)
        matrix[0, 1:] = -weights[channel]
        matrix[1:, 0] = 1.0
        matrix[np.arange(rank) + 1, np.arange(rank) + 1] = -rates[channel]
        initial = np.zeros(rank + 1, dtype=float)
        initial[0] = 1.0
        for index, time in enumerate(times):
            output[index, channel] = (expm(time * matrix) @ initial)[0]
    return output


def independent_decision(pair: dict, thresholds: dict) -> tuple[str, str]:
    if not pair["numerical_eligible"]:
        return "INDETERMINATE", "NUMERICAL_AXIS_INELIGIBLE"
    structural = pair["structural_score"]
    if structural is not None and structural > thresholds["structural_threshold"]:
        return "REFUSE", "STRONG_STRUCTURAL_REFUSAL"
    validation = pair["validation_score"]
    if (
        validation is not None
        and validation > thresholds["validation_threshold"]
        and pair["structural_refuse_votes"] >= 1
    ):
        return "REFUSE", "STRONG_VALIDATION_WITH_STRUCTURAL_SUPPORT"
    if pair["validation_axis"] == "RETAIN" and pair["structural_axis"] == "RETAIN":
        return "RETAIN", "CONCORDANT_RETENTION"
    return "INDETERMINATE", "INSUFFICIENT_CONCORDANT_EVIDENCE"


def numerical_trials() -> list[dict]:
    rng = np.random.default_rng(259001)
    trials = []
    for channels, rank, horizon in ((2, 2, 1.0), (5, 3, 2.5), (8, 2, 5.0)):
        times = np.linspace(0.0, horizon, 9)
        weights = rng.uniform(0.03, 0.20, size=(channels, rank))
        rates = rng.uniform(0.15, 2.0, size=(channels, rank))
        torch_rates = torch.tensor(rates, dtype=DTYPE, device=DEVICE, requires_grad=True)
        torch_weights = torch.tensor(weights, dtype=DTYPE, device=DEVICE)
        torch_times = torch.tensor(times, dtype=DTYPE, device=DEVICE)
        output = independent_lifted_response(torch_times, torch_weights, torch_rates)
        reference = scipy_lifted_response(times, weights, rates)
        value_abs = float(np.max(np.abs(output.detach().cpu().numpy() - reference)))
        value_rel = value_abs / max(float(np.max(np.abs(reference))), 1e-15)

        probe = rng.normal(size=reference.shape)
        scalar = torch.sum(output * torch.tensor(probe, dtype=DTYPE, device=DEVICE))
        scalar.backward()
        ad_gradient = torch_rates.grad.detach().cpu().numpy()
        fd_gradient = np.empty_like(rates)
        step = 2.0e-6
        for index in np.ndindex(rates.shape):
            plus = rates.copy()
            minus = rates.copy()
            plus[index] += step
            minus[index] -= step
            fd_gradient[index] = (
                np.sum(scipy_lifted_response(times, weights, plus) * probe)
                - np.sum(scipy_lifted_response(times, weights, minus) * probe)
            ) / (2.0 * step)
        gradient_abs = float(np.max(np.abs(ad_gradient - fd_gradient)))
        gradient_rel = gradient_abs / max(float(np.max(np.abs(fd_gradient))), 1e-12)
        trials.append({
            "channels": channels,
            "rank": rank,
            "horizon": horizon,
            "value_absolute_error": value_abs,
            "value_relative_error": value_rel,
            "gradient_absolute_error": gradient_abs,
            "gradient_relative_error": gradient_rel,
        })
    return trials


def decision_replay() -> dict:
    comparisons = []
    for path, pair_key, threshold_key in (
        (STAGE57, "evaluation_pairs", "thresholds"),
        (STAGE58, "pairs", "frozen_thresholds"),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for pair in payload[pair_key]:
            decision, reason = independent_decision(pair, payload[threshold_key])
            comparisons.append({
                "source": path.name,
                "expected_decision": pair["morphology_calibrated_class"],
                "replayed_decision": decision,
                "expected_reason": pair["morphology_calibrated_reason"],
                "replayed_reason": reason,
                "match": decision == pair["morphology_calibrated_class"] and reason == pair["morphology_calibrated_reason"],
            })
    return {
        "comparisons": comparisons,
        "count": len(comparisons),
        "concordance": sum(row["match"] for row in comparisons) / len(comparisons),
    }


def build_payload() -> dict:
    trials = numerical_trials()
    replay = decision_replay()
    maxima = {
        "value_relative_error": max(row["value_relative_error"] for row in trials),
        "gradient_relative_error": max(row["gradient_relative_error"] for row in trials),
    }
    checks = {
        "scipy_value_relative_error_at_most_1e_10": maxima["value_relative_error"] <= 1e-10,
        "finite_difference_gradient_relative_error_at_most_1e_6": maxima["gradient_relative_error"] <= 1e-6,
        "independent_decision_concordance_equals_one": replay["concordance"] == 1.0,
        "replayed_at_least_70_decisions": replay["count"] >= 70,
    }
    return {
        "experiment": "stage59_cross_implementation_confirmation",
        "numerical_reference": "scipy.linalg.expm plus central finite differences",
        "decision_reference": "standalone pure-Python rule with no import from the production decision module",
        "numerical_trials": trials,
        "maxima": maxima,
        "decision_replay": replay,
        "checks": checks,
        "route_pass": all(checks.values()),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cross_implementation_confirmation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Cross-implementation confirmation",
        "",
        f"Route pass: **{payload['route_pass']}**.",
        "",
        f"- Maximum value relative error: {payload['maxima']['value_relative_error']:.3e}",
        f"- Maximum gradient relative error: {payload['maxima']['gradient_relative_error']:.3e}",
        f"- Decision replay concordance: {payload['decision_replay']['concordance']:.4f}",
        f"- Replayed decisions: {payload['decision_replay']['count']}",
    ]
    (RESULTS / "cross_implementation_confirmation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = build_payload()
    write_outputs(payload)
    print(json.dumps({"maxima": payload["maxima"], "checks": payload["checks"], "route_pass": payload["route_pass"]}, indent=2))


if __name__ == "__main__":
    main()
