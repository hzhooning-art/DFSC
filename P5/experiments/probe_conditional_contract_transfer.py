"""Stage 53: transfer a frozen Stage 52 contract without recalibration."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from probe_budget_consensus_abstention import consensus_decision
from probe_cross_start_consistency_transfer import select_consistent_candidate
from probe_decomposed_tolerance_transfer import (
    BLOCKS,
    HORIZON,
    NOISE_CORRELATION,
    PROXY_SCOPE,
    TRUE_CENTRAL_RATES,
    frozen_total_tolerance,
)
from probe_extended_refinement_transfer import ADAM_STEPS, load_frozen_stage48
from probe_high_dimensional_shared_spectrum import (
    DEVICE,
    DTYPE,
    fit_candidate,
    independent_lifted_response,
)
from probe_nested_group_sharing_gate import fit_grouped_candidate
from probe_noise_aware_sharing_gate import classify_with_limit, second_difference_correlation_proxy
from probe_noise_scale_optimizer_transfer import (
    GROUPED_STARTS,
    SHARED_STARTS,
    mixed_difference_noise_scale,
    scale_correction,
)
from probe_optimizer_budget_stability import LBFGS_BUDGETS, decision_class


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STAGE52_RESULT = RESULTS / "high_noise_conditional_calibration.json"
SEED_BASE = 181000
CHANNELS = (48, 192)
CORE_NOISE_STDS = (8.0e-4, 1.4e-3)
STRESS_NOISE_STD = 2.0e-3
CONSTRUCTIONS = ("antisymmetric", "rotated")
DRIFTS = (0.0, 0.05, 0.15)
REPEATS = 2

MAX_CORE_FALSE_REFUSAL = 0.05
MIN_CORE_RETAIN = 0.80
MIN_CORE_SEVERE_REFUSAL = 0.80
MAX_CORE_BUDGET_SENSITIVE = 0.20


def score_matrix(construction: str) -> np.ndarray:
    base = np.asarray([-1.5, -0.5, 0.5, 1.5])
    if construction == "antisymmetric":
        scores = np.column_stack([base, -base])
    elif construction == "rotated":
        second = np.asarray([0.5, -1.5, 1.5, -0.5])
        scores = np.column_stack([0.75 * base + 0.25 * second, 0.25 * base - 0.75 * second])
        target = float(np.sqrt(np.mean(np.column_stack([base, -base]) ** 2)))
        scores *= target / float(np.sqrt(np.mean(scores**2)))
    else:
        raise ValueError(f"unknown construction: {construction}")
    return scores


def build_transfer_observation(
    channels: int,
    noise_std: float,
    construction: str,
    drift: float,
    seed: int,
):
    if channels % BLOCKS:
        raise ValueError("channels must be divisible by four")
    rng = np.random.default_rng(seed)
    times = torch.linspace(0.0, HORIZON, 65, dtype=DTYPE, device=DEVICE)
    labels = np.repeat(np.arange(BLOCKS), channels // BLOCKS)
    scores = score_matrix(construction)[labels]
    rates = np.exp(np.log(TRUE_CENTRAL_RATES)[None, :] + drift * scores)
    amplitude = np.linspace(0.28, 0.78, channels)[:, None]
    tilt = np.linspace(0.80, 1.20, channels)[:, None]
    weights = np.concatenate([0.58 * amplitude / tilt, 0.42 * amplitude * tilt], axis=1)
    clean = independent_lifted_response(
        times,
        torch.tensor(weights, dtype=DTYPE, device=DEVICE),
        torch.tensor(rates, dtype=DTYPE, device=DEVICE),
    )
    common = rng.standard_normal((times.numel(), 1))
    independent = rng.standard_normal((times.numel(), channels))
    noise = math.sqrt(NOISE_CORRELATION) * common + math.sqrt(1.0 - NOISE_CORRELATION) * independent
    observations = clean + noise_std * torch.tensor(noise, dtype=DTYPE, device=DEVICE)
    split = 48
    train = np.sort(rng.choice(np.arange(1, split), size=34, replace=False))
    return (
        times,
        observations,
        torch.tensor(train, dtype=torch.long, device=DEVICE),
        torch.tensor(np.arange(split, times.numel()), dtype=torch.long, device=DEVICE),
        labels,
    )


def fit_record(
    channels: int,
    noise_std: float,
    construction: str,
    drift: float,
    repeat: int,
    budget: int,
    stage52: dict,
    noise_calibration: dict,
    consistency_calibration: dict,
) -> dict:
    code = CONSTRUCTIONS.index(construction)
    seed = SEED_BASE + channels * 10 + int(noise_std * 1e6) + code * 1000 + int(drift * 1000) + repeat
    times, observations, train_idx, val_idx, labels = build_transfer_observation(
        channels, noise_std, construction, drift, seed
    )
    shared_candidates = [
        fit_candidate(
            times, observations, train_idx, val_idx, 2, True, seed * 10 + start,
            adam_steps=ADAM_STEPS, lbfgs_steps=budget,
        )
        for start in range(SHARED_STARTS)
    ]
    shared, diagnostics, adequate_starts, signal_scale = select_consistent_candidate(
        shared_candidates, times, observations, consistency_calibration
    )
    grouped = min(
        [
            fit_grouped_candidate(
                times, observations, train_idx, val_idx,
                torch.tensor(labels, dtype=torch.long, device=DEVICE),
                seed * 10 + 20 + start, adam_steps=ADAM_STEPS, lbfgs_steps=budget,
            )
            for start in range(GROUPED_STARTS)
        ],
        key=lambda candidate: candidate.bic,
    )
    noise_proxy = mixed_difference_noise_scale(observations)
    correlation_proxy = second_difference_correlation_proxy(observations)
    base = frozen_total_tolerance(correlation_proxy) + scale_correction(noise_proxy, noise_calibration)
    adjusted = base * stage52["validation_tolerance_multiplier"]
    in_scope = (
        noise_calibration["noise_proxy_min"] <= noise_proxy <= noise_calibration["noise_proxy_max"]
        and PROXY_SCOPE[0] <= correlation_proxy <= PROXY_SCOPE[1]
    )
    if shared is None:
        support = None
        shared_val = None
        decision = "INDETERMINATE_OPTIMIZATION"
    else:
        support = shared.bic - grouped.bic
        shared_val = shared.val_rmse
        decision = classify_with_limit(support, shared.val_rmse, grouped.val_rmse, adjusted)
    return {
        "channels": channels,
        "noise_std": noise_std,
        "noise_domain": "core" if noise_std in CORE_NOISE_STDS else "stress",
        "construction": construction,
        "drift": drift,
        "repeat": repeat,
        "seed": seed,
        "lbfgs_steps": budget,
        "decision": decision,
        "decision_class": decision_class(decision),
        "group_bic_support": support,
        "shared_val_rmse": shared_val,
        "grouped_val_rmse": grouped.val_rmse,
        "base_tolerance_before_stage52": base,
        "adjusted_tolerance": adjusted,
        "noise_scale_proxy": noise_proxy,
        "correlation_proxy": correlation_proxy,
        "diagnostics_in_calibration_scope": in_scope,
        "adequate_shared_starts": adequate_starts,
        "cross_start_diagnostics": diagnostics,
        "signal_scale": signal_scale,
    }


def aggregate_pairs(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (
            record["channels"], record["noise_std"], record["construction"],
            record["drift"], record["repeat"], record["seed"],
        )
        grouped[key].append(record)
    pairs = []
    for key, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: item["lbfgs_steps"])
        classes = [item["decision_class"] for item in ordered]
        consensus, reason = consensus_decision(classes)
        pairs.append({
            "channels": key[0], "noise_std": key[1],
            "noise_domain": "core" if key[1] in CORE_NOISE_STDS else "stress",
            "construction": key[2], "drift": key[3], "repeat": key[4], "seed": key[5],
            "budget_classes": classes, "class_counts": dict(Counter(classes)),
            "consensus_class": consensus, "consensus_reason": reason,
            "budget_sensitive": reason == "BUDGET_SENSITIVE_BINARY_CONFLICT",
        })
    return pairs


def domain_metrics(pairs: list[dict], domain: str) -> dict:
    local = [item for item in pairs if item["noise_domain"] == domain]
    acceptable = [item for item in local if item["drift"] <= 0.05]
    severe = [item for item in local if item["drift"] == 0.15]
    return {
        "pairs": len(local),
        "false_refusal_fraction": float(np.mean([x["consensus_class"] == "REFUSE" for x in acceptable])),
        "retain_fraction": float(np.mean([x["consensus_class"] == "RETAIN" for x in acceptable])),
        "severe_refusal_fraction": float(np.mean([x["consensus_class"] == "REFUSE" for x in severe])),
        "budget_sensitive_fraction": float(np.mean([x["budget_sensitive"] for x in local])),
    }


def summarize(records: list[dict]) -> dict:
    pairs = aggregate_pairs(records)
    core = domain_metrics(pairs, "core")
    stress = domain_metrics(pairs, "stress")
    core_records = [r for r in records if r["noise_domain"] == "core"]
    checks = {
        "complete_transfer_matrix": len(pairs) == 72,
        "core_false_refusal_at_most_0_05": core["false_refusal_fraction"] <= MAX_CORE_FALSE_REFUSAL,
        "core_retain_at_least_0_80": core["retain_fraction"] >= MIN_CORE_RETAIN,
        "core_severe_refusal_at_least_0_80": core["severe_refusal_fraction"] >= MIN_CORE_SEVERE_REFUSAL,
        "core_budget_sensitive_at_most_0_20": core["budget_sensitive_fraction"] <= MAX_CORE_BUDGET_SENSITIVE,
        "core_diagnostics_in_scope": all(r["diagnostics_in_calibration_scope"] for r in core_records),
    }
    return {"pairs": pairs, "core": core, "stress": stress, "checks": checks, "route_pass": all(checks.values())}


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "conditional_contract_transfer.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    s = payload["summary"]
    lines = [
        "# Frozen conditional-contract transfer",
        "",
        f"Core route pass: **{s['route_pass']}**.",
        f"Core: false refusal {s['core']['false_refusal_fraction']:.3f}, retain {s['core']['retain_fraction']:.3f}, severe refusal {s['core']['severe_refusal_fraction']:.3f}, budget-sensitive {s['core']['budget_sensitive_fraction']:.3f}.",
        f"Stress: false refusal {s['stress']['false_refusal_fraction']:.3f}, retain {s['stress']['retain_fraction']:.3f}, severe refusal {s['stress']['severe_refusal_fraction']:.3f}, budget-sensitive {s['stress']['budget_sensitive_fraction']:.3f}.",
        "",
        "The 2.0e-3 noise condition is a declared extrapolation stress test and is not pooled into the core pass criterion.",
    ]
    (RESULTS / "conditional_contract_transfer.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    stage52_payload = json.loads(STAGE52_RESULT.read_text(encoding="utf-8"))
    stage52 = stage52_payload["calibration"]
    noise_calibration, consistency_calibration = load_frozen_stage48()
    records = []
    for channels in CHANNELS:
        for noise_std in (*CORE_NOISE_STDS, STRESS_NOISE_STD):
            for construction in CONSTRUCTIONS:
                for drift in DRIFTS:
                    for repeat in range(REPEATS):
                        for budget in LBFGS_BUDGETS:
                            record = fit_record(
                                channels, noise_std, construction, drift, repeat, budget,
                                stage52, noise_calibration, consistency_calibration,
                            )
                            records.append(record)
                            print(
                                f"stage53 channels={channels} noise={noise_std:.1e} construction={construction} "
                                f"drift={drift:.2f} repeat={repeat} budget={budget} decision={record['decision']}",
                                flush=True,
                            )
    summary = summarize(records)
    payload = {
        "experiment": "conditional_contract_transfer",
        "device": str(DEVICE), "dtype": str(DTYPE),
        "protocol": {
            "seed_base": SEED_BASE, "channels": list(CHANNELS),
            "core_noise_stds": list(CORE_NOISE_STDS), "stress_noise_std": STRESS_NOISE_STD,
            "constructions": list(CONSTRUCTIONS), "drifts": list(DRIFTS),
            "repeats": REPEATS, "budgets": list(LBFGS_BUDGETS),
            "stage52_calibration_reused_without_recalibration": True,
        },
        "stage52_calibration": stage52,
        "records": records, "summary": summary,
        "exit_rule": {
            "failure_action": "do not add another empirical threshold; retain only a scoped or negative reliability result",
        },
    }
    write_outputs(payload)
    print(json.dumps({"core": summary["core"], "stress": summary["stress"], "checks": summary["checks"], "route_pass": summary["route_pass"]}, indent=2))


if __name__ == "__main__":
    main()
