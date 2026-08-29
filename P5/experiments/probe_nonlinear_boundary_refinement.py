"""Refine the nonlinear-refusal bracket and expose its noise-normalized driver."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from probe_controlled_misspecification import (
    LATE_AUDIT_RMSE_LIMIT,
    RATES,
    simulate_channel,
)
from probe_mechanism_vs_trajectory_baselines import (
    CHANNELS,
    DEVICE,
    DTYPE,
    HORIZON,
    NOISE_STD,
    NUM_POINTS,
    RESULTS,
    TRAIN_COUNT,
    TRAIN_END_FRACTION,
    fit_mechanism,
    rmse,
)
from probe_nonlinear_transition_boundary import wilson_interval


STRENGTHS = (0.075, 0.080, 0.085, 0.090, 0.095, 0.100)
REPEATS = 8
TRAIN_RMSE_LIMIT = max(4.0 * NOISE_STD, 3.0e-3)


def make_dataset(strength: float, repeat: int) -> dict:
    seed = 479000 + 10007 * STRENGTHS.index(strength) + 101 * repeat
    rng = np.random.default_rng(seed)
    times_np = np.linspace(0.0, HORIZON, NUM_POINTS)
    channel_scale = np.linspace(0.28, 0.82, CHANNELS)[:, None]
    pole_scale = np.linspace(0.78, 1.22, len(RATES))[None, :]
    weights = channel_scale * pole_scale / len(RATES)
    clean_np = np.column_stack(
        [
            simulate_channel(
                times_np,
                weights[channel],
                "nonlinear_feedback",
                strength,
            )
            for channel in range(CHANNELS)
        ]
    )
    observations_np = clean_np + NOISE_STD * rng.standard_normal(clean_np.shape)
    split = int(round(TRAIN_END_FRACTION * (NUM_POINTS - 1)))
    pool = np.arange(1, split + 1)
    sampled = np.sort(rng.choice(pool, size=TRAIN_COUNT - 1, replace=False))
    train_np = np.concatenate(([0], sampled))
    interpolation_np = np.setdiff1d(np.arange(split + 1), train_np)
    extrapolation_np = np.arange(split + 1, NUM_POINTS)
    return {
        "seed": seed,
        "strength": strength,
        "times": torch.tensor(times_np, dtype=DTYPE, device=DEVICE),
        "clean": torch.tensor(clean_np, dtype=DTYPE, device=DEVICE),
        "observations": torch.tensor(observations_np, dtype=DTYPE, device=DEVICE),
        "train_idx": torch.tensor(train_np, dtype=torch.long, device=DEVICE),
        "interpolation_idx": torch.tensor(
            interpolation_np, dtype=torch.long, device=DEVICE
        ),
        "extrapolation_idx": torch.tensor(
            extrapolation_np, dtype=torch.long, device=DEVICE
        ),
        "true_rank": 2,
    }


def selected_candidate(metadata: dict) -> dict:
    winners = [
        candidate
        for candidate in metadata["candidate_fits"]
        if candidate["rank"] == metadata["selected_rank"]
        and np.isclose(candidate["bic"], metadata["bic"], rtol=0.0, atol=1.0e-9)
    ]
    if len(winners) != 1:
        raise RuntimeError(f"Expected one selected candidate, found {len(winners)}")
    return winners[0]


def evaluate(strength: float, repeat: int) -> dict:
    started = time.perf_counter()
    data = make_dataset(strength, repeat)
    metadata, prediction = fit_mechanism(data)
    winner = selected_candidate(metadata)
    clean = data["clean"].detach().cpu().numpy()
    train_idx = data["train_idx"].detach().cpu().numpy()
    extrapolation_idx = data["extrapolation_idx"].detach().cpu().numpy()
    train_rmse_to_clean = rmse(prediction[train_idx], clean[train_idx])
    extrapolation_rmse = rmse(
        prediction[extrapolation_idx], clean[extrapolation_idx]
    )
    reasons = []
    if metadata["condition"] > 1.0e8:
        reasons.append("ill_conditioned_jacobian")
    if winner["train_rmse"] > TRAIN_RMSE_LIMIT:
        reasons.append("training_fit_quality")
    if extrapolation_rmse > LATE_AUDIT_RMSE_LIMIT:
        reasons.append("late_audit_error")
    refusal = bool(reasons)
    if refusal != bool(metadata["quality_failure"] or extrapolation_rmse > LATE_AUDIT_RMSE_LIMIT):
        raise RuntimeError("Refusal decomposition disagrees with the frozen gate")
    return {
        "strength": strength,
        "repeat": repeat,
        "seed": data["seed"],
        "selected_rank": metadata["selected_rank"],
        "condition": metadata["condition"],
        "selected_train_rmse_to_observations": winner["train_rmse"],
        "selected_train_rmse_over_noise": winner["train_rmse"] / NOISE_STD,
        "train_rmse_to_clean": train_rmse_to_clean,
        "extrapolation_rmse": extrapolation_rmse,
        "refusal": refusal,
        "refusal_reasons": reasons,
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for strength in STRENGTHS:
        group = [record for record in records if record["strength"] == strength]
        refusals = sum(record["refusal"] for record in group)
        lower, upper = wilson_interval(refusals, len(group))
        rows.append(
            {
                "strength": strength,
                "trials": len(group),
                "refusals": refusals,
                "refusal_rate": refusals / len(group),
                "refusal_rate_wilson_95": [lower, upper],
                "rank3_rate": sum(record["selected_rank"] == 3 for record in group)
                / len(group),
                "median_train_rmse_over_noise": float(
                    np.median(
                        [record["selected_train_rmse_over_noise"] for record in group]
                    )
                ),
                "train_rmse_over_noise_range": [
                    float(
                        np.min(
                            [
                                record["selected_train_rmse_over_noise"]
                                for record in group
                            ]
                        )
                    ),
                    float(
                        np.max(
                            [
                                record["selected_train_rmse_over_noise"]
                                for record in group
                            ]
                        )
                    ),
                ],
                "median_train_rmse_to_clean": float(
                    np.median([record["train_rmse_to_clean"] for record in group])
                ),
                "median_extrapolation_rmse": float(
                    np.median([record["extrapolation_rmse"] for record in group])
                ),
            }
        )
    return rows


def assess(records: list[dict], summary: list[dict]) -> dict:
    rates = [row["refusal_rate"] for row in summary]
    first_majority_refusal = next(
        (row["strength"] for row in summary if row["refusal_rate"] >= 0.5), None
    )
    last_majority_acceptance = next(
        (
            row["strength"]
            for row in reversed(summary)
            if row["refusal_rate"] <= 0.5
        ),
        None,
    )
    bracket_width = (
        first_majority_refusal - last_majority_acceptance
        if first_majority_refusal is not None
        and last_majority_acceptance is not None
        and first_majority_refusal >= last_majority_acceptance
        else None
    )
    silent_failures = [
        record
        for record in records
        if record["extrapolation_rmse"] > LATE_AUDIT_RMSE_LIMIT
        and not record["refusal"]
    ]
    rank_correlation = spearmanr(
        [row["strength"] for row in summary],
        [row["median_train_rmse_over_noise"] for row in summary],
    )
    checks = {
        "both_acceptance_and_refusal_are_observed": min(rates) < max(rates),
        "majority_transition_bracket_is_at_most_0p01": (
            bracket_width is not None and bracket_width <= 0.0100001
        ),
        "observed_refusal_rate_is_nondecreasing": all(
            later >= earlier for earlier, later in zip(rates, rates[1:])
        ),
        "no_silent_late_audit_failures": not silent_failures,
    }
    return {
        "checks": checks,
        "route_pass": all(checks.values()),
        "first_strength_with_at_least_half_refused": first_majority_refusal,
        "last_strength_with_at_least_half_accepted": last_majority_acceptance,
        "majority_transition_bracket_width": bracket_width,
        "silent_failure_count": len(silent_failures),
        "exploratory_spearman_strength_vs_group_median_train_rmse_over_noise": {
            "rho": float(rank_correlation.statistic),
            "pvalue": float(rank_correlation.pvalue),
            "groups": len(summary),
            "note": "Exploratory group-level association; six groups are insufficient for calibration.",
        },
        "interpretation": (
            "The boundary is a frozen protocol threshold under one noise/horizon design, "
            "not a physical phase transition."
        ),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "nonlinear_boundary_refinement.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Refined nonlinear-refusal boundary",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        f"- Frozen training-fit threshold: {TRAIN_RMSE_LIMIT:.3e} ({TRAIN_RMSE_LIMIT / NOISE_STD:.2f} noise standard deviations)",
        "- Eight repeats per strength; Wilson intervals remain descriptive.",
        "",
        "| Strength | Refusals | Rate (95% Wilson) | Rank-3 rate | Median train RMSE/noise | Range | Median clean-train RMSE | Median extrap. RMSE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        lower, upper = row["refusal_rate_wilson_95"]
        ratio_min, ratio_max = row["train_rmse_over_noise_range"]
        lines.append(
            f"| {row['strength']:.3f} | {row['refusals']}/{row['trials']} | "
            f"{row['refusal_rate']:.3f} [{lower:.3f}, {upper:.3f}] | "
            f"{row['rank3_rate']:.3f} | {row['median_train_rmse_over_noise']:.3f} | "
            f"[{ratio_min:.3f}, {ratio_max:.3f}] | "
            f"{row['median_train_rmse_to_clean']:.3e} | "
            f"{row['median_extrapolation_rmse']:.3e} |"
        )
    lines.extend(["", "## Prespecified checks", ""])
    lines.extend(
        f"- {name}: **{value}**"
        for name, value in payload["assessment"]["checks"].items()
    )
    correlation = payload["assessment"][
        "exploratory_spearman_strength_vs_group_median_train_rmse_over_noise"
    ]
    lines.extend(
        [
            "",
            "## Exploratory diagnostic",
            "",
            f"Across the six group medians, strength and train RMSE/noise had "
            f"Spearman rho={correlation['rho']:.3f} "
            f"(nominal p={correlation['pvalue']:.4g}). This is a descriptive "
            "six-group diagnostic, not a calibrated inferential result.",
            "",
            payload["assessment"]["interpretation"],
        ]
    )
    (RESULTS / "nonlinear_boundary_refinement.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    started = time.perf_counter()
    records = []
    for strength in STRENGTHS:
        for repeat in range(REPEATS):
            record = evaluate(strength, repeat)
            records.append(record)
            print(
                f"strength={strength:.3f} repeat={repeat} "
                f"rank={record['selected_rank']} "
                f"train/noise={record['selected_train_rmse_over_noise']:.3f} "
                f"extra={record['extrapolation_rmse']:.3e} "
                f"refuse={record['refusal']} "
                f"elapsed={record['elapsed_seconds']:.1f}s",
                flush=True,
            )
    summary = summarize(records)
    payload = {
        "experiment": "nonlinear_boundary_refinement",
        "design": {
            "strengths": STRENGTHS,
            "repeats_per_strength": REPEATS,
            "noise_std": NOISE_STD,
            "train_rmse_limit": TRAIN_RMSE_LIMIT,
            "train_rmse_limit_over_noise": TRAIN_RMSE_LIMIT / NOISE_STD,
            "late_audit_rmse_limit": LATE_AUDIT_RMSE_LIMIT,
            "training_support": (
                f"{TRAIN_COUNT} points in first {TRAIN_END_FRACTION:.0%} of horizon"
            ),
        },
        "records": records,
        "summary": summary,
        "assessment": assess(records, summary),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_outputs(payload)
    print(json.dumps(payload["assessment"], indent=2), flush=True)
    print(f"elapsed_seconds={payload['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
