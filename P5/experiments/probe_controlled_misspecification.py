"""Probe usefulness and refusal under controlled mechanism misspecification."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

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
    fit_modal_baseline,
    rmse,
)
from probe_unconstrained_spline_baseline import fit_smoothing_spline


REPEATS = 2
RATES = np.array([0.16, 1.25], dtype=float)
LEVELS = {
    "rate_drift": {"control": 0.0, "mild": 0.25, "strong": 0.75},
    "nonlinear_feedback": {"control": 0.0, "mild": 0.05, "strong": 0.20},
}
LATE_AUDIT_RMSE_LIMIT = 1.0e-2


def simulate_channel(
    times: np.ndarray,
    weights: np.ndarray,
    family: str,
    strength: float,
) -> np.ndarray:
    rank = len(RATES)

    def dynamics(t: float, state: np.ndarray) -> np.ndarray:
        x = state[0]
        memory = state[1:]
        dx = -float(weights @ memory)
        local_rates = RATES
        if family == "rate_drift":
            local_rates = RATES * (1.0 + strength * t / HORIZON)
        elif family == "nonlinear_feedback":
            dx -= strength * x**3
        else:
            raise ValueError(f"Unknown family: {family}")
        dz = x - local_rates * memory
        return np.concatenate(([dx], dz))

    initial = np.zeros(rank + 1, dtype=float)
    initial[0] = 1.0
    solution = solve_ivp(
        dynamics,
        (float(times[0]), float(times[-1])),
        initial,
        t_eval=times,
        method="DOP853",
        rtol=2.0e-11,
        atol=2.0e-13,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[0]


def make_misspecified_dataset(family: str, level: str, repeat: int) -> dict:
    strength = LEVELS[family][level]
    seed = (
        451000
        + 5003 * list(LEVELS).index(family)
        + 1009 * list(LEVELS[family]).index(level)
        + 101 * repeat
    )
    rng = np.random.default_rng(seed)
    times_np = np.linspace(0.0, HORIZON, NUM_POINTS)
    channel_scale = np.linspace(0.28, 0.82, CHANNELS)[:, None]
    pole_scale = np.linspace(0.78, 1.22, len(RATES))[None, :]
    weights = channel_scale * pole_scale / len(RATES)
    clean_np = np.column_stack([
        simulate_channel(times_np, weights[channel], family, strength)
        for channel in range(CHANNELS)
    ])
    observations_np = clean_np + NOISE_STD * rng.standard_normal(clean_np.shape)

    split = int(round(TRAIN_END_FRACTION * (NUM_POINTS - 1)))
    pool = np.arange(1, split + 1)
    sampled = np.sort(rng.choice(pool, size=TRAIN_COUNT - 1, replace=False))
    train_np = np.concatenate(([0], sampled))
    interpolation_np = np.setdiff1d(np.arange(split + 1), train_np)
    extrapolation_np = np.arange(split + 1, NUM_POINTS)
    return {
        "seed": seed,
        "family": family,
        "level": level,
        "strength": strength,
        "times": torch.tensor(times_np, dtype=DTYPE, device=DEVICE),
        "clean": torch.tensor(clean_np, dtype=DTYPE, device=DEVICE),
        "observations": torch.tensor(observations_np, dtype=DTYPE, device=DEVICE),
        "train_idx": torch.tensor(train_np, dtype=torch.long, device=DEVICE),
        "interpolation_idx": torch.tensor(interpolation_np, dtype=torch.long, device=DEVICE),
        "extrapolation_idx": torch.tensor(extrapolation_np, dtype=torch.long, device=DEVICE),
        "true_rank": 2,
    }


def evaluate(family: str, level: str, repeat: int) -> dict:
    started = time.perf_counter()
    data = make_misspecified_dataset(family, level, repeat)
    clean = data["clean"].detach().cpu().numpy()
    indices = {
        "train": data["train_idx"].detach().cpu().numpy(),
        "interpolation": data["interpolation_idx"].detach().cpu().numpy(),
        "extrapolation": data["extrapolation_idx"].detach().cpu().numpy(),
    }
    methods = {}
    for name, fitter in (
        ("positive_real_memory", fit_mechanism),
        ("regularized_damped_modal", fit_modal_baseline),
        ("smoothing_spline", fit_smoothing_spline),
    ):
        result = fitter(data)
        if name == "smoothing_spline":
            prediction, metadata = result
        else:
            metadata, prediction = result
        methods[name] = {
            **metadata,
            "train_rmse_to_clean": rmse(prediction[indices["train"]], clean[indices["train"]]),
            "interpolation_rmse_to_clean": rmse(
                prediction[indices["interpolation"]], clean[indices["interpolation"]]
            ),
            "extrapolation_rmse_to_clean": rmse(
                prediction[indices["extrapolation"]], clean[indices["extrapolation"]]
            ),
        }
    mechanism = methods["positive_real_memory"]
    mechanism["late_audit_refusal"] = (
        mechanism["quality_failure"]
        or mechanism["extrapolation_rmse_to_clean"] > LATE_AUDIT_RMSE_LIMIT
    )
    best_trajectory = min(
        methods["regularized_damped_modal"]["extrapolation_rmse_to_clean"],
        methods["smoothing_spline"]["extrapolation_rmse_to_clean"],
    )
    mechanism["extrapolation_ratio_to_best_trajectory"] = (
        mechanism["extrapolation_rmse_to_clean"] / best_trajectory
    )
    return {
        "family": family,
        "level": level,
        "strength": data["strength"],
        "repeat": repeat,
        "seed": data["seed"],
        "methods": methods,
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for family, levels in LEVELS.items():
        for level in levels:
            group = [
                record for record in records
                if record["family"] == family and record["level"] == level
            ]
            row = {
                "family": family,
                "level": level,
                "strength": levels[level],
                "trials": len(group),
                "selected_rank_counts": {
                    str(rank): sum(
                        record["methods"]["positive_real_memory"]["selected_rank"] == rank
                        for record in group
                    )
                    for rank in (1, 2, 3)
                },
                "late_audit_refusals": sum(
                    record["methods"]["positive_real_memory"]["late_audit_refusal"]
                    for record in group
                ),
                "methods": {},
            }
            for method in (
                "positive_real_memory",
                "regularized_damped_modal",
                "smoothing_spline",
            ):
                row["methods"][method] = {
                    metric: float(np.median([
                        record["methods"][method][metric] for record in group
                    ]))
                    for metric in (
                        "train_rmse_to_clean",
                        "interpolation_rmse_to_clean",
                        "extrapolation_rmse_to_clean",
                    )
                }
            trajectory_error = min(
                row["methods"]["regularized_damped_modal"]["extrapolation_rmse_to_clean"],
                row["methods"]["smoothing_spline"]["extrapolation_rmse_to_clean"],
            )
            row["mechanism_to_best_trajectory_extrapolation_ratio"] = (
                row["methods"]["positive_real_memory"]["extrapolation_rmse_to_clean"]
                / trajectory_error
            )
            rows.append(row)
    return rows


def assess(records: list[dict], summary: list[dict]) -> dict:
    controls = [row for row in summary if row["level"] == "control"]
    mild = [row for row in summary if row["level"] == "mild"]
    strong_records = [record for record in records if record["level"] == "strong"]
    silent_failures = [
        record for record in records
        if record["methods"]["positive_real_memory"]["extrapolation_rmse_to_clean"]
        > LATE_AUDIT_RMSE_LIMIT
        and not record["methods"]["positive_real_memory"]["late_audit_refusal"]
    ]
    strong_safe_or_useful = [
        record["methods"]["positive_real_memory"]["late_audit_refusal"]
        or record["methods"]["positive_real_memory"]["extrapolation_ratio_to_best_trajectory"]
        <= 1.25
        for record in strong_records
    ]
    checks = {
        "all_control_instances_have_sub_0p005_mechanism_extrapolation": all(
            row["methods"]["positive_real_memory"]["extrapolation_rmse_to_clean"] <= 5.0e-3
            for row in controls
        ),
        "mild_misspecification_mechanism_not_over_25_percent_worse_than_best_trajectory": all(
            row["mechanism_to_best_trajectory_extrapolation_ratio"] <= 1.25 for row in mild
        ),
        "every_strong_instance_is_useful_or_refused": all(strong_safe_or_useful),
        "no_silent_late_audit_failures": len(silent_failures) == 0,
    }
    return {
        "checks": checks,
        "route_pass": all(checks.values()),
        "silent_failure_count": len(silent_failures),
        "strong_useful_or_refused_count": sum(strong_safe_or_useful),
        "strong_trial_count": len(strong_records),
    }


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "controlled_misspecification.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Controlled mechanism misspecification",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        f"- Late-audit RMSE refusal limit: {LATE_AUDIT_RMSE_LIMIT:.3e}",
        "- The late audit requires observations beyond the training horizon; it is not an a-priori guarantee.",
        "",
        "| Family | Level | Strength | Rank counts | Refusals | Memory extrap. | Modal extrap. | Spline extrap. | Ratio to best |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        methods = row["methods"]
        lines.append(
            f"| {row['family']} | {row['level']} | {row['strength']:.3f} | "
            f"{row['selected_rank_counts']} | {row['late_audit_refusals']}/{row['trials']} | "
            f"{methods['positive_real_memory']['extrapolation_rmse_to_clean']:.4e} | "
            f"{methods['regularized_damped_modal']['extrapolation_rmse_to_clean']:.4e} | "
            f"{methods['smoothing_spline']['extrapolation_rmse_to_clean']:.4e} | "
            f"{row['mechanism_to_best_trajectory_extrapolation_ratio']:.3f} |"
        )
    lines.extend(["", "## Prespecified checks", ""])
    lines.extend(
        f"- {name}: **{value}**" for name, value in payload["assessment"]["checks"].items()
    )
    lines.extend([
        "",
        "A refusal means that an observed late-time audit exceeded the frozen error",
        "limit or the numerical fit failed its quality gate. It does not predict",
        "failure before late observations become available.",
    ])
    (RESULTS / "controlled_misspecification.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    started = time.perf_counter()
    records = []
    for family, levels in LEVELS.items():
        for level in levels:
            for repeat in range(REPEATS):
                record = evaluate(family, level, repeat)
                records.append(record)
                mechanism = record["methods"]["positive_real_memory"]
                print(
                    f"{family}/{level} repeat={repeat} "
                    f"rank={mechanism['selected_rank']} "
                    f"extra={mechanism['extrapolation_rmse_to_clean']:.3e} "
                    f"refuse={mechanism['late_audit_refusal']} "
                    f"elapsed={record['elapsed_seconds']:.1f}s"
                )
    summary = summarize(records)
    payload = {
        "experiment": "controlled_misspecification",
        "design": {
            "families_and_strengths": LEVELS,
            "repeats_per_cell": REPEATS,
            "base_rates": RATES.tolist(),
            "late_audit_rmse_limit": LATE_AUDIT_RMSE_LIMIT,
            "training_support": f"{TRAIN_COUNT} points in first {TRAIN_END_FRACTION:.0%} of horizon",
            "interpretation_boundary": (
                "Late-audit refusal requires post-training observations and is not an a-priori guarantee."
            ),
        },
        "records": records,
        "summary": summary,
        "assessment": assess(records, summary),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_outputs(payload)
    print(json.dumps(payload["assessment"], indent=2))
    print(f"elapsed_seconds={payload['elapsed_seconds']:.1f}")


if __name__ == "__main__":
    main()
