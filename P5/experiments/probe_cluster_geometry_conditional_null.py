"""Test geometry-conditional null calibration for clustered observations."""

from __future__ import annotations

import json

import numpy as np
import torch

from probe_memory_rank import DEVICE, DTYPE, fit_rank
from probe_out_of_class_refusal import mean_lag1, prediction_from_fit
from probe_refusal_calibration import RESULTS, wilson_interval
from probe_multiwindow_external_calibration import (
    HORIZON,
    NOISE_STD,
    WINDOW_FRACTIONS,
    response,
)
from probe_sampling_process_stress import FIT_CONDITION_LIMIT, FIT_RMSE_LIMIT, split_indices


SOURCE_ARTIFACT = RESULTS / "sampling_stratified_null_law.json"
CALIBRATION_REPEATS_PER_FAMILY = 90
EVALUATION_REPEATS_PER_FAMILY = 60
CALIBRATION_OFFSET = 211000
EVALUATION_OFFSET = 241000
FAMILYWISE_ALPHA = 0.05
NUM_GEOMETRY_BINS = 3
GEOMETRY_HISTOGRAM_BINS = 12


def seed_for(case: str, repeat: int, split: str) -> int:
    offset = CALIBRATION_OFFSET if split == "calibration" else EVALUATION_OFFSET
    return offset + 101 * repeat + sum(map(ord, case))


def make_variable_clustered_times(seed: int) -> torch.Tensor:
    """Sample a continuum of clustered designs while preserving audit support."""
    rng = np.random.default_rng(seed)
    uniform_count = int(rng.integers(28, 51))
    cluster_count = int(rng.integers(8, 25))
    cluster_scale = float(rng.uniform(0.018, 0.080)) * HORIZON
    centers = np.asarray([0.22, 0.54, 0.84]) * HORIZON
    centers += rng.uniform(-0.055, 0.055, size=3) * HORIZON
    support = np.concatenate([
        np.linspace(0.02, 0.14, 6) * HORIZON,
        np.linspace(0.36, 0.44, 6) * HORIZON,
        np.linspace(0.67, 0.78, 6) * HORIZON,
    ])
    clustered = np.concatenate([
        rng.normal(center, cluster_scale, size=cluster_count) for center in centers
    ])
    times = np.unique(np.clip(
        np.concatenate((
            [0.0, HORIZON],
            support,
            rng.uniform(0.0, HORIZON, size=uniform_count),
            clustered,
        )),
        0.0,
        HORIZON,
    ))
    return torch.tensor(np.sort(times), dtype=DTYPE, device=DEVICE)


def geometry_features(times: torch.Tensor, windows: dict[str, np.ndarray]) -> dict:
    normalized = times.detach().cpu().numpy() / HORIZON
    counts, _ = np.histogram(normalized, bins=GEOMETRY_HISTOGRAM_BINS, range=(0.0, 1.0))
    mean_count = float(np.mean(counts))
    return {
        "maximum_normalized_gap": float(np.max(np.diff(normalized))),
        "histogram_concentration_cv": float(np.std(counts) / max(mean_count, 1.0e-12)),
        "minimum_window_count": int(min(index.size for index in windows.values())),
    }


def fit_null(case: str, repeat: int, split: str) -> dict:
    seed = seed_for(case, repeat, split)
    times = make_variable_clustered_times(seed)
    clean = response(case, times)
    rng = np.random.default_rng(seed + 17)
    observations = clean + NOISE_STD * torch.tensor(
        rng.standard_normal(clean.shape), dtype=DTYPE, device=DEVICE
    )
    train_np, diagnostic_np, windows_np = split_indices(times, seed + 29)
    train_idx = torch.tensor(train_np, dtype=torch.long, device=DEVICE)
    diagnostic_idx = torch.tensor(diagnostic_np, dtype=torch.long, device=DEVICE)
    candidates = [
        fit_rank(
            times,
            observations,
            train_idx,
            diagnostic_idx,
            rank=1,
            seed=seed * 100 + start,
            adam_steps=165,
            lbfgs_steps=48,
        )
        for start in range(2)
    ]
    winner = min(candidates, key=lambda item: item.bic)
    prediction = prediction_from_fit(times, winner)
    statistics = {}
    for name, index_np in windows_np.items():
        index = torch.tensor(index_np, dtype=torch.long, device=DEVICE)
        statistics[name] = mean_lag1(prediction[index] - observations[index])
    quality_pass = (
        np.isfinite(winner.val_rmse)
        and np.isfinite(winner.jacobian_condition)
        and winner.val_rmse <= FIT_RMSE_LIMIT
        and winner.jacobian_condition <= FIT_CONDITION_LIMIT
    )
    return {
        "split": split,
        "case": case,
        "repeat": repeat,
        "seed": seed,
        "observation_count": int(len(times)),
        "training_count": int(len(train_np)),
        "validation_rmse": winner.val_rmse,
        "jacobian_condition": winner.jacobian_condition,
        "fit_quality_pass": bool(quality_pass),
        "geometry": geometry_features(times, windows_np),
        "statistics": statistics,
        "max_abs_statistic": max(abs(value) for value in statistics.values()),
        "strongest_window": max(statistics, key=lambda name: abs(statistics[name])),
    }


def rank_correlation(x: list[float], y: list[float]) -> float:
    x_rank = np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)
    y_rank = np.argsort(np.argsort(np.asarray(y, dtype=float))).astype(float)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def fit_geometry_partition(records: list[dict]) -> dict:
    valid = [record for record in records if record["fit_quality_pass"]]
    feature_names = (
        "maximum_normalized_gap",
        "histogram_concentration_cv",
        "minimum_window_count",
    )
    location = {}
    scale = {}
    for name in feature_names:
        values = np.asarray([record["geometry"][name] for record in valid], dtype=float)
        location[name] = float(np.mean(values))
        scale[name] = float(max(np.std(values), 1.0e-12))

    def score(record: dict) -> float:
        geometry = record["geometry"]
        return float(
            (geometry["maximum_normalized_gap"] - location["maximum_normalized_gap"])
            / scale["maximum_normalized_gap"]
            + (geometry["histogram_concentration_cv"] - location["histogram_concentration_cv"])
            / scale["histogram_concentration_cv"]
            - (geometry["minimum_window_count"] - location["minimum_window_count"])
            / scale["minimum_window_count"]
        )

    scores = np.asarray([score(record) for record in valid])
    cutpoints = [
        float(np.quantile(scores, index / NUM_GEOMETRY_BINS))
        for index in range(1, NUM_GEOMETRY_BINS)
    ]
    return {
        "features": list(feature_names),
        "score_definition": "z(max_gap) + z(concentration_cv) - z(min_window_count)",
        "location": location,
        "scale": scale,
        "cutpoints": cutpoints,
    }


def assign_geometry(records: list[dict], partition: dict) -> None:
    for record in records:
        geometry = record["geometry"]
        score = (
            (geometry["maximum_normalized_gap"] - partition["location"]["maximum_normalized_gap"])
            / partition["scale"]["maximum_normalized_gap"]
            + (geometry["histogram_concentration_cv"] - partition["location"]["histogram_concentration_cv"])
            / partition["scale"]["histogram_concentration_cv"]
            - (geometry["minimum_window_count"] - partition["location"]["minimum_window_count"])
            / partition["scale"]["minimum_window_count"]
        )
        record["geometry_risk_score"] = float(score)
        record["geometry_bin"] = int(np.digitize(score, partition["cutpoints"]))


def empirical_threshold(records: list[dict], bootstrap_seed: int) -> dict:
    valid = [record for record in records if record["fit_quality_pass"]]
    maxima = np.asarray([record["max_abs_statistic"] for record in valid])
    threshold = float(np.quantile(maxima, 1.0 - FAMILYWISE_ALPHA, method="higher"))
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = []
    for _ in range(2000):
        sample = rng.choice(maxima, size=maxima.size, replace=True)
        bootstrap.append(float(np.quantile(
            sample, 1.0 - FAMILYWISE_ALPHA, method="higher"
        )))
    lo = float(np.quantile(bootstrap, 0.025))
    hi = float(np.quantile(bootstrap, 0.975))
    return {
        "independent_fits": len(records),
        "valid_fits": len(valid),
        "threshold": threshold,
        "bootstrap_interval95": [lo, hi],
        "bootstrap_width": hi - lo,
    }


def apply_decisions(
    records: list[dict], legacy: float, global_calibration: dict, bin_calibration: dict
) -> None:
    for record in records:
        thresholds = {
            "legacy_clustered": legacy,
            "new_global_clustered": global_calibration["threshold"],
            "geometry_conditional": bin_calibration[str(record["geometry_bin"])]["threshold"],
        }
        record["thresholds"] = thresholds
        record["decisions"] = {
            name: (
                "ACCEPT_CONTRACT"
                if record["fit_quality_pass"] and record["max_abs_statistic"] <= threshold
                else "REFUSE_CONTRACT"
            )
            for name, threshold in thresholds.items()
        }


def decision_summary(records: list[dict]) -> list[dict]:
    rows = []
    for bin_id in ("all", 0, 1, 2):
        group = records if bin_id == "all" else [
            record for record in records if record["geometry_bin"] == bin_id
        ]
        methods = {}
        for method in ("legacy_clustered", "new_global_clustered", "geometry_conditional"):
            refused = sum(record["decisions"][method] == "REFUSE_CONTRACT" for record in group)
            methods[method] = {
                "refusals": refused,
                "fraction": refused / len(group),
                "wilson95": wilson_interval(refused, len(group)),
            }
        rows.append({
            "geometry_bin": bin_id,
            "trials": len(group),
            "invalid_fits": sum(not record["fit_quality_pass"] for record in group),
            "family_counts": {
                case: sum(record["case"] == case for record in group)
                for case in ("signed_zero", "oscillation_zero")
            },
            "methods": methods,
        })
    return rows


def assess(bin_calibration: dict, summary: list[dict]) -> dict:
    lookup = {str(row["geometry_bin"]): row for row in summary}
    overall = lookup["all"]["methods"]["geometry_conditional"]
    checks = {
        "all_180_calibration_fits_valid": all(
            calibration["valid_fits"] == calibration["independent_fits"]
            for calibration in bin_calibration.values()
        ),
        "each_geometry_bin_has_at_least_55_calibration_fits": all(
            calibration["valid_fits"] >= 55 for calibration in bin_calibration.values()
        ),
        "each_bootstrap_width_at_most_0.20": all(
            calibration["bootstrap_width"] <= 0.20
            for calibration in bin_calibration.values()
        ),
        "each_threshold_at_most_0.40": all(
            calibration["threshold"] <= 0.40 for calibration in bin_calibration.values()
        ),
        "overall_heldout_refusals_at_most_6_of_120": overall["refusals"] <= 6,
        "overall_wilson_upper_at_most_0.10": overall["wilson95"][1] <= 0.10,
        "each_bin_heldout_refusals_at_most_3": all(
            lookup[str(bin_id)]["methods"]["geometry_conditional"]["refusals"] <= 3
            for bin_id in range(NUM_GEOMETRY_BINS)
        ),
        "each_bin_wilson_upper_at_most_0.20": all(
            lookup[str(bin_id)]["methods"]["geometry_conditional"]["wilson95"][1] <= 0.20
            for bin_id in range(NUM_GEOMETRY_BINS)
        ),
    }
    return {"checks": checks, "route_pass": all(checks.values())}


def write_outputs(payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cluster_geometry_conditional_null.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Geometry-conditional clustered null calibration",
        "",
        f"- Route pass: **{payload['assessment']['route_pass']}**",
        f"- Legacy clustered threshold: {payload['design']['legacy_clustered_threshold']:.6f}",
        f"- New global threshold: {payload['global_calibration']['threshold']:.6f}",
        "",
        "| Geometry bin | Calibration n | Bin threshold | Bootstrap 95% | Held-out n | Legacy FR | Global FR | Conditional FR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summary = {str(row["geometry_bin"]): row for row in payload["evaluation_summary"]}
    for bin_id in range(NUM_GEOMETRY_BINS):
        calibration = payload["bin_calibration"][str(bin_id)]
        row = summary[str(bin_id)]
        lines.append(
            f"| {bin_id} | {calibration['valid_fits']} | {calibration['threshold']:.6f} | "
            f"{calibration['bootstrap_interval95']} | {row['trials']} | "
            f"{row['methods']['legacy_clustered']['refusals']} | "
            f"{row['methods']['new_global_clustered']['refusals']} | "
            f"{row['methods']['geometry_conditional']['refusals']} |"
        )
    overall = summary["all"]
    lines.extend([
        "",
        "## Overall held-out false refusals",
        "",
        f"- Legacy: {overall['methods']['legacy_clustered']['refusals']}/{overall['trials']}",
        f"- New global: {overall['methods']['new_global_clustered']['refusals']}/{overall['trials']}",
        f"- Geometry conditional: {overall['methods']['geometry_conditional']['refusals']}/{overall['trials']}",
    ])
    (RESULTS / "cluster_geometry_conditional_null.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    source = json.loads(SOURCE_ARTIFACT.read_text(encoding="utf-8"))
    legacy = float(source["calibration"]["clustered"]["threshold"])
    calibration_records = [
        fit_null(case, repeat, "calibration")
        for case in ("signed_zero", "oscillation_zero")
        for repeat in range(CALIBRATION_REPEATS_PER_FAMILY)
    ]
    evaluation_records = [
        fit_null(case, repeat, "evaluation")
        for case in ("signed_zero", "oscillation_zero")
        for repeat in range(EVALUATION_REPEATS_PER_FAMILY)
    ]
    partition = fit_geometry_partition(calibration_records)
    assign_geometry(calibration_records, partition)
    assign_geometry(evaluation_records, partition)
    global_calibration = empirical_threshold(calibration_records, 31091)
    bin_calibration = {
        str(bin_id): empirical_threshold([
            record for record in calibration_records if record["geometry_bin"] == bin_id
        ], 31103 + bin_id)
        for bin_id in range(NUM_GEOMETRY_BINS)
    }
    apply_decisions(evaluation_records, legacy, global_calibration, bin_calibration)
    summary = decision_summary(evaluation_records)
    valid_calibration = [record for record in calibration_records if record["fit_quality_pass"]]
    correlations = {
        name: rank_correlation(
            [record["geometry"][name] for record in valid_calibration],
            [record["max_abs_statistic"] for record in valid_calibration],
        )
        for name in partition["features"]
    }
    correlations["geometry_risk_score"] = rank_correlation(
        [record["geometry_risk_score"] for record in valid_calibration],
        [record["max_abs_statistic"] for record in valid_calibration],
    )
    assessment = assess(bin_calibration, summary)
    payload = {
        "experiment": "cluster_geometry_conditional_null",
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "design": {
            "legacy_clustered_threshold": legacy,
            "legacy_source": str(SOURCE_ARTIFACT),
            "calibration_fits": len(calibration_records),
            "heldout_fits": len(evaluation_records),
            "calibration_and_evaluation_seeds_disjoint": True,
            "candidate_rank": 1,
            "starts_per_fit": 2,
            "geometry_bins": NUM_GEOMETRY_BINS,
            "geometry_partition_uses_residual_labels": False,
            "threshold_unit": "independent per-fit three-window maximum",
        },
        "geometry_partition": partition,
        "geometry_residual_rank_correlations": correlations,
        "global_calibration": global_calibration,
        "bin_calibration": bin_calibration,
        "calibration_records": calibration_records,
        "evaluation_records": evaluation_records,
        "evaluation_summary": summary,
        "assessment": assessment,
    }
    write_outputs(payload)
    print(json.dumps({
        "geometry_partition": partition,
        "geometry_residual_rank_correlations": correlations,
        "global_calibration": global_calibration,
        "bin_calibration": bin_calibration,
        "evaluation_summary": summary,
        "assessment": assessment,
    }, indent=2))


if __name__ == "__main__":
    main()
