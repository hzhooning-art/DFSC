import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import f, ncf

from finite_local_limit import (
    ALPHA, C, C2W, KAPPAS, LAMBDA, N, P, R_SCALE, S, TIMES, V, profiled_rss,
)
from phase_matched_score import score_energy


HERE = Path(__file__).resolve().parent
TARGETS = (4.0, 8.0, 12.0, 18.0)
MS_CURVE = (4096, 16384)
TRIALS = 1024


def target_condition(path, m, target):
    if path == "diversity":
        r = R_SCALE
        delta = np.sqrt(target / (S * R_SCALE * V)) * m ** -0.5
    elif path == "crossover":
        r = R_SCALE * m ** -0.5
        aa = S * R_SCALE * V
        bb = S * C2W
        x = (-aa + np.sqrt(aa * aa + 4 * bb * target)) / (2 * bb)
        delta = np.sqrt(x) * m ** -0.25
    elif path == "proportional":
        r = 0.0
        delta = (target / (S * C2W)) ** 0.25 * m ** -0.25
    else:
        raise ValueError(path)
    s = np.ones(C) * np.sqrt(S / C)
    u = np.zeros(C)
    u[0], u[1] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    b = P * s + np.sqrt(S * r) * u
    a = s - b
    mean = a[:, None] * np.exp(-LAMBDA * TIMES)[None, :]
    mean += b[:, None] * np.exp(-(LAMBDA + delta) * TIMES)[None, :]
    return float(delta), mean, a, b


def population(path, m, target):
    delta, mean, a, b = target_condition(path, m, target)
    distance2, rate = profiled_rss(mean)
    design = np.column_stack((np.ones(N), np.exp(-rate * TIMES)))
    amplitude = np.linalg.lstsq(design, mean.T, rcond=None)[0][1]
    energy, _, _ = score_energy(mean, rate, amplitude)
    return {
        "delta": delta, "mean": mean,
        "minimum_amplitude": float(min(a.min(), b.min())),
        "distance_nc": float(m * distance2), "score_nc": float(m * energy),
        "retention": float(energy / distance2),
    }


def trial(mean, m, seed):
    rng = np.random.default_rng(seed)
    ybar = mean + rng.standard_normal(mean.shape) / np.sqrt(m)
    nu = C * N * (m - 1)
    s2 = rng.chisquare(nu) / nu
    rss, rate = profiled_rss(ybar)
    design = np.column_stack((np.ones(N), np.exp(-rate * TIMES)))
    amplitude = np.linalg.lstsq(design, ybar.T, rcond=None)[0][1]
    energy, _, _ = score_energy(ybar, rate, amplitude)
    union_critical = f.ppf(1 - ALPHA, C * (N - 2), nu)
    score_critical = f.ppf(1 - ALPHA, C, nu)
    return {
        "union": bool(m * rss / (C * (N - 2) * s2) > union_critical),
        "score": bool(m * energy / (C * s2) > score_critical),
    }


def run_cell(path, m, target, index):
    pop = population(path, m, target)
    nu = C * N * (m - 1)
    d = C * (N - 2)
    r0 = d - 1
    score_critical = f.ppf(1 - ALPHA, C, nu)
    union_critical = d * f.ppf(1 - ALPHA, d, nu) / r0
    score_prediction = float(1 - ncf.cdf(score_critical, C, nu, pop["score_nc"]))
    union_prediction = float(1 - ncf.cdf(union_critical, r0, nu, pop["distance_nc"]))
    base_seed = 6091050000000 + 100000 * index
    records = [trial(pop["mean"], m, base_seed + j) for j in range(TRIALS)]
    score_power = float(np.mean([row["score"] for row in records]))
    union_power = float(np.mean([row["union"] for row in records]))
    return {
        "path": path, "m": m, "target_nc": target, "trials": TRIALS,
        **{key: value for key, value in pop.items() if key != "mean"},
        "score_power": score_power, "union_power": union_power,
        "power_gain": score_power - union_power,
        "score_prediction": score_prediction, "union_prediction": union_prediction,
        "score_prediction_error": abs(score_power - score_prediction),
        "union_prediction_error": abs(union_power - union_prediction),
        "replay": [
            {"seed": base_seed, **records[0]},
            {"seed": base_seed + TRIALS - 1, **records[-1]},
        ],
    }


def run():
    rows = []
    index = 0
    for target in TARGETS:
        for path in ("diversity", "crossover", "proportional"):
            for m in MS_CURVE:
                rows.append(run_cell(path, m, target, index))
                index += 1
    largest = [row for row in rows if row["m"] == MS_CURVE[-1]]
    gates = {
        "score_prediction_error_at_most_005": all(row["score_prediction_error"] <= .05 for row in largest),
        "union_prediction_error_at_most_006": all(row["union_prediction_error"] <= .06 for row in largest),
        "score_power_gain_at_least_012": all(row["power_gain"] >= .12 for row in largest),
        "noncentrality_retention_at_least_092": all(row["retention"] >= .92 for row in largest),
        "all_finite_and_positive": all(
            row["minimum_amplitude"] > 0 and np.isfinite([
                row["distance_nc"], row["score_nc"], row["score_power"],
                row["union_power"], row["score_prediction"], row["union_prediction"]
            ]).all() for row in rows
        ),
    }
    return {
        "settings": {"targets": list(TARGETS), "m_values": list(MS_CURVE), "trials": TRIALS},
        "rows": rows, "gates": gates,
        "hashes": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                   for name in ("SCORE_POWER_CURVE_PROTOCOL.md", "score_power_curve.py")},
    }


if __name__ == "__main__":
    result = run()
    (HERE / "score_power_curve_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["gates"], indent=2))
