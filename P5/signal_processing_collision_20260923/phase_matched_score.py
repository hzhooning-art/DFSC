import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import f, ncf

from finite_local_limit import (
    ALPHA, ALT_TRIALS, C, MS, N, NULL_TRIALS, TIMES, condition, profiled_rss,
)


HERE = Path(__file__).resolve().parent


def score_basis(rate, amplitude):
    e = np.exp(-rate * TIMES)
    ep = -TIMES * e
    epp = TIMES * TIMES * e
    x2 = np.column_stack((np.ones(N), e))
    q2, _ = np.linalg.qr(x2)
    g1 = ep - q2 @ (q2.T @ ep)
    g1 /= np.linalg.norm(g1)
    x3 = np.column_stack((np.ones(N), e, ep))
    q3, _ = np.linalg.qr(x3)
    g2 = epp - q3 @ (q3.T @ epp)
    g2 /= np.linalg.norm(g2)
    snorm = np.linalg.norm(amplitude)
    if not np.isfinite(snorm) or snorm <= 0:
        raise FloatingPointError("invalid fitted amplitude norm")
    return g1, g2, amplitude / snorm


def null_fit(y):
    _, rate = profiled_rss(y)
    design = np.column_stack((np.ones(N), np.exp(-rate * TIMES)))
    coefficients = np.linalg.lstsq(design, y.T, rcond=None)[0]
    amplitude = coefficients[1]
    return rate, amplitude


def score_energy(y, rate=None, amplitude=None):
    if rate is None or amplitude is None:
        rate, amplitude = null_fit(y)
    g1, g2, sunit = score_basis(rate, amplitude)
    z1 = y @ g1
    z1_perp = z1 - sunit * float(sunit @ z1)
    z2 = float(sunit @ (y @ g2))
    energy = float(z1_perp @ z1_perp + z2 * z2)
    return energy, float(rate), float(np.linalg.norm(amplitude))


def population_row(path, m):
    _, _, mean, _, _ = condition(path, m)
    distance2, closest_rate = profiled_rss(mean)
    design = np.column_stack((np.ones(N), np.exp(-closest_rate * TIMES)))
    amplitude = np.linalg.lstsq(design, mean.T, rcond=None)[0][1]
    energy, _, _ = score_energy(mean, closest_rate, amplitude)
    return {
        "distance_noncentrality": float(m * distance2),
        "score_noncentrality": float(m * energy),
        "retention": float(energy / distance2) if distance2 > 0 else 1.0,
    }


def one_trial(path, m, seed):
    _, _, mean, _, _ = condition(path, m)
    rng = np.random.default_rng(seed)
    ybar = mean + rng.standard_normal(mean.shape) / np.sqrt(m)
    nu = C * N * (m - 1)
    s2 = rng.chisquare(nu) / nu
    energy, rate, amplitude_norm = score_energy(ybar)
    statistic = m * energy / (C * s2)
    critical = f.ppf(1 - ALPHA, C, nu)
    return {
        "seed": seed, "energy": energy, "rate": rate,
        "amplitude_norm": amplitude_norm, "s2": float(s2),
        "statistic": float(statistic), "reject": bool(statistic > critical),
    }


def run_cell(path, m, trials, index):
    population = population_row(path, m)
    nu = C * N * (m - 1)
    critical = f.ppf(1 - ALPHA, C, nu)
    nc = population["score_noncentrality"]
    prediction = float(1 - ncf.cdf(critical, C, nu, nc))
    base_seed = 6091030000000 + 100000 * index
    records = [one_trial(path, m, base_seed + j) for j in range(trials)]
    empirical = float(np.mean([row["reject"] for row in records]))
    return {
        "path": path, "m": m, "trials": trials, **population,
        "empirical_rejection": empirical, "noncentral_f_prediction": prediction,
        "absolute_prediction_error": abs(empirical - prediction),
        "replay": [records[0], records[-1]],
    }


def run():
    rows = []
    index = 0
    for path in ("diversity", "crossover", "proportional"):
        for m in MS:
            rows.append(run_cell(path, m, ALT_TRIALS, index))
            index += 1
    for m in MS:
        rows.append(run_cell("null", m, NULL_TRIALS, index))
        index += 1
    old = json.loads((HERE / "finite_local_limit_results.json").read_text(encoding="utf-8"))
    old_largest = {
        row["path"]: row["empirical_rejection"] for row in old["rows"]
        if row["m"] == MS[-1] and row["path"] != "null"
    }
    largest = [row for row in rows if row["m"] == MS[-1] and row["path"] != "null"]
    nulls = [row for row in rows if row["path"] == "null"]
    finite = all(
        np.isfinite([row["score_noncentrality"], row["distance_noncentrality"],
                     row["empirical_rejection"], row["noncentral_f_prediction"]]).all()
        for row in rows
    )
    gates = {
        "null_rejection_at_most_007": all(row["empirical_rejection"] <= 0.07 for row in nulls),
        "largest_prediction_error_at_most_005": all(row["absolute_prediction_error"] <= 0.05 for row in largest),
        "largest_power_gain_at_least_015": all(
            row["empirical_rejection"] - old_largest[row["path"]] >= 0.15 for row in largest
        ),
        "largest_noncentrality_retention_within_8pct": all(
            abs(row["retention"] - 1) <= 0.08 for row in largest
        ),
        "all_finite": finite,
    }
    return {
        "settings": {"C": C, "n": N, "m_values": list(MS),
                     "alternative_trials": ALT_TRIALS, "null_trials": NULL_TRIALS},
        "rows": rows, "union_power_at_largest_m": old_largest, "gates": gates,
        "hashes": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("PHASE_MATCHED_SCORE_PROTOCOL.md", "PHASE_MATCHED_SCORE_THEORY.md", "phase_matched_score.py")
        },
    }


if __name__ == "__main__":
    result = run()
    (HERE / "phase_matched_score_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["gates"], indent=2))
