import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import f

from finite_local_limit import ALPHA, C, LAMBDA, N, TIMES, condition, profiled_rss
from phase_matched_score import score_basis
from score_power_curve import TARGETS, TRIALS, target_condition


HERE = Path(__file__).resolve().parent
M = 16384
NULL_TRIALS = 4096


def critical_value(nu):
    def cdf(q):
        return 0.5 * f.cdf(q / (C - 1), C - 1, nu) + 0.5 * f.cdf(q / C, C, nu)
    return float(brentq(lambda q: cdf(q) - (1 - ALPHA), 0.0, 100.0))


def components(y):
    _, rate = profiled_rss(y)
    design = np.column_stack((np.ones(N), np.exp(-rate * TIMES)))
    amplitude = np.linalg.lstsq(design, y.T, rcond=None)[0][1]
    g1, g2, sunit = score_basis(rate, amplitude)
    z1 = y @ g1
    diversity = float(np.sum((z1 - sunit * float(sunit @ z1)) ** 2))
    curvature = float(sunit @ (y @ g2))
    return diversity, curvature


def paired_trial(mean, m, seed, critical):
    rng = np.random.default_rng(seed)
    ybar = mean + rng.standard_normal(mean.shape) / np.sqrt(m)
    nu = C * N * (m - 1)
    s2 = rng.chisquare(nu) / nu
    diversity, curvature = components(ybar)
    raw_two = m * (diversity + curvature * curvature) / s2
    raw_cone = m * (diversity + max(curvature, 0.0) ** 2) / s2
    two_critical = C * f.ppf(1 - ALPHA, C, nu)
    return bool(raw_two > two_critical), bool(raw_cone > critical)


def run():
    nu = C * N * (M - 1)
    critical = critical_value(nu)
    rows = []
    index = 0
    for target in TARGETS:
        for path in ("diversity", "crossover", "proportional"):
            _, mean, _, _ = target_condition(path, M, target)
            base_seed = 6091050000000 + 100000 * (index * 2 + 1)
            records = [paired_trial(mean, M, base_seed + j, critical) for j in range(TRIALS)]
            two = float(np.mean([row[0] for row in records]))
            cone = float(np.mean([row[1] for row in records]))
            rows.append({"target_nc": target, "path": path, "m": M,
                         "two_sided_power": two, "cone_power": cone,
                         "gain": cone - two})
            index += 1

    null_rows = []
    for j, m in enumerate((4096, 16384)):
        _, _, mean, _, _ = condition("null", m)
        nu0 = C * N * (m - 1)
        crit0 = critical_value(nu0)
        base_seed = 6091060000000 + 100000 * j
        values = [paired_trial(mean, m, base_seed + k, crit0)[1] for k in range(NULL_TRIALS)]
        null_rows.append({"m": m, "trials": NULL_TRIALS, "critical": crit0,
                          "rejection": float(np.mean(values))})

    prop_low = [row for row in rows if row["path"] == "proportional" and row["target_nc"] in (4.0, 8.0)]
    gates = {
        "null_between_003_and_007": all(.03 <= row["rejection"] <= .07 for row in null_rows),
        "proportional_low_nonnegative_and_mean_gain_0015": (
            all(row["gain"] >= 0 for row in prop_low)
            and float(np.mean([row["gain"] for row in prop_low])) >= .015
        ),
        "no_cell_loses_more_than_003": all(row["gain"] >= -.03 for row in rows),
        "all_finite": np.isfinite([critical] + [row["gain"] for row in rows]
                                   + [row["rejection"] for row in null_rows]).all().item(),
    }
    return {
        "settings": {"m": M, "targets": list(TARGETS), "trials": TRIALS,
                     "null_trials": NULL_TRIALS, "critical": critical},
        "rows": rows, "null_rows": null_rows, "gates": gates,
        "hashes": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                   for name in ("CONE_MATCHED_SCORE_PROTOCOL.md", "CONE_MATCHED_SCORE_THEORY.md", "cone_matched_score.py")},
    }


if __name__ == "__main__":
    result = run()
    (HERE / "cone_matched_score_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["gates"], indent=2))
