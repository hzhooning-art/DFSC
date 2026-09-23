import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import f

from cone_matched_score import critical_value as cone_critical
from finite_local_limit import ALPHA, C, KLOW, KHIGH, N, TIMES, profiled_rss
from phase_matched_score import score_basis
from score_power_curve import target_condition


HERE = Path(__file__).resolve().parent
M = 16384
CAL_TRIALS = 2048
NULL_TRIALS = 2048
ALT_TRIALS = 512


def centered(rate):
    value = np.exp(-rate * TIMES)
    return value - value.mean()


def pair_basis(center_rate, half_gap):
    if center_rate - half_gap < KLOW or center_rate + half_gap > KHIGH:
        return None
    if half_gap < 1e-7:
        v1 = centered(center_rate)
        raw = -TIMES * np.exp(-center_rate * TIMES)
        v2 = raw - raw.mean()
    else:
        lo = centered(center_rate - half_gap)
        hi = centered(center_rate + half_gap)
        v1 = 0.5 * (lo + hi)
        v2 = (hi - lo) / (2 * half_gap)
    q1 = v1 / np.linalg.norm(v1)
    v2 = v2 - q1 * float(q1 @ v2)
    norm2 = np.linalg.norm(v2)
    if norm2 < 1e-12:
        return None
    return q1, v2 / norm2


def grid_bases(points):
    rates = np.linspace(KLOW, KHIGH, points)
    parameters, q1s, q2s = [], [], []
    for i, left in enumerate(rates):
        basis = pair_basis(left, 0.0)
        parameters.append((left, 0.0))
        q1s.append(basis[0]); q2s.append(basis[1])
        for right in rates[i + 1:]:
            center_rate = 0.5 * (left + right)
            half_gap = 0.5 * (right - left)
            basis = pair_basis(center_rate, half_gap)
            if basis is None:
                continue
            parameters.append((center_rate, half_gap))
            q1s.append(basis[0]); q2s.append(basis[1])
    return np.asarray(parameters), np.asarray(q1s), np.asarray(q2s)


GRID_CACHE = {points: grid_bases(points) for points in (41, 81)}


def two_rate_rss(y, points=41):
    yc = y - y.mean(axis=1, keepdims=True)
    total = float(np.sum(yc * yc))
    params, q1s, q2s = GRID_CACHE[points]
    z1 = yc @ q1s.T
    z2 = yc @ q2s.T
    energies = np.sum(z1 * z1 + z2 * z2, axis=0)
    selected = list(np.argsort(energies)[-5:])
    step = (KHIGH - KLOW) / (points - 1)
    masks = (
        params[:, 1] <= 1e-12,
        params[:, 0] - params[:, 1] <= KLOW + step / 2,
        params[:, 0] + params[:, 1] >= KHIGH - step / 2,
        params[:, 1] >= 0.2,
    )
    for mask in masks:
        indices = np.flatnonzero(mask)
        selected.append(int(indices[np.argmax(energies[indices])]))
    best_indices = np.asarray(sorted(set(selected)), dtype=int)

    def objective(x):
        basis = pair_basis(float(x[0]), float(x[1]))
        if basis is None:
            overflow = max(0.0, KLOW - (x[0] - x[1])) + max(0.0, x[0] + x[1] - KHIGH)
            return 1e6 + 1e6 * overflow
        a = yc @ basis[0]
        b = yc @ basis[1]
        return total - float(a @ a + b @ b)

    candidates = []
    for best in best_indices:
        start = params[int(best)]
        fit = minimize(
            objective, start, method="Nelder-Mead",
            options={"xatol": 2e-10, "fatol": 1e-13, "maxiter": 350},
        )
        candidates.append((float(min(total - energies[int(best)], fit.fun)),
                           fit.x if fit.success else start))
    rss2, rates = min(candidates, key=lambda item: item[0])
    return max(0.0, rss2), [float(x) for x in rates]


def score_decisions(ybar, s2, m):
    _, rate = profiled_rss(ybar)
    design = np.column_stack((np.ones(N), np.exp(-rate * TIMES)))
    amplitude = np.linalg.lstsq(design, ybar.T, rcond=None)[0][1]
    g1, g2, sunit = score_basis(rate, amplitude)
    z1 = ybar @ g1
    diversity = float(np.sum((z1 - sunit * float(sunit @ z1)) ** 2))
    curvature = float(sunit @ (ybar @ g2))
    nu = C * N * (m - 1)
    phase = m * (diversity + curvature * curvature) / (C * s2)
    cone = m * (diversity + max(curvature, 0.0) ** 2) / s2
    return (
        bool(phase > f.ppf(1 - ALPHA, C, nu)),
        bool(cone > cone_critical(nu)),
    )


def realization(mean, seed, points=41):
    rng = np.random.default_rng(seed)
    ybar = mean + rng.standard_normal(mean.shape) / np.sqrt(M)
    nu = C * N * (M - 1)
    s2 = rng.chisquare(nu) / nu
    rss1, _ = profiled_rss(ybar)
    rss2, rates = two_rate_rss(ybar, points)
    glr = M * max(0.0, rss1 - rss2) / s2
    phase, cone = score_decisions(ybar, s2, M)
    return {"glr": float(glr), "rss1": float(rss1), "rss2": float(rss2),
            "rates": rates, "phase": phase, "cone": cone}


def wilson(successes, trials, z=1.96):
    p = successes / trials
    den = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / den
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return [float(center - half), float(center + half)]


def run():
    _, null_mean, _, _ = target_condition("proportional", M, 0.0)
    calibration = [realization(null_mean, 6091070000000 + j) for j in range(CAL_TRIALS)]
    threshold = float(np.quantile([row["glr"] for row in calibration], .95, method="higher"))
    null_eval = [realization(null_mean, 6091080000000 + j) for j in range(NULL_TRIALS)]
    null_rejects = sum(row["glr"] > threshold for row in null_eval)
    rows, replay_diffs = [], []
    index = 0
    for target in (4.0, 8.0):
        for path in ("diversity", "crossover", "proportional"):
            _, mean, _, _ = target_condition(path, M, target)
            base = 6091090000000 + 100000 * index
            records = [realization(mean, base + j) for j in range(ALT_TRIALS)]
            glrt = [row["glr"] > threshold for row in records]
            phase = [row["phase"] for row in records]
            cone = [row["cone"] for row in records]
            for offset in (0, ALT_TRIALS - 1):
                fine = realization(mean, base + offset, points=81)
                replay_diffs.append({"target": target, "path": path, "seed": base + offset,
                                     "absolute_glr_difference": abs(fine["glr"] - records[offset]["glr"])})
            rows.append({
                "target_nc": target, "path": path, "trials": ALT_TRIALS,
                "glrt_power": float(np.mean(glrt)), "phase_power": float(np.mean(phase)),
                "cone_power": float(np.mean(cone)),
                "glrt_wilson": wilson(sum(glrt), ALT_TRIALS),
                "phase_wilson": wilson(sum(phase), ALT_TRIALS),
                "cone_wilson": wilson(sum(cone), ALT_TRIALS),
                "paired_cone_minus_glrt": float(np.mean(np.asarray(cone, float) - np.asarray(glrt, float))),
            })
            index += 1
    all_fit_rows = calibration + null_eval
    gates = {
        "held_null_between_003_and_007": .03 <= null_rejects / NULL_TRIALS <= .07,
        "grid_refinement_glr_difference_at_most_005": all(row["absolute_glr_difference"] <= .05 for row in replay_diffs),
        "all_rss_nested_and_finite": all(
            row["rss2"] <= row["rss1"] + 1e-9 and np.isfinite([row["glr"], row["rss1"], row["rss2"]]).all()
            for row in all_fit_rows
        ),
        "paired_intervals_reported": all(len(row["glrt_wilson"]) == 2 and len(row["cone_wilson"]) == 2 for row in rows),
    }
    return {
        "settings": {"m": M, "calibration_trials": CAL_TRIALS, "null_trials": NULL_TRIALS,
                     "alternative_trials": ALT_TRIALS, "threshold": threshold},
        "null": {"rejections": null_rejects, "rate": null_rejects / NULL_TRIALS,
                 "wilson": wilson(null_rejects, NULL_TRIALS)},
        "rows": rows, "grid_replays": replay_diffs, "gates": gates,
        "hashes": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                   for name in ("MATCHED_GLRT_PROTOCOL.md", "matched_glrt_benchmark.py")},
    }


if __name__ == "__main__":
    result = run()
    (HERE / "matched_glrt_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["gates"], indent=2))
