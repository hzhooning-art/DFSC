import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, f, kstest, ncf, ncx2


HERE = Path(__file__).resolve().parent
C, N = 6, 24
TIMES = np.linspace(0.0, 8.0, N)
KLOW, KHIGH = 0.2, 0.8
LAMBDA = 0.4
P = 0.5
S = 10_000.0
R_SCALE = 0.04
SIGMA = 1.0
ALPHA = 0.05
TARGET_NC = 36.0
MS = (256, 1024, 4096, 16384)
ALT_TRIALS = 1024
NULL_TRIALS = 2048
GRID = np.linspace(KLOW, KHIGH, 801)


def centered_exp(rate):
    value = np.exp(-rate * TIMES)
    return value - value.mean()


GRID_V = np.stack([centered_exp(rate) for rate in GRID])
GRID_V /= np.linalg.norm(GRID_V, axis=1, keepdims=True)


def residual_constants():
    e = np.exp(-LAMBDA * TIMES)
    ep = -TIMES * e
    epp = TIMES * TIMES * e
    x2 = np.column_stack((np.ones(N), e))
    x3 = np.column_stack((np.ones(N), e, ep))
    p2 = np.eye(N) - x2 @ np.linalg.inv(x2.T @ x2) @ x2.T
    p3 = np.eye(N) - x3 @ np.linalg.inv(x3.T @ x3) @ x3.T
    return float(np.linalg.norm(p2 @ ep) ** 2), float(np.linalg.norm(p3 @ epp) ** 2)


V, W = residual_constants()
C2W = (P * (1 - P) / 2) ** 2 * W


def kappas():
    kd = np.sqrt(TARGET_NC / (S * R_SCALE * V))
    kc = (TARGET_NC / (S * C2W)) ** 0.25
    aa = S * R_SCALE * V
    bb = S * C2W
    x = (-aa + np.sqrt(aa * aa + 4 * bb * TARGET_NC)) / (2 * bb)
    return {"diversity": float(kd), "crossover": float(np.sqrt(x)), "proportional": float(kc)}


KAPPAS = kappas()


def condition(path, m):
    if path == "diversity":
        r = R_SCALE
        delta = KAPPAS[path] * m ** -0.5
    elif path == "crossover":
        r = R_SCALE * m ** -0.5
        delta = KAPPAS[path] * m ** -0.25
    elif path == "proportional":
        r = 0.0
        delta = KAPPAS[path] * m ** -0.25
    elif path == "null":
        r = 0.0
        delta = 0.0
    else:
        raise ValueError(path)
    s = np.ones(C) * np.sqrt(S / C)
    u = np.zeros(C)
    u[0], u[1] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    b = P * s + np.sqrt(S * r) * u
    a = s - b
    mean = a[:, None] * np.exp(-LAMBDA * TIMES)[None, :]
    mean += b[:, None] * np.exp(-(LAMBDA + delta) * TIMES)[None, :]
    return r, float(delta), mean, a, b


def profiled_rss(y):
    yc = y - y.mean(axis=1, keepdims=True)
    total = float(np.sum(yc * yc))
    scores = yc @ GRID_V.T
    rss = total - np.sum(scores * scores, axis=0)
    index = int(np.argmin(rss))

    def objective(rate):
        v = centered_exp(rate)
        v /= np.linalg.norm(v)
        z = yc @ v
        return max(0.0, total - float(z @ z))

    candidates = [(float(rss[index]), float(GRID[index]))]
    lo = GRID[max(0, index - 1)]
    hi = GRID[min(len(GRID) - 1, index + 1)]
    if hi > lo:
        fit = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-13})
        candidates.append((float(fit.fun), float(fit.x)))
    candidates.extend((objective(rate), rate) for rate in (KLOW, KHIGH))
    return min(candidates)


def prediction(noncentrality, m):
    d = C * (N - 2)
    r0 = d - 1
    nu = C * N * (m - 1)
    critical = f.ppf(1 - ALPHA, d, nu)
    scaled_critical = d * critical / r0
    # scipy.stats.ncf.sf can return a negative value at nc=0 in some SciPy
    # builds; the CDF is stable for the probability range used here.
    return float(1.0 - ncf.cdf(scaled_critical, r0, nu, noncentrality))


def one_trial(path, m, seed):
    _, _, mean, _, _ = condition(path, m)
    rng = np.random.default_rng(seed)
    ybar = mean + rng.standard_normal(mean.shape) / np.sqrt(m)
    s2 = rng.chisquare(C * N * (m - 1)) / (C * N * (m - 1))
    rss, rate = profiled_rss(ybar)
    d = C * (N - 2)
    critical = f.ppf(1 - ALPHA, d, C * N * (m - 1))
    statistic = m * rss / (d * s2)
    return {"seed": seed, "rss": rss, "rate": rate, "s2": s2, "statistic": statistic,
            "reject": bool(statistic > critical)}


def run_cell(path, m, trials, cell_index):
    r, delta, mean, a, b = condition(path, m)
    distance2, nearest_rate = profiled_rss(mean)
    nc_exact = m * distance2
    d = C * (N - 2)
    r0 = d - 1
    nu = C * N * (m - 1)
    critical = f.ppf(1 - ALPHA, d, nu)
    base_seed = 6091030000000 + 100000 * cell_index
    records = [one_trial(path, m, base_seed + j) for j in range(trials)]
    transformed = np.asarray([row["statistic"] * d / r0 for row in records])
    ks = kstest(transformed, lambda x: ncf.cdf(x, r0, nu, nc_exact))
    empirical = float(np.mean([row["reject"] for row in records]))
    local_exact = prediction(nc_exact, m)
    local_target = prediction(0.0 if path == "null" else TARGET_NC, m)
    return {
        "path": path,
        "m": m,
        "trials": trials,
        "r": r,
        "delta": delta,
        "minimum_amplitude": float(min(a.min(), b.min())),
        "distance2": distance2,
        "nearest_null_rate": nearest_rate,
        "exact_noncentrality": nc_exact,
        "target_noncentrality": 0.0 if path == "null" else TARGET_NC,
        "empirical_rejection": empirical,
        "local_prediction_exact_distance": local_exact,
        "local_prediction_target": local_target,
        "absolute_power_error": abs(empirical - local_exact),
        "ks_distance": float(ks.statistic),
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
    largest = [row for row in rows if row["m"] == MS[-1] and row["path"] != "null"]
    nulls = [row for row in rows if row["path"] == "null"]
    gates = {
        "largest_exact_nc_within_8pct": all(abs(row["exact_noncentrality"] / TARGET_NC - 1) <= 0.08 for row in largest),
        "largest_power_error_at_most_006": all(row["absolute_power_error"] <= 0.06 for row in largest),
        "largest_ks_at_most_007": all(row["ks_distance"] <= 0.07 for row in largest),
        "null_rejection_at_most_007": all(row["empirical_rejection"] <= 0.07 for row in nulls),
        "all_finite_and_positive_amplitudes": all(
            np.isfinite([row["distance2"], row["nearest_null_rate"], row["ks_distance"]]).all()
            and row["distance2"] >= 0 and row["minimum_amplitude"] > 0 for row in rows
        ),
    }
    return {
        "settings": {"C": C, "n": N, "lambda": LAMBDA, "S": S, "target_nc": TARGET_NC,
                     "m_values": list(MS), "alternative_trials": ALT_TRIALS, "null_trials": NULL_TRIALS,
                     "V": V, "W": W, "kappas": KAPPAS},
        "rows": rows,
        "gates": gates,
        "hashes": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                   for name in ("FINITE_LOCAL_LIMIT_PROTOCOL.md", "finite_local_limit.py", "LOCAL_LIMIT_POWER_THEORY.md")},
    }


if __name__ == "__main__":
    result = run()
    (HERE / "finite_local_limit_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["gates"], indent=2))
