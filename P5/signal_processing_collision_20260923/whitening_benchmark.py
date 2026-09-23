import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, ncx2

HERE = Path(__file__).resolve().parent
C, N, D = 4, 12, 48
TIMES = np.linspace(0.0, 8.0, N)
RATE = 0.4
M = 4096
KS = (64, 128, 512)
TRIALS = 512
ALPHA = 0.05
S = np.array([20.0, 25.0, 30.0, 35.0])


def covariance():
    channel = np.full((C, C), 0.25)
    np.fill_diagonal(channel, 1.0)
    time = 0.60 ** np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    return np.kron(channel, time)


SIGMA = covariance()
CHOL_TRUE = np.linalg.cholesky(SIGMA)


def atom(rate):
    e = np.exp(-rate * TIMES)
    return e, -TIMES * e, TIMES * TIMES * e


def null_design(rate):
    e, _, _ = atom(rate)
    design = np.zeros((D, 2 * C))
    for c in range(C):
        sl = slice(c * N, (c + 1) * N)
        design[sl, 2 * c] = 1.0
        design[sl, 2 * c + 1] = e
    return design


def whiten(chol, x):
    return solve_triangular(chol, x, lower=True, check_finite=False)


def gls_fit(y, chol):
    wy = whiten(chol, y)
    def objective(rate):
        design = null_design(rate)
        wd = whiten(chol, design)
        beta = np.linalg.lstsq(wd, wy, rcond=None)[0]
        residual = wy - wd @ beta
        return float(residual @ residual)
    fit = minimize_scalar(objective, bounds=(0.10, 0.80), method="bounded",
                          options={"xatol": 2e-10})
    rate = float(fit.x)
    design = null_design(rate)
    wd = whiten(chol, design)
    beta = np.linalg.lstsq(wd, wy, rcond=None)[0]
    residual = y - design @ beta
    amplitude = beta[1::2]
    return float(fit.fun), rate, residual, amplitude


def collision_frame(rate, amplitude, chol):
    e, ep, epp = atom(rate)
    tangent = null_design(rate)
    rate_col = np.concatenate([amplitude[c] * ep for c in range(C)])[:, None]
    tangent = np.column_stack((tangent, rate_col))
    raw = np.zeros((D, C + 1))
    for c in range(C):
        raw[c * N:(c + 1) * N, c] = ep
    raw[:, -1] = np.concatenate([amplitude[c] * epp for c in range(C)])
    wt = whiten(chol, tangent)
    qt, _ = np.linalg.qr(wt, mode="reduced")
    wr = whiten(chol, raw)
    normal_raw = wr - qt @ (qt.T @ wr)
    u, singular, _ = np.linalg.svd(normal_raw, full_matrices=False)
    scale = singular[0] if singular.size else 0.0
    rank = int(np.sum(singular > max(1e-10, 1e-9 * scale)))
    if rank != C:
        raise np.linalg.LinAlgError(f"collision rank {rank}, expected {C}")
    return u[:, :C], rank


def score(y, chol, correction=1.0):
    _, rate, residual, amplitude = gls_fit(y, chol)
    frame, rank = collision_frame(rate, amplitude, chol)
    wr = whiten(chol, residual)
    return float(correction * M * np.sum((frame.T @ wr) ** 2)), rank


def alt_mean(delta):
    first = 0.5 * S[:, None] * np.exp(-RATE * TIMES)[None, :]
    second = 0.5 * S[:, None] * np.exp(-(RATE + delta) * TIMES)[None, :]
    return (first + second).reshape(-1)


NULL_MEAN = (S[:, None] * np.exp(-RATE * TIMES)[None, :]).reshape(-1)


def target_delta(target=8.0):
    def objective(delta):
        distance, *_ = gls_fit(alt_mean(delta), CHOL_TRUE)
        return M * distance - target
    return float(brentq(objective, 1e-5, 0.35))


def one_trial(mean, K, seed):
    rng = np.random.default_rng(seed)
    training = rng.standard_normal((K, D)) @ CHOL_TRUE.T
    training -= training.mean(axis=0, keepdims=True)
    sample_cov = training.T @ training / (K - 1)
    chol_hat = np.linalg.cholesky(sample_cov)
    y = mean + CHOL_TRUE @ rng.standard_normal(D) / np.sqrt(M)
    correction = (K - D - 2) / (K - 1)
    white, rank = score(y, chol_hat, correction)
    raw, raw_rank = score(y, np.eye(D), 1.0)
    return white, raw, rank, raw_rank


def cell(mean, K, seed0):
    records = [one_trial(mean, K, seed0 + j) for j in range(TRIALS)]
    white = np.array([x[0] for x in records])
    raw = np.array([x[1] for x in records])
    critical = chi2.ppf(1 - ALPHA, C)
    return {
        "K": K,
        "trials": TRIALS,
        "correction": (K - D - 2) / (K - 1),
        "whitened_rate": float(np.mean(white > critical)),
        "unwhitened_rate": float(np.mean(raw > critical)),
        "whitened_quantiles": [float(v) for v in np.quantile(white, [.5, .9, .95, .99])],
        "unwhitened_quantiles": [float(v) for v in np.quantile(raw, [.5, .9, .95, .99])],
        "minimum_rank": int(min(min(x[2], x[3]) for x in records)),
        "all_finite": bool(np.isfinite(white).all() and np.isfinite(raw).all()),
    }


def run():
    delta = target_delta(8.0)
    prediction = float(1 - ncx2.cdf(chi2.ppf(.95, C), C, 8.0))
    rows = []
    for i, K in enumerate(KS):
        null = cell(NULL_MEAN, K, 6091130000000 + i * 100000)
        alt = cell(alt_mean(delta), K, 6091130500000 + i * 100000)
        rows.append({"K": K, "null": null, "alternative": alt})
    last = rows[-1]
    gates = {
        "all_finite_and_full_rank": all(
            r[x]["all_finite"] and r[x]["minimum_rank"] == C
            for r in rows for x in ("null", "alternative")
        ),
        "K512_whitened_null_between_0025_and_0075": .025 <= last["null"]["whitened_rate"] <= .075,
        "K512_power_prediction_error_at_most_008": abs(last["alternative"]["whitened_rate"] - prediction) <= .08,
        "K512_whitening_improves_null_size": abs(last["null"]["whitened_rate"] - ALPHA) < abs(last["null"]["unwhitened_rate"] - ALPHA),
        "population_collision_rank_C": collision_frame(RATE, S, CHOL_TRUE)[1] == C,
    }
    return {
        "settings": {"C": C, "n": N, "d": D, "m": M, "K_values": KS,
                     "trials_per_cell": TRIALS, "target_nc": 8.0,
                     "delta": delta, "chi_square_power_prediction": prediction},
        "rows": rows,
        "gates": gates,
        "hashes": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                   for name in ("WHITENING_PROTOCOL.md", "COVARIANCE_WHITENING_THEORY.tex", "whitening_benchmark.py")},
    }


if __name__ == "__main__":
    result = run()
    (HERE / "whitening_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gates": result["gates"], "rows": result["rows"]}, indent=2))
