import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, ncx2

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor_samp"
N = 24
TIMES = np.linspace(0.0, 8.0, N)
DT = float(TIMES[1] - TIMES[0])
RATE = 0.4
AMP = 100.0
M = 16384
ALPHA = 0.05
NULL_TRIALS = 2048
ALT_TRIALS = 1024


def profiled_one_rate(y):
    def rss(rate):
        x = np.column_stack((np.ones(N), np.exp(-rate * TIMES)))
        residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        return float(residual @ residual)
    fit = minimize_scalar(rss, bounds=(0.05, 1.2), method="bounded",
                          options={"xatol": 1e-13})
    return float(fit.fun), float(fit.x)


def phase_c1(y):
    _, rate = profiled_one_rate(y)
    e = np.exp(-rate * TIMES)
    ep = -TIMES * e
    epp = TIMES * TIMES * e
    q3, _ = np.linalg.qr(np.column_stack((np.ones(N), e, ep)))
    g2 = epp - q3 @ (q3.T @ epp)
    g2 /= np.linalg.norm(g2)
    return float(M * (y @ g2) ** 2)


def effective_rank(s):
    p = s / s.sum()
    p = p[p > 0]
    return int(np.round(np.exp(-np.sum(p * np.log(p)))))


def hankel(y, L):
    return np.stack([y[i:i + L + 1] for i in range(N - L)])


def mp_modes(y):
    L = round(N / 3)
    H = hankel(y, L)
    y0, y1 = H[:, :-1], H[:, 1:]
    u, s, vh = np.linalg.svd(y0, full_matrices=False)
    r = max(1, effective_rank(s))
    if r == min(y0.shape):
        r -= 1
    ur, sr, vr = u[:, :r], s[:r], vh.conj().T[:, :r]
    a = (ur.conj().T @ y1 @ vr) / sr[:, None]
    poles, q = np.linalg.eig(a)
    left = ur @ (sr[:, None] * q)
    return left, poles


def feature_peak(left):
    nfft = 1 << int(np.ceil(np.log2(left.shape[0])))
    centered = left - left.mean(axis=0, keepdims=True)
    return np.max(np.abs(np.fft.fft(centered, n=nfft, axis=0)), axis=0)


def exact_two_means_high_group(values):
    values = np.asarray(values, float)
    if len(values) == 1:
        return np.array([0], dtype=int)
    order = np.argsort(values)
    x = values[order]
    best = None
    for split in range(1, len(x)):
        loss = np.sum((x[:split] - x[:split].mean()) ** 2)
        loss += np.sum((x[split:] - x[split:].mean()) ** 2)
        candidate = (float(loss), split)
        if best is None or candidate < best:
            best = candidate
    split = best[1]
    low, high = order[:split], order[split:]
    return high if values[high].mean() >= values[low].mean() else low


def samp_decay_order(y):
    left, _ = mp_modes(y)
    feature = feature_peak(left)
    if not np.isfinite(feature).all() or np.max(feature) <= 0:
        return 0
    feature /= np.max(feature)
    return int(len(exact_two_means_high_group(feature)))


def official_applicability_on_noiseless(mean):
    left, poles = mp_modes(mean)
    freqs = np.angle(poles) / DT
    denominator = np.sum(np.abs(freqs[:, None] - freqs[None, :]) ** 2, axis=1)
    return {
        "effective_rank": int(len(poles)),
        "pole_frequencies": [float(v) for v in freqs],
        "frequency_denominators": [float(v) for v in denominator],
        "applicable": bool(np.all(denominator > 1e-12)),
    }


def mean_for_delta(delta):
    return 0.5 * AMP * np.exp(-RATE * TIMES) + 0.5 * AMP * np.exp(-(RATE + delta) * TIMES)


def target_delta(target):
    def objective(delta):
        rss, _ = profiled_one_rate(mean_for_delta(delta))
        return M * rss - target
    return float(brentq(objective, 1e-6, 0.5))


def run_cell(mean, trials, seed0):
    phase_reject, decay_order2 = 0, 0
    phase_stats, orders = [], []
    for j in range(trials):
        rng = np.random.default_rng(seed0 + j)
        y = mean + rng.standard_normal(N) / np.sqrt(M)
        stat = phase_c1(y)
        order = samp_decay_order(y)
        phase_stats.append(stat); orders.append(order)
        phase_reject += stat > chi2.ppf(1 - ALPHA, 1)
        decay_order2 += order >= 2
    return {
        "trials": trials,
        "phase_rejections": int(phase_reject),
        "phase_rate": float(phase_reject / trials),
        "samp_decay_order2": int(decay_order2),
        "samp_decay_rate": float(decay_order2 / trials),
        "order_histogram": {str(k): int(orders.count(k)) for k in sorted(set(orders))},
        "phase_stat_quantiles": [float(v) for v in np.quantile(phase_stats, [.5, .9, .95, .99])],
    }


def run():
    null = AMP * np.exp(-RATE * TIMES)
    applicability = {
        "null": official_applicability_on_noiseless(null),
        "two_rate": official_applicability_on_noiseless(mean_for_delta(target_delta(8.0))),
    }
    rows = [{"target_nc": 0.0, **run_cell(null, NULL_TRIALS, 6091120000000)}]
    for i, target in enumerate((4.0, 8.0), 1):
        delta = target_delta(target)
        prediction = float(1 - ncx2.cdf(chi2.ppf(.95, 1), 1, target))
        rows.append({"target_nc": target, "delta": delta,
                     "phase_prediction": prediction,
                     **run_cell(mean_for_delta(delta), ALT_TRIALS, 6091120000000 + i * 100000)})
    source_files = [
        VENDOR / "toolbox" / "compute_SAMP_signal_detection_features.m",
        VENDOR / "toolbox" / "get_MP_components.m",
        VENDOR / "toolbox" / "get_true_idx.m",
    ]
    gates = {
        "official_formula_verified_in_source": all(p.exists() for p in source_files),
        "official_samp_flagged_inapplicable": not applicability["null"]["applicable"] and not applicability["two_rate"]["applicable"],
        "phase_null_between_003_and_007": .03 <= rows[0]["phase_rate"] <= .07,
        "samp_decay_all_rates_reported": all(np.isfinite(r["samp_decay_rate"]) for r in rows),
        "no_posthoc_calibration": True,
    }
    return {
        "settings": {"C": 1, "n": N, "m": M, "null_trials": NULL_TRIALS,
                     "alternative_trials": ALT_TRIALS, "alpha": ALPHA},
        "official_SAMP": applicability,
        "rows": rows,
        "gates": gates,
        "source": {"commit": "3f8cf58a32d7c7849da89ef911827fbfd57a11bd",
                   "hashes": {str(p.relative_to(VENDOR)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in source_files}},
    }


if __name__ == "__main__":
    result = run()
    (HERE / "samp_direct_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gates": result["gates"], "rows": result["rows"]}, indent=2))
