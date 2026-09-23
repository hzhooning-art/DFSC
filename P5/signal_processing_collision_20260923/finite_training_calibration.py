import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.stats import chi2, f


HERE = Path(__file__).resolve().parent
D, P, R = 48, 9, 4
K_VALUES = (64, 128, 512)
TRIALS = 1500
ALPHA = 0.05
SEED = 6091140000000


def covariances():
    identity = np.eye(D)
    c, n = 4, 12
    channel = np.full((c, c), 0.25)
    np.fill_diagonal(channel, 1.0)
    time = 0.60 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return {"identity": identity, "colored": np.kron(channel, time)}


def fixed_spaces():
    rng = np.random.default_rng(609113991)
    orthogonal, _ = np.linalg.qr(rng.standard_normal((D, D)))
    a = orthogonal[:, :P]
    ell = orthogonal[:, :P + R]
    # Euclidean complements are sufficient for the residual block identity.
    return a, ell, null_space(a.T), null_space(ell.T)


def residual_quadratic(x, scatter, complement, nu):
    projected_x = complement.T @ x
    projected_w = complement.T @ scatter @ complement
    return float(nu * projected_x @ np.linalg.solve(projected_w, projected_x))


def cell(sigma, k_training, seed):
    nu = k_training - 1
    k = D - P
    _, _, a_perp, ell_perp = fixed_spaces()
    chol = np.linalg.cholesky(sigma)
    rng = np.random.default_rng(seed)
    exact_reject = 0
    old_reject = 0
    exact_values = np.empty(TRIALS)
    old_values = np.empty(TRIALS)
    ql_values = np.empty(TRIALS)
    exact_critical = f.ppf(1.0 - ALPHA, R, nu - k + 1)
    old_critical = chi2.ppf(1.0 - ALPHA, R)
    old_correction = (nu - D - 1) / nu
    for trial in range(TRIALS):
        training = rng.standard_normal((nu, D)) @ chol.T
        scatter = training.T @ training
        x = chol @ rng.standard_normal(D)
        qa = residual_quadratic(x, scatter, a_perp, nu)
        ql = residual_quadratic(x, scatter, ell_perp, nu)
        improvement = max(0.0, qa - ql)
        exact = (nu - k + 1) / R * improvement / (nu + ql)
        old = old_correction * improvement
        exact_values[trial] = exact
        old_values[trial] = old
        ql_values[trial] = ql
        exact_reject += exact > exact_critical
        old_reject += old > old_critical
    return {
        "K": k_training,
        "nu": nu,
        "exact_F_df": [R, nu - k + 1],
        "old_scalar_correction": old_correction,
        "exact_F_rejection_rate": exact_reject / TRIALS,
        "old_chi_square_rejection_rate": old_reject / TRIALS,
        "exact_F_quantiles": [float(v) for v in np.quantile(exact_values, [0.5, 0.9, 0.95, 0.99])],
        "theoretical_F_95": float(exact_critical),
        "old_corrected_quantiles": [float(v) for v in np.quantile(old_values, [0.5, 0.9, 0.95, 0.99])],
        "median_Q_L": float(np.median(ql_values)),
        "all_finite": bool(np.isfinite(exact_values).all() and np.isfinite(old_values).all()),
    }


def run():
    rows = []
    for covariance_index, (name, sigma) in enumerate(covariances().items()):
        for k_index, k_training in enumerate(K_VALUES):
            result = cell(sigma, k_training, SEED + covariance_index * 100000 + k_index * 10000)
            result["covariance"] = name
            rows.append(result)
    exact_rates = [row["exact_F_rejection_rate"] for row in rows]
    gates = {
        "all_finite": bool(all(row["all_finite"] for row in rows)),
        "all_exact_sizes_between_003_and_007": bool(all(0.03 <= rate <= 0.07 for rate in exact_rates)),
        "max_exact_size_error_at_most_002": bool(max(abs(rate - ALPHA) for rate in exact_rates) <= 0.02),
        "old_scalar_is_liberal_at_K64_for_both_covariances": bool(all(
            row["old_chi_square_rejection_rate"] > 0.08
            for row in rows if row["K"] == 64
        )),
        "invariance_rate_gap_at_most_0025": bool(max(
            abs(next(row["exact_F_rejection_rate"] for row in rows if row["covariance"] == "identity" and row["K"] == k_training)
                - next(row["exact_F_rejection_rate"] for row in rows if row["covariance"] == "colored" and row["K"] == k_training))
            for k_training in K_VALUES
        ) <= 0.025),
    }
    names = (
        "FINITE_TRAINING_CALIBRATION_THEORY.tex",
        "finite_training_calibration.py",
    )
    return {
        "settings": {
            "d": D, "fixed_null_dimension": P, "added_dimension": R,
            "tangent_complement_dimension": D - P, "K_values": K_VALUES,
            "trials_per_cell": TRIALS, "alpha": ALPHA,
        },
        "rows": rows,
        "gates": gates,
        "hashes": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names},
    }


if __name__ == "__main__":
    output = run()
    (HERE / "finite_training_calibration_results.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gates": output["gates"], "rows": output["rows"]}, indent=2))
