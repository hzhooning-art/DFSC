import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2


HERE = Path(__file__).resolve().parent


def power(rho, df, critical):
    return float(ncx2.sf(critical, df, rho * rho))


def rho_for_power(target, df, critical):
    return float(brentq(lambda rho: power(rho, df, critical) - target, 0.0, 100.0))


def build(C=6, n=24, alpha=0.05, target=0.70):
    d = C * (n - 2)
    r0 = d - 1
    critical_profile = float(chi2.ppf(1 - alpha, r0))
    critical_union = float(chi2.ppf(1 - alpha, d))
    rho_profile = rho_for_power(target, r0, critical_profile)
    rho_union = rho_for_power(target, r0, critical_union)
    old_triangle_limit = float(
        np.sqrt(chi2.ppf(0.85, C * (n - 1))) + np.sqrt(critical_union)
    )
    grid = [0.0, 1.3, 2.0, 3.0, 4.0, rho_profile, rho_union, 6.0, 10.0]
    rows = [
        {
            "rho": rho,
            "profile_power": power(rho, r0, critical_profile),
            "exact_union_local_power": power(rho, r0, critical_union),
            "pinsker_upper_bound": min(1.0, alpha + rho / 2.0),
        }
        for rho in sorted(set(grid))
    ]
    return {
        "C": C,
        "n": n,
        "alpha": alpha,
        "target_power": target,
        "profile_residual_df": r0,
        "finite_union_reference_df": d,
        "profile_critical": critical_profile,
        "finite_union_asymptotic_critical": critical_union,
        "rho_for_target_profile": rho_profile,
        "rho_for_target_exact_union_local_limit": rho_union,
        "old_triangle_sufficient_rho_limit": old_triangle_limit,
        "rows": rows,
        "hashes": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in (
                "LOCAL_LIMIT_POWER_THEORY.md",
                "local_limit_power.py",
                "MATCHED_DETECTION_BOUNDARY_THEORY.md",
            )
        },
    }


if __name__ == "__main__":
    result = build()
    (HERE / "local_limit_power_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: result[k] for k in (
        "rho_for_target_profile",
        "rho_for_target_exact_union_local_limit",
        "old_triangle_sufficient_rho_limit",
    )}, indent=2))
