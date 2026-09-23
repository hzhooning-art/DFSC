import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, ncx2, norm


HERE = Path(__file__).resolve().parent
C = 6
D = C - 1
ALPHA = 0.05
RHO2_VALUES = (2.0, 4.0, 8.0, 12.0, 18.0)


def cone_critical():
    return brentq(
        lambda x: .5 * chi2.cdf(x, D) + .5 * chi2.cdf(x, C) - (1 - ALPHA),
        0.0, 40.0,
    )


def cone_power(rho, theta, critical):
    diversity = rho * np.cos(theta)
    curvature = rho * np.sin(theta)
    base = norm.cdf(-curvature) * ncx2.sf(critical, D, diversity**2)
    middle = quad(
        lambda z: norm.pdf(z - curvature)
        * ncx2.sf(critical - z*z, D, diversity**2),
        0.0, np.sqrt(critical), epsabs=1e-11,
    )[0]
    upper = norm.sf(np.sqrt(critical) - curvature)
    return float(base + middle + upper)


def run():
    critical = cone_critical()
    phase_critical = chi2.ppf(1 - ALPHA, C)
    rows = []
    for rho2 in RHO2_VALUES:
        rho = np.sqrt(rho2)
        fit = minimize_scalar(
            lambda theta: cone_power(rho, theta, critical),
            bounds=(0.0, np.pi / 2), method="bounded",
            options={"xatol": 1e-11},
        )
        rows.append({
            "rho2": rho2,
            "phase_power": float(ncx2.sf(phase_critical, C, rho2)),
            "cone_min_power": float(fit.fun),
            "worst_angle": float(fit.x),
            "pure_diversity_power": cone_power(rho, 0.0, critical),
            "pure_curvature_power": cone_power(rho, np.pi / 2, critical),
        })
    checks = {
        "null_critical_finite": bool(np.isfinite(critical)),
        "minimum_at_diversity_face_numerically": all(row["worst_angle"] <= 1e-5 for row in rows),
        "cone_min_exceeds_phase_in_checked_cases": all(row["cone_min_power"] > row["phase_power"] for row in rows),
        "curvature_power_exceeds_diversity_power": all(row["pure_curvature_power"] > row["pure_diversity_power"] for row in rows),
    }
    return {"critical": critical, "rows": rows, "checks": checks,
            "interpretation": "deterministic limit-experiment diagnostic, not proof of exact cone maximinity"}


if __name__ == "__main__":
    result = run()
    (HERE / "cone_direction_power_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["checks"], indent=2))
