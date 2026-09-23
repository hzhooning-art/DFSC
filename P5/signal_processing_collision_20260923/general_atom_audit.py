from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.optimize import minimize_scalar


def atom(family, theta, t):
    if family == "gaussian_scale":
        f = np.exp(-theta * t * t)
        return f, -t * t * f, t**4 * f
    if family == "rational_decay":
        z = 1.0 + theta * t
        return 1.0 / z, -t / z**2, 2.0 * t * t / z**3
    raise ValueError(family)


def residual_norm(x, basis):
    q, _ = np.linalg.qr(basis, mode="reduced")
    z = x - q @ (q.T @ x)
    return float(z @ z)


def coefficients(family, theta, t):
    f, fp, fpp = atom(family, theta, t)
    v = residual_norm(fp, np.column_stack([np.ones(t.size), f]))
    w = residual_norm(fpp, np.column_stack([np.ones(t.size), f, fp]))
    return v, w


def profiled(family, theta, delta, p, r, t, bounds):
    f, _, _ = atom(family, theta, t)
    f2, _, _ = atom(family, theta + delta, t)
    g = f2 - f
    rows = np.vstack([f + p * g, np.sqrt(r) * g])

    def objective(k):
        fk, _, _ = atom(family, k, t)
        basis = np.column_stack([np.ones(t.size), fk])
        q, _ = np.linalg.qr(basis, mode="reduced")
        z = rows - (rows @ q) @ q.T
        return float(np.sum(z * z))

    grid = np.linspace(bounds[0], bounds[1], 2001)
    values = np.array([objective(k) for k in grid])
    j = int(np.argmin(values))
    lo = grid[max(j - 2, 0)]
    hi = grid[min(j + 2, grid.size - 1)]
    fit = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-14})
    candidates = [(grid[j], values[j]), (fit.x, fit.fun),
                  (bounds[0], objective(bounds[0])),
                  (bounds[1], objective(bounds[1]))]
    return min(candidates, key=lambda x: x[1])


def run():
    designs = {
        "gaussian_scale": (np.linspace(0.0, 2.0, 24), 0.35, (0.10, 0.80)),
        "rational_decay": (np.linspace(0.0, 4.0, 24), 0.55, (0.20, 1.20)),
    }
    rows = []
    for family, (t, theta, bounds) in designs.items():
        v, w = coefficients(family, theta, t)
        for p in (0.2, 0.5, 0.8):
            c2 = (p * (1.0 - p) / 2.0) ** 2
            for delta in (0.002, 0.005, 0.01, 0.02):
                paths = {
                    "proportional": 0.0,
                    "crossover": c2 * w * delta**2 / v,
                    "fixed": 0.01,
                }
                for path, r in paths.items():
                    k, exact = profiled(family, theta, delta, p, r, t, bounds)
                    leading = r * v * delta**2 + c2 * w * delta**4
                    rows.append({
                        "family": family, "p": p, "delta": delta,
                        "path": path, "B_over_S": r, "V": v, "W": w,
                        "profiled_rate": k, "distance_squared": exact,
                        "leading": leading,
                        "relative_error": abs(exact / leading - 1.0),
                        "interior": bool(bounds[0] < k < bounds[1]),
                    })

    small = [x for x in rows if x["delta"] <= 0.005]
    monotone = True
    for family in designs:
        for p in (0.2, 0.5, 0.8):
            for path in ("proportional", "crossover", "fixed"):
                group = [x for x in rows if x["family"] == family
                         and x["p"] == p and x["path"] == path]
                err = {x["delta"]: x["relative_error"] for x in group}
                monotone &= err[0.002] <= err[0.02]

    root = Path(__file__).resolve().parent
    gates = {
        "positive_projected_coefficients": bool(all(x["V"] > 0 and x["W"] > 0 for x in rows)),
        "all_minimizers_interior": bool(all(x["interior"] for x in rows)),
        "max_small_gap_relative_error_at_most_005": bool(max(x["relative_error"] for x in small) <= 0.05),
        "smallest_gap_improves_every_group": bool(monotone),
    }
    return {
        "designs": {k: {"theta": v[1], "bounds": v[2], "n": len(v[0])}
                    for k, v in designs.items()},
        "rows": rows,
        "summary": {
            "cells": len(rows),
            "max_small_gap_relative_error": max(x["relative_error"] for x in small),
            "max_all_relative_error": max(x["relative_error"] for x in rows),
        },
        "gates": gates,
        "hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ("GENERAL_ATOM_PROTOCOL.md", "GENERAL_ATOM_EXTENSION.tex",
                                "general_atom_audit.py")},
    }


if __name__ == "__main__":
    result = run()
    Path(__file__).with_name("general_atom_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "gates": result["gates"]}, indent=2))
