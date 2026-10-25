"""Prior-free positive loading anchor: compact moment bounds and fast witnesses."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog

HERE = Path(__file__).resolve().parent
SPEC = HERE / "specs/observation_weighted_prediction_20261011.json"
OUTPUT = HERE.parent / "results/p4_observation_weighted_prediction_20261011.json"
ARCHIVE = OUTPUT.with_suffix(".npz")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def anchor(rate):
    rates = np.asarray(rate, dtype=float)
    safe = np.where(rates == 0, 1., rates)
    return np.where(rates == 0, 1., -np.expm1(-rates) / safe)


def normalized_target(x, target):
    x = np.asarray(x, dtype=float)
    if np.any((x < 0) | (x > 1)):
        raise ValueError("Compact coordinate outside [0,1]")
    if target == "G20":
        result = np.zeros_like(x)
        middle = (x > 0) & (x < 1)
        result[middle] = -np.log(x[middle]) * x[middle] ** 20 / (1-x[middle])
        result[x == 1] = 1.
        return result
    if target == "quiet_half_ramp":
        return 2*x/(1+np.sqrt(x))
    if target == "active_half_ramp":
        return 2/(1+np.sqrt(x))
    if target == "reversal_sigma":
        return x*x-.5*x
    raise ValueError(target)


def target_cell_bounds(edges, target):
    left, right = edges[:-1], edges[1:]
    ql, qr = normalized_target(left, target), normalized_target(right, target)
    low, high = np.minimum(ql, qr), np.maximum(ql, qr)
    if target == "reversal_sigma":
        crossing = (left <= .25) & (right >= .25)
        low[crossing] = -.0625
    return low, high


def physical_target(rate, target):
    """Independent loading-time integrals, not compact-coordinate ratios."""
    if target == "G20":
        return float(np.exp(-20*rate))
    if target in ("quiet_half_ramp", "active_half_ramp"):
        at = 1.5 if target == "quiet_half_ramp" else .5
        return 2*quad(lambda s: np.exp(-rate*(at-s)), 0., .5, epsabs=1e-12)[0]
    if target == "reversal_sigma":
        return quad(lambda s: np.exp(-rate*(3-s)), 0., 1., epsabs=1e-12)[0] - .5*quad(lambda s: np.exp(-rate*(3-s)), 1., 2., epsabs=1e-12)[0]
    raise ValueError(target)


def integrate_truth(info, function):
    if info["kind"] == "atoms":
        return float(sum(w*function(r) for w, r in zip(info["weights"], info["rates"])))
    lo, hi = info["rate_interval"]
    if info["kind"] == "uniform":
        continuous = quad(function, lo, hi, epsabs=1e-11, epsrel=1e-11)[0]/(hi-lo)
    elif info["kind"] == "log_uniform":
        continuous = quad(lambda z: function(np.exp(z)), np.log(lo), np.log(hi), epsabs=1e-11, epsrel=1e-11)[0]/np.log(hi/lo)
    else:
        raise ValueError("Unknown truth")
    return info["equilibrium"]*function(0.) + info["mass"]*continuous


def representation(cell_count, times, target, inner_minimum_x):
    edges = np.linspace(0., 1., cell_count+1)
    powers = np.asarray(times)-1.
    if np.any(powers < 0):
        raise ValueError("Anchor precedes all observation times")
    low = edges[:-1][None, :] ** powers[:, None]
    high = edges[1:][None, :] ** powers[:, None]
    # 0**0 = 1 represents the positive anchor, including x=0 closure.
    qlow, qhigh = target_cell_bounds(edges, target)
    nodes = np.r_[inner_minimum_x, edges[1:]]
    exact = nodes[None, :] ** powers[:, None]
    return {"edges": edges, "outer_a": np.vstack((low, -high)), "q_low": qlow, "q_high": qhigh,
            "inner_a": np.vstack((exact, -exact)), "nodes": nodes,
            "inner_q": normalized_target(nodes, target)}


def verify(c, a, rhs, answer):
    x, y, z = answer.x, answer.ineqlin.marginals, answer.lower.marginals
    return {"primal_violation": float(max(0., np.max(a@x-rhs), -np.min(x))),
            "dual_stationarity": float(np.max(np.abs(c-a.T@y-z))),
            "dual_sign": float(max(0., np.max(y), -np.min(z))),
            "duality_gap": float(abs(c@x-rhs@y))}


def solve(c, a, rhs, label, archive, logs, tolerance):
    answer = linprog(c, A_ub=a, b_ub=rhs, bounds=(0., None), method="highs-ds",
                     options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    if not answer.success:
        raise RuntimeError(f"{label}: {answer.message}")
    checks = verify(c, a, rhs, answer)
    if max(checks.values()) > tolerance:
        raise RuntimeError(f"{label}: bad LP replay {checks}")
    for key, value in (("x", answer.x), ("y", answer.ineqlin.marginals), ("z", answer.lower.marginals)):
        archive[f"{label}_{key}"] = value
    value = float(c@answer.x)
    logs.append({"label": label, "objective": value, "checks": checks})
    return value


def assess(spec):
    rows, witnesses, logs, payload = [], [], [], {}
    times = np.array(spec["observation_times"])
    b = spec["response_error_bound"]
    reps = {(count, target): representation(count, times, target, spec["inner_minimum_x"])
            for count in spec["cell_counts"] for target in spec["bounded_targets"]}
    for info in spec["truths"]:
        values = np.array([integrate_truth(info, lambda rate: float(anchor(rate))*np.exp(-rate*(at-1))) for at in times])
        targets = {target: integrate_truth(info, lambda rate: physical_target(rate, target)) for target in spec["bounded_targets"]}
        payload[f"{info['name']}_observations"] = values
        rhs = np.r_[values+b, -values+b]
        for count in spec["cell_counts"]:
            for target in spec["bounded_targets"]:
                rep = reps[count, target]
                stem = f"{info['name']}_{count}_{target}"
                outer_low = solve(rep["q_low"], rep["outer_a"], rhs, stem+"_outer_low", payload, logs, spec["lp_replay_tolerance"])
                outer_high = -solve(-rep["q_high"], rep["outer_a"], rhs, stem+"_outer_high", payload, logs, spec["lp_replay_tolerance"])
                inner_low = solve(rep["inner_q"], rep["inner_a"], rhs, stem+"_inner_low", payload, logs, spec["lp_replay_tolerance"])
                inner_high = -solve(-rep["inner_q"], rep["inner_a"], rhs, stem+"_inner_high", payload, logs, spec["lp_replay_tolerance"])
                assert outer_low <= inner_low+2e-7 and outer_high >= inner_high-2e-7
                contains = outer_low-2e-7 <= targets[target] <= outer_high+2e-7
                if not contains:
                    raise AssertionError((info["name"], count, target, targets[target]))
                rows.append({"truth": info["name"], "cells": count, "target": target,
                             "outer_interval": [outer_low, outer_high], "inner_interval": [inner_low, inner_high],
                             "endpoint_bracket": max(inner_low-outer_low, outer_high-inner_high),
                             "outer_width": outer_high-outer_low, "truth_target": targets[target], "contains_truth": bool(contains)})
        # Add arbitrarily fast physical finite-mass atoms inside the declared
        # observation bands. This is an adversarial construction, not a new
        # draw from an alleged material distribution.
        alpha = spec["fast_witness_weighted_mass"]
        for rate in spec["fast_witness_rates"]:
            mass = alpha/float(anchor(rate))
            response_change = alpha*np.exp(-rate*(times-1))
            assert np.max(response_change) < b
            target_change = {target: alpha*float(normalized_target(np.array([np.exp(-rate)]), target)[0]) for target in spec["bounded_targets"]}
            covered = []
            for target in spec["bounded_targets"]:
                row = next(r for r in rows if r["truth"] == info["name"] and r["target"] == target and r["cells"] == spec["cell_counts"][-1])
                value = targets[target]+target_change[target]
                low, high = row["outer_interval"]
                covered.append(low-2e-7 <= value <= high+2e-7)
            assert all(covered)
            witnesses.append({"truth": info["name"], "rate": rate, "added_physical_mass": mass,
                              "weighted_mass": alpha, "max_observation_change": float(np.max(response_change)),
                              "bounded_target_changes": target_change, "all_bounded_targets_covered": bool(all(covered)),
                              "initial_modulus_and_instantaneous_step_change": mass})
        print(json.dumps({"completed_truth": info["name"], "LPs": len(logs)}), flush=True)
    # Nested cell outer sets must shrink; inner point sets must expand.
    monotonicity = 0
    for info in spec["truths"]:
        for target in spec["bounded_targets"]:
            group = [r for r in rows if r["truth"] == info["name"] and r["target"] == target]
            for old, new in zip(group[:-1], group[1:]):
                assert new["outer_interval"][0] >= old["outer_interval"][0]-2e-7
                assert new["outer_interval"][1] <= old["outer_interval"][1]+2e-7
                assert new["inner_interval"][0] <= old["inner_interval"][0]+2e-7
                assert new["inner_interval"][1] >= old["inner_interval"][1]-2e-7
                monotonicity += 4
    finest = [r for r in rows if r["cells"] == spec["cell_counts"][-1]]
    bracket = all(r["endpoint_bracket"] <= spec["endpoint_bracket_tolerance"] for r in finest)
    useful = all(r["outer_width"] <= spec["utility_limits"][r["target"]] for r in finest if r["target"] in spec["utility_limits"])
    return {"schema": spec["schema"], "rows": rows, "fast_witnesses": witnesses, "lp_logs": logs,
            "monotonicity_checks": monotonicity, "bracket_gate_pass": bool(bracket), "utility_gate_pass": bool(useful),
            "gate_pass": bool(bracket and useful), "limits": spec["limits"]}, payload


def main():
    if OUTPUT.exists() or ARCHIVE.exists():
        raise FileExistsError("Frozen result exists")
    start = time.perf_counter()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    result, payload = assess(spec)
    np.savez_compressed(ARCHIVE, **payload)
    result.update({"spec_sha256": digest(SPEC), "source_sha256": digest(__file__), "witness_archive_sha256": digest(ARCHIVE), "duration_seconds": time.perf_counter()-start})
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"gate_pass": result["gate_pass"], "LPs": len(result["lp_logs"]), "max_finest_bracket": max(r["endpoint_bracket"] for r in result["rows"] if r["cells"] == spec["cell_counts"][-1]), "duration_seconds": result["duration_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
