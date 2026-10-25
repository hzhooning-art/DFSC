"""Equal-dose endpoint-design screen against the admissible constant baseline."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import gammainc, gamma
from observation_weighted_prediction import digest, integrate_truth, solve

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "experiments/specs/matched_endpoint_design_20261011.json"
OUT = ROOT / "results/p4_matched_endpoint_design_20261011.json"
ARCHIVE = OUT.with_suffix(".npz")
# Coefficients of the positive rate profile in lag r=1-t, not a signed input.
LAG_COEFF = {"constant": [1.], "early_linear_endpoint": [0., 3., -6., 6.],
             "late_linear_start": [3., -9., 12., -6.],
             "early_quadratic_endpoint": [0., 0., 3.],
             "late_quadratic_start": [3., -6., 3.]}


def kernel(rates, design):
    rates = np.asarray(rates, dtype=float)
    if np.any(rates < 0):
        raise ValueError("Negative relaxation rate")
    safe = np.where(rates == 0, 1., rates)
    total = np.zeros_like(rates)
    for p, coefficient in enumerate(LAG_COEFF[design]):
        val = gamma(p+1)*gammainc(p+1, safe)/safe**(p+1)
        total += coefficient*np.where(rates == 0, 1/(p+1), val)
    if np.any(total <= 0):
        raise ArithmeticError("Positive loading kernel lost positivity")
    return total


def target_kernel(rates, delay):
    rates = np.asarray(rates, dtype=float)
    safe = np.where(rates == 0, 1., rates)
    return np.exp(-delay*rates)*np.where(rates == 0, 1., -2*np.expm1(-.5*rates)/safe)


def tail_bounds(design, delay, start):
    if start <= 3 or delay < 0:
        raise ValueError("Tail formula requires start>3 and nonnegative delay")
    if design == "constant":
        if delay == 0:
            return 2/(1+np.exp(-start/2)), 2.
        return 0., 2*np.exp(-delay*start)
    if design.startswith("late_"):
        k = 3 if design == "late_linear_start" else 2
        high = (2/3)/(1-k/start)*np.exp(-delay*start)
        low = (2/3)*(1-np.exp(-start/2)) if delay == 0 else 0.
        return float(low), float(high)
    if delay == 0:
        raise ValueError("Tapered reference has an unbounded active target")
    if design == "early_linear_endpoint":
        a, factor = 1, 2/(1.5*gammainc(2, start))
    else:
        a, factor = 2, 2/(6*gammainc(3, start))
    at = max(start, a/delay)
    return 0., float(factor*at**a*np.exp(-delay*at))


def representation(spec, design, target, cells):
    edges = np.r_[0., np.geomspace(spec["minimum_positive_rate"], spec["tail_start"], cells-1), np.inf]
    left, right = edges[:-1], edges[1:]
    delays = np.array(spec["observation_delays"])
    low = np.exp(-delays[:, None]*left[None, :])
    high = np.empty_like(low)
    high[:, :-1] = np.exp(-delays[:, None]*right[None, :-1])
    high[:, -1] = (delays == 0).astype(float)
    # low is the larger observation kernel; high is the smaller one.
    delay = spec["targets"][target]
    qlo = np.empty(cells)
    qhi = np.empty(cells)
    qlo[:-1] = target_kernel(right[:-1], delay)/kernel(left[:-1], design)
    qhi[:-1] = target_kernel(left[:-1], delay)/kernel(right[:-1], design)
    if design == "constant":
        # Exact monotonicity for delay=0 and delay=1/4 avoids loose ratio cells.
        a = target_kernel(left[:-1], delay)/kernel(left[:-1], design)
        b = target_kernel(right[:-1], delay)/kernel(right[:-1], design)
        qlo[:-1], qhi[:-1] = np.minimum(a, b), np.maximum(a, b)
    elif design.startswith("late_"):
        # Increasing unit-area loading stochastically dominates uniform loading.
        qhi[:-1] = np.minimum(qhi[:-1], 2.)
    qlo[-1], qhi[-1] = tail_bounds(design, delay, spec["tail_start"])
    pad = 2e-12
    qlo = np.maximum(0., qlo-pad)
    qhi += pad
    nodes = edges[:-1]  # All finite; closure at infinity appears in the outer only.
    exact = np.exp(-delays[:, None]*nodes[None, :])
    return {"edges": edges, "nodes": nodes, "outer_a": np.vstack((high, -low)),
            "inner_a": np.vstack((exact, -exact)), "q_low": qlo, "q_high": qhi,
            "inner_q": target_kernel(nodes, delay)/kernel(nodes, design)}


def compare(rows, spec):
    finest = [r for r in rows if r["cells"] == max(spec["cell_counts"])]
    summaries = []
    for mode in spec["band_modes"]:
        for target in spec["targets"]:
            base = {r["truth"]: r for r in finest if r["band_mode"] == mode and r["target"] == target and r["design"] == "constant"}
            for design in spec["designs"]:
                if design == "constant":
                    continue
                group = [r for r in finest if r["band_mode"] == mode and r["target"] == target and r["design"] == design]
                if not group:
                    summaries.append({"band_mode": mode, "target": target, "design": design, "bounded": False,
                                      "decision": "Unbounded by theorem; no finite-width performance comparison"})
                    continue
                mean_inner = np.mean([r["inner_width"] for r in group])
                mean_outer = np.mean([r["outer_width"] for r in group])
                mean_base_inner = np.mean([base[r["truth"]]["inner_width"] for r in group])
                mean_base_outer = np.mean([base[r["truth"]]["outer_width"] for r in group])
                gain = [float(1-mean_outer/mean_base_inner), float(1-mean_inner/mean_base_outer)]
                precision = all(r["endpoint_bracket"] <= spec["endpoint_bracket_tolerance"] and base[r["truth"]]["endpoint_bracket"] <= spec["endpoint_bracket_tolerance"] for r in group)
                useful = all(r["outer_width"] <= spec["utility_width_limits"][target] for r in group)
                summaries.append({"band_mode": mode, "target": target, "design": design, "bounded": True,
                                  "mean_width_bracket": [float(mean_inner), float(mean_outer)],
                                  "baseline_mean_width_bracket": [float(mean_base_inner), float(mean_base_outer)],
                                  "mean_gain_bracket": gain, "all_endpoint_precision_pass": bool(precision),
                                  "utility_pass": bool(useful),
                                  "gain_gate_pass": bool(precision and useful and gain[0] >= spec["required_mean_gain_lower"]),
                                  "gain_definitely_below_gate": bool(gain[1] < spec["required_mean_gain_lower"])})
    return summaries


def main():
    if OUT.exists() or ARCHIVE.exists():
        raise FileExistsError("Frozen matched-design result exists")
    start = time.perf_counter()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    parent_path = SPEC.parent/spec["parent_truth_spec"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    theory = ROOT/"docs/P4_LOADING_ENDPOINT_PHASE_THEORY_20261011.md"
    assert digest(theory) == spec["parent_theory_sha256"]
    rows, logs, payload, data, witnesses = [], [], {}, [], []
    reps = {(design, target, cells): representation(spec, design, target, cells)
            for design in spec["designs"] for target, delay in spec["targets"].items()
            if delay > 0 or not design.startswith("early_") for cells in spec["cell_counts"]}
    for info in parent["truths"]:
        target_values = {target: integrate_truth(info, lambda rate: float(target_kernel(rate, delay))) for target, delay in spec["targets"].items()}
        for design in spec["designs"]:
            values = np.array([integrate_truth(info, lambda rate: float(kernel(rate, design))*np.exp(-d*rate)) for d in spec["observation_delays"]])
            for mode in spec["band_modes"]:
                band = spec["absolute_response_band"] if mode == "absolute" else spec["relative_anchor_band"]*values[0]
                data.append({"truth": info["name"], "design": design, "band_mode": mode,
                             "observations": values.tolist(), "band": float(band), "anchor_relative_band": float(band/values[0])})
                rhs = np.r_[values+band, -values+band]
                if design.startswith("early_"):
                    alpha = band/2
                    for rate in spec["adversarial_rates"]:
                        mass = alpha/float(kernel(rate, design))
                        changes = alpha*np.exp(-rate*np.array(spec["observation_delays"]))
                        change = mass*float(target_kernel(rate, 0.))
                        assert changes.max() < band
                        witnesses.append({"truth": info["name"], "design": design, "band_mode": mode,
                                          "rate": rate, "physical_mass": mass, "observation_changes": changes.tolist(),
                                          "active_target_change": change, "minimax_pair_lower": change/2})
                for target, delay in spec["targets"].items():
                    if delay == 0 and design.startswith("early_"):
                        continue
                    for cells in spec["cell_counts"]:
                        rep = reps[design, target, cells]
                        stem = f"{info['name']}_{design}_{mode}_{target}_{cells}"
                        outer_low = solve(rep["q_low"], rep["outer_a"], rhs, stem+"_outer_low", payload, logs, spec["lp_replay_tolerance"])
                        outer_high = -solve(-rep["q_high"], rep["outer_a"], rhs, stem+"_outer_high", payload, logs, spec["lp_replay_tolerance"])
                        inner_low = solve(rep["inner_q"], rep["inner_a"], rhs, stem+"_inner_low", payload, logs, spec["lp_replay_tolerance"])
                        inner_high = -solve(-rep["inner_q"], rep["inner_a"], rhs, stem+"_inner_high", payload, logs, spec["lp_replay_tolerance"])
                        assert outer_low-2e-7 <= inner_low <= inner_high <= outer_high+2e-7
                        assert outer_low-2e-7 <= target_values[target] <= outer_high+2e-7
                        rows.append({"truth": info["name"], "design": design, "band_mode": mode, "target": target,
                                     "cells": cells, "outer_interval": [outer_low, outer_high], "inner_interval": [inner_low, inner_high],
                                     "inner_width": inner_high-inner_low, "outer_width": outer_high-outer_low,
                                     "endpoint_bracket": max(inner_low-outer_low, outer_high-inner_high),
                                     "truth_target": target_values[target]})
            print(json.dumps({"completed_truth": info["name"], "design": design, "LPs": len(logs)}), flush=True)
    summaries = compare(rows, spec)
    np.savez_compressed(ARCHIVE, **payload)
    result = {"schema": "P4-matched-endpoint-design-result-v1", "spec_sha256": digest(SPEC),
              "source_sha256": digest(__file__), "parent_truth_spec_sha256": digest(parent_path),
              "parent_theory_sha256": digest(theory), "archive_sha256": digest(ARCHIVE),
              "data": data, "rows": rows, "physical_unboundedness_examples": witnesses,
              "lp_logs": logs, "summaries": summaries,
              "negative_control": {"reference_dead_time": spec["negative_control_dead_time"],
                                   "active_target_unbounded_for_all_designs": True,
                                   "basis": "target delay=0 < earliest observation dead time; proof, not finite rate search"},
              "all_internal_checks_pass": True, "duration_seconds": time.perf_counter()-start,
              "limits": spec["limits"]}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"LPs": len(logs), "rows": len(rows), "physical_witnesses": len(witnesses),
                      "summaries": summaries, "duration_seconds": result["duration_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
