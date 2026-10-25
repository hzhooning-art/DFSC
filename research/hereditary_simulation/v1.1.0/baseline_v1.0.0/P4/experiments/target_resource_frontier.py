"""Frozen resource-boundary diagnosis following a failed loading library gate."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from target_loading_headroom_gate import (
    TARGETS, build_library, matrices, modulus_for, pair_problem,
    posterior_problem, solve, stress_truth_integral,
)

HERE = Path(__file__).resolve().parent
SPEC = HERE / "specs" / "target_resource_frontier_20261011.json"
OUTPUT = HERE.parent / "results" / "p4_target_resource_frontier_20261011.json"
ARCHIVE = OUTPUT.with_suffix(".npz")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    parent_path = SPEC.parent / spec["parent_spec"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    previous_path = OUTPUT.parent / spec["parent_result"]
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    for name, expected in previous["source_hashes"].items():
        assert digest(HERE.parent/name) == expected, name
    previous_archive = OUTPUT.parent / spec["parent_archive"]
    assert digest(previous_archive) == previous["witness_archive_sha256"]
    heldout = json.loads((SPEC.parent/parent["parent_spec"]).read_text(
        encoding="utf-8"))["heldout_input_segments"]
    return spec, parent, previous, previous_archive, [tuple(s) for s in heldout]


def resources(spec):
    return [{"index": i, "horizon": t, "future_noise": b,
             "total_window_duration": 8+2*t}
            for i, (t, b) in enumerate((t, b)
                for t in spec["confirmation_horizons"]
                for b in spec["future_noise_amplitudes"])]


def representations(spec, parent, heldout):
    histories, _ = build_library(parent)
    pilot = [(np.linspace(.5, 8, parent["pilot_count"]), histories["ramp_hold"])]
    schedules = {str(t): [(np.linspace(.25, t, parent["confirmation_count_each"]),
                           histories[name]) for name in ("ramp_hold", "cycle")]
                 for t in spec["confirmation_horizons"]}
    assert all(t < spec["target_time"] for t in spec["confirmation_horizons"])
    grid = dict(parent, inner_rate_count=spec["inner_rate_count"],
                outer_cell_count=spec["outer_cell_count"])
    reps = {}
    for rep in ("inner", "outer"):
        rates, q, f, block = matrices(grid, schedules, heldout, rep == "outer")
        reps[rep] = {"rates": rates, "q": q, "f": f, "pilot": block(pilot),
                     "future": {k: block(v) for k, v in schedules.items()}}
    return reps, schedules


def pair_with_distinct_noise(ap, ep, yp, af, ef, pilot_b, future_b, q, f):
    c, a, _, eq, rhs = pair_problem(ap, ep, yp, af, ef, future_b, q, f)
    d = np.r_[yp+pilot_b, -yp+pilot_b, yp+pilot_b, -yp+pilot_b,
              np.full(2*len(af), 2*future_b)]
    return c, a, d, eq, rhs


def posterior_with_distinct_noise(ap, ep, yp, af, ef, yf,
                                  pilot_b, future_b, q, f, lower):
    a, e, y = np.vstack((ap, af)), np.vstack((ep, ef)), np.r_[yp, yf]
    bands = np.r_[np.full(len(yp), pilot_b), np.full(len(yf), future_b)]
    return posterior_problem(a, e, y, bands, q, f, lower)


def main():
    if OUTPUT.exists() or ARCHIVE.exists():
        raise FileExistsError("Frozen frontier output exists; do not overwrite")
    start = time.perf_counter()
    spec, parent, previous, previous_archive, heldout = inputs()
    reps, schedules = representations(spec, parent, heldout)
    cells = resources(spec)
    witnesses, logs, rows = {}, [], []
    old_pairs = {}
    with np.load(previous_archive, allow_pickle=False) as old:
        for case in previous["cases"]:
            cid = case["id"]
            selected = next(r for r in previous["rows"] if r["case"] == cid and
                            r["candidate"] == case["selected"])
            target = max(TARGETS, key=lambda t: selected["diameter"]["inner"][t] /
                         spec["utility_limits"][t])
            x = old[f"pair_{cid}_{selected['candidate_index']}_inner_{target}_x"]
            first, second = np.split(x, 2)
            old_pairs[cid] = first-second
            witnesses[f"case{cid}_pilot"] = old[f"case{cid}_pilot"]
    for case in previous["cases"]:
        cid, truth = case["id"], case["truth"]
        yp = witnesses[f"case{cid}_pilot"]
        modulus = modulus_for(parent["truths"][truth])
        rng = np.random.default_rng(np.random.SeedSequence([spec["seed_base"], cid, 1]))
        standard_noise = rng.uniform(-1., 1., 2*parent["confirmation_count_each"])
        witnesses[f"case{cid}_standard_future_noise"] = standard_noise
        for cell in cells:
            rid, t, bf = cell["index"], cell["horizon"], cell["future_noise"]
            key = str(t)
            widths = {"inner": {}, "outer": {}}
            for rep, data in reps.items():
                ap, ep = data["pilot"]; af, ef = data["future"][key]
                for target in TARGETS:
                    problem = pair_with_distinct_noise(ap, ep, yp, af, ef,
                        spec["pilot_noise_amplitude"], bf, data["q"][target], data["f"][target])
                    label = f"pair_{cid}_{rid}_{rep}_{target}"
                    solution = solve(problem, label, witnesses, logs)
                    if solution is None:
                        raise RuntimeError(f"Unexpected infeasible pair: {label}")
                    primal, dual = solution
                    widths[rep][target] = max(0., -primal-1e-8 if rep == "inner" else -dual+1e-8)
            assert all(widths["inner"][tgt] <= widths["outer"][tgt]+2e-7 for tgt in TARGETS)
            scores = {r: max(widths[r][target]/spec["utility_limits"][target]
                            for target in TARGETS) for r in widths}
            if t == 8. and bf == .01:
                baseline = next(r for r in previous["rows"] if r["case"] == cid and
                                r["candidate"] == parent["fixed_baseline"])
                for rep in widths:
                    for target in TARGETS:
                        assert abs(widths[rep][target]-baseline["diameter"][rep][target]) < 2e-7
            clean = np.concatenate([stress_truth_integral(times, pieces, modulus)
                                    for times, pieces in schedules[key]])
            yf = clean+bf*standard_noise
            witnesses[f"case{cid}_future_{rid}"] = yf
            ap, ep = reps["outer"]["pilot"]; af, ef = reps["outer"]["future"][key]
            intervals = {}
            for target in TARGETS:
                bounds = []
                for lower in (True, False):
                    problem = posterior_with_distinct_noise(ap, ep, yp, af, ef, yf,
                        spec["pilot_noise_amplitude"], bf, reps["outer"]["q"][target],
                        reps["outer"]["f"][target], lower)
                    solution = solve(problem, f"post_{cid}_{rid}_{target}_{int(lower)}",
                                     witnesses, logs)
                    if solution is None:
                        raise RuntimeError("Supported truth infeasible in posterior")
                    _, dual = solution
                    bounds.append(dual-1e-8 if lower else -dual+1e-8)
                if target == "G20":
                    bounds = [max(0., bounds[0]), min(1., bounds[1])]
                real = previous["truth_targets"][truth][target]
                assert bounds[0]-2e-7 <= real <= bounds[1]+2e-7
                assert bounds[1]-bounds[0] <= widths["outer"][target]+2e-7
                intervals[target] = {"interval": bounds, "width": bounds[1]-bounds[0],
                                     "contains_truth": True}
            # Previously saved witnesses can disappear; fresh pair LPs above check
            # whether different ambiguity remains under the changed resources.
            separation = float(np.max(np.abs(reps["inner"]["future"][key][0]@old_pairs[cid])))
            rows.append({"case": cid, "truth": truth, "seed": case["seed"],
                "resource_index": rid, "diameter": widths, "score": scores,
                "posterior": intervals, "old_pair_maximum_separation": separation,
                "old_pair_survives": separation <= 2*bf+2e-7})
        print(json.dumps({"completed_case": cid+1, "truth": truth}), flush=True)
    # Fixed-horizon constraints shrink with the future bound. Stretched grids
    # across horizons are not nested, so no monotonic-window assertion is made.
    for case in previous["cases"]:
        for t in spec["confirmation_horizons"]:
            group = [r for r in rows if r["case"] == case["id"] and cells[r["resource_index"]]["horizon"] == t]
            for coarse, fine in zip(group[:-1], group[1:]):
                for rep in ("inner", "outer"):
                    for target in TARGETS:
                        assert fine["diameter"][rep][target] <= coarse["diameter"][rep][target]+2e-7
    summaries = []
    for cell in cells:
        group = [r for r in rows if r["resource_index"] == cell["index"]]
        by_truth = {}
        for truth in parent["truths"]:
            subset = [r for r in group if r["truth"] == truth]
            base = [r for r in rows if r["truth"] == truth and r["resource_index"] == 0]
            upper = float(np.mean([r["score"]["outer"] for r in subset]))
            baseline_lower = float(np.mean([r["score"]["inner"] for r in base]))
            ratio = upper/baseline_lower
            by_truth[truth] = {"mean_inner_score": float(np.mean([r["score"]["inner"] for r in subset])),
                "mean_outer_score": upper, "maximum_outer_score": max(r["score"]["outer"] for r in subset),
                "upper_to_baseline_lower_ratio": ratio, "clear_20_percent_reduction": ratio <= spec["resource_gate"]["maximum_ratio"],
                "mean_outer_diameters": {target: float(np.mean([r["diameter"]["outer"][target] for r in subset])) for target in TARGETS},
                "mean_realized_widths": {target: float(np.mean([r["posterior"][target]["width"] for r in subset])) for target in TARGETS}}
        passing = [truth for truth, v in by_truth.items() if v["clear_20_percent_reduction"]]
        slow_pass = by_truth["slow_three"]["maximum_outer_score"] <= 1.
        summaries.append(dict(cell, by_truth=by_truth, reduced_groups=passing,
            slow_group_all_pilots_utility_pass=slow_pass,
            resource_gate_pass=len(passing) >= 2 and "continuous_uniform" in passing and slow_pass,
            old_pairs_surviving=sum(r["old_pair_survives"] for r in group),
            all_pilot_utility_pass_count=sum(r["score"]["outer"] <= 1. for r in group)))
    np.savez_compressed(ARCHIVE, **witnesses)
    source_paths = [SPEC, Path(__file__), HERE/"target_loading_headroom_gate.py",
                    HERE/"calibrated_multiload_gate.py", HERE/"constitutive_response_gate.py",
                    HERE/"offgrid_curvature_envelope_gate.py", SPEC.parent/spec["parent_spec"],
                    SPEC.parent/parent["parent_spec"]]
    result = {"scope": spec["scope"], "spec_sha256": digest(SPEC),
        "parent_result_sha256": digest(OUTPUT.parent/spec["parent_result"]),
        "source_hashes": {str(p.relative_to(HERE.parent)): digest(p) for p in source_paths},
        "witness_archive_sha256": digest(ARCHIVE), "resource_cells": cells,
        "case_count": len(previous["cases"]), "row_count": len(rows), "solver_count": len(logs),
        "duration_seconds": time.perf_counter()-start, "summaries": summaries,
        "all_baseline_brackets_reproduced": True, "all_fixed_horizon_noise_brackets_monotone": True,
        "all_posterior_targets_contained": True, "rows": rows, "lp_replay_logs": logs,
        "limitations": spec["limitations"]}
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": [{k: v for k, v in s.items() if k != "by_truth"} for s in summaries],
        "solver_count": len(logs), "duration_seconds": result["duration_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
