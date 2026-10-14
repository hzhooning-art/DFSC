"""Frozen nested-support prediction test, including the zero-rate limit."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from calibrated_multiload_gate import basis
from constitutive_response_gate import strain
from offgrid_curvature_envelope_gate import response_curvature_envelope
from target_loading_headroom_gate import TARGETS, solve
from target_resource_frontier import (
    digest, inputs as frontier_inputs, pair_with_distinct_noise,
    posterior_with_distinct_noise, representations as frontier_representations,
)

HERE = Path(__file__).resolve().parent
SPEC = HERE/"specs"/"target_support_sensitivity_20261011.json"
OUTPUT = HERE.parent/"results"/"p4_target_support_sensitivity_20261011.json"
ARCHIVE = OUTPUT.with_suffix(".npz")


def inputs():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    parent = json.loads((SPEC.parent/spec["parent_spec"]).read_text(encoding="utf-8"))
    prior = json.loads((OUTPUT.parent/spec["parent_result"]).read_text(encoding="utf-8"))
    for name, expected in prior["source_hashes"].items():
        assert digest(HERE.parent/name) == expected, name
    prior_archive = OUTPUT.parent/spec["parent_archive"]
    assert digest(prior_archive) == prior["witness_archive_sha256"]
    _, original, headroom, _, heldout = frontier_inputs()
    _, schedules = frontier_representations(parent, original, heldout)
    histories = original["histories"]
    pilot = [(np.linspace(.5, 8, original["pilot_count"]), histories["ramp_hold"])]
    return spec, parent, original, prior, headroom, prior_archive, heldout, pilot, schedules


def nested_grids(original, floor):
    edges = np.geomspace(*original["rate_support"], original["outer_cell_count"]+1)
    rates = np.geomspace(*original["rate_support"], original["inner_rate_count"])
    for lower, upper in ((.002, .02), (.0002, .002), (.00002, .0002)):
        if floor <= lower:
            edges = np.r_[np.geomspace(lower, upper, 17)[:-1], edges]
            rates = np.r_[np.geomspace(lower, upper, 33)[:-1], rates]
    if floor == 0.:
        edges = np.r_[0., edges]
    assert edges[0] == floor and np.all(np.diff(edges) > 0)
    return edges, rates


def zero_safe_basis(times, segments, rates):
    """Zero-rate exponential response equals the instantaneous strain column."""
    zero = rates == 0.
    result = np.empty((len(times), 1+len(rates)))
    result[:, 0] = strain(times, segments)
    result[:, 1:][:, zero] = result[:, :1]
    if np.any(~zero):
        result[:, 1:][:, ~zero] = basis(times, segments, rates[~zero])[:, 1:]
    return result


def zero_safe_envelope(times, segments, edges):
    if edges[0] > 0.:
        return response_curvature_envelope(times, segments, edges)
    assert edges[1] > 0.
    t = np.asarray(times)
    zero_curvature = np.zeros(len(t))
    for a, b, v in segments:
        # At lambda=0, the positive integrand u^2 is integrated exactly.
        zero_curvature += abs(v)*(np.maximum(t-a, 0.)**3-np.maximum(t-b, 0.)**3)/3
    first = edges[1]**2/8*(zero_curvature*(1+1e-10)+1e-11)
    rest = response_curvature_envelope(times, segments, edges[1:])
    return np.column_stack((first, rest))


def matrices(original, floor, pilot, future, heldout):
    edges, inner_rates = nested_grids(original, floor)
    reps = {}
    for rep in ("inner", "outer"):
        outer = rep == "outer"
        rates = np.column_stack((edges[:-1], edges[1:])).ravel() if outer else inner_rates
        def block(pieces):
            a = np.vstack([zero_safe_basis(t, s, rates) for t, s in pieces])
            if outer:
                e = np.vstack([zero_safe_envelope(t, s, edges) for t, s in pieces])
                e = np.column_stack((np.zeros(len(e)), np.repeat(e, 2, axis=1)))
            else:
                e = np.zeros_like(a)
            return a, e
        q = {"G20": np.r_[1., np.exp(-20*rates)],
             "heldout_sigma20": zero_safe_basis(np.array([20.]), heldout, rates)[0]}
        f = {target: np.zeros_like(v) for target, v in q.items()}
        if outer:
            eg = np.diff(edges)**2/8*(400*np.exp(-20*edges[:-1])*(1+1e-10)+1e-11)
            es = zero_safe_envelope(np.array([20.]), heldout, edges)[0]
            f = {"G20": np.r_[0., np.repeat(eg, 2)],
                 "heldout_sigma20": np.r_[0., np.repeat(es, 2)]}
        reps[rep] = {"rates": rates, "q": q, "f": f, "pilot": block(pilot),
                     "future": block(future)}
    return reps


def monotonicity(rows):
    count = 0
    for cid in sorted({r["case"] for r in rows}):
        for horizon in (8., 16.):
            group = sorted((r for r in rows if r["case"] == cid and r["horizon"] == horizon),
                           key=lambda r: -r["support_floor"])
            for restricted, expanded in zip(group[:-1], group[1:]):
                for target in TARGETS:
                    for rep in ("inner", "outer"):
                        assert expanded["diameter"][rep][target]+2e-7 >= restricted["diameter"][rep][target]
                        count += 1
                    old, new = restricted["posterior"][target]["interval"], expanded["posterior"][target]["interval"]
                    assert new[0] <= old[0]+2e-7 and new[1] >= old[1]-2e-7
                    count += 2
    return count


def main():
    if OUTPUT.exists() or ARCHIVE.exists():
        raise FileExistsError("Frozen support output exists")
    start = time.perf_counter()
    spec, parent, original, prior, headroom, old_archive, heldout, pilot, schedules = inputs()
    reps = {(h, f): matrices(original, f, pilot, schedules[str(h)], heldout)
            for h in spec["confirmation_horizons"] for f in spec["support_floors"]}
    archive, logs, rows = {}, [], []
    with np.load(old_archive, allow_pickle=False) as old:
        for case in headroom["cases"]:
            cid = case["id"]
            yp = old[f"case{cid}_pilot"]
            archive[f"case{cid}_pilot"] = yp
            for hi, horizon in enumerate(spec["confirmation_horizons"]):
                ri = next(c["index"] for c in prior["resource_cells"] if
                          c["horizon"] == horizon and c["future_noise"] == spec["future_noise_amplitude"])
                yf = old[f"case{cid}_future_{ri}"]
                archive[f"case{cid}_future_{hi}"] = yf
                for fi, floor in enumerate(spec["support_floors"]):
                    data_reps = reps[(horizon, floor)]
                    diameters = {rep: {} for rep in data_reps}
                    for rep, data in data_reps.items():
                        ap, ep = data["pilot"]; af, ef = data["future"]
                        for target in TARGETS:
                            problem = pair_with_distinct_noise(ap, ep, yp, af, ef,
                                spec["pilot_noise_amplitude"], spec["future_noise_amplitude"],
                                data["q"][target], data["f"][target])
                            solution = solve(problem, f"pair_{cid}_{hi}_{fi}_{rep}_{target}", archive, logs)
                            if solution is None:
                                raise RuntimeError("Supported pair unexpectedly infeasible")
                            primal, dual = solution
                            diameters[rep][target] = max(0., -primal-1e-8 if rep == "inner" else -dual+1e-8)
                    intervals = {}
                    data = data_reps["outer"]
                    ap, ep = data["pilot"]; af, ef = data["future"]
                    for target in TARGETS:
                        values = []
                        for lower in (True, False):
                            problem = posterior_with_distinct_noise(ap, ep, yp, af, ef, yf,
                                spec["pilot_noise_amplitude"], spec["future_noise_amplitude"],
                                data["q"][target], data["f"][target], lower)
                            solution = solve(problem, f"post_{cid}_{hi}_{fi}_{target}_{int(lower)}", archive, logs)
                            if solution is None:
                                raise RuntimeError("Supported posterior unexpectedly infeasible")
                            _, dual = solution
                            values.append(dual-1e-8 if lower else -dual+1e-8)
                        if target == "G20":
                            values = [max(0., values[0]), min(1., values[1])]
                        real = headroom["truth_targets"][case["truth"]][target]
                        assert values[0]-2e-7 <= real <= values[1]+2e-7
                        assert values[1]-values[0] <= diameters["outer"][target]+2e-7
                        assert diameters["inner"][target] <= diameters["outer"][target]+2e-7
                        intervals[target] = {"interval": values, "width": values[1]-values[0], "contains_truth": True}
                    scores = {rep: max(v[t]/spec["utility_limits"][t] for t in TARGETS)
                              for rep, v in diameters.items()}
                    if floor == original["rate_support"][0]:
                        old_row = next(r for r in prior["rows"] if r["case"] == cid and r["resource_index"] == ri)
                        for rep in data_reps:
                            for target in TARGETS:
                                assert abs(diameters[rep][target]-old_row["diameter"][rep][target]) < 2e-7
                        for target in TARGETS:
                            np.testing.assert_allclose(intervals[target]["interval"], old_row["posterior"][target]["interval"], atol=2e-7, rtol=0)
                    rows.append({"case": cid, "truth": case["truth"], "seed": case["seed"],
                        "horizon": horizon, "horizon_index": hi, "support_floor": floor, "floor_index": fi,
                        "diameter": diameters, "score": scores, "posterior": intervals})
            print(json.dumps({"completed_case": cid+1, "truth": case["truth"]}), flush=True)
    checks = monotonicity(rows)
    summaries = []
    for floor in spec["support_floors"]:
        by_truth = {}
        for truth in original["truths"]:
            long = [r for r in rows if r["truth"] == truth and r["support_floor"] == floor and r["horizon"] == 16.]
            short = [r for r in rows if r["truth"] == truth and r["support_floor"] == floor and r["horizon"] == 8.]
            restricted = [r for r in rows if r["truth"] == truth and r["support_floor"] == .02 and r["horizon"] == 16.]
            upper = float(np.mean([r["score"]["outer"] for r in long]))
            ratio = upper/np.mean([r["score"]["inner"] for r in short])
            retention = upper/np.mean([r["score"]["inner"] for r in restricted])
            by_truth[truth] = {"long_mean_inner_score": float(np.mean([r["score"]["inner"] for r in long])),
                "long_mean_outer_score": upper, "long_maximum_outer_score": max(r["score"]["outer"] for r in long),
                "long_upper_to_matched_short_lower_ratio": float(ratio),
                "long_upper_to_restricted_long_lower_ratio": float(retention),
                "long_mean_outer_diameters": {t: float(np.mean([r["diameter"]["outer"][t] for r in long])) for t in TARGETS},
                "long_mean_realized_widths": {t: float(np.mean([r["posterior"][t]["width"] for r in long])) for t in TARGETS}}
        passing = [t for t, v in by_truth.items() if v["long_upper_to_matched_short_lower_ratio"] <= spec["gate"]["headroom_ratio_max"]]
        longs = [r for r in rows if r["support_floor"] == floor and r["horizon"] == 16.]
        summaries.append({"support_floor": floor, "outer_cell_count": len(nested_grids(original, floor)[0])-1,
            "inner_positive_rate_count": len(nested_grids(original, floor)[1]), "by_truth": by_truth,
            "long_utility_pass_count": sum(r["score"]["outer"] <= 1. for r in longs),
            "resource_gate_pass": all(r["score"]["outer"] <= 1. for r in longs) and
                len(passing) >= 2 and "continuous_uniform" in passing,
            "retention_check_pass": all(v["long_upper_to_restricted_long_lower_ratio"] <= spec["gate"]["retention_ratio_max"] for v in by_truth.values())})
    np.savez_compressed(ARCHIVE, **archive)
    sources = [SPEC, Path(__file__), HERE/"target_resource_frontier.py", HERE/"target_loading_headroom_gate.py",
               HERE/"calibrated_multiload_gate.py", HERE/"constitutive_response_gate.py",
               HERE/"offgrid_curvature_envelope_gate.py", SPEC.parent/spec["parent_spec"]]
    result = {"scope": spec["scope"], "spec_sha256": digest(SPEC),
        "parent_result_sha256": digest(OUTPUT.parent/spec["parent_result"]),
        "source_hashes": {str(p.relative_to(HERE.parent)): digest(p) for p in sources},
        "witness_archive_sha256": digest(ARCHIVE), "row_count": len(rows), "solver_count": len(logs),
        "case_count": 16, "duration_seconds": time.perf_counter()-start,
        "support_expansion_monotonicity_checks": checks, "summaries": summaries,
        "zero_floor_resource_gate_pass": summaries[-1]["resource_gate_pass"],
        "zero_floor_retention_check_pass": summaries[-1]["retention_check_pass"],
        "rows": rows, "lp_replay_logs": logs, "limitations": spec["limitations"]}
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summaries": summaries, "solver_count": len(logs),
        "duration_seconds": result["duration_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
