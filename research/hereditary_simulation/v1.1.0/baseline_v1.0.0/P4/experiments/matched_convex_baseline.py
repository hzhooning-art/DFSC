"""Matched continuum baseline screen; elementary convexity is prior art."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from target_loading_headroom_gate import TARGETS, solve
from target_support_sensitivity import zero_safe_envelope, nested_grids, matrices, digest
from ultraslow_calibrated_validation import inputs as parent_inputs

HERE = Path(__file__).resolve().parent
SPEC = HERE / "specs" / "matched_convex_baseline_20261011.json"
OUTPUT = HERE.parent / "results" / "p4_matched_convex_baseline_20261011.json"
ARCHIVE = OUTPUT.with_suffix(".npz")
METHODS = ("classical_sign_convex", "finite_grid_diagnostic")


def inputs():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    parent_path = OUTPUT.parent / spec["parent_result"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    assert digest(parent_path.with_suffix(".npz")) == parent["witness_archive_sha256"]
    for name, expected in parent["source_hashes"].items():
        assert digest(HERE.parent / name) == expected, name
    pspec, original, heldout, pilot, schedules = parent_inputs()
    assert spec["rate_support"] == pspec["rate_support"]
    assert spec["utility_limits"] == pspec["utility_limits"]
    return spec, parent, pspec, original, heldout, pilot, schedules


def sign_errors(times, segments, edges):
    """For signed histories, f is secant minus positive remainder plus negative."""
    positive = [s for s in segments if s[2] > 0]
    negative = [s for s in segments if s[2] < 0]
    def error(parts):
        # Empty histories have exactly zero response, without padding.
        e = zero_safe_envelope(times, parts, edges) if parts else np.zeros((len(times), len(edges)-1))
        return np.column_stack((np.zeros(len(times)), np.repeat(e, 2, axis=1)))
    return error(positive), error(negative)


def representations(original, heldout, pilot, future):
    old = matrices(original, 0., pilot, future, heldout)
    edges, _ = nested_grids(original, 0.)
    def make_symmetric(data):
        ap, ep = data["pilot"]; af, ef = data["future"]
        return {"a": np.vstack((ap, af)), "down": np.vstack((ep, ef)),
                "up": np.vstack((ep, ef)), "q": data["q"], "qdown": data["f"], "qup": data["f"]}
    current = make_symmetric(old["outer"])
    point = make_symmetric(old["inner"])
    down, up = [], []
    for times, segments in pilot + future:
        d, u = sign_errors(times, segments, edges)
        down.append(d); up.append(u)
    stress_down, stress_up = sign_errors(np.array([20.]), heldout, edges)
    classic = {"a": current["a"], "down": np.vstack(down), "up": np.vstack(up),
               "q": current["q"],
               "qdown": {"G20": current["qdown"]["G20"], "heldout_sigma20": stress_down[0]},
               "qup": {"G20": np.zeros_like(current["q"]["G20"]), "heldout_sigma20": stress_up[0]}}
    # Only roundoff allowances differ when the positive and negative sums are split.
    assert np.max(classic["down"]-current["down"]) < 1e-12
    assert np.max(classic["up"]-current["up"]) < 1e-12
    return {"current_symmetric": current, "classical_sign_convex": classic, "finite_grid_diagnostic": point}


def problem(data, y, noise_bound, target, lower):
    q = data["q"][target]
    c = q-data["qdown"][target] if lower else -(q+data["qup"][target])
    a = np.vstack((data["a"]-data["down"], -data["a"]-data["up"]))
    return c, a, np.r_[y+noise_bound, -y+noise_bound], np.ones((1, len(q))), np.ones(1)


def intervals(data, y, bound, prefix, archive, logs, infeasible):
    values = {}
    for target in TARGETS:
        endpoints = []
        for lower in (True, False):
            label = f"{prefix}__{target}__{int(lower)}"
            answer = solve(problem(data, y, bound, target, lower), label, archive, logs)
            if answer is None:
                infeasible.append(label); endpoints.append(None)
            else:
                dual = answer[1]
                endpoints.append(dual-1e-8 if lower else -dual+1e-8)
        if None in endpoints:
            values[target] = None
        else:
            if target == "G20": endpoints = [max(0., endpoints[0]), min(1., endpoints[1])]
            assert endpoints[1] >= endpoints[0]-2e-7
            values[target] = {"interval": endpoints, "width": endpoints[1]-endpoints[0]}
    return values


def summarize(spec, parent, rows):
    summary = {}
    qualified = []
    for truth in parent["summaries"]:
        group = [r for r in rows if r["truth"] == truth and r["horizon"] == 16.]
        valid = [r for r in group if r["calibration_valid"]]
        methods = {}
        for method in ("current_symmetric",) + METHODS:
            covered = sum(all(r["methods"][method][t] is not None and r["methods"][method][t]["contains_truth"] for t in TARGETS) for r in group)
            valid_covered = sum(all(r["methods"][method][t] is not None and r["methods"][method][t]["contains_truth"] for t in TARGETS) for r in valid)
            means = {t: float(np.mean([r["methods"][method][t]["width"] for r in valid if r["methods"][method][t] is not None])) for t in TARGETS}
            methods[method] = {"unconditional_joint_covered": covered, "trial_count": len(group),
                "valid_joint_covered": valid_covered, "valid_count": len(valid), "valid_mean_width": means,
                "infeasible_case_count": sum(any(r["methods"][method][t] is None for t in TARGETS) for r in group)}
        ratio = {t: methods["current_symmetric"]["valid_mean_width"][t] / methods["classical_sign_convex"]["valid_mean_width"][t] for t in TARGETS}
        if all(ratio[t] <= spec["gate"]["current_mean_width_ratio_max"] for t in TARGETS): qualified.append(truth)
        summary[truth] = {"methods": methods, "current_to_classical_width_ratio": ratio,
            "classical_width_reduction_percent": {t: 100*(1-1/ratio[t]) for t in TARGETS}}
    continuous = {t for t, d in inputs()[2]["truths"].items() if d["kind"] != "discrete"}
    coverage = all(v["methods"][m]["valid_joint_covered"] == v["methods"][m]["valid_count"] for v in summary.values() for m in ("current_symmetric", "classical_sign_convex"))
    passed = coverage and len(qualified) >= spec["gate"]["minimum_truth_groups"] and bool(continuous.intersection(qualified))
    return summary, passed


def main():
    if OUTPUT.exists() or ARCHIVE.exists(): raise FileExistsError("Frozen comparison exists")
    started = time.perf_counter()
    spec, parent, pspec, original, heldout, pilot, schedules = inputs()
    reps = {h: representations(original, heldout, pilot, schedules[str(h)]) for h in pspec["confirmation_horizons"]}
    archive, logs, rows, infeasible, replays = {}, [], [], [], []
    nested_checks = 0
    parent_path = OUTPUT.parent / spec["parent_result"]
    with np.load(parent_path.with_suffix(".npz"), allow_pickle=False) as saved:
        for old in parent["rows"]:
            cid, hi, h = old["case"], old["horizon_index"], old["horizon"]
            y = np.r_[saved[f"case{cid}_pilot"], saved[f"case{cid}_future_{hi}"]]
            methods = {"current_symmetric": old["posterior"]}
            for method in METHODS:
                values = intervals(reps[h][method], y, old["calibrated_bound"], f"{cid}__{hi}__{method}", archive, logs, infeasible)
                for t in TARGETS:
                    if values[t] is not None:
                        lo, upper = values[t]["interval"]
                        real = parent["truth_targets"][old["truth"]][t]
                        values[t]["contains_truth"] = bool(lo-2e-7 <= real <= upper+2e-7)
                        if method == "classical_sign_convex":
                            current = old["posterior"][t]["interval"]
                            assert lo >= current[0]-2e-7 and upper <= current[1]+2e-7
                            nested_checks += 2
                            if old["calibration_valid"]: assert values[t]["contains_truth"]
                methods[method] = values
            if old["seed"] < 4 and h == 16.:
                replay = intervals(reps[h]["current_symmetric"], y, old["calibrated_bound"], f"{cid}__{hi}__current_symmetric", archive, logs, infeasible)
                for t in TARGETS:
                    np.testing.assert_allclose(replay[t]["interval"], old["posterior"][t]["interval"], atol=2e-7, rtol=0)
                replays.append({"case": cid, "horizon_index": hi, "posterior": replay})
            rows.append({k: old[k] for k in ("case", "truth", "truth_index", "seed", "horizon", "horizon_index", "calibrated_bound", "calibration_valid")} | {"methods": methods})
            if hi == 1 and (old["seed"]+1) % 16 == 0:
                print(json.dumps({"truth": old["truth"], "completed": old["seed"]+1}), flush=True)
    summary, passed = summarize(spec, parent, rows)
    np.savez_compressed(ARCHIVE, **archive)
    sources = [SPEC, Path(__file__), HERE/"target_loading_headroom_gate.py", HERE/"target_support_sensitivity.py", HERE/"ultraslow_calibrated_validation.py"]
    result = {"spec_sha256": digest(SPEC), "parent_result_sha256": digest(parent_path),
        "source_hashes": {str(p.relative_to(HERE.parent)): digest(p) for p in sources},
        "witness_archive_sha256": digest(ARCHIVE), "case_count": len(parent["cases"]), "row_count": len(rows),
        "solver_count": len(logs), "infeasible_lp_labels": infeasible, "nested_endpoint_checks": nested_checks,
        "current_replay_case_count": len(replays), "summaries": summary, "technical_increment_gate_pass": passed,
        "rows": rows, "current_replays": replays, "lp_replay_logs": logs,
        "duration_seconds": time.perf_counter()-started, "limitations": spec["limitations"]}
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summaries": summary, "technical_increment_gate_pass": passed, "solver_count": len(logs),
        "infeasible_count": len(infeasible), "duration_seconds": result["duration_seconds"]}), flush=True)


if __name__ == "__main__": main()
