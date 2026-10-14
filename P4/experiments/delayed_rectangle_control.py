"""Strong simple loading control; parent results remain frozen."""
import json
from pathlib import Path
import time
import numpy as np
from observation_weighted_prediction import digest, integrate_truth, solve
from matched_endpoint_design import target_kernel, ROOT

SPEC = ROOT/"experiments/specs/delayed_rectangle_control_20261011.json"
OUT = ROOT/"results/p4_delayed_rectangle_control_20261011.json"
ARCHIVE = OUT.with_suffix(".npz")


def rectangle_kernel(rate):
    rate = np.asarray(rate, dtype=float)
    if np.any(rate < 0):
        raise ValueError("Negative rate")
    safe = np.where(rate == 0, 1., rate)
    return np.where(rate == 0, 1., -3*np.expm1(-rate/3)/safe)


def ratio(rate, delay):
    return target_kernel(rate, delay)/rectangle_kernel(rate)


def representation(parent_spec, target, cells):
    edges = np.r_[0., np.geomspace(parent_spec["minimum_positive_rate"],
                                   parent_spec["tail_start"], cells-1), np.inf]
    nodes = edges[:-1]
    delays = np.asarray(parent_spec["observation_delays"])
    larger = np.exp(-delays[:, None]*edges[:-1])
    smaller = np.empty_like(larger)
    smaller[:, :-1] = np.exp(-delays[:, None]*edges[1:-1])
    smaller[:, -1] = (delays == 0).astype(float)
    delay = parent_spec["targets"][target]
    qlo = np.empty(cells)
    qhi = np.empty(cells)
    # (1-exp(-a*x))/(1-exp(-b*x)) decreases for a>b>0:
    # log derivative is a/(exp(a*x)-1)-b/(exp(b*x)-1)<0.
    # Multiplication by exp(-delay*x) preserves decreasing order.
    qlo[:-1] = ratio(edges[1:-1], delay)
    qhi[:-1] = ratio(edges[:-2], delay)
    qlo[-1] = 2/3 if delay == 0 else 0.
    qhi[-1] = ratio(edges[-2], delay)
    qlo = np.maximum(qlo-2e-12, 0.)
    qhi += 2e-12
    exact = np.exp(-delays[:, None]*nodes)
    return {"edges": edges, "nodes": nodes, "outer_a": np.vstack((smaller, -larger)),
            "inner_a": np.vstack((exact, -exact)), "q_low": qlo, "q_high": qhi,
            "inner_q": ratio(nodes, delay)}


def comparisons(rows, settings, spec):
    finest = [r for r in rows if r["cells"] == max(settings["cell_counts"])]
    reports = []
    for mode in settings["band_modes"]:
        for target in settings["targets"]:
            candidates = [r for r in finest if r["design"] == spec["new_design"]
                          and r["band_mode"] == mode and r["target"] == target]
            for base in spec["comparison_baselines"]:
                baseline = {r["truth"]:r for r in finest if r["design"] == base
                            and r["band_mode"] == mode and r["target"] == target}
                mi = float(np.mean([r["inner_width"] for r in candidates]))
                mo = float(np.mean([r["outer_width"] for r in candidates]))
                bi = float(np.mean([baseline[r["truth"]]["inner_width"] for r in candidates]))
                bo = float(np.mean([baseline[r["truth"]]["outer_width"] for r in candidates]))
                gain = [1-mo/bi, 1-mi/bo]
                reports.append({"band_mode":mode, "target":target, "baseline":base,
                                "rectangle_mean_width_bracket":[mi, mo],
                                "baseline_mean_width_bracket":[bi, bo],
                                "rectangle_gain_bracket":gain,
                                "rectangle_utility_cases":sum(r["outer_width"] <= settings["utility_width_limits"][target] for r in candidates),
                                "case_count":len(candidates),
                                "precision_pass":all(r["endpoint_bracket"] <= settings["endpoint_bracket_tolerance"] for r in candidates),
                                "gain_at_least_10_percent":gain[0] >= spec["required_mean_gain_lower"]})
    decisions = []
    for r in finest:
        limit = settings["utility_width_limits"][r["target"]]
        state = ("accept" if r["outer_width"] <= limit-2e-7 else
                 "reject" if r["inner_width"] > limit+2e-7 else "unresolved")
        decisions.append({"truth":r["truth"], "design":r["design"], "band_mode":r["band_mode"],
                          "target":r["target"], "width_bracket":[r["inner_width"], r["outer_width"]],
                          "limit":limit, "state":state})
    return reports, decisions


def main():
    if OUT.exists() or ARCHIVE.exists():
        raise FileExistsError("Frozen rectangle control already exists")
    start = time.perf_counter()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    parent_spec_path = SPEC.parent/spec["parent_spec"]
    settings = json.loads(parent_spec_path.read_text(encoding="utf-8"))
    parent_path = ROOT/"results/p4_matched_endpoint_design_20261011.json"
    assert digest(parent_path) == spec["parent_result_sha256"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    assert digest(parent_spec_path) == parent["spec_sha256"]
    truth_path = SPEC.parent/settings["parent_truth_spec"]
    truths = json.loads(truth_path.read_text(encoding="utf-8"))["truths"]
    rows = [dict(r, provenance="frozen_parent") for r in parent["rows"] if r["design"] in spec["retained_designs"]]
    logs, payload, data = [], {}, []
    reps = {(target,c):representation(settings,target,c) for target in settings["targets"] for c in settings["cell_counts"]}
    for info in truths:
        observations = np.array([integrate_truth(info, lambda rate: float(rectangle_kernel(rate))*np.exp(-d*rate))
                                 for d in settings["observation_delays"]])
        targets = {t:integrate_truth(info, lambda rate: float(target_kernel(rate,d))) for t,d in settings["targets"].items()}
        for mode in settings["band_modes"]:
            band = settings["absolute_response_band"] if mode == "absolute" else settings["relative_anchor_band"]*observations[0]
            data.append({"truth":info["name"], "band_mode":mode, "design":spec["new_design"],
                         "observations":observations.tolist(), "band":float(band)})
            rhs = np.r_[observations+band, -observations+band]
            for target in settings["targets"]:
                for cells in settings["cell_counts"]:
                    rep = reps[target,cells]
                    stem = f"{info['name']}_delayed_rectangle_{mode}_{target}_{cells}"
                    ol = solve(rep["q_low"],rep["outer_a"],rhs,stem+"_outer_low",payload,logs,settings["lp_replay_tolerance"])
                    oh = -solve(-rep["q_high"],rep["outer_a"],rhs,stem+"_outer_high",payload,logs,settings["lp_replay_tolerance"])
                    il = solve(rep["inner_q"],rep["inner_a"],rhs,stem+"_inner_low",payload,logs,settings["lp_replay_tolerance"])
                    ih = -solve(-rep["inner_q"],rep["inner_a"],rhs,stem+"_inner_high",payload,logs,settings["lp_replay_tolerance"])
                    assert ol-2e-7 <= il <= ih <= oh+2e-7
                    assert ol-2e-7 <= targets[target] <= oh+2e-7
                    rows.append({"truth":info["name"],"design":spec["new_design"],"band_mode":mode,
                                 "target":target,"cells":cells,"outer_interval":[ol,oh],"inner_interval":[il,ih],
                                 "inner_width":ih-il,"outer_width":oh-ol,"endpoint_bracket":max(il-ol,oh-ih),
                                 "truth_target":targets[target],"provenance":"new_rectangle_control"})
        print(json.dumps({"truth_completed":info["name"],"new_LPs":len(logs)}),flush=True)
    summaries,decisions = comparisons(rows,settings,spec)
    np.savez_compressed(ARCHIVE,**payload)
    result = {"schema":"P4-delayed-rectangle-control-result-v1","spec_sha256":digest(SPEC),
              "source_sha256":digest(__file__),"parent_result_sha256":digest(parent_path),
              "parent_spec_sha256":digest(parent_spec_path),"truth_spec_sha256":digest(truth_path),
              "dependencies_sha256":{name:digest(ROOT/"experiments"/name) for name in
                                     ("matched_endpoint_design.py","observation_weighted_prediction.py")},
              "archive_sha256":digest(ARCHIVE),"rows":rows,"data":data,"lp_logs":logs,
              "summaries":summaries,"decisions":decisions,"duration_seconds":time.perf_counter()-start,
              "all_internal_checks_pass":True,"limits":spec["limits"]}
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"new_LPs":len(logs),"new_rows":32,"summaries":summaries,
                      "duration_seconds":result["duration_seconds"]}),flush=True)


if __name__ == "__main__":
    main()

