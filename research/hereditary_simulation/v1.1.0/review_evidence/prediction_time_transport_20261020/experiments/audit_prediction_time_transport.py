"""Replay time-transport LPs, witness containments and off-grid time cells."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import numpy as np

from observation_weighted_prediction import integrate_truth
from prediction_time_search import Information, log_laplace, ratio, representation
from prediction_time_transport import transport_bracket, transferred_decision
from prediction_time_curvature import curvature_bound, joint_cell_test

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/prediction_time_transport_20261020/run_02"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result = json.loads((OUT/"result.json").read_text(encoding="utf-8"))
    receipt = json.loads((OUT/"protocol_receipt.json").read_text(encoding="utf-8"))
    assert sha(OUT/"protocol_receipt.json") == result["protocol_receipt_sha256"]
    assert all(sha(ROOT/p) == h for p, h in receipt["bindings"].items())
    assert sha(OUT/"lp_witnesses.npz") == result["lp_witnesses_sha256"]
    spec = json.loads((ROOT/"experiments/specs/prediction_time_transport_20261020.json").read_text())
    truths = {t["name"]: t for t in json.loads((ROOT/"experiments/specs"/spec["truth_spec"]).read_text())["truths"]}
    arrays = np.load(OUT/"lp_witnesses.npz")
    logmap = {r["label"]: r for r in result["lp_logs"]}
    scenarios = {r["scenario"]: r for r in result["rows"]}
    reps = {}
    replayed, max_residual = 0, 0.
    for item in result["lp_logs"]:
        m = re.fullmatch(r"(.+_p\d+_g\d+)_w([\d.]+)_c(\d+)_(ol|oh|il|ih)", item["label"])
        assert m, item["label"]
        name, wait, level, endpoint = m.group(1), float(m.group(2)), int(m.group(3)), m.group(4)
        row = scenarios[name]
        key = name, wait, level
        if key not in reps:
            info = Information(row["beta"], spec["reference_dead_time"], tuple(spec["observation_delays_from_anchor"]),
                               tuple(row["centers"]), (spec["observation_half_width"],)*len(row["centers"]))
            reps[key] = representation(info, row["gamma"], wait, level)
        rep = reps[key]
        a = rep["outer_a"] if endpoint.startswith("o") else rep["inner_a"]
        c = rep["q_low"] if endpoint == "ol" else -rep["q_high"] if endpoint == "oh" else rep["inner_q"]*(1 if endpoint == "il" else -1)
        rhs = np.r_[np.array(row["centers"])+spec["observation_half_width"], -np.array(row["centers"])+spec["observation_half_width"]]
        x, y, z = [arrays[item["label"]+"_"+s] for s in ("x", "y", "z")]
        residual = max(0., float(np.max(a@x-rhs)), float(-np.min(x)), float(np.max(np.abs(c-a.T@y-z))),
                       float(np.max(y)), float(-np.min(z)), float(abs(c@x-rhs@y)))
        assert residual <= spec["numerical_margin"]
        assert abs(float(c@x)-item["objective"]) < 1e-8
        replayed += 1
        max_residual = max(max_residual, residual)
    acceptance_checks = 0
    for check in result["accepted_prediction_checks"]:
        assert check["enclosed"]
        assert check["absolute_error"] <= spec["absolute_error_tolerance"]+1e-7
        assert check["relative_error"] <= spec["relative_error_tolerance"]+1e-7
        acceptance_checks += 1
    opposite = [d for d in result["differences"] if "unresolved" not in (d["baseline"], d["adaptive"])]
    lost = [d for d in result["differences"] if d["baseline"] != "unresolved" and d["adaptive"] == "unresolved"]
    assert not opposite and not lost
    def endpoints(name, time, level):
        stem = f"{name}_w{time}_c{level}_"
        v = lambda suffix: logmap[stem+suffix]["objective"]
        pad = spec["numerical_margin"]
        return {"outer": [max(0., v("ol")-pad), -v("oh")+pad],
                "inner": [v("il")+pad, -v("ih")-pad]}
    cell_checks = 0
    varying_cells = 0
    statuses = {"sufficient": 0, "insufficient": 0}
    fractions = [0., (3-np.sqrt(3))/6, .5, (3+np.sqrt(3))/6, 1.]
    for row in result["rows"]:
        info = Information(row["beta"], spec["reference_dead_time"], tuple(spec["observation_delays_from_anchor"]),
                           tuple(row["centers"]), (spec["observation_half_width"],)*len(row["centers"]))
        for algorithm in spec["algorithms"][1:]:
            for cell in row["algorithms"][algorithm]["time_cells"]:
                s, t = cell["start"], cell["end"]
                left = endpoints(row["scenario"], s, cell["left_level"])
                right = endpoints(row["scenario"], t, cell["right_level"])
                statuses[cell["state"]] += 1
                m = curvature_bound(info, row["gamma"], s) if "curvature" in algorithm else None
                if cell.get("varying_prediction"):
                    assert joint_cell_test(left, right, s, t, m, spec["absolute_error_tolerance"],
                                            spec["relative_error_tolerance"], row["rho"])
                    varying_cells += 1
                # Rebuild the TWO finite physical endpoint spectra. Their
                # target ordering bounds the transported inner overlap even
                # when intermediate times were never optimized.
                leftstem = f"{row['scenario']}_w{s}_c{cell['left_level']}_il"
                rightstem = f"{row['scenario']}_w{t}_c{cell['right_level']}_ih"
                lr = reps[row["scenario"], s, cell["left_level"]]
                rr = reps[row["scenario"], t, cell["right_level"]]
                for f in fractions:
                    at = s+(t-s)*f
                    actual = integrate_truth(truths[row["truth"]], lambda x: float(np.exp(-at*x+log_laplace(x, row["gamma"]))))
                    bracket = transport_bracket(left, right, s, t, at if cell.get("varying_prediction") else None)
                    if cell.get("varying_prediction"):
                        w = (t-at)/(t-s)
                        lo = w*left["outer"][0]+(1-w)*right["outer"][0]-m*(at-s)*(t-at)/2
                        bracket["outer"][0] = max(bracket["outer"][0], lo, 0.)
                    value = transferred_decision(bracket, spec["absolute_error_tolerance"], spec["relative_error_tolerance"], row["rho"])
                    assert value["state"] == cell["state"]
                    ql = float(arrays[leftstem+"_x"]@ratio(lr["nodes"], row["beta"], row["gamma"], at-info.dead_time))
                    qr = float(arrays[rightstem+"_x"]@ratio(rr["nodes"], row["beta"], row["gamma"], at-info.dead_time))
                    assert ql <= left["inner"][0]+1e-6 and qr >= right["inner"][1]-1e-6
                    if value["state"] == "sufficient":
                        for scale in [1., 1/(1+row["rho"]), 1/(1-row["rho"])]:
                            q, z = actual*scale, value["joint_prediction"]
                            assert abs(q-z) <= spec["absolute_error_tolerance"]+1e-7
                            assert abs(q-z)/q <= spec["relative_error_tolerance"]+1e-7
                    cell_checks += 1
    for rel in receipt["bindings"]:
        destination = OUT/"source_snapshot"/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, destination)
    value = {"schema": "P4-time-transport-audit-v1", "all_checks_pass": True,
        "result_sha256": sha(OUT/"result.json"), "checked_lp_replays": replayed,
        "maximum_replay_residual": max_residual, "accepted_joint_estimate_checks": acceptance_checks,
        "off_grid_and_endpoint_time_cell_checks": cell_checks,
        "whole_time_cell_records": statuses, "varying_predictor_cell_records": varying_cells,
        "opposite_decisive_labels": len(opposite), "lost_resolved_candidates": len(lost),
        "independent_observation_sets": 12, "synthetic_generating_spectra": 4,
        "same_developer_audit": True, "independent_external_review": False,
        "directed_rounding_certificate": False, "actually_submitted": False}
    (OUT/"audit.json").write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(value))


if __name__ == "__main__":
    main()
