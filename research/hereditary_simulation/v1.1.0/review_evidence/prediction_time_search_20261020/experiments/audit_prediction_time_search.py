"""Replay final search evidence, including finite physical pairs at high precision."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import mpmath as mp
import numpy as np

from prediction_time_search import Information, representation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/prediction_time_search_20261020/run_05"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result = json.loads((OUT/"result.json").read_text(encoding="utf-8"))
    receipt = json.loads((OUT/"protocol_receipt.json").read_text(encoding="utf-8"))
    assert sha(OUT/"protocol_receipt.json") == result["protocol_receipt_sha256"]
    assert all(sha(ROOT/p) == h for p, h in receipt["bindings"].items())
    assert sha(OUT/"lp_witnesses.npz") == result["lp_witnesses_sha256"]
    spec = json.loads((ROOT/"experiments/specs/prediction_time_search_20261020.json").read_text())
    mp.mp.dps = 80
    def lap(order, rate):
        if order == 0:
            return mp.mpf(1)
        if rate == 0:
            return 1/mp.mpf(order)
        return mp.gammainc(order, 0, rate)/rate**order
    proposed = spec["algorithms"][-1]
    pair_checks = acceptance_checks = 0
    max_pair_relative_error = 0.
    for row in result["rows"]:
        witness = row["strict_witness"]
        nodes, masses = np.array(witness["nodes"]), np.array(witness["weighted_masses"])
        actual = np.exp(-np.asarray(spec["observation_delays_from_anchor"])[:, None]*nodes)@masses
        assert np.max(np.abs(actual-row["centers"])) < spec["observation_half_width"]-witness["margin"]+1e-10
        for request in row["algorithms"][proposed]["rows"]:
            if request["route"] == "physical_pair":
                pair = request["witness"]
                lam = mp.mpf(str(pair["rate"]))
                alpha = mp.mpf(str(pair["added_weighted_mass"]))
                # Independent incomplete-gamma implementation, not log_laplace.
                value = alpha*mp.exp(-mp.mpf(str(pair["excess_wait"]))*lam)*lap(row["gamma"], lam)/lap(row["beta"], lam)
                discrepancy = abs(float(value)/pair["target_increment"]-1)
                max_pair_relative_error = max(max_pair_relative_error, discrepancy)
                assert discrepancy < 1e-10
                assert float(value) > 2*spec["absolute_error_tolerance"]+2*spec["numerical_margin"]
                assert request["wait"] <= pair["through_wait"]
                increments = float(alpha)*np.exp(-np.array(spec["observation_delays_from_anchor"])*float(lam))
                assert np.all(np.abs(actual+increments-row["centers"]) < spec["observation_half_width"])
                assert np.isfinite(pair["log_added_physical_mass"])
                pair_checks += 1
        for algorithm in spec["algorithms"]:
            for request in row["algorithms"][algorithm]["rows"]:
                if request["state"] != "sufficient":
                    continue
                z = request["joint_prediction"]
                truth = row["truth_targets"][str(request["wait"])]
                # The endpoint gains preserve observations but change the
                # target, so test them as well as the nominal generating pair.
                for scale in [1., 1/(1+row["rho"]), 1/(1-row["rho"])]:
                    q = truth*scale
                    assert request["outer"][0]-1e-7 <= q <= request["outer"][1]+1e-7
                    assert abs(z-q) <= spec["absolute_error_tolerance"]+1e-7
                    assert abs(z-q)/q <= spec["relative_error_tolerance"]+1e-7
                    acceptance_checks += 1
    lp_archive = np.load(OUT/"lp_witnesses.npz")
    scenarios = {r["scenario"]: r for r in result["rows"]}
    replays = 0
    max_replay = 0.
    representations = {}
    for item in result["lp_logs"]:
        m = re.fullmatch(r"(.+_p\d+_g\d+)_w([\d.]+)_c(\d+)_(ol|oh|il|ih)", item["label"])
        assert m, item["label"]
        name, wait, cells, endpoint = m.group(1), float(m.group(2)), int(m.group(3)), m.group(4)
        row = scenarios[name]
        key = (name, wait, cells)
        if key not in representations:
            info = Information(row["beta"], spec["reference_dead_time"],
                tuple(spec["observation_delays_from_anchor"]), tuple(row["centers"]),
                (spec["observation_half_width"],)*len(row["centers"]))
            representations[key] = representation(info, row["gamma"], wait, cells)
        rep = representations[key]
        a = rep["outer_a"] if endpoint.startswith("o") else rep["inner_a"]
        c = rep["q_low"] if endpoint == "ol" else -rep["q_high"] if endpoint == "oh" else rep["inner_q"]*(1 if endpoint == "il" else -1)
        b = np.r_[np.array(row["centers"])+spec["observation_half_width"],
                  -np.array(row["centers"])+spec["observation_half_width"]]
        x, y, z = [lp_archive[item["label"]+"_"+s] for s in ("x", "y", "z")]
        residuals = [max(0., float(np.max(a@x-b)), float(-np.min(x))),
                     float(np.max(np.abs(c-a.T@y-z))),
                     max(0., float(np.max(y)), float(-np.min(z))),
                     float(abs(c@x-b@y))]
        assert max(residuals) <= spec["numerical_margin"]
        assert abs(float(c@x)-item["objective"]) < 1e-8
        max_replay = max(max_replay, *residuals)
        replays += 1
    assert not result["opposite_decisive_labels"]
    for other in spec["algorithms"][:-1]:
        for row in result["rows"]:
            for a, b in zip(row["algorithms"][proposed]["rows"], row["algorithms"][other]["rows"]):
                assert not (b["state"] != "unresolved" and a["state"] == "unresolved"), (row["scenario"], a, b)
    # Save exact executable snapshots for this final run; earlier pilots are
    # retained as exploratory records, not claimed as current-source replays.
    for rel in receipt["bindings"]:
        destination = OUT/"source_snapshot"/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, destination)
    audit = {"schema": "P4-prediction-time-search-audit-v1", "all_checks_pass": True,
        "result_sha256": sha(OUT/"result.json"), "same_developer_audit": True,
        "independent_external_review": False, "high_precision_digits": 80,
        "physical_pair_request_checks": pair_checks,
        "maximum_pair_increment_relative_error": max_pair_relative_error,
        "accepted_joint_estimate_checks_including_input_endpoints": acceptance_checks,
        "checked_lp_replays": replays, "maximum_lp_replay_residual": max_replay,
        "opposite_decisive_labels": 0, "lost_resolved_decisions": 0,
        "candidate_requests": len(result["rows"])*len(spec["target_delays"]),
        "distinct_information_sets": 12, "distinct_target_profile_scenarios": 24,
        "independent_synthetic_generators": 4,
        "actually_submitted": False, "directed_rounding_certificate": False}
    (OUT/"audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
