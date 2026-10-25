"""Execute the frozen paired search-work study; retain all numerical failures."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from observation_weighted_prediction import integrate_truth
from prediction_time_search import (EnclosureFailure, Information, enclose,
                                    log_laplace, search, strict_witness)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT/"experiments/specs/prediction_time_search_20261020.json"
OUT = ROOT/"results/prediction_time_search_20261020/run_05"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    def native(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              allow_nan=False, default=native)+"\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = OUT/"protocol_receipt.json"
    if receipt.exists():
        raise FileExistsError("Preserve the frozen run; choose a new protocol for further experiments")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    truth_path = SPEC.parent/spec["truth_spec"]
    bindings = [SPEC, Path(__file__), ROOT/"experiments/prediction_time_search.py",
                ROOT/"experiments/observation_weighted_prediction.py", truth_path]
    put(receipt, {"schema": "P4-prediction-time-search-protocol-receipt-v1",
                  "bindings": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in bindings},
                  "scope": spec["scope"], "frozen_before_numerical_execution": True})
    truths = json.loads(truth_path.read_text(encoding="utf-8"))["truths"]
    rows, logs, archive, numerical_failures, check_rows = [], [], {}, [], []
    start = time.perf_counter()
    for truth in truths:
        for p in spec["reference_orders"]:
            beta, dead = p+1., spec["reference_dead_time"]
            delays = tuple(spec["observation_delays_from_anchor"])
            y = tuple(integrate_truth(truth, lambda x: float(np.exp(
                -(dead+d)*x+log_laplace(x, beta)))) for d in delays)
            info = Information(beta, dead, delays, y, (spec["observation_half_width"],)*len(y))
            feasible = strict_witness(info)
            if feasible is None:
                raise RuntimeError("Synthetic information unexpectedly lacks a strict physical witness")
            witness_record = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in feasible.items()}
            for gamma in spec["target_orders_gamma"]:
                cache, cache_errors = {}, {}
                name = f"{truth['name']}_p{p}_g{gamma}"
                def oracle(wait, cells):
                    key = (wait, cells)
                    if key in cache_errors:
                        e = cache_errors[key]
                        raise EnclosureFailure(e["error"], e["programs"], e["solver_calls"], e["cells"])
                    if key not in cache:
                        try:
                            cache[key] = enclose(info, gamma, wait, cells,
                                f"{name}_w{wait}_c{cells}", archive, logs, spec["numerical_margin"])
                        except RuntimeError as error:
                            cache_errors[key] = {"error": str(error),
                                "programs": getattr(error, "programs", 0),
                                "solver_calls": getattr(error, "solver_calls", 0),
                                "cells": getattr(error, "cells", 0)}
                            numerical_failures.append({"scenario": name, "wait": wait,
                                                       "cells": cells, "error": str(error)})
                            raise
                    return cache[key]
                for rho in spec["common_input_relative_envelopes"]:
                    variants = {}
                    for algorithm in spec["algorithms"]:
                        variants[algorithm] = search(info, gamma, spec["target_delays"],
                            spec["absolute_error_tolerance"], spec["relative_error_tolerance"], rho,
                            spec["mesh_levels"], oracle, feasible,
                            use_witness="rate_pair" in algorithm,
                            fixed_fine="fixed_fine" in algorithm,
                            margin=spec["numerical_margin"])
                    truth_targets = {str(w): integrate_truth(truth, lambda x: float(np.exp(
                        -w*x+log_laplace(x, gamma)))) for w in spec["target_delays"]}
                    for algorithm, outcome in variants.items():
                        for row in outcome["rows"]:
                            if row["state"] == "sufficient":
                                actual = truth_targets[str(row["wait"])]
                                l, u = row["outer"]
                                check_rows.append({"scenario": name, "rho": rho, "algorithm": algorithm,
                                    "wait": row["wait"], "truth_enclosed": l-1e-7 <= actual <= u+1e-7,
                                    "absolute_error": abs(row["absolute_prediction"]-actual),
                                    "relative_error": abs(row["relative_prediction"]-actual)/actual,
                                    "joint_absolute_error": abs(row["joint_prediction"]-actual),
                                    "joint_relative_error": abs(row["joint_prediction"]-actual)/actual})
                    rows.append({"scenario": name, "truth": truth["name"], "beta": beta,
                                 "gamma": gamma, "rho": rho, "centers": list(y),
                                 "strict_witness": witness_record, "truth_targets": truth_targets,
                                 "algorithms": variants})
                print(json.dumps({"scenario": name, "unique_checked_LPs": len(logs),
                                  "retained_failures": len(numerical_failures)}), flush=True)
    totals = {}
    for algorithm in spec["algorithms"]:
        items = [r["algorithms"][algorithm] for r in rows]
        candidates = [v for item in items for v in item["rows"]]
        totals[algorithm] = {"lp_calls": sum(v["lp_calls"] for v in items),
            "solver_calls": sum(v["solver_calls"] for v in items),
            "cell_variables": sum(v["cell_variables"] for v in items),
            "states": {state: sum(v["state"] == state for v in candidates)
                       for state in ("unbounded", "insufficient", "sufficient", "unresolved")},
            "routes": {route: sum(v["route"] == route for v in candidates)
                       for route in sorted({v["route"] for v in candidates})}}
    proposed = spec["algorithms"][-1]
    disagreements = []
    resolved_differences = []
    for row in rows:
        own = row["algorithms"][proposed]["rows"]
        for other in spec["algorithms"][:-1]:
            for a, b in zip(own, row["algorithms"][other]["rows"]):
                if a["state"] != b["state"]:
                    item = {"scenario": row["scenario"], "rho": row["rho"], "wait": a["wait"],
                            "comparator": other, "proposed": a["state"], "baseline": b["state"]}
                    (resolved_differences if "unresolved" in (a["state"], b["state"])
                     else disagreements).append(item)
    np.savez_compressed(OUT/"lp_witnesses.npz", **archive)
    result = {"schema": "P4-prediction-time-search-result-v1", "date": spec["date"],
              "protocol_receipt_sha256": sha(receipt), "spec_sha256": sha(SPEC),
              "rows": rows, "summary": totals, "opposite_decisive_labels": disagreements,
              "resolved_vs_unresolved_differences": resolved_differences,
              "accepted_prediction_checks": check_rows, "numerical_failures": numerical_failures,
              "lp_logs": logs, "unique_checked_LPs": len(logs),
              "lp_witnesses_sha256": sha(OUT/"lp_witnesses.npz"),
              "elapsed_seconds_including_shared_baselines": time.perf_counter()-start,
              "limits": spec["limitations"]}
    put(OUT/"result.json", result)
    print(json.dumps({"summary": totals, "opposite_decisive_labels": len(disagreements),
                      "resolved_vs_unresolved": len(resolved_differences),
                      "elapsed_seconds": result["elapsed_seconds_including_shared_baselines"]}), flush=True)


if __name__ == "__main__":
    main()
