"""Compare bracket transport to the frozen v1 method at two candidate densities."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import scipy
import mpmath

from observation_weighted_prediction import integrate_truth
from prediction_time_search import (Information, EnclosureFailure, enclose, log_laplace,
                                    search, strict_witness)
from prediction_time_transport import adaptive_search
from prediction_time_curvature import curvature_search

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT/"experiments/specs/prediction_time_transport_20261020.json"
OUT = ROOT/"results/prediction_time_transport_20261020/run_02"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    def native(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=native)+"\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/"protocol_receipt.json").exists():
        raise FileExistsError("Do not overwrite a frozen transport run")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    truth_path = SPEC.parent/spec["truth_spec"]
    bindings = [SPEC, Path(__file__), ROOT/"experiments/prediction_time_transport.py",
                ROOT/"experiments/prediction_time_curvature.py",
                ROOT/"experiments/prediction_time_search.py",
                ROOT/"experiments/observation_weighted_prediction.py", truth_path]
    put(OUT/"protocol_receipt.json", {"bindings": {p.relative_to(ROOT).as_posix(): sha(p) for p in bindings},
        "scope": spec["scope"], "frozen_before_execution": True,
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "mpmath": mpmath.__version__}})
    truths = json.loads(truth_path.read_text(encoding="utf-8"))["truths"]
    grids = {str(n): sorted(set(spec["always_include_delays"]+[round(spec["reference_dead_time"]+float(e), 12)
             for e in np.geomspace(*spec["excess_range"], n)])) for n in spec["geometric_grid_sizes"]}
    archive, logs, failures, rows, truthchecks = {}, [], [], [], []
    start = time.perf_counter()
    for truth in truths:
        for p in spec["reference_orders"]:
            beta, dead = p+1., spec["reference_dead_time"]
            delays = tuple(spec["observation_delays_from_anchor"])
            centers = tuple(integrate_truth(truth, lambda x: float(np.exp(-(dead+d)*x+log_laplace(x, beta)))) for d in delays)
            info = Information(beta, dead, delays, centers, (spec["observation_half_width"],)*len(delays))
            feasible = strict_witness(info)
            assert feasible is not None
            for gamma in spec["target_orders_gamma"]:
                name = f"{truth['name']}_p{p}_g{gamma}"
                cache, errors = {}, {}
                def oracle(wait, cells):
                    key = (wait, cells)
                    if key in errors:
                        e = errors[key]
                        raise EnclosureFailure(e["error"], e["programs"], e["solver_calls"], e["cells"])
                    if key not in cache:
                        try:
                            cache[key] = enclose(info, gamma, wait, cells, f"{name}_w{wait}_c{cells}", archive, logs, spec["numerical_margin"])
                        except RuntimeError as error:
                            errors[key] = {"error": str(error), "programs": getattr(error, "programs", 0),
                                "solver_calls": getattr(error, "solver_calls", 0), "cells": getattr(error, "cells", 0)}
                            failures.append({"scenario": name, "wait": wait, "level": cells, **errors[key]})
                            raise
                    return cache[key]
                for density, waits in grids.items():
                    actual = {str(w): integrate_truth(truth, lambda x: float(np.exp(-w*x+log_laplace(x, gamma)))) for w in waits}
                    for rho in spec["common_input_relative_envelopes"]:
                        variants = {}
                        for algorithm in spec["algorithms"]:
                            args = (info, gamma, waits, spec["absolute_error_tolerance"], spec["relative_error_tolerance"],
                                    rho, spec["mesh_levels"], oracle, feasible)
                            variants[algorithm] = (search(*args) if algorithm.startswith("v1_") else
                                curvature_search(*args) if "curvature" in algorithm else
                                adaptive_search(*args, use_secant="secant" in algorithm))
                            for request in variants[algorithm]["rows"]:
                                if request["state"] == "sufficient":
                                    q = actual[str(request["wait"])]
                                    for scale in (1., 1/(1+rho), 1/(1-rho)):
                                        value, z = q*scale, request["joint_prediction"]
                                        truthchecks.append({"scenario": name, "density": density, "rho": rho,
                                            "algorithm": algorithm, "wait": request["wait"], "scale": scale,
                                            "enclosed": request["outer"][0]-1e-7 <= value <= request["outer"][1]+1e-7,
                                            "absolute_error": abs(z-value), "relative_error": abs(z-value)/value})
                        rows.append({"scenario": name, "truth": truth["name"], "beta": beta, "gamma": gamma,
                                     "rho": rho, "density": density, "candidate_count": len(waits),
                                     "centers": centers, "strict_witness": feasible, "algorithms": variants})
                print(json.dumps({"scenario": name, "unique_checked_LPs": len(logs), "failed_brackets": len(failures)}), flush=True)
    summary, differences = {}, []
    for density in grids:
        summary[density] = {}
        for algorithm in spec["algorithms"]:
            results = [row["algorithms"][algorithm] for row in rows if row["density"] == density]
            requests = [item for result in results for item in result["rows"]]
            summary[density][algorithm] = {
                "lp_calls": sum(x["lp_calls"] for x in results),
                "solver_calls": sum(x["solver_calls"] for x in results),
                "cell_variables": sum(x["cell_variables"] for x in results),
                "states": {s: sum(x["state"] == s for x in requests) for s in ("unbounded", "insufficient", "sufficient", "unresolved")},
                "routes": {r: sum(x["route"] == r for x in requests) for r in sorted({x["route"] for x in requests})},
                "whole_time_cells": sum(len(x.get("time_cells", [])) for x in results)}
    for row in rows:
        base = row["algorithms"][spec["primary_comparator"]]["rows"]
        for name in spec["algorithms"][1:]:
            for a, b in zip(base, row["algorithms"][name]["rows"]):
                if a["state"] != b["state"]:
                    differences.append({"scenario": row["scenario"], "density": row["density"], "rho": row["rho"],
                        "algorithm": name, "wait": a["wait"], "baseline": a["state"], "adaptive": b["state"]})
    np.savez_compressed(OUT/"lp_witnesses.npz", **archive)
    value = {"schema": "P4-time-transport-result-v1", "spec_sha256": sha(SPEC),
        "protocol_receipt_sha256": sha(OUT/"protocol_receipt.json"), "grids": grids,
        "rows": rows, "summary": summary, "differences": differences,
        "accepted_prediction_checks": truthchecks, "numerical_failures": failures, "lp_logs": logs,
        "lp_witnesses_sha256": sha(OUT/"lp_witnesses.npz"), "unique_checked_LPs": len(logs),
        "elapsed_seconds_with_shared_numerical_cache": time.perf_counter()-start,
        "limits": spec["limitations"]}
    put(OUT/"result.json", value)
    print(json.dumps({"summary": summary, "differences": len(differences),
                      "elapsed_seconds": value["elapsed_seconds_with_shared_numerical_cache"]}), flush=True)


if __name__ == "__main__":
    main()
