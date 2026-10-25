"""Local research CLI: compute target-delay decisions from a declared request."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
from prediction_time_search import Information, enclose, search, strict_witness
from prediction_time_transport import adaptive_search
from prediction_time_curvature import curvature_search
from prediction_time_dual import enclose_dual
from prediction_time_polish import polish_unresolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["v1", "transport", "curvature", "transport-dual"], default="transport-dual")
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if args.output.exists():
        raise FileExistsError("Choose a new output name; research records are not overwritten")
    declaration = request["information"]
    for key in ("delays", "centers", "bands"):
        declaration[key] = tuple(declaration[key])
    info = Information(**declaration)
    gamma = request["target_gamma"]
    archive, logs = {}, []
    def oracle(wait, cells):
        return enclose(info, gamma, wait, cells, f"w{wait}_c{cells}", archive, logs)
    algorithm = {"v1": search, "transport": adaptive_search, "curvature": curvature_search,
                 "transport-dual":adaptive_search}[args.method]
    feasible = strict_witness(info)
    output = algorithm(info, gamma, request["target_delays"], request["absolute_error"],
                       request["relative_error"], request.get("input_rho", 0.),
                       request.get("mesh_levels", [128, 512, 2048]), oracle, feasible)
    if args.method == "transport-dual":
        def dual_oracle(wait,cells):
            return enclose_dual(info,gamma,wait,cells,f"dual_w{wait}_c{cells}",archive,logs,
                                feasible=feasible)
        output = polish_unresolved(info,gamma,output,request["absolute_error"],
            request["relative_error"],request.get("input_rho",0.),
            request.get("mesh_levels",[128,512,2048]),dual_oracle,feasible)
    output["method"] = args.method
    output["scope"] = "Declared polynomial positive-spectral model; finite candidate list; numerical checks only"
    output["lp_logs"] = logs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist())+"\n", encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".npz"), **archive)
    print(str(args.output))


if __name__ == "__main__":
    main()
