"""Frozen P4 endpoint-interpolation outer LP with a rate-curvature bound.

This is a conditional set-membership comparison, not a novelty claim. Its
guarantee requires a nonnegative rate measure on the frozen compact support,
the exact strain history, known G(0), and a valid independent noise bound.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import gammainc

from calibrated_multiload_gate import basis
from offgrid_cell_envelope_gate import interval_lp
from offgrid_mesh_refinement_gate import design_data, prepare_observations


HERE = Path(__file__).resolve().parent
SPEC = HERE / "specs" / "offgrid_curvature_envelope_gate_20261010.json"
PARENT_SPEC = HERE / "specs" / "calibrated_multiload_gate_20261010.json"
PARENT_RESULT = HERE.parent / "results" / "p4_calibrated_multiload_gate_20261010.json"
PRIOR_OUTER = HERE.parent / "results" / "p4_offgrid_mesh_refinement_gate_20261010.json"
PRIOR_INNER = HERE.parent / "results" / "p4_offgrid_inner_outer_sandwich_20261010.json"
RESULT = HERE.parent / "results" / "p4_offgrid_curvature_envelope_gate_20261010.json"
TIMING = HERE.parent / "results" / "p4_offgrid_curvature_envelope_timing_20261010.json"


def u2_exponential_integral(low: np.ndarray, high: np.ndarray,
                            rate: np.ndarray) -> np.ndarray:
    """Integral_low^high u^2 exp(-rate*u) du for positive rates.

    Gamma(3)=2 and scipy's gammainc is regularized. A small nonnegative
    floating-point allowance is added by curvature_envelope, not here.
    """
    values = (2.0 / rate**3) * (gammainc(3.0, rate * high) -
                                 gammainc(3.0, rate * low))
    return np.maximum(values, 0.0)


def response_curvature_envelope(times: np.ndarray,
                                segments: list[tuple[float, float, float]],
                                edges: np.ndarray) -> np.ndarray:
    """Uniform absolute interpolation remainder for each rate cell.

    For a cell [a,b], |f(rate)-linear_endpoints(f)(rate)| is at most
    (b-a)^2/8 times sup_[a,b] |f''|. The segment integral's absolute second
    derivative is bounded at a, since u^2 exp(-rate*u) decreases in rate.
    """
    left = edges[:-1][None, :]
    widths = np.diff(edges)[None, :]
    bound = np.zeros((len(times), len(widths.ravel())))
    t = np.asarray(times)[:, None]
    for start, end, slope in segments:
        low = np.maximum(t - end, 0.0)
        high = np.maximum(t - start, 0.0)
        bound += abs(slope) * u2_exponential_integral(low, high, left)
    # The allowance dominates roundoff in the incomplete-gamma difference.
    return (widths**2 / 8.0) * (bound * (1.0 + 1e-10) + 1e-11)


def matrices_for_count(count: int, support: list[float], designs: dict,
                       heldout: list) -> tuple:
    edges = np.geomspace(*support, count + 1)
    endpoints = np.column_stack((edges[:-1], edges[1:])).reshape(-1)
    matrices = {}
    errors = {}
    for name, pieces in designs.items():
        matrices[name] = np.vstack([basis(times, segments, endpoints)
                                    for times, segments in pieces])
        cell_error = np.vstack([response_curvature_envelope(
            times, segments, edges) for times, segments in pieces])
        errors[name] = np.column_stack((np.zeros(len(cell_error)),
                                        np.repeat(cell_error, 2, axis=1)))
    q_g20 = np.r_[1.0, np.exp(-20.0 * endpoints)]
    f_g20 = (np.diff(edges)**2 / 8.0) * (
        400.0 * np.exp(-20.0 * edges[:-1]) * (1.0 + 1e-10) + 1e-11)
    q_stress = basis(np.array([20.0]), heldout, endpoints)[0]
    f_stress = response_curvature_envelope(np.array([20.0]), heldout, edges)[0]
    targets = {
        "G20": (q_g20, np.r_[0.0, np.repeat(f_g20, 2)]),
        "heldout_sigma20": (q_stress, np.r_[0.0, np.repeat(f_stress, 2)]),
    }
    return edges, endpoints, matrices, errors, targets


def interpolate_truth(parent: dict, edges: np.ndarray, truth: str) -> np.ndarray:
    weights = np.zeros(1 + 2 * (len(edges) - 1))
    if truth == "on_grid":
        info = parent["truths"]["on_grid"]
        old = parent["positive_grid_rates"]
        original_rates = np.geomspace(old["start"], old["stop"], old["count"])
        atoms = [(original_rates[int(j)], mass)
                 for j, mass in info["transient_grid_weights"].items()]
    else:
        info = parent["truths"]["off_grid"]
        atoms = zip(info["rates"], info["weights"])
    weights[0] = info["equilibrium"]
    for rate, mass in atoms:
        if not edges[0] <= rate <= edges[-1]:
            raise AssertionError("Truth rate outside the frozen support")
        j = min(np.searchsorted(edges, rate, side="right") - 1, len(edges) - 2)
        fraction_right = (rate - edges[j]) / (edges[j + 1] - edges[j])
        weights[1 + 2*j] += mass * (1.0 - fraction_right)
        weights[2 + 2*j] += mass * fraction_right
    if abs(weights.sum() - 1.0) > 1e-12 or weights.min() < -1e-12:
        raise AssertionError("Interpolated spectrum does not conserve mass")
    return weights


def main() -> None:
    raw_spec = SPEC.read_bytes()
    spec = json.loads(raw_spec)
    raw_parent = PARENT_SPEC.read_bytes()
    parent = json.loads(raw_parent)
    frozen = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    raw_outer = PRIOR_OUTER.read_bytes()
    old_outer = json.loads(raw_outer)
    raw_inner = PRIOR_INNER.read_bytes()
    old_inner = json.loads(raw_inner)
    parent_sha = hashlib.sha256(raw_parent).hexdigest()
    if parent_sha != frozen["spec_sha256"] or parent_sha != old_outer["parent_spec_sha256"]:
        raise AssertionError("Parent protocol differs from frozen results")
    if spec["parent_spec"] != PARENT_SPEC.name or spec["prior_outer_result"] != PRIOR_OUTER.name:
        raise AssertionError("Specified parent/outer file changed")
    if spec["prior_inner_result"] != PRIOR_INNER.name:
        raise AssertionError("Specified inner file changed")
    if spec["rate_support"] != [parent["positive_grid_rates"]["start"],
                                parent["positive_grid_rates"]["stop"]]:
        raise AssertionError("Spectral support differs from parent")
    if spec["replicates"] != parent["replicates"]:
        raise AssertionError("Replicate count differs from parent")
    if spec["truths"] != ["on_grid", "off_grid"] or spec["designs"] != [
            "single_ramp_hold", "ramp_hold_plus_cycle"]:
        raise AssertionError("Truth/design order differs from seeded parent")

    designs, heldout = design_data(parent)
    prepared = prepare_observations(parent, frozen, designs)
    outer_rows = {(r["truth"], r["design"], r["seed"]): r for r in old_outer["rows"]
                  if r["grid_count"] == 96}
    inner_rows = {(r["truth"], r["design"], r["seed"]): r for r in old_inner["rows"]
                  if r["inner_grid_count"] == 192}
    expected = 4 * parent["replicates"]
    if len(outer_rows) != expected or len(inner_rows) != expected:
        raise AssertionError("Missing comparison row")

    rows = []
    summary = {}
    timing = {}
    for count in spec["cell_counts"]:
        begin = time.perf_counter()
        edges, _, matrices, errors, targets = matrices_for_count(
            count, spec["rate_support"], designs, heldout)
        for truth in spec["truths"]:
            mapped = interpolate_truth(parent, edges, truth)
            for name in spec["designs"]:
                clean = prepared[truth, name, 0][3]
                excess = np.max(np.abs(matrices[name] @ mapped - clean) -
                                errors[name] @ mapped)
                if excess > 1e-9:
                    raise AssertionError(f"Clean response escapes curvature envelope: {excess}")
            for target, (q, f) in targets.items():
                real = frozen["true_targets"][truth][target]
                if abs(q @ mapped - real) > f @ mapped + 1e-9:
                    raise AssertionError("True target escapes curvature envelope")
        group_rows = []
        for seed in range(spec["replicates"]):
            for truth in spec["truths"]:
                mapped = interpolate_truth(parent, edges, truth)
                for name in spec["designs"]:
                    observed, tolerance, valid, _ = prepared[truth, name, seed]
                    if valid and np.max(np.abs(matrices[name] @ mapped - observed) -
                                        errors[name] @ mapped) > tolerance + 1e-9:
                        raise AssertionError("Valid-calibration truth violates outer LP")
                    found = {}
                    for target, (q, f) in targets.items():
                        interval = interval_lp(matrices[name], errors[name],
                                               observed, tolerance, q, f)
                        if interval is not None and target == "G20":
                            interval = max(0.0, interval[0]), interval[1]
                        real = frozen["true_targets"][truth][target]
                        included = (interval is not None and
                                    interval[0] - 2e-7 <= real <= interval[1] + 2e-7)
                        if valid and not included:
                            raise AssertionError("Valid-calibration true target excluded")
                        prior_inner = inner_rows[truth, name, seed]["targets"][target]["inner_interval"]
                        if prior_inner is not None:
                            if (interval is None or prior_inner[0] < interval[0] - 2e-7 or
                                    prior_inner[1] > interval[1] + 2e-7):
                                raise AssertionError("Finite-grid inner interval escapes new outer")
                        found[target] = {"interval": list(interval) if interval is not None else None,
                                         "width": float(interval[1] - interval[0])
                                         if interval is not None else None,
                                         "contains_truth": bool(included)}
                    feasible = found["G20"]["interval"] is not None
                    if feasible != (found["heldout_sigma20"]["interval"] is not None):
                        raise AssertionError("Target-dependent feasibility")
                    row = {"cell_count": count, "seed": seed, "truth": truth,
                           "design": name, "calibration_valid": bool(valid),
                           "feasible": feasible, "targets": found}
                    group_rows.append(row)
                    rows.append(row)
        summary[str(count)] = {}
        for truth in spec["truths"]:
            summary[str(count)][truth] = {}
            for name in spec["designs"]:
                group = [r for r in group_rows if r["truth"] == truth and r["design"] == name]
                feasible = [r for r in group if r["feasible"]]
                valid = [r for r in group if r["calibration_valid"]]
                info = {"feasible_count": len(feasible), "valid_calibration_count": len(valid)}
                for target in spec["targets"]:
                    info[target] = {
                        "coverage_valid_count": sum(r["targets"][target]["contains_truth"]
                                                    for r in valid),
                        "mean_width_feasible": float(np.mean([
                            r["targets"][target]["width"] for r in feasible]))
                            if feasible else None,
                    }
                summary[str(count)][truth][name] = info
        timing[str(count)] = round(time.perf_counter() - begin, 3)
        print(f"cells={count}: {json.dumps(summary[str(count)])}", flush=True)

    limits = spec["off_grid_dual_mean_width_limits"]
    gate = {}
    for count in spec["cell_counts"]:
        s = summary[str(count)]
        covered = all(s[truth][name][target]["coverage_valid_count"] ==
                      s[truth][name]["valid_calibration_count"]
                      for truth in spec["truths"] for name in spec["designs"]
                      for target in spec["targets"])
        dual = s["off_grid"]["ramp_hold_plus_cycle"]
        gate[str(count)] = {
            "valid_calibration_coverage": covered,
            "off_grid_dual_meets_both_width_limits": all(
                dual[target]["mean_width_feasible"] <= limits[target]
                for target in spec["targets"]),
        }
    qualifying = [c for c in spec["cell_counts"] if all(gate[str(c)].values())]
    verdict = {
        "smallest_qualifying_cell_count": min(qualifying) if qualifying else None,
        "desirable_gate_pass": any(c <= spec["desirable_cell_count_max"] for c in qualifying),
        "hard_gate_pass": any(c <= spec["hard_cell_count_max"] for c in qualifying),
    }
    output = {
        "scope": "conditional curvature-bounded endpoint-interpolation outer LP",
        "spec_sha256": hashlib.sha256(raw_spec).hexdigest(),
        "parent_spec_sha256": parent_sha,
        "prior_outer_result_sha256": hashlib.sha256(raw_outer).hexdigest(),
        "prior_inner_result_sha256": hashlib.sha256(raw_inner).hexdigest(),
        "summary": summary,
        "gate_by_count": gate,
        "verdict": verdict,
        "rows": rows,
    }
    RESULT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    TIMING.write_text(json.dumps({"seconds_by_cell_count": timing}, indent=2),
                      encoding="utf-8")
    print(json.dumps({"summary": summary, "gate_by_count": gate,
                      "verdict": verdict, "timing": timing}, indent=2))


if __name__ == "__main__":
    main()
