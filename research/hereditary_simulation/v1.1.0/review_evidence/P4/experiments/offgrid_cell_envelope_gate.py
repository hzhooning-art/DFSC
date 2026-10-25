"""Frozen P4 response gate: conditional off-grid interval outer bounds.

This is an auditable set-membership baseline, not a claim of a new algorithm.
The guarantee requires a positive spectrum supported on the frozen grid span,
the exact recorded strain histories, and a valid independently calibrated
sensor-noise amplitude. It does not apply to the stretched-exponential truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from calibrated_multiload_gate import basis
from constitutive_response_gate import exponential_basis, strain, stress_truth_integral


HERE = Path(__file__).resolve().parent
PARENT_SPEC = HERE / "specs" / "calibrated_multiload_gate_20261010.json"
THIS_SPEC = HERE / "specs" / "offgrid_cell_envelope_gate_20261010.json"
PARENT_RESULT = HERE.parent / "results" / "p4_calibrated_multiload_gate_20261010.json"
OUTPUT = HERE.parent / "results" / "p4_offgrid_cell_envelope_gate_20261010.json"


def grid_cells(rates: np.ndarray) -> np.ndarray:
    """Support is exactly [rates[0], rates[-1]]; do not infer a wider support."""
    return np.r_[rates[0], np.sqrt(rates[:-1] * rates[1:]), rates[-1]]


def response_envelope(times: np.ndarray, segments: list[tuple[float, float, float]],
                      rates: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Termwise rigorous rate-cell envelope for piecewise-constant strain rate.

    The response to each signed segment is monotone in lambda. Its largest
    absolute difference from a representative occurs at a cell endpoint.
    Summing segment-wise absolute differences handles loading and unloading.
    """
    result = np.zeros((len(times), len(rates) + 1))
    for segment in segments:
        for j, rate in enumerate(rates):
            columns = exponential_basis(times, [segment],
                                        np.array([edges[j], rate, edges[j + 1]]))
            result[:, j + 1] += np.maximum(
                np.abs(columns[:, 0] - columns[:, 1]),
                np.abs(columns[:, 2] - columns[:, 1]),
            )
    return result


def interval_lp(design: np.ndarray, error: np.ndarray, observed: np.ndarray,
                tolerance: float, functional: np.ndarray,
                target_error: np.ndarray) -> tuple[float, float] | None:
    n = design.shape[1]
    upper = np.vstack((design - error, -design - error))
    limits = np.r_[observed + tolerance, -observed + tolerance]
    equality = np.ones((1, n))
    options = {"primal_feasibility_tolerance": 1e-9,
               "dual_feasibility_tolerance": 1e-9}
    problems = ((functional - target_error, 1.0),
                (functional + target_error, -1.0))
    answers = []
    for objective, sign in problems:
        answer = linprog(sign * objective, A_ub=upper, b_ub=limits,
                         A_eq=equality, b_eq=[1.0], bounds=(0.0, None),
                         method="highs", options=options)
        if answer.status == 2:
            return None
        if not answer.success:
            raise RuntimeError(f"Envelope LP failed: {answer.message}")
        if np.max(upper @ answer.x - limits) > 2e-7:
            raise AssertionError("LP witness violates envelope constraint")
        answers.append(float(objective @ answer.x))
    return answers[0], answers[1]


def main() -> None:
    raw_parent = PARENT_SPEC.read_bytes()
    raw_self = THIS_SPEC.read_bytes()
    parent = json.loads(raw_parent)
    local = json.loads(raw_self)
    previous = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    parent_hash = hashlib.sha256(raw_parent).hexdigest()
    if parent_hash != previous["spec_sha256"]:
        raise AssertionError("The parent frozen spec changed")
    if local["replicates"] != parent["replicates"]:
        raise AssertionError("Replicate count differs from parent gate")
    grid = parent["positive_grid_rates"]
    rates = np.geomspace(grid["start"], grid["stop"], grid["count"])
    edges = grid_cells(rates)
    off = parent["truths"]["off_grid"]
    if not all(edges[0] <= rate <= edges[-1] for rate in off["rates"]):
        raise AssertionError("Off-grid truth violates the frozen support assumption")
    single = [tuple(x) for x in parent["single_design"]["ramp_hold_segments"]]
    cyclic = [tuple(x) for x in parent["dual_design"]["cyclic_segments"]]
    heldout = [tuple(x) for x in parent["heldout_input_segments"]]
    t_single = np.arange(1, 81) * parent["single_design"]["sample_step"]
    t_dual = np.arange(1, 41) * parent["dual_design"]["sample_step_each"]
    designs = {
        "single_ramp_hold": [(t_single, single)],
        "ramp_hold_plus_cycle": [(t_dual, single), (t_dual, cyclic)],
    }
    A = {name: np.vstack([basis(times, segments, rates)
                          for times, segments in pieces])
         for name, pieces in designs.items()}
    E = {name: np.vstack([response_envelope(times, segments, rates, edges)
                          for times, segments in pieces])
         for name, pieces in designs.items()}
    old_rows = {(r["truth"], r["design"], r["seed"]): r
                for r in previous["rows"]}
    targets = {
        "G20": (np.r_[1.0, np.exp(-20.0 * rates)],
                np.r_[0.0, np.maximum(
                    np.abs(np.exp(-20.0 * edges[:-1]) - np.exp(-20.0 * rates)),
                    np.abs(np.exp(-20.0 * edges[1:]) - np.exp(-20.0 * rates)))]),
        "heldout_sigma20": (
            basis(np.array([20.0]), heldout, rates)[0],
            response_envelope(np.array([20.0]), heldout, rates, edges)[0],
        ),
    }
    on = parent["truths"]["on_grid"]
    true_on = np.zeros(len(rates) + 1)
    true_on[0] = on["equilibrium"]
    for index, mass in on["transient_grid_weights"].items():
        true_on[int(index) + 1] = mass
    true_off = np.zeros_like(true_on)
    true_off[0] = off["equilibrium"]
    for rate, mass in zip(off["rates"], off["weights"]):
        j = np.searchsorted(edges, rate, side="right") - 1
        j = min(j, len(rates) - 1)
        true_off[j + 1] += mass
    mapped = {"on_grid": true_on, "off_grid": true_off}
    moduli = {
        "on_grid": lambda t: true_on[0] +
            np.exp(-np.asarray(t)[..., None] * rates).dot(true_on[1:]),
        "off_grid": lambda t: off["equilibrium"] +
            np.exp(-np.asarray(t)[..., None] * np.array(off["rates"]))
            .dot(np.array(off["weights"])),
    }
    clean = {
        truth: {name: np.concatenate([stress_truth_integral(times, segments, modulus)
                                      for times, segments in pieces])
                for name, pieces in designs.items()}
        for truth, modulus in moduli.items()
    }
    # An independent noiseless containment check tests the numerical envelope.
    for truth, response_by_design in clean.items():
        for name, response in response_by_design.items():
            discrepancy = np.abs(A[name] @ mapped[truth] - response)
            permitted = E[name] @ mapped[truth]
            if np.max(discrepancy - permitted) > 1e-9:
                raise AssertionError("Clean true response escapes the cell envelope")
        for target_name, (functional, target_error) in targets.items():
            real = previous["true_targets"][truth][target_name]
            if abs(functional @ mapped[truth] - real) > target_error @ mapped[truth] + 1e-9:
                raise AssertionError("True target escapes the cell envelope")
    noise = parent["sensor_noise"]
    true_amplitude = noise["unknown_amplitude_truth"]
    inflation = (1.0 - noise["upper_amplitude_confidence"]) ** (
        -1.0 / noise["independent_zero_input_calibration_count"])
    rows = []
    for seed in range(parent["replicates"]):
        calibration_rng = np.random.default_rng(np.random.SeedSequence(
            [parent["seed_base"], seed, 999]))
        calibration = calibration_rng.uniform(-true_amplitude, true_amplitude,
            noise["independent_zero_input_calibration_count"])
        tolerance = float(np.max(np.abs(calibration)) * inflation)
        valid = tolerance >= true_amplitude
        for truth_index, truth in enumerate(local["truths"]):
            for design_index, name in enumerate(designs):
                old = old_rows[truth, name, seed]
                if abs(tolerance - old["calibrated_amplitude"]) > 1e-12:
                    raise AssertionError("Calibration seed differs from frozen gate")
                rng = np.random.default_rng(np.random.SeedSequence(
                    [parent["seed_base"], seed, truth_index, design_index]))
                response = clean[truth][name]
                observed = response + rng.uniform(-true_amplitude, true_amplitude,
                                                   response.size)
                found = {}
                for target_name, (functional, target_error) in targets.items():
                    interval = interval_lp(A[name], E[name], observed,
                                           tolerance, functional, target_error)
                    if interval is not None and target_name == "G20":
                        interval = max(0.0, interval[0]), interval[1]
                    found[target_name] = interval
                if (found["G20"] is None) != (found["heldout_sigma20"] is None):
                    raise AssertionError("Same feasible set disagrees on feasibility")
                feasible = found["G20"] is not None
                if valid and not feasible:
                    raise AssertionError("Admissible truth excluded under valid calibration")
                assessed = {}
                for target_name, interval in found.items():
                    real = previous["true_targets"][truth][target_name]
                    included = interval is not None and interval[0] - 2e-7 <= real <= interval[1] + 2e-7
                    if valid and not included:
                        raise AssertionError("Admissible true target excluded")
                    old_interval = old["targets"][target_name]["interval"]
                    if interval is not None and old_interval is not None and (
                        interval[0] > old_interval[0] + 2e-7 or
                        interval[1] < old_interval[1] - 2e-7
                    ):
                        raise AssertionError("Outer interval fails to contain frozen LP interval")
                    assessed[target_name] = {
                        "truth": real,
                        "interval": list(interval) if interval is not None else None,
                        "width": interval[1] - interval[0] if interval is not None else None,
                        "contains_truth": bool(included),
                    }
                rows.append({"seed": seed, "truth": truth, "design": name,
                             "calibration_valid": bool(valid), "feasible": feasible,
                             "targets": assessed})
    summary = {}
    for truth in local["truths"]:
        summary[truth] = {}
        for name in designs:
            group = [r for r in rows if r["truth"] == truth and r["design"] == name]
            feasible = [r for r in group if r["feasible"]]
            valid = [r for r in group if r["calibration_valid"]]
            item = {
                "calibration_valid_count": len(valid),
                "feasible_count": len(feasible),
                "feasible_valid_count": sum(r["feasible"] for r in valid),
            }
            for target_name in targets:
                item[target_name + "_coverage_valid_count"] = sum(
                    r["targets"][target_name]["contains_truth"] for r in valid)
                item[target_name + "_mean_width_feasible"] = float(np.mean([
                    r["targets"][target_name]["width"] for r in feasible]))
            summary[truth][name] = item
    thresholds = local["utility_gate"]
    utility = {name: {
        "G20": summary["off_grid"][name]["G20_mean_width_feasible"] <=
            thresholds["off_grid_G20_mean_width_feasible_max"],
        "heldout_sigma20": summary["off_grid"][name]["heldout_sigma20_mean_width_feasible"] <=
            thresholds["off_grid_heldout_sigma20_mean_width_feasible_max"],
    } for name in designs}
    output = {
        "scope": "conditional outer interval for positive spectra on frozen bounded rate support",
        "parent_spec_sha256": parent_hash,
        "this_spec_sha256": hashlib.sha256(raw_self).hexdigest(),
        "rate_support": [float(edges[0]), float(edges[-1])],
        "cell_edges": edges.tolist(),
        "summary": summary,
        "utility_gate_pass": utility,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("parent_spec_sha256", "this_spec_sha256",
                                           "summary", "utility_gate_pass")}, indent=2))


if __name__ == "__main__":
    main()
