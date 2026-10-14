"""Frozen mesh-refinement audit of the P4 positive-spectrum outer intervals.

The loading histories, noise calibration, random seeds, targets, and support
are inherited unchanged. Only the number of log-rate cells changes. This
checks the price of a rigorous off-grid envelope; it does not establish a
novel estimator or unconditional coverage.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from calibrated_multiload_gate import basis
from constitutive_response_gate import exponential_basis, stress_truth_integral
from offgrid_cell_envelope_gate import grid_cells, interval_lp, response_envelope


HERE = Path(__file__).resolve().parent
PARENT_SPEC = HERE / "specs" / "calibrated_multiload_gate_20261010.json"
THIS_SPEC = HERE / "specs" / "offgrid_mesh_refinement_gate_20261010.json"
PARENT_RESULT = HERE.parent / "results" / "p4_calibrated_multiload_gate_20261010.json"
REFERENCE_RESULT = HERE.parent / "results" / "p4_offgrid_cell_envelope_gate_20261010.json"
OUTPUT = HERE.parent / "results" / "p4_offgrid_mesh_refinement_gate_20261010.json"
TIMING = HERE.parent / "results" / "p4_offgrid_mesh_refinement_timing_20261010.json"


def design_data(spec: dict) -> tuple[dict, list]:
    single = [tuple(x) for x in spec["single_design"]["ramp_hold_segments"]]
    cyclic = [tuple(x) for x in spec["dual_design"]["cyclic_segments"]]
    heldout = [tuple(x) for x in spec["heldout_input_segments"]]
    t_single = np.arange(1, 81) * spec["single_design"]["sample_step"]
    t_dual = np.arange(1, 41) * spec["dual_design"]["sample_step_each"]
    designs = {
        "single_ramp_hold": [(t_single, single)],
        "ramp_hold_plus_cycle": [(t_dual, single), (t_dual, cyclic)],
    }
    return designs, heldout


def prepare_observations(spec: dict, parent_result: dict, designs: dict) -> dict:
    old_rates = np.geomspace(spec["positive_grid_rates"]["start"],
                            spec["positive_grid_rates"]["stop"],
                            spec["positive_grid_rates"]["count"])
    on = spec["truths"]["on_grid"]
    off = spec["truths"]["off_grid"]
    old_masses = np.zeros(len(old_rates))
    for index, mass in on["transient_grid_weights"].items():
        old_masses[int(index)] = mass
    moduli = {
        "on_grid": lambda t: on["equilibrium"] +
            np.exp(-np.asarray(t)[..., None] * old_rates).dot(old_masses),
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
    noise = spec["sensor_noise"]
    amplitude = noise["unknown_amplitude_truth"]
    inflation = (1.0 - noise["upper_amplitude_confidence"]) ** (
        -1.0 / noise["independent_zero_input_calibration_count"])
    prior_rows = {(row["truth"], row["design"], row["seed"]): row
                  for row in parent_result["rows"]}
    prepared = {}
    for seed in range(spec["replicates"]):
        calibration_rng = np.random.default_rng(np.random.SeedSequence(
            [spec["seed_base"], seed, 999]))
        calibration = calibration_rng.uniform(-amplitude, amplitude,
            noise["independent_zero_input_calibration_count"])
        tolerance = float(np.max(np.abs(calibration)) * inflation)
        for truth_index, truth in enumerate(("on_grid", "off_grid")):
            for design_index, name in enumerate(designs):
                old = prior_rows[truth, name, seed]
                if abs(tolerance - old["calibrated_amplitude"]) > 1e-12:
                    raise AssertionError("Calibration differs from frozen parent")
                rng = np.random.default_rng(np.random.SeedSequence(
                    [spec["seed_base"], seed, truth_index, design_index]))
                response = clean[truth][name]
                observed = response + rng.uniform(-amplitude, amplitude,
                                                   response.size)
                prepared[truth, name, seed] = (
                    observed, tolerance, tolerance >= amplitude, response)
    return prepared


def grid_matrices(rates: np.ndarray, designs: dict, heldout: list) -> tuple:
    edges = grid_cells(rates)
    matrices = {name: np.vstack([basis(times, segments, rates)
                                 for times, segments in pieces])
                for name, pieces in designs.items()}
    errors = {name: np.vstack([response_envelope(times, segments, rates, edges)
                               for times, segments in pieces])
              for name, pieces in designs.items()}
    targets = {
        "G20": (np.r_[1.0, np.exp(-20.0 * rates)],
                np.r_[0.0, np.maximum(
                    np.abs(np.exp(-20.0 * edges[:-1]) - np.exp(-20.0 * rates)),
                    np.abs(np.exp(-20.0 * edges[1:]) - np.exp(-20.0 * rates)))]),
        "heldout_sigma20": (
            basis(np.array([20.0]), heldout, rates)[0],
            response_envelope(np.array([20.0]), heldout, rates, edges)[0]),
    }
    return edges, matrices, errors, targets


def mapped_true_mass(parent: dict, rates: np.ndarray, edges: np.ndarray,
                     truth: str) -> np.ndarray:
    old_rates = np.geomspace(parent["positive_grid_rates"]["start"],
                            parent["positive_grid_rates"]["stop"],
                            parent["positive_grid_rates"]["count"])
    on = parent["truths"]["on_grid"]
    off = parent["truths"]["off_grid"]
    if truth == "on_grid":
        pairs = [(old_rates[int(j)], mass)
                 for j, mass in on["transient_grid_weights"].items()]
        equilibrium = on["equilibrium"]
    else:
        pairs = list(zip(off["rates"], off["weights"]))
        equilibrium = off["equilibrium"]
    mass_vector = np.zeros(len(rates) + 1)
    mass_vector[0] = equilibrium
    for rate, mass in pairs:
        if not edges[0] <= rate <= edges[-1]:
            raise AssertionError("Truth falls outside fixed spectral support")
        j = min(np.searchsorted(edges, rate, side="right") - 1, len(rates) - 1)
        mass_vector[j + 1] += mass
    if abs(mass_vector.sum() - 1.0) > 1e-12:
        raise AssertionError("Known G(0)=1 violated")
    return mass_vector


def main() -> None:
    raw_parent = PARENT_SPEC.read_bytes()
    raw_self = THIS_SPEC.read_bytes()
    raw_reference = REFERENCE_RESULT.read_bytes()
    parent = json.loads(raw_parent)
    local = json.loads(raw_self)
    old = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    reference = json.loads(raw_reference)
    if hashlib.sha256(raw_parent).hexdigest() != old["spec_sha256"]:
        raise AssertionError("Parent frozen specification changed")
    if reference["parent_spec_sha256"] != old["spec_sha256"]:
        raise AssertionError("Reference result does not match parent")
    if local["grid_counts"][0] != parent["positive_grid_rates"]["count"]:
        raise AssertionError("First mesh is not the preceding gate")
    if local["rate_support"] != [parent["positive_grid_rates"]["start"],
                                 parent["positive_grid_rates"]["stop"]]:
        raise AssertionError("Spectral support changed")
    if local["replicates"] != parent["replicates"]:
        raise AssertionError("Replicate count changed")
    designs, heldout = design_data(parent)
    prepared = prepare_observations(parent, old, designs)
    reference_rows = {(row["truth"], row["design"], row["seed"]): row
                      for row in reference["rows"]}
    all_rows = []
    summaries = {}
    timing = {}
    for count in local["grid_counts"]:
        start = time.perf_counter()
        rates = np.geomspace(*local["rate_support"], count)
        edges, matrices, errors, targets = grid_matrices(rates, designs, heldout)
        group_rows = []
        for truth in local["truths"]:
            mapped = mapped_true_mass(parent, rates, edges, truth)
            for name in designs:
                clean = prepared[truth, name, 0][3]
                if np.max(np.abs(matrices[name] @ mapped - clean)
                          - errors[name] @ mapped) > 1e-9:
                    raise AssertionError("True clean stress outside cell envelope")
            for target_name, (q, f) in targets.items():
                real = old["true_targets"][truth][target_name]
                if abs(q @ mapped - real) > f @ mapped + 1e-9:
                    raise AssertionError("True target outside cell envelope")
        for seed in range(parent["replicates"]):
            for truth in local["truths"]:
                for name in designs:
                    observed, tolerance, valid, _ = prepared[truth, name, seed]
                    mapped = mapped_true_mass(parent, rates, edges, truth)
                    if valid and np.max(np.abs(matrices[name] @ mapped - observed)
                                        - errors[name] @ mapped) > tolerance + 1e-9:
                        raise AssertionError("Calibrated truth violates LP constraints")
                    found = {}
                    for target_name, (q, f) in targets.items():
                        interval = interval_lp(matrices[name], errors[name],
                                               observed, tolerance, q, f)
                        if interval is not None and target_name == "G20":
                            interval = max(0.0, interval[0]), interval[1]
                        real = old["true_targets"][truth][target_name]
                        included = (interval is not None and
                                    interval[0] - 2e-7 <= real <= interval[1] + 2e-7)
                        if valid and not included:
                            raise AssertionError("Valid-calibration target excluded")
                        if count == local["grid_counts"][0]:
                            prior = reference_rows[truth, name, seed]["targets"][target_name]["interval"]
                            if interval is None or prior is None or np.max(
                                np.abs(np.asarray(interval) - np.asarray(prior))) > 1e-9:
                                raise AssertionError("12-rate result differs from previous gate")
                        found[target_name] = {
                            "interval": list(interval) if interval is not None else None,
                            "width": interval[1] - interval[0] if interval is not None else None,
                            "contains_truth": bool(included),
                        }
                    row = {"grid_count": count, "seed": seed, "truth": truth,
                           "design": name, "calibration_valid": bool(valid),
                           "feasible": found["G20"]["interval"] is not None,
                           "targets": found}
                    group_rows.append(row)
                    all_rows.append(row)
        summaries[str(count)] = {}
        for truth in local["truths"]:
            summaries[str(count)][truth] = {}
            for name in designs:
                group = [r for r in group_rows if r["truth"] == truth and r["design"] == name]
                feasible = [r for r in group if r["feasible"]]
                valid = [r for r in group if r["calibration_valid"]]
                info = {"calibration_valid_count": len(valid),
                        "feasible_count": len(feasible)}
                for target_name in targets:
                    info[target_name + "_coverage_valid_count"] = sum(
                        r["targets"][target_name]["contains_truth"] for r in valid)
                    info[target_name + "_mean_width_feasible"] = (
                        float(np.mean([r["targets"][target_name]["width"]
                                       for r in feasible])) if feasible else None)
                summaries[str(count)][truth][name] = info
        timing[str(count)] = round(time.perf_counter() - start, 3)
        print(f"grid={count}: {json.dumps(summaries[str(count)])}", flush=True)
    limits = local["utility_gate"]
    utility = {str(count): {name: {
        "G20": summaries[str(count)]["off_grid"][name]["G20_mean_width_feasible"] <=
            limits["off_grid_G20_mean_width_feasible_max"],
        "heldout_sigma20": summaries[str(count)]["off_grid"][name]
            ["heldout_sigma20_mean_width_feasible"] <=
            limits["off_grid_heldout_sigma20_mean_width_feasible_max"],
    } for name in designs} for count in local["grid_counts"]}
    output = {
        "scope": "bounded-support conditional outer intervals; fixed data, refined log-rate cells",
        "parent_spec_sha256": old["spec_sha256"],
        "this_spec_sha256": hashlib.sha256(raw_self).hexdigest(),
        "reference_result_sha256": hashlib.sha256(raw_reference).hexdigest(),
        "summary": summaries,
        "utility_gate_pass": utility,
        "rows": all_rows,
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    TIMING.write_text(json.dumps({"seconds_by_grid": timing}, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summaries, "utility_gate_pass": utility,
                      "seconds_by_grid": timing}, indent=2))


if __name__ == "__main__":
    main()
