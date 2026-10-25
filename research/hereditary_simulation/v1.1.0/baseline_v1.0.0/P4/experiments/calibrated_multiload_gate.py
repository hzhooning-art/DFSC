"""Frozen multi-loading identified-set gate with separate noise calibration.

All inference is conditional on a *known* fixed-rate nonnegative Prony model,
an exact instantaneous modulus, and stationary uniform sensor noise.  This
script explicitly tests failure under model misspecification; it is not a
new confidence-set method or a transport of P2's Mittag-Leffler controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from constitutive_response_gate import exponential_basis, strain, stress_truth_integral


def basis(times: np.ndarray, segments: list[tuple[float, float, float]],
          rates: np.ndarray) -> np.ndarray:
    return np.column_stack((strain(times, segments),
                            exponential_basis(times, segments, rates)))


def interval_lp(design: np.ndarray, observed: np.ndarray, tolerance: float,
                functional: np.ndarray) -> tuple[float, float] | None:
    n = design.shape[1]
    upper = np.vstack((design, -design))
    limits = np.r_[observed + tolerance, -observed + tolerance]
    equality = np.ones((1, n))
    options = {"primal_feasibility_tolerance": 1e-9,
               "dual_feasibility_tolerance": 1e-9}
    solutions = [linprog(sign * functional, A_ub=upper, b_ub=limits,
                         A_eq=equality, b_eq=[1.0], bounds=(0.0, None),
                         method="highs", options=options)
                 for sign in (1.0, -1.0)]
    if any(result.status == 2 for result in solutions):
        return None
    if not all(result.success for result in solutions):
        raise RuntimeError(f"LP did not solve: {[r.message for r in solutions]}")
    for result in solutions:
        if np.max(np.abs(design @ result.x - observed)) > tolerance + 2e-7:
            raise AssertionError("LP witness exceeds response band")
        if abs(result.x.sum() - 1.0) > 1e-7 or result.x.min() < -1e-7:
            raise AssertionError("LP witness violates nonnegative unit modulus")
    return float(functional @ solutions[0].x), float(functional @ solutions[1].x)


def best_supnorm_discrepancy(design: np.ndarray, clean: np.ndarray) -> float:
    """Best achievable deterministic response error within the frozen grid."""
    n = design.shape[1]
    upper = np.block([[design, -np.ones((len(clean), 1))],
                      [-design, -np.ones((len(clean), 1))]])
    objective = np.r_[np.zeros(n), 1.0]
    result = linprog(objective, A_ub=upper, b_ub=np.r_[clean, -clean],
                     A_eq=np.r_[np.ones(n), 0.0][None, :], b_eq=[1.0],
                     bounds=[(0.0, None)] * (n + 1), method="highs")
    if not result.success:
        raise RuntimeError(f"minimax discrepancy problem failed: {result.message}")
    return float(result.x[-1])


def mean_or_none(items: list[float]) -> float | None:
    return float(np.mean(items)) if items else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=Path(__file__).parent /
                        "specs" / "calibrated_multiload_gate_20261010.json")
    parser.add_argument("--out", type=Path, default=Path(__file__).parents[1] /
                        "results" / "p4_calibrated_multiload_gate_20261010.json")
    args = parser.parse_args()
    raw_spec = args.spec.read_bytes()
    spec = json.loads(raw_spec)
    spec_sha256 = hashlib.sha256(raw_spec).hexdigest()
    assert spec["samples_per_design"] == 80 and spec["replicates"] == 128
    assert spec["training_end"] == 8.0
    rate_spec = spec["positive_grid_rates"]
    rates = np.geomspace(rate_spec["start"], rate_spec["stop"], rate_spec["count"])
    single = [tuple(x) for x in spec["single_design"]["ramp_hold_segments"]]
    cyclic = [tuple(x) for x in spec["dual_design"]["cyclic_segments"]]
    heldout = [tuple(x) for x in spec["heldout_input_segments"]]
    t_single = np.arange(1, 81) * spec["single_design"]["sample_step"]
    t_dual = np.arange(1, 41) * spec["dual_design"]["sample_step_each"]
    assert np.isclose(t_single[-1], 8.0) and np.isclose(t_dual[-1], 8.0)
    designs = {
        "single_ramp_hold": [(t_single, single)],
        "ramp_hold_plus_cycle": [(t_dual, single), (t_dual, cyclic)],
    }
    matrices = {name: np.vstack([basis(t, segments, rates) for t, segments in pieces])
                for name, pieces in designs.items()}
    assert all(matrix.shape == (80, 13) for matrix in matrices.values())
    on = spec["truths"]["on_grid"]
    true_weights = np.zeros(13)
    true_weights[0] = on["equilibrium"]
    for grid_index, value in on["transient_grid_weights"].items():
        true_weights[1 + int(grid_index)] = value
    assert np.isclose(true_weights.sum(), 1.0)
    off = spec["truths"]["off_grid"]
    stretched = spec["truths"]["stretched"]
    moduli = {
        "on_grid": lambda t: true_weights[0]
                   + np.exp(-np.asarray(t)[..., None] * rates).dot(true_weights[1:]),
        "off_grid": lambda t: off["equilibrium"]
                    + np.exp(-np.asarray(t)[..., None] * np.array(off["rates"]))
                    .dot(np.array(off["weights"])),
        "stretched": lambda t: stretched["equilibrium"]
                     + (1.0 - stretched["equilibrium"])
                     * np.exp(-(np.asarray(t) / stretched["tau"]) ** stretched["beta"]),
    }
    functionals = {
        "G20": np.r_[1.0, np.exp(-20.0 * rates)],
        "heldout_sigma20": basis(np.array([20.0]), heldout, rates)[0],
    }
    true_targets = {
        name: {"G20": float(modulus(20.0)),
               "heldout_sigma20": float(stress_truth_integral(
                   np.array([20.0]), heldout, modulus)[0])}
        for name, modulus in moduli.items()
    }
    clean = {
        truth: {name: np.concatenate([stress_truth_integral(t, segments, modulus)
                                      for t, segments in pieces])
                for name, pieces in designs.items()}
        for truth, modulus in moduli.items()
    }
    reference_discrepancy = {
        name: float(np.max(np.abs(matrices[name] @ true_weights - clean["on_grid"][name])))
        for name in designs
    }
    if max(reference_discrepancy.values()) > 1e-9:
        raise AssertionError("independent convolution and in-grid modal response disagree")
    model_discrepancy = {
        truth: {name: best_supnorm_discrepancy(matrices[name], clean[truth][name])
                for name in designs}
        for truth in moduli
    }
    noise_spec = spec["sensor_noise"]
    true_amplitude = noise_spec["unknown_amplitude_truth"]
    calibration_n = noise_spec["independent_zero_input_calibration_count"]
    confidence = noise_spec["upper_amplitude_confidence"]
    delta = 1.0 - confidence
    inflation = delta ** (-1.0 / calibration_n)
    rows = []
    for seed in range(spec["replicates"]):
        calibration_rng = np.random.default_rng(np.random.SeedSequence(
            [spec["seed_base"], seed, 999]))
        calibration = calibration_rng.uniform(-true_amplitude, true_amplitude,
                                               calibration_n)
        calibrated_amplitude = float(np.max(np.abs(calibration)) * inflation)
        calibration_valid = calibrated_amplitude >= true_amplitude
        for truth_index, truth in enumerate(moduli):
            for design_index, design_name in enumerate(designs):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [spec["seed_base"], seed, truth_index, design_index]))
                response = clean[truth][design_name]
                observed = response + rng.uniform(-true_amplitude, true_amplitude,
                                                   response.size)
                matrix = matrices[design_name]
                found = {}
                for target_name, functional in functionals.items():
                    found[target_name] = interval_lp(matrix, observed,
                                                     calibrated_amplitude, functional)
                if (found["G20"] is None) != (found["heldout_sigma20"] is None):
                    raise AssertionError("same feasible set disagrees on feasibility")
                feasible = found["G20"] is not None
                if truth == "on_grid" and calibration_valid and not feasible:
                    raise AssertionError("true on-grid model excluded under valid noise bound")
                targets = {}
                for target_name, interval in found.items():
                    target = true_targets[truth][target_name]
                    targets[target_name] = {
                        "truth": target,
                        "interval": list(interval) if interval is not None else None,
                        "width": interval[1] - interval[0] if interval is not None else None,
                        "contains_truth": (interval[0] - 2e-7 <= target <= interval[1] + 2e-7)
                                          if interval is not None else False,
                    }
                    if truth == "on_grid" and calibration_valid and not targets[target_name]["contains_truth"]:
                        raise AssertionError("in-grid target omitted under valid calibration")
                rows.append({
                    "seed": seed, "truth": truth, "design": design_name,
                    "calibrated_amplitude": calibrated_amplitude,
                    "calibration_valid": calibration_valid,
                    "max_observation_noise": float(np.max(np.abs(observed - response))),
                    "feasible": feasible, "targets": targets,
                })
    summary = {}
    for truth in moduli:
        summary[truth] = {}
        for design_name in designs:
            group = [r for r in rows if r["truth"] == truth and r["design"] == design_name]
            feasible = [r for r in group if r["feasible"]]
            summary[truth][design_name] = {
                "calibration_valid_fraction": float(np.mean([r["calibration_valid"] for r in group])),
                "feasible_fraction": len(feasible) / len(group),
                "G20_coverage_all_trials": float(np.mean(
                    [r["targets"]["G20"]["contains_truth"] for r in group])),
                "heldout_sigma20_coverage_all_trials": float(np.mean(
                    [r["targets"]["heldout_sigma20"]["contains_truth"] for r in group])),
                "G20_mean_width_feasible": mean_or_none(
                    [r["targets"]["G20"]["width"] for r in feasible]),
                "heldout_sigma20_mean_width_feasible": mean_or_none(
                    [r["targets"]["heldout_sigma20"]["width"] for r in feasible]),
                "best_clean_response_supnorm_mismatch": model_discrepancy[truth][design_name],
            }
        paired = []
        for seed in range(spec["replicates"]):
            by_name = {r["design"]: r for r in rows if r["truth"] == truth and r["seed"] == seed}
            if all(r["feasible"] for r in by_name.values()):
                paired.append(by_name)
        summary[truth]["paired_feasible_count"] = len(paired)
        summary[truth]["dual_narrower_fraction_G20"] = mean_or_none([
            float(pair["ramp_hold_plus_cycle"]["targets"]["G20"]["width"]
                  < pair["single_ramp_hold"]["targets"]["G20"]["width"])
            for pair in paired])
        summary[truth]["dual_narrower_fraction_heldout_sigma20"] = mean_or_none([
            float(pair["ramp_hold_plus_cycle"]["targets"]["heldout_sigma20"]["width"]
                  < pair["single_ramp_hold"]["targets"]["heldout_sigma20"]["width"])
            for pair in paired])
    output = {
        "purpose": "Frozen physical response-only identified-set gate; conditional, not novel",
        "spec_sha256": spec_sha256,
        "noise_upper_formula": "M / (0.05 ** (1/64)); exact 95% amplitude upper confidence only under iid U[-a,a] blanks",
        "true_noise_amplitude_for_evaluation_only": true_amplitude,
        "calibration_inflation_factor": inflation,
        "reference_modal_vs_adaptive_quad_max_abs": reference_discrepancy,
        "true_targets": true_targets,
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"spec_sha256": spec_sha256, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
