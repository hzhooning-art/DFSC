"""Loading endpoint/dead-time phase test; no spectrum reconstruction or LP."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import gammainc, gammaln

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "experiments/specs/loading_endpoint_phase_20261011.json"
OUT = ROOT / "results/p4_loading_endpoint_phase_20261011.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log_kernel(rate, order, dead_time, duration=1.0):
    """c=1: exp(-dead*rate) integral_0^L r^order exp(-rate*r) dr.

    order=None is a unit ideal step, not a finite-rate loading history.
    Positive input ends at duration, and is observed duration+dead_time.
    """
    if rate < 0 or dead_time < 0 or duration <= 0:
        raise ValueError("Nonnegative rates/dead time and positive duration required")
    if order is None:
        return -dead_time * rate
    if order < 0:
        raise ValueError("Finite-rate profiles require nonnegative endpoint order")
    beta = order + 1.0
    if rate == 0:
        return beta * np.log(duration) - np.log(beta)
    return float(-dead_time*rate + gammaln(beta)
                 + np.log(gammainc(beta, rate*duration)) - beta*np.log(rate))


def is_bounded(reference_order, reference_dead, target_order, target_dead):
    gamma = 0.0 if target_order is None else target_order+1.0
    if target_dead != reference_dead:
        return target_dead > reference_dead
    return gamma >= reference_order+1.0


def ratio_upper(reference_order, reference_dead, target_order, target_dead, duration=1.0):
    """Analytic whole-half-line bound for polynomial profiles, c=1.

    Splits rate <=1 (integral comparison) and rate >=1 (incomplete gamma).
    Returns None for cases proved unbounded, not a numeric optimizer result.
    """
    if not is_bounded(reference_order, reference_dead, target_order, target_dead):
        return None
    beta = reference_order+1.0
    gamma = 0.0 if target_order is None else target_order+1.0
    qzero = 1.0 if target_order is None else duration**gamma/gamma
    compact = qzero*np.exp(reference_dead+duration)*beta/duration**beta
    coefficient = np.exp((0.0 if gamma == 0 else gammaln(gamma))-gammaln(beta)) / gammainc(beta, duration)
    a, gap = beta-gamma, target_dead-reference_dead
    if a <= 0:
        tail = coefficient
    else:
        # gap>0 follows from boundedness when a>0.
        peak = max(1.0, a/gap)
        tail = coefficient*peak**a*np.exp(-gap*peak)
    return float(max(compact, tail))


def witness(spec, p, dead, target_order, target_dead, rate):
    lg = log_kernel(rate, p, dead, spec["duration"])
    lq = log_kernel(rate, target_order, target_dead, spec["duration"])
    alpha = spec["added_weighted_mass"]
    mass = float(np.exp(np.log(alpha)-lg))
    changes = [float(alpha*np.exp(-d*rate)) for d in spec["observation_delays"]]
    target_change = float(alpha*np.exp(lq-lg))
    assert max(changes) < spec["declared_response_band"]
    assert np.isfinite(mass) and mass > 0
    return {"rate": rate, "physical_added_mass": mass,
            "observation_changes": changes, "target_change": target_change,
            "deterministic_minimax_pair_lower": target_change/2}


def main():
    if OUT.exists():
        raise FileExistsError("Refuse to overwrite frozen endpoint-phase result")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    parent = ROOT / "docs" / spec["parent_theory"]
    assert sha(parent) == spec["parent_theory_sha256"]
    data, checks, phases, frontiers = [], [], [], []
    for p in spec["reference_orders"]:
        for dead in spec["reference_dead_times"]:
            observations = [float(sum(w*np.exp(log_kernel(l, p, dead+d, spec["duration"]))
                                     for l, w in zip(spec["base_rates"], spec["base_weights"])))
                            for d in spec["observation_delays"]]
            data.append({"reference_order": p, "reference_dead_time": dead,
                         "observations": observations, "anchor_upper": observations[0]+spec["declared_response_band"]})
            for rate in spec["kernel_check_rates"]:
                checks.append({"reference_order": p, "reference_dead_time": dead,
                               "rate": rate, "kernel": float(np.exp(log_kernel(rate, p, dead, spec["duration"])))})
            for target_order in spec["target_orders"]:
                gamma = 0 if target_order is None else target_order+1
                for offset in spec["target_dead_time_offsets"]:
                    target_dead = dead+offset
                    if target_dead < 0:
                        continue
                    bounded = is_bounded(p, dead, target_order, target_dead)
                    upper = ratio_upper(p, dead, target_order, target_dead, spec["duration"])
                    witnesses = [witness(spec, p, dead, target_order, target_dead, rate)
                                 for rate in spec["adversarial_rates"]]
                    if upper is not None:
                        assert all(w["target_change"] <= spec["added_weighted_mass"]*upper*(1+1e-12) for w in witnesses)
                    phases.append({"reference_order": p, "reference_dead_time": dead,
                                   "target_order": target_order, "target_dead_time": target_dead,
                                   "proved_bounded": bounded, "ratio_whole_half_line_upper": upper,
                                   "witnesses": witnesses})
                a = p+1-gamma
                if a <= 0:
                    continue
                constant = float(np.exp((0 if gamma == 0 else gammaln(gamma))-gammaln(p+1))*(a/np.e)**a)
                records = []
                for gap in spec["frontier_gaps"]:
                    rate = a/gap
                    w = witness(spec, p, dead, target_order, dead+gap, rate)
                    scaled = w["target_change"]*gap**a
                    whole_upper = ratio_upper(p, dead, target_order, dead+gap, spec["duration"])
                    records.append({"gap": gap, **w, "scaled_target_change": scaled,
                                    "diameter_whole_class_upper": (observations[0]+spec["declared_response_band"])*whole_upper})
                limit = spec["added_weighted_mass"]*constant
                rel = abs(records[-1]["scaled_target_change"]/limit-1)
                slope = float(-np.polyfit(np.log(spec["frontier_gaps"][-3:]),
                                         np.log([r["target_change"] for r in records[-3:]]), 1)[0])
                assert rel < spec["frontier_relative_limit_tolerance"]
                assert abs(slope-a) < spec["frontier_slope_tolerance"]
                assert all(r["target_change"] <= r["diameter_whole_class_upper"]*(1+1e-12) for r in records)
                frontiers.append({"reference_order": p, "reference_dead_time": dead,
                                  "target_order": target_order, "proved_exponent": a,
                                  "observed_pair_slope": slope, "scaled_pair_limit": limit,
                                  "last_scaled_relative_error": rel, "records": records})
    output = {"schema": "P4-loading-endpoint-phase-result-v1", "spec_sha256": sha(SPEC),
              "source_sha256": sha(__file__), "parent_theory_sha256": sha(parent),
              "data": data, "kernel_checks": checks, "phases": phases, "frontiers": frontiers,
              "summary": {"reference_conditions": len(data), "phase_conditions": len(phases),
                          "bounded_conditions": sum(r["proved_bounded"] for r in phases),
                          "physical_pairs": sum(len(r["witnesses"]) for r in phases)+sum(len(r["records"]) for r in frontiers),
                          "frontiers": len(frontiers), "max_slope_error": max(abs(r["observed_pair_slope"]-r["proved_exponent"]) for r in frontiers)},
              "all_internal_checks_pass": True,
              "limitations": ["Declared deterministic bands; no sensor calibration or new random validation",
                              "Asymptotics and unboundedness follow from proof, not a finite sample of rates",
                              "No outward rounding, external proof review, or global novelty clearance",
                              "Known positive polynomial endpoint inputs only; arbitrary absolute rate errors can change endpoint order"]}
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output["summary"]), flush=True)


if __name__ == "__main__":
    main()
