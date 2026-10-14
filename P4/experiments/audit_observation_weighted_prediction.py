"""Replay saved LPs and adversarial physical spectra without rerunning fits."""
import json
from pathlib import Path
import numpy as np

from observation_weighted_prediction import SPEC, OUTPUT, ARCHIVE, HERE, digest, representation, anchor, integrate_truth

AUDIT = OUTPUT.with_name("p4_observation_weighted_prediction_audit_20261011.json")
FRONTIER_SPEC = HERE / "specs/observation_weighted_delay_frontier_20261011.json"


def time_integral(rate, duration, delay):
    if rate == 0:
        return duration
    return np.exp(-rate*delay)*(-np.expm1(-rate*duration))/rate


def target_from_rate(rate, target):
    if target == "G20":
        return np.exp(-20*rate)
    if target == "quiet_half_ramp":
        return 2*time_integral(rate, .5, 1.)
    if target == "active_half_ramp":
        return 2*time_integral(rate, .5, 0.)
    if target == "reversal_sigma":
        return time_integral(rate, 1., 2.)-.5*time_integral(rate, 1., 1.)
    raise ValueError(target)


def main():
    if AUDIT.exists():
        raise FileExistsError("Frozen audit exists")
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    frontier = json.loads(FRONTIER_SPEC.read_text(encoding="utf-8"))
    assert result["spec_sha256"] == digest(SPEC)
    assert result["source_sha256"] == digest(HERE / "observation_weighted_prediction.py")
    assert result["witness_archive_sha256"] == digest(ARCHIVE)
    assert result["gate_pass"] and result["monotonicity_checks"] == 256
    times = np.array(spec["observation_times"])
    residuals = {"primal": 0., "stationarity": 0., "gap": 0., "sign": 0.}
    target_checks, witness_checks, frontier_rows = 0, 0, []
    with np.load(ARCHIVE, allow_pickle=False) as saved:
        for info in spec["truths"]:
            observations = saved[f"{info['name']}_observations"]
            independent = np.array([integrate_truth(info, lambda rate: time_integral(rate, 1., at-1)) for at in times])
            np.testing.assert_allclose(independent, observations, atol=1e-12, rtol=0)
            rhs = np.r_[observations+spec["response_error_bound"], -observations+spec["response_error_bound"]]
            for row in [r for r in result["rows"] if r["truth"] == info["name"]]:
                count, target = row["cells"], row["target"]
                rep = representation(count, times, target, spec["inner_minimum_x"])
                stem = f"{info['name']}_{count}_{target}"
                independent_target = integrate_truth(info, lambda rate: target_from_rate(rate, target))
                assert abs(independent_target-row["truth_target"]) < 2e-11
                target_checks += 1
                for method, a, objectives, expected in (
                        ("outer", rep["outer_a"], (rep["q_low"], -rep["q_high"]), row["outer_interval"]),
                        ("inner", rep["inner_a"], (rep["inner_q"], -rep["inner_q"]), row["inner_interval"])):
                    for index, (side, c) in enumerate(zip(("low", "high"), objectives)):
                        name = f"{stem}_{method}_{side}"
                        x, y, z = (saved[f"{name}_{key}"] for key in ("x", "y", "z"))
                        residuals["primal"] = max(residuals["primal"], float(max(0., np.max(a@x-rhs), -np.min(x))))
                        residuals["stationarity"] = max(residuals["stationarity"], float(np.max(np.abs(c-a.T@y-z))))
                        residuals["gap"] = max(residuals["gap"], float(abs(c@x-rhs@y)))
                        residuals["sign"] = max(residuals["sign"], float(max(0., np.max(y), -np.min(z))))
                        assert abs(float(c@x)*(1 if index == 0 else -1)-expected[index]) < 1e-11
                # A finite positive-node inner spectrum must also pass the
                # original rate-coordinate observation/target calculation.
                for side in ("low", "high"):
                    weights = saved[f"{stem}_inner_{side}_x"]
                    rates = -np.log(rep["nodes"])
                    physical_mass = weights/anchor(rates)
                    physical_observations = np.array([sum(m*time_integral(r, 1., at-1) for m, r in zip(physical_mass, rates)) for at in times])
                    assert np.max(np.abs(physical_observations-observations)) <= spec["response_error_bound"]+2e-7
                    physical_target = sum(m*target_from_rate(r, target) for m, r in zip(physical_mass, rates))
                    assert abs(physical_target-row["inner_interval"][0 if side == "low" else 1]) < 2e-9
            for witness in [w for w in result["fast_witnesses"] if w["truth"] == info["name"]]:
                rate, mass = witness["rate"], witness["added_physical_mass"]
                changes = mass*np.array([time_integral(rate, 1., at-1) for at in times])
                assert np.max(changes) < spec["response_error_bound"]
                for target in spec["bounded_targets"]:
                    independent_change = mass*target_from_rate(rate, target)
                    assert abs(independent_change-witness["bounded_target_changes"][target]) < 2e-12
                    witness_checks += 1
            # Fresh consequence check: fixed-data compatible pair across delays.
            for delay in frontier["delays"]:
                rate = 1/delay
                alpha = spec["fast_witness_weighted_mass"]
                mass = alpha/time_integral(rate, 1., 0.)
                observation_change = mass*np.array([time_integral(rate, 1., at-1) for at in times])
                difference = frontier["step_amplitude"]*mass*np.exp(-rate*delay)
                lower = alpha*frontier["step_amplitude"]/(np.e*delay*(-np.expm1(-1/delay)))
                upper = (observations[0]+spec["response_error_bound"])*frontier["step_amplitude"]*(1/(np.e*delay)+1.)
                assert np.max(observation_change) < spec["response_error_bound"]
                assert abs(difference-lower) < 2e-12 and lower <= upper
                frontier_rows.append({"truth": info["name"], "delay": delay, "rate": rate,
                    "physical_added_mass": mass, "max_observation_change": float(np.max(observation_change)),
                    "compatible_pair_target_difference": difference, "diameter_lower_bound": lower,
                    "diameter_upper_bound": upper, "delay_times_difference": delay*difference})
    assert max(residuals.values()) <= spec["lp_replay_tolerance"]
    report = {"status": "PASS", "result_sha256": digest(OUTPUT), "audit_source_sha256": digest(__file__),
              "theory_sha256": digest(HERE.parent / "docs/P4_OBSERVATION_WEIGHTED_PREDICTION_THEORY_20261011.md"),
              "delay_frontier_spec_sha256": digest(FRONTIER_SPEC), "LPs_replayed": len(result["lp_logs"]),
              "max_residuals": residuals, "independent_reference_target_checks": target_checks,
              "physical_inner_witness_checks": 2*target_checks, "fast_witness_target_checks": witness_checks,
              "delay_frontier_rows": frontier_rows,
              "scope": "Same-agent local floating-point replay and alternative loading-time algebra; not external proof review, noise calibration or experimental material validation"}
    AUDIT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "delay_frontier_rows"}, indent=2))


if __name__ == "__main__":
    main()
