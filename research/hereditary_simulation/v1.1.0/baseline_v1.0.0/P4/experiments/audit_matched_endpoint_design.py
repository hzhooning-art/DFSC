"""Replay saved LPs; independently integrate positive time-domain loadings."""
from functools import lru_cache
import json
import numpy as np
from scipy.integrate import quad
from matched_endpoint_design import ROOT, SPEC, OUT, ARCHIVE, representation, compare
from observation_weighted_prediction import digest, integrate_truth

AUDIT = OUT.with_name("p4_matched_endpoint_design_audit_20261011.json")


def time_profile(t, design):
    if design == "constant":
        return 1.
    if design == "early_linear_endpoint":
        return 3*(1-t)*(t*t+(1-t)**2)
    if design == "late_linear_start":
        return 3*t*(t*t+(1-t)**2)
    if design == "early_quadratic_endpoint":
        return 3*(1-t)**2
    if design == "late_quadratic_start":
        return 3*t*t
    raise ValueError(design)


@lru_cache(None)
def direct_kernel(rate, design):
    if rate == 0:
        return quad(lambda t: time_profile(t, design), 0, 1, epsabs=1e-13)[0]
    # Change variable for rapid endpoint layers; omitted tail at 80 has
    # absolute error <=3*exp(-80)/rate, negligible here, not directed rounding.
    end = min(float(rate), 80.)
    val = quad(lambda z: np.exp(-z)*time_profile(1-z/rate, design), 0, end,
               epsabs=2e-13, epsrel=2e-13, points=[z for z in (1, 8, 32) if z < end])[0]
    return val/rate


def direct_target(rate, delay):
    return np.exp(-delay*rate)*(1. if rate == 0 else 2*(-np.expm1(-.5*rate))/rate)


def main():
    if AUDIT.exists():
        raise FileExistsError("Frozen matched-design audit exists")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    result = json.loads(OUT.read_text(encoding="utf-8"))
    parent_path = SPEC.parent/spec["parent_truth_spec"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    assert result["spec_sha256"] == digest(SPEC)
    assert result["source_sha256"] == digest(ROOT/"experiments/matched_endpoint_design.py")
    assert result["archive_sha256"] == digest(ARCHIVE)
    assert result["parent_truth_spec_sha256"] == digest(parent_path)
    data = {(d["truth"], d["design"], d["band_mode"]): d for d in result["data"]}
    truth_map = {v["name"]: v for v in parent["truths"]}
    maximum = {"primal": 0., "stationarity": 0., "gap": 0., "sign": 0.,
               "physical_observation_excess": 0., "physical_target_error": 0.}
    source_checks, physical, lp_count, samples = 0, 0, 0, 0
    reps = {}
    for key, row in data.items():
        info = truth_map[key[0]]
        for d, observed in zip(spec["observation_delays"], row["observations"]):
            direct = integrate_truth(info, lambda rate: direct_kernel(rate, key[1])*np.exp(-d*rate))
            assert abs(direct-observed) < 2e-11
            source_checks += 1
        expected_band = spec["absolute_response_band"] if key[2] == "absolute" else spec["relative_anchor_band"]*row["observations"][0]
        assert abs(row["band"]-expected_band) < 1e-14
    with np.load(ARCHIVE, allow_pickle=False) as saved:
        for row in result["rows"]:
            design, target, cells = row["design"], row["target"], row["cells"]
            key = (design, target, cells)
            if key not in reps:
                rep = representation(spec, design, target, cells)
                reps[key] = rep
                # Selected log-rate interiors and the infinite-cell tail.
                for index in np.linspace(0, cells-2, 96, dtype=int):
                    a, b = rep["edges"][index:index+2]
                    for rate in (.75*a+.25*b, .25*a+.75*b):
                        ratio = direct_target(rate, spec["targets"][target])/direct_kernel(rate, design)
                        assert rep["q_low"][index]-2e-10 <= ratio <= rep["q_high"][index]+2e-10
                        samples += 1
                for rate in (spec["tail_start"], spec["tail_start"]*1.3, spec["tail_start"]*4):
                    ratio = direct_target(rate, spec["targets"][target])/direct_kernel(rate, design)
                    assert rep["q_low"][-1]-2e-10 <= ratio <= rep["q_high"][-1]+2e-10
                    samples += 1
            rep = reps[key]
            datum = data[row["truth"], design, row["band_mode"]]
            observations = np.array(datum["observations"])
            rhs = np.r_[observations+datum["band"], -observations+datum["band"]]
            expected_target = integrate_truth(truth_map[row["truth"]], lambda rate: direct_target(rate, spec["targets"][target]))
            assert abs(expected_target-row["truth_target"]) < 2e-11
            stem = f"{row['truth']}_{design}_{row['band_mode']}_{target}_{cells}"
            for method, a, costs in (("outer", rep["outer_a"], (rep["q_low"], -rep["q_high"])),
                                     ("inner", rep["inner_a"], (rep["inner_q"], -rep["inner_q"]))):
                for side_i, (side, c) in enumerate(zip(("low", "high"), costs)):
                    label = stem+"_"+method+"_"+side
                    x, y, z = [saved[label+"_"+k] for k in ("x", "y", "z")]
                    maximum["primal"] = max(maximum["primal"], float(max(0., np.max(a@x-rhs), -np.min(x))))
                    maximum["stationarity"] = max(maximum["stationarity"], float(np.max(np.abs(c-a.T@y-z))))
                    maximum["gap"] = max(maximum["gap"], float(abs(c@x-rhs@y)))
                    maximum["sign"] = max(maximum["sign"], float(max(0., np.max(y), -np.min(z))))
                    assert abs(float(c@x)*(1 if side_i == 0 else -1)-row[method+"_interval"][side_i]) < 2e-10
                    lp_count += 1
                    if method == "inner":
                        active = np.flatnonzero(x > 0)
                        rates = rep["nodes"][active]
                        # Derive physical mass from the weighted measure using
                        # independent quadrature, then replay physical response.
                        masses = np.array([x[k]/direct_kernel(float(rep["nodes"][k]), design) for k in active])
                        actual = np.array([sum(m*direct_kernel(float(rate), design)*np.exp(-d*rate) for m, rate in zip(masses, rates)) for d in spec["observation_delays"]])
                        maximum["physical_observation_excess"] = max(maximum["physical_observation_excess"], float(np.max(np.abs(actual-observations))-datum["band"]))
                        actual_target = float(sum(m*direct_target(rate, spec["targets"][target]) for m, rate in zip(masses, rates)))
                        maximum["physical_target_error"] = max(maximum["physical_target_error"], abs(actual_target-row["inner_interval"][side_i]))
                        assert np.all(np.isfinite(masses)) and np.all(masses >= 0)
                        physical += 1
    assert max(maximum.values()) <= spec["lp_replay_tolerance"]
    for w in result["physical_unboundedness_examples"]:
        datum = data[w["truth"], w["design"], w["band_mode"]]
        change = w["physical_mass"]*direct_kernel(w["rate"], w["design"])
        assert abs(change-datum["band"]/2) < 2e-11
        assert change < datum["band"]
        target = w["physical_mass"]*direct_target(w["rate"], 0.)
        assert abs(target-w["active_target_change"]) < 2e-10*(1+target)
    assert compare(result["rows"], spec) == result["summaries"]
    # A common positive dead time breaks the active-target bound for all
    # designs, even keeping the other original observation times and horizon.
    dead = spec["negative_control_dead_time"]
    control = []
    observation_lags = [dead]+spec["observation_delays"][1:]
    for design in spec["designs"]:
        for rate in spec["adversarial_rates"]:
            alpha = spec["absolute_response_band"]/2
            mass = alpha*np.exp(dead*rate)/direct_kernel(rate, design)
            changes = [mass*direct_kernel(rate, design)*np.exp(-lag*rate) for lag in observation_lags]
            assert max(changes) < spec["absolute_response_band"]
            control.append({"design": design, "rate": rate, "physical_mass": mass,
                            "max_observation_change": float(max(changes)),
                            "active_target_change": float(mass*direct_target(rate, 0.))})
    report = {"schema": "P4-matched-endpoint-design-audit-v1", "result_sha256": digest(OUT),
              "audit_source_sha256": digest(__file__), "LPs_replayed": lp_count,
              "independent_observation_checks": source_checks, "physical_inner_endpoints": physical,
              "independent_cell_ratio_checks": samples, "unboundedness_examples_replayed": len(result["physical_unboundedness_examples"]),
              "dead_time_control_witnesses": control, "max_residuals": maximum, "all_checks_pass": True,
              "limits": "Same developer, alternative quadrature and saved LP replay; no external review, random validation, measured calibration or directed rounding"}
    AUDIT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "dead_time_control_witnesses"}), flush=True)


if __name__ == "__main__":
    main()
