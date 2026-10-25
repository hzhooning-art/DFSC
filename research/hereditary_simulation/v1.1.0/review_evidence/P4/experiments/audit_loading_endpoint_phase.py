"""Independent 60-digit quadrature and physical-pair replay; no SciPy kernels."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "experiments/specs/loading_endpoint_phase_20261011.json"
RESULT = ROOT / "results/p4_loading_endpoint_phase_20261011.json"
OUT = ROOT / "results/p4_loading_endpoint_phase_audit_20261011.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(None)
def direct_kernel(rate, order, dead, duration):
    rate, dead, duration = map(lambda x: mp.mpf(str(x)), (rate, dead, duration))
    if order is None:
        return mp.exp(-dead*rate)
    if rate == 0:
        return duration**(order+1)/(order+1)
    end = rate*duration
    cuts = sorted(set([mp.mpf(0), end]+[min(end, mp.mpf(v)) for v in (1, 8, 32, 128)]))
    integral = mp.quad(lambda y: mp.exp(-y)*y**order, cuts)
    return mp.exp(-dead*rate)*integral/rate**(order+1)


def main():
    if OUT.exists():
        raise FileExistsError("Refuse to overwrite frozen audit")
    mp.mp.dps = 60
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["spec_sha256"] == sha(SPEC)
    assert result["source_sha256"] == sha(ROOT / "experiments/loading_endpoint_phase.py")
    tol = spec["numeric_relative_tolerance"]
    max_rel, max_band, kernel_count, pairs, targets, upper_checks = 0., 0., 0, 0, 0, 0

    def check_relative(saved, reference):
        nonlocal max_rel
        err = float(abs(mp.mpf(str(saved))/reference-1))
        max_rel = max(max_rel, err)
        assert err < tol

    for row in result["kernel_checks"]:
        direct = direct_kernel(row["rate"], row["reference_order"], row["reference_dead_time"], spec["duration"])
        check_relative(row["kernel"], direct)
        kernel_count += 1
    by_key = {(d["reference_order"], d["reference_dead_time"]): d for d in result["data"]}
    for (p, dead), row in by_key.items():
        for delay, saved in zip(spec["observation_delays"], row["observations"]):
            direct = sum(mp.mpf(str(w))*direct_kernel(rate, p, dead+delay, spec["duration"])
                         for rate, w in zip(spec["base_rates"], spec["base_weights"]))
            check_relative(saved, direct)

    def replay(row, w, target_dead):
        nonlocal max_band, pairs, targets
        p, dead, rate = row["reference_order"], row["reference_dead_time"], w["rate"]
        mass = mp.mpf(str(w["physical_added_mass"]))
        for delay, saved in zip(spec["observation_delays"], w["observation_changes"]):
            change = mass*direct_kernel(rate, p, dead+delay, spec["duration"])
            max_band = max(max_band, float(change))
            assert change < spec["declared_response_band"]
            # Very late responses may underflow in double; absolute check then.
            assert abs(change-mp.mpf(str(saved))) < tol*spec["added_weighted_mass"]
        target = mass*direct_kernel(rate, row["target_order"], target_dead, spec["duration"])
        check_relative(w["target_change"], target)
        check_relative(w["deterministic_minimax_pair_lower"], target/2)
        pairs += 1
        targets += 1

    for row in result["phases"]:
        for w in row["witnesses"]:
            replay(row, w, row["target_dead_time"])
            if row["proved_bounded"]:
                ratio = mp.mpf(str(w["target_change"]))/spec["added_weighted_mass"]
                assert ratio <= row["ratio_whole_half_line_upper"]*(1+tol)
                upper_checks += 1
    for row in result["frontiers"]:
        a = row["proved_exponent"]
        gamma = 0 if row["target_order"] is None else row["target_order"]+1
        limit = mp.mpf(str(spec["added_weighted_mass"]))*mp.gamma(max(gamma, 1))/mp.gamma(row["reference_order"]+1)*(mp.mpf(a)/mp.e)**a
        check_relative(row["scaled_pair_limit"], limit)
        for w in row["records"]:
            replay(row, w, row["reference_dead_time"]+w["gap"])
            check_relative(w["scaled_target_change"], mp.mpf(str(w["target_change"]))*mp.mpf(str(w["gap"]))**a)
            assert w["target_change"] <= w["diameter_whole_class_upper"]*(1+tol)
            upper_checks += 1
        last = row["records"][-1]
        assert abs(mp.mpf(str(last["scaled_target_change"]))/limit-1) < spec["frontier_relative_limit_tolerance"]
    output = {"schema": "P4-loading-endpoint-phase-audit-v1", "result_sha256": sha(RESULT),
              "source_sha256": sha(__file__), "precision_digits": mp.mp.dps,
              "quadrature_kernel_checks": kernel_count, "base_observation_checks": len(by_key)*len(spec["observation_delays"]),
              "physical_pair_checks": pairs, "physical_target_checks": targets,
              "whole_class_upper_comparisons": upper_checks, "max_relative_error": max_rel,
              "max_observation_change": max_band, "all_checks_pass": True,
              "limits": "Independent numerical implementation only; same developer, not external review or an exhaustive proof/novelty certificate"}
    OUT.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
