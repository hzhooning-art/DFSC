"""Frozen finite-library P4 target-loading headroom screen.

Pair LPs bracket the worst possible posterior target diameter conditional on
a noisy pilot. The point-grid inner problem gives a lower bound, never a
continuum coverage interval. This implements established set-membership
design as a feasibility comparator, not a new algorithm claim.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from calibrated_multiload_gate import basis
from constitutive_response_gate import strain, stress_truth_integral
from offgrid_curvature_envelope_gate import response_curvature_envelope

HERE = Path(__file__).resolve().parent
SPEC = HERE / "specs" / "target_loading_headroom_gate_20261011.json"
OUTPUT = HERE.parent / "results" / "p4_target_loading_headroom_gate_20261011.json"
ARCHIVE = OUTPUT.with_suffix(".npz")
TARGETS = ("G20", "heldout_sigma20")


def modulus_for(info):
    def evaluate(t):
        x = np.asarray(t)
        if "rates" in info:
            value = info["equilibrium"] + np.exp(
                -x[..., None] * np.array(info["rates"])) @ np.array(info["weights"])
        else:
            a, b = info["rate_low"], info["rate_high"]
            value = info["equilibrium"] + info["transient_mass"] * np.where(
                x == 0, 1.0, np.exp(-a*x) * (-np.expm1(-(b-a)*x)) /
                ((b-a)*np.where(x == 0, 1.0, x)))
        return float(value) if x.ndim == 0 else value
    return evaluate


def build_library(spec):
    histories = {k: [tuple(s) for s in v] for k, v in spec["histories"].items()}
    x = np.linspace(0, 1, spec["confirmation_count_each"])
    times = {"uniform": .25 + 7.75*x, "early": .25 + 7.75*x*x,
             "late": .25 + 7.75*np.sqrt(x)}
    library = {f"{h}__{rule}": [(t, histories["ramp_hold"]), (t, pieces)]
               for h, pieces in histories.items() for rule, t in times.items()}
    for pieces in histories.values():
        corners = sorted({0.0, 8.0, *(v for s in pieces for v in s[:2])})
        eps = strain(np.array(corners), pieces)
        max_rate = max(abs(sum(v for a, b, v in pieces if a <= (l+r)/2 < b))
                       for l, r in zip(corners[:-1], corners[1:]))
        cost = spec["cost"]
        assert eps.min() >= cost["minimum_strain"] - 1e-12
        assert eps.max() <= cost["maximum_strain"] + 1e-12
        assert max_rate <= cost["maximum_absolute_strain_rate"] + 1e-12
        assert sum(abs(v)*(b-a) for a, b, v in pieces) <= cost[
            "maximum_total_variation_per_run"] + 1e-12
        assert all(0 <= a < b <= cost["duration_per_run"] for a, b, _ in pieces)
    return histories, library


def matrices(spec, library, heldout, outer):
    edges = np.geomspace(*spec["rate_support"], spec["outer_cell_count"]+1)
    rates = (np.column_stack((edges[:-1], edges[1:])).ravel() if outer else
             np.geomspace(*spec["rate_support"], spec["inner_rate_count"]))
    def block(pieces):
        a = np.vstack([basis(t, s, rates) for t, s in pieces])
        if outer:
            e = np.vstack([response_curvature_envelope(t, s, edges) for t, s in pieces])
            e = np.column_stack((np.zeros(len(e)), np.repeat(e, 2, axis=1)))
        else:
            e = np.zeros_like(a)
        return a, e
    q = {"G20": np.r_[1., np.exp(-20*rates)],
         "heldout_sigma20": basis(np.array([20.]), heldout, rates)[0]}
    f = {k: np.zeros_like(v) for k, v in q.items()}
    if outer:
        eg = np.diff(edges)**2/8 * (400*np.exp(-20*edges[:-1])*(1+1e-10)+1e-11)
        es = response_curvature_envelope(np.array([20.]), heldout, edges)[0]
        f = {"G20": np.r_[0., np.repeat(eg, 2)],
             "heldout_sigma20": np.r_[0., np.repeat(es, 2)]}
    return rates, q, f, block


def pair_problem(ap, ep, yp, af, ef, b, q, f):
    """Each measure fits pilot; future observation bands must intersect."""
    n = ap.shape[1]
    cp = np.vstack((ap-ep, -ap-ep))
    z = np.zeros_like(cp)
    a = np.vstack((np.column_stack((cp, z)), np.column_stack((z, cp)),
                   np.column_stack((af-ef, -af-ef)),
                   np.column_stack((-af-ef, af-ef))))
    d = np.r_[yp+b, -yp+b, yp+b, -yp+b, np.full(2*len(af), 2*b)]
    eq = np.zeros((2, 2*n)); eq[0, :n] = 1.; eq[1, n:] = 1.
    # Symmetry under exchange of the two measures covers both difference signs.
    c = -np.r_[q+f, -q+f]
    return c, a, d, eq, np.ones(2)


def posterior_problem(a, e, y, b, q, f, lower):
    c = q-f if lower else -(q+f)
    return c, np.vstack((a-e, -a-e)), np.r_[y+b, -y+b], np.ones((1, len(q))), np.ones(1)


def verify_lp(problem, x, yu, ye, zl):
    c, a, d, eq, rhs = problem
    primal = float(c@x); dual = float(d@yu + rhs@ye)
    diagnostics = {
        "inequality": float(max(0., np.max(a@x-d))),
        "equality": float(np.max(np.abs(eq@x-rhs))),
        "negative_mass": float(max(0., -np.min(x))),
        "stationarity": float(np.max(np.abs(c-a.T@yu-eq.T@ye-zl))),
        "dual_sign": float(max(0., np.max(yu), -np.min(zl))),
        "duality_gap": float(abs(primal-dual)),
    }
    return diagnostics, primal, dual


def solve(problem, label, archive, logs):
    c, a, d, eq, rhs = problem
    options = {"primal_feasibility_tolerance": 1e-10,
               "dual_feasibility_tolerance": 1e-10}
    for method in ("highs-ds", "highs-ipm"):
        answer = linprog(c, A_ub=a, b_ub=d, A_eq=eq, b_eq=rhs,
                         bounds=(0., None), method=method, options=options)
        if answer.status == 2:
            return None
        if not answer.success:
            continue
        diagnostic, primal, dual = verify_lp(problem, answer.x,
            answer.ineqlin.marginals, answer.eqlin.marginals, answer.lower.marginals)
        if max(diagnostic.values()) > 2e-7:
            continue
        for name, vector in (("x", answer.x), ("yu", answer.ineqlin.marginals),
                             ("ye", answer.eqlin.marginals), ("zl", answer.lower.marginals)):
            archive[f"{label}_{name}"] = vector
        logs.append({"label": label, "method": method, "primal": primal,
                     "dual": dual, "diagnostics": diagnostic})
        return primal, dual
    raise RuntimeError(f"LP failed primal/dual replay: {label}: {answer.message}")


def information_choices(spec, pilot, library, heldout):
    rates = np.geomspace(*spec["rate_support"], spec["information_baseline_grid_count"])
    ap = basis(pilot[0][0], pilot[0][1], rates)
    bp = ap[:, 1:]-ap[:, :1]
    q = {"G20": np.r_[1., np.exp(-20*rates)],
         "heldout_sigma20": basis(np.array([20.]), heldout, rates)[0]}
    cs = {k: (v[1:]-v[0])/spec["utility_limits"][k] for k, v in q.items()}
    scores = {}
    for name, pieces in library.items():
        af = np.vstack([basis(t, s, rates) for t, s in pieces])
        bf = af[:, 1:]-af[:, :1]
        m = (bp.T@bp + bf.T@bf)/spec["known_noise_amplitude"]**2
        m += np.eye(len(rates))*spec["information_baseline_ridge"]
        eigen, vectors = np.linalg.eigh(m)
        eigen = np.maximum(eigen, spec["information_baseline_ridge"])
        scores[name] = {"D": float(-np.log(eigen).sum()),
                        "c": max(float(np.sum((vectors.T@v)**2/eigen)) for v in cs.values())}
    return {k: min(scores, key=lambda n: (scores[n][k], n)) for k in ("D", "c")}, scores


def main():
    if OUTPUT.exists() or ARCHIVE.exists():
        raise FileExistsError("Frozen output already exists; do not overwrite research outcomes")
    start = time.perf_counter()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    parent_path = HERE / "specs" / spec["parent_spec"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    heldout = [tuple(x) for x in parent["heldout_input_segments"]]
    histories, library = build_library(spec)
    pilot = [(np.linspace(.5, 8, spec["pilot_count"]), histories["ramp_hold"])]
    assert len(library) == 18
    assert spec["pilot_count"] + 2*spec["confirmation_count_each"] == spec["total_response_observations"]
    representations = {}
    for rep in ("inner", "outer"):
        rates, q, f, block = matrices(spec, library, heldout, rep == "outer")
        representations[rep] = {"rates": rates, "q": q, "f": f,
            "pilot": block(pilot), "future": {k: block(v) for k, v in library.items()}}
    info_choices, info_scores = information_choices(spec, pilot, library, heldout)
    archive, logs, rows, cases = {}, [], [], []
    b = spec["known_noise_amplitude"]
    truth_targets = {}
    # Candidate builders use no true parameters. Truth only generates input/output data.
    for ti, (truth, details) in enumerate(spec["truths"].items()):
        modulus = modulus_for(details)
        assert abs(modulus(0.)-1.) < 1e-12
        truth_targets[truth] = {"G20": modulus(20.), "heldout_sigma20": float(
            stress_truth_integral(np.array([20.]), heldout, modulus)[0])}
        clean_pilot = stress_truth_integral(pilot[0][0], pilot[0][1], modulus)
        cache = {}
        def clean_future(pieces):
            answers = []
            for t, segments in pieces:
                key = (tuple(t), tuple(segments))
                if key not in cache:
                    cache[key] = stress_truth_integral(t, segments, modulus)
                answers.append(cache[key])
            return np.concatenate(answers)
        clean = {k: clean_future(v) for k, v in library.items()}
        for seed in range(spec["replicates_per_truth"]):
            case_id = len(cases)
            pilot_rng = np.random.default_rng(np.random.SeedSequence(
                [spec["seed_base"], ti, seed, 0]))
            yp = clean_pilot + pilot_rng.uniform(-b, b, len(clean_pilot))
            archive[f"case{case_id}_pilot"] = yp
            assert np.max(np.abs(yp-clean_pilot)) <= b
            candidate_rows = []
            for ci, (name, pieces) in enumerate(library.items()):
                bounds = {"inner": {}, "outer": {}}
                for rep, data in representations.items():
                    ap, ep = data["pilot"]; af, ef = data["future"][name]
                    for target in TARGETS:
                        problem = pair_problem(ap, ep, yp, af, ef, b,
                                               data["q"][target], data["f"][target])
                        label = f"pair_{case_id}_{ci}_{rep}_{target}"
                        solved = solve(problem, label, archive, logs)
                        if solved is None:
                            raise RuntimeError(f"Infeasible {rep} pair bracket: {label}; no gate conclusion")
                        primal, dual = solved
                        # Conservative numerical allowance, not outward-rounded arithmetic.
                        bounds[rep][target] = max(0., -primal-1e-8 if rep == "inner" else -dual+1e-8)
                assert all(bounds["inner"][t] <= bounds["outer"][t]+2e-7 for t in TARGETS)
                score = {r: max(bounds[r][t]/spec["utility_limits"][t] for t in TARGETS)
                         for r in bounds}
                item = {"case": case_id, "truth": truth, "seed": seed,
                        "candidate_index": ci, "candidate": name, "diameter": bounds, "score": score}
                rows.append(item); candidate_rows.append(item)
            selected = min(candidate_rows, key=lambda r: (r["score"]["outer"], r["candidate"]))
            baseline = next(r for r in candidate_rows if r["candidate"] == spec["fixed_baseline"])
            choices = {"set_minimax": selected["candidate"], "fixed_cycle": spec["fixed_baseline"],
                       "fixed_ramp": "ramp_hold__uniform", "D_optimal": info_choices["D"],
                       "task_c_optimal": info_choices["c"]}
            # All candidates use the same two-run noise vector for a paired replay.
            # These outcomes are generated only after the design is chosen.
            future_rng = np.random.default_rng(np.random.SeedSequence(
                [spec["seed_base"], ti, seed, 1]))
            future_noise = future_rng.uniform(-b, b, 2*spec["confirmation_count_each"])
            posterior = {}
            for name in dict.fromkeys(choices.values()):
                ci = list(library).index(name)
                y = np.r_[yp, clean[name]+future_noise]
                archive[f"case{case_id}_future_{ci}"] = y[len(yp):]
                data = representations["outer"]
                ap, ep = data["pilot"]; af, ef = data["future"][name]
                a, e = np.vstack((ap, af)), np.vstack((ep, ef))
                intervals = {}
                for target in TARGETS:
                    values = []
                    for lower in (True, False):
                        problem = posterior_problem(a, e, y, b, data["q"][target], data["f"][target], lower)
                        label = f"post_{case_id}_{ci}_{target}_{int(lower)}"
                        solution = solve(problem, label, archive, logs)
                        if solution is None:
                            raise RuntimeError("Supported truth posterior unexpectedly infeasible")
                        _, dual = solution
                        values.append(dual-1e-8 if lower else -dual+1e-8)
                    if target == "G20":
                        values = [max(0., values[0]), min(1., values[1])]
                    real = truth_targets[truth][target]
                    assert values[0]-2e-7 <= real <= values[1]+2e-7
                    # Outer-pair supremum must cover the width of this same outer posterior.
                    bound_row = candidate_rows[ci]
                    assert values[1]-values[0] <= bound_row["diameter"]["outer"][target]+2e-7
                    intervals[target] = {"interval": values, "width": values[1]-values[0], "contains_truth": True}
                posterior[name] = intervals
            cases.append({"id": case_id, "truth": truth, "seed": seed,
                "selected": selected["candidate"], "selected_outer_score": selected["score"]["outer"],
                "best_inner_score": min(r["score"]["inner"] for r in candidate_rows),
                "baseline_inner_score": baseline["score"]["inner"],
                "baseline_outer_score": baseline["score"]["outer"],
                "selected_score_gap": selected["score"]["outer"]-selected["score"]["inner"],
                "policy_choices": choices, "posterior": posterior})
            print(json.dumps({"completed_case": case_id+1, "truth": truth, "seed": seed,
                "selected": selected["candidate"], "selected_upper_score": selected["score"]["outer"],
                "baseline_lower_score": baseline["score"]["inner"]}), flush=True)
    summaries = {}
    for truth in spec["truths"]:
        group = [c for c in cases if c["truth"] == truth]
        ratio = float(np.mean([c["selected_outer_score"] for c in group]) /
                      np.mean([c["baseline_inner_score"] for c in group]))
        optimistic_ratio = float(np.mean([c["best_inner_score"] for c in group]) /
                                 np.mean([c["baseline_outer_score"] for c in group]))
        summaries[truth] = {"selected_upper_to_fixed_lower_ratio": ratio,
            "best_lower_to_fixed_upper_ratio": optimistic_ratio,
            "clear_10_percent_headroom": ratio <= spec["headroom_ratio_max"],
            "no_10_percent_headroom_within_library": optimistic_ratio >= spec["headroom_ratio_max"],
            "selected_outer_score_mean": float(np.mean([c["selected_outer_score"] for c in group])),
            "baseline_inner_score_mean": float(np.mean([c["baseline_inner_score"] for c in group])),
            "selected_score_gap_mean": float(np.mean([c["selected_score_gap"] for c in group])),
            "selection_counts": {n: sum(c["selected"] == n for c in group)
                                  for n in sorted({c["selected"] for c in group})},
            "realized_width_means": {policy: {t: float(np.mean([
                c["posterior"][c["policy_choices"][policy]][t]["width"] for c in group]))
                for t in TARGETS} for policy in group[0]["policy_choices"]}}
    passing = [t for t, v in summaries.items() if v["clear_10_percent_headroom"]]
    gate = len(passing) >= 2 and "continuous_uniform" in passing
    np.savez_compressed(ARCHIVE, **archive)
    files = [SPEC, parent_path, Path(__file__), HERE/"calibrated_multiload_gate.py",
             HERE/"constitutive_response_gate.py", HERE/"offgrid_curvature_envelope_gate.py"]
    result = {"scope": spec["scope"], "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        "source_hashes": {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "witness_archive": ARCHIVE.name, "witness_archive_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "noise_calibration": "Exactly known bound, no independent amplitude-calibration claim",
        "candidate_count": len(library), "case_count": len(cases), "pair_row_count": len(rows),
        "solver_count": len(logs), "duration_seconds": time.perf_counter()-start,
        "truth_targets": truth_targets, "information_baseline_choices": info_choices,
        "information_scores": info_scores, "summaries": summaries, "headroom_gate_pass": gate,
        "all_posterior_targets_contained": True, "all_inner_outer_brackets_ordered": True,
        "cases": cases, "rows": rows, "lp_replay_logs": logs,
        "limitations": ["Only the frozen finite action library and 16 exploratory pilot outcomes",
            "Known stationary noise bound and known G(0), compact positive spectral support",
            "Floating-point primal/dual checks do not certify outward-rounded bounds",
            "The selector is an established set-minimax comparator, not new method innovation",
            "Worst-case diameter headroom does not imply mean-estimator or sequential superiority"]}
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summaries": summaries, "headroom_gate_pass": gate,
                      "solver_count": len(logs), "duration_seconds": result["duration_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
