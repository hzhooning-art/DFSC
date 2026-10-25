"""Loading-aware search over a FINITE list of target delays.

Polynomial reference/target profiles on [0, 1], positive spectra, no shared
mass/rate cap. Exact-envelope arguments are mathematical; LP replay below is
floating-point validation, not outward-rounded certification. Historical
experiments are imported only for the checked LP engine, never overwritten.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import linprog
from scipy.special import gammainc, gammaln

from observation_weighted_prediction import verify


class EnclosureFailure(RuntimeError):
    def __init__(self, message, programs, solver_calls, cells):
        super().__init__(message)
        self.programs, self.solver_calls, self.cells = programs, solver_calls, cells


@dataclass(frozen=True)
class Information:
    beta: float
    dead_time: float
    delays: tuple[float, ...]
    centers: tuple[float, ...]
    bands: tuple[float, ...]
    common_mass_cap: float | None = None
    common_rate_cap: float | None = None

    def __post_init__(self):
        if self.common_mass_cap is not None or self.common_rate_cap is not None:
            raise ValueError("Uncapped search cannot silently discard supplied caps")
        if (not math.isfinite(self.beta) or self.beta < 1 or
                not math.isfinite(self.dead_time) or self.dead_time < 0):
            raise ValueError("Reference order beta>=1 and nonnegative dead time required")
        if not (len(self.delays) == len(self.centers) == len(self.bands) > 0):
            raise ValueError("Inconsistent observation arrays")
        if self.delays[0] != 0 or any(d < 0 for d in self.delays):
            raise ValueError("First observation must be the dominating anchor")
        if (not all(math.isfinite(v) for v in self.delays+self.centers+self.bands)
                or min(self.bands) <= 0):
            raise ValueError("Finite observations and positive deterministic bands required")


def log_laplace(rates, order):
    """log integral_0^1 r^(order-1) exp(-lambda r) dr; order=0 is a step."""
    x = np.asarray(rates, dtype=float)
    if order < 0 or np.any(x < 0) or np.any(~np.isfinite(x)):
        raise ValueError("Finite nonnegative rates and orders required")
    if order == 0:
        return np.zeros_like(x)
    result = np.full_like(x, -np.log(order))
    mask = x > 0
    z = x[mask]
    result[mask] = gammaln(order)+np.log(gammainc(order, z))-order*np.log(z)
    return result


def ratio(rates, beta, gamma, excess):
    x = np.asarray(rates, dtype=float)
    return np.exp(-excess*x+log_laplace(x, gamma)-log_laplace(x, beta))


def bounded(info, gamma, wait):
    if not math.isfinite(gamma) or gamma < 0 or not math.isfinite(wait) or wait < 0:
        raise ValueError("Nonnegative finite target order and delay required")
    # No isclose: equality is a mathematical boundary, not a numerical band.
    return wait > info.dead_time or (wait == info.dead_time and gamma >= info.beta)


def strict_witness(info, margin_tolerance=2e-7):
    """Find a finite physical strict-feasibility witness without generating truth."""
    nodes = np.r_[0., np.geomspace(1e-6, 128., 256)]
    moments = np.exp(-np.asarray(info.delays)[:, None]*nodes)
    y, b = np.asarray(info.centers), np.asarray(info.bands)
    a = np.vstack((np.c_[moments, np.ones(len(y))],
                   np.c_[-moments, np.ones(len(y))]))
    answer = linprog(np.r_[np.zeros(len(nodes)), -1.], A_ub=a,
                     b_ub=np.r_[y+b, -y+b], bounds=(0, None), method="highs-ds",
                     options={"primal_feasibility_tolerance": 1e-10,
                              "dual_feasibility_tolerance": 1e-10})
    if not answer.success:
        return None
    weights = answer.x[:-1]
    slack = b-np.abs(moments@weights-y)
    eta = float(slack.min()-margin_tolerance)
    if eta <= margin_tolerance or np.min(weights) < -margin_tolerance:
        return None
    keep = weights > 0
    return {"nodes": nodes[keep], "weighted_masses": weights[keep],
            "margin": eta, "alpha": eta/2,
            "observed_responses": (moments@weights).tolist(),
            "maximum_log_physical_mass": float(np.max(
                np.log(weights[keep])+info.dead_time*nodes[keep]
                -log_laplace(nodes[keep], info.beta)))}


def rate_witness(info, gamma, wait, feasible):
    """Finite-scale evaluation at the rate suggested by the recovery exponent.

    The increment is evaluated using the actual kernels; a Theta constant is
    never inferred. Physical added mass is stored in logarithmic form because
    it can exceed floating-point range while still being mathematically finite.
    """
    if feasible is None or not bounded(info, gamma, wait):
        return None
    excess, a = wait-info.dead_time, info.beta-gamma
    if excess <= 0 or a <= 0:
        return None
    rate = a/excess
    increment = float(feasible["alpha"]*ratio(rate, info.beta, gamma, excess))
    return {"rate": rate, "excess_wait": excess, "exponent": a,
            "target_increment": increment,
            "added_weighted_mass": feasible["alpha"],
            "log_added_physical_mass": float(np.log(feasible["alpha"])
                +info.dead_time*rate-log_laplace(rate, info.beta)),
            "maximum_observation_increment": feasible["alpha"]}


def representation(info, gamma, wait, cells):
    """Classical interval relaxation including the entire infinite-rate tail."""
    if not bounded(info, gamma, wait):
        raise ValueError("An unbounded target must not enter a finite LP")
    if cells < 16:
        raise ValueError("At least 16 cells required")
    excess, a = wait-info.dead_time, info.beta-gamma
    peak = a/excess if a > 0 and excess > 0 else 1.
    tail = max(128., 4*peak)
    # Rate law locates a potentially important scale; all cells and tail still
    # receive whole-interval bounds. There is no assumed target monotonicity.
    edges = np.unique(np.r_[0., np.geomspace(1e-6, tail, cells),
                             peak, np.inf])
    left, right = edges[:-2], edges[1:-1]
    # All positive polynomial Laplace factors decrease with rate.
    low = np.exp(-excess*right+log_laplace(right, gamma)
                 -log_laplace(left, info.beta))
    high = np.exp(-excess*left+log_laplace(left, gamma)
                  -log_laplace(right, info.beta))
    # For lambda>=tail: F_beta >= Gamma(beta) P(beta,tail) lambda^-beta;
    # target <= Gamma(gamma) lambda^-gamma (or 1 for an ideal step).
    coefficient = np.exp((gammaln(gamma) if gamma else 0.)
                         -gammaln(info.beta))/gammainc(info.beta, tail)
    maximizer = max(tail, a/excess) if a > 0 and excess > 0 else tail
    tail_high = coefficient*maximizer**a*np.exp(-excess*maximizer)
    qlow = np.r_[low, 0.]
    qhigh = np.r_[high, tail_high]
    if gamma == info.beta:
        # The polynomial loading factors cancel exactly. In particular the
        # zero-excess tail limit is ONE, not zero; retaining a zero tail lower
        # bound there would prevent convergence of the endpoint bracket.
        qlow = np.r_[np.exp(-excess*right), 1. if excess == 0 else 0.]
        qhigh = np.exp(-excess*edges[:-1])
    # Tiny padding is a numerical allowance, not directed rounding.
    qlow = np.maximum(0., qlow-1e-12)
    qhigh += 1e-12
    d = np.asarray(info.delays)[:, None]
    obs_high = np.exp(-d*edges[:-1])
    obs_low = np.empty_like(obs_high)
    obs_low[:, :-1] = np.exp(-d*edges[1:-1])
    obs_low[:, -1] = (np.asarray(info.delays) == 0)
    nodes = edges[:-1]
    exact = np.exp(-d*nodes)
    return {"edges": edges, "nodes": nodes,
            "outer_a": np.vstack((obs_low, -obs_high)),
            "inner_a": np.vstack((exact, -exact)),
            "q_low": qlow, "q_high": qhigh,
            "inner_q": ratio(nodes, info.beta, gamma, excess)}


def enclose(info, gamma, wait, cells, label, archive, logs, tolerance=2e-7):
    rep = representation(info, gamma, wait, cells)
    rhs = np.r_[np.asarray(info.centers)+info.bands,
                -np.asarray(info.centers)+info.bands]
    programs = solver_calls = 0
    def checked(c, a, suffix):
        nonlocal programs, solver_calls
        programs += 1
        errors = []
        for method in ("highs-ds", "highs-ipm"):
            solver_calls += 1
            answer = linprog(c, A_ub=a, b_ub=rhs, bounds=(0., None), method=method,
                             options={"primal_feasibility_tolerance": 1e-10,
                                      "dual_feasibility_tolerance": 1e-10})
            checks = verify(c, a, rhs, answer) if answer.success else None
            if answer.success and max(checks.values()) <= tolerance:
                key = label+suffix
                for name, value in (("x", answer.x), ("y", answer.ineqlin.marginals),
                                    ("z", answer.lower.marginals)):
                    archive[f"{key}_{name}"] = value
                value = float(c@answer.x)
                logs.append({"label": key, "objective": value, "checks": checks,
                             "method": method, "failed_attempts": errors})
                return value
            errors.append({"method": method, "status": int(answer.status),
                           "message": answer.message, "checks": checks})
        raise EnclosureFailure(str(errors), programs, solver_calls, len(rep["nodes"]))
    ol = checked(rep["q_low"], rep["outer_a"], "_ol")
    oh = -checked(-rep["q_high"], rep["outer_a"], "_oh")
    il = checked(rep["inner_q"], rep["inner_a"], "_il")
    ih = -checked(-rep["inner_q"], rep["inner_a"], "_ih")
    if not ol-tolerance <= il <= ih+tolerance <= oh+2*tolerance:
        raise RuntimeError("Inner/outer endpoint ordering failed")
    return {"outer": [max(0., ol-tolerance), oh+tolerance],
            "inner": [il+tolerance, ih-tolerance],
            "cells": len(rep["nodes"]), "requested_cells": cells,
            "endpoint_gap": max(il-ol, oh-ih), "programs": programs,
            "solver_calls": solver_calls}


def decision(bracket, abs_error, relative_error=None, rho=0., margin=2e-7):
    if abs_error <= 0 or not math.isfinite(abs_error):
        raise ValueError("Positive finite absolute tolerance required")
    if not 0 <= rho < 1 or (relative_error is not None and
                           not 0 < relative_error < 1):
        raise ValueError("Invalid relative envelope or tolerance")
    def measures(interval):
        l, u = interval[0]/(1+rho), interval[1]/(1-rho)
        radius = (u-l)/2
        relative = (u-l)/(u+l) if l > 0 else 1.
        return l, u, radius, relative
    outer, inner = measures(bracket["outer"]), measures(bracket["inner"])
    def allowed(values, absolute, relative):
        l, u = values[:2]
        lower, upper = u-absolute, l+absolute
        if relative is not None:
            lower, upper = max(lower, (1-relative)*u), min(upper, (1+relative)*l)
        return lower, upper
    # One estimate must meet BOTH tolerances. Separate arithmetic/harmonic
    # midpoint tests alone do not establish this joint requirement.
    outer_allowed = allowed(outer, abs_error-margin,
                            relative_error-margin if relative_error is not None else None)
    inner_allowed = allowed(inner, abs_error+margin,
                            relative_error+margin if relative_error is not None else None)
    fail = inner_allowed[0] > inner_allowed[1]
    passed = outer_allowed[0] <= outer_allowed[1]
    state = "insufficient" if fail else "sufficient" if passed else "unresolved"
    return {"state": state, "absolute_radius_bracket": [inner[2], outer[2]],
            "relative_radius_bracket": [inner[3], outer[3]],
            "joint_prediction": sum(outer_allowed)/2 if passed else None,
            "joint_estimate_interval": list(outer_allowed) if passed else None,
            "absolute_prediction": (outer[0]+outer[1])/2 if passed else None,
            "relative_prediction": 2*outer[0]*outer[1]/(outer[0]+outer[1])
                if passed and outer[0] > 0 else None,
            "outer": list(outer[:2]), "inner": list(inner[:2])}


def search(info, gamma, waits, abs_error, relative_error, rho, levels, oracle,
           feasible, use_witness=True, fixed_fine=False, margin=2e-7):
    """Return every candidate's status, never infer monotonicity of diameter.

    oracle(wait,cells) returns checked brackets. Its failures stay unresolved.
    A missing physical strict witness blocks phase and rate exclusion, because
    strict feasibility is part of the theorem, not a solver-success flag.
    """
    if len(set(waits)) != len(waits) or not levels or sorted(set(levels)) != list(levels):
        raise ValueError("Unique target candidates and increasing mesh levels required")
    if any(not math.isfinite(w) or w < 0 for w in waits):
        raise ValueError("Invalid target delay")
    if abs_error <= 0 or not 0 <= rho < 1 or not 0 < relative_error < 1:
        raise ValueError("Invalid tolerance or input envelope")
    rows, region = [], None
    if feasible is not None and use_witness:
        for w in sorted(waits, reverse=True):
            witness = rate_witness(info, gamma, w, feasible)
            if witness and witness["target_increment"]/2 > abs_error+margin:
                region = {"through_wait": w, **witness}
                break  # same positive added atom excludes ALL earlier targets
    for w in sorted(waits):
        base = {"wait": w, "levels_used": [], "lp_calls": 0,
                "solver_calls": 0, "cell_variables": 0}
        if feasible is None:
            rows.append({**base, "state": "unresolved", "route": "strict_feasibility_unresolved"})
            continue
        if not bounded(info, gamma, w):
            rows.append({**base, "state": "unbounded", "route": "phase"})
            continue
        if rho > relative_error+margin:
            rows.append({**base, "state": "insufficient", "route": "input_floor"})
            continue
        if region is not None and w <= region["through_wait"]:
            # Only an absolute-error lower witness is propagated. Neither a
            # relative-error decision nor a diameter is assumed monotone.
            rows.append({**base, "state": "insufficient", "route": "physical_pair",
                         "witness": region})
            continue
        result = None
        for cells in [levels[-1]] if fixed_fine else levels:
            try:
                bracket = oracle(w, cells)
                result = decision(bracket, abs_error, relative_error, rho, margin)
            except RuntimeError as error:
                result = {"state": "unresolved", "numerical_failure": str(error)}
                base["levels_used"].append(cells)
                base["lp_calls"] += getattr(error, "programs", 0)
                base["solver_calls"] += getattr(error, "solver_calls", 0)
                base["cell_variables"] += getattr(error, "programs", 0)*getattr(error, "cells", 0)
                break
            base["levels_used"].append(cells)
            base["lp_calls"] += bracket.get("programs", 4)
            base["solver_calls"] += bracket.get("solver_calls", 4)
            base["cell_variables"] += bracket.get("programs", 4)*bracket["cells"]
            if result["state"] != "unresolved":
                break
        rows.append({**base, **result, "route": "inner_outer"})
    passing = [r["wait"] for r in rows if r["state"] == "sufficient"]
    first = min(passing) if passing else None
    earlier_decided = first is not None and all(
        r["state"] in ("unbounded", "insufficient") for r in rows if r["wait"] < first)
    return {"rows": rows, "first_sufficient_candidate": first,
            "earliest_candidate_proved": earlier_decided,
            "lp_calls": sum(r["lp_calls"] for r in rows),
            "solver_calls": sum(r["solver_calls"] for r in rows),
            "cell_variables": sum(r["cell_variables"] for r in rows),
            "physical_witness_region": region,
            "continuous_time_minimum_claimed": False}
