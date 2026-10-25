"""Adaptive reuse of target-time brackets for a fixed positive loading profile.

The frozen v1 numerical kernel is not modified. Lower and upper target endpoints
are nonincreasing in delay; the upper endpoint is also convex. The diameter need
not be monotone. Exact bracket transport is separated from floating-point LP
replay, which is not a directed-rounding certificate.
"""
from __future__ import annotations

import math
import numpy as np
from prediction_time_search import bounded, decision, rate_witness


def transport_bracket(left, right, start, end, at=None):
    """Whole-time-cell hull (at=None), or sharper secant outer at one time.

    Inner overlap may be empty. This means no transferred lower witness, NOT
    an empty physical information class or an insufficient prediction.
    """
    if not all(math.isfinite(v) for v in (start, end)) or start >= end:
        raise ValueError("Ordered finite bracketing times required")
    if at is not None and not start <= at <= end:
        raise ValueError("No extrapolation outside the two computed times")
    low = right["outer"][0]
    high = left["outer"][1]
    if at is not None:
        weight = (end-at)/(end-start)
        high = weight*left["outer"][1]+(1-weight)*right["outer"][1]
    inner_low, inner_high = left["inner"][0], right["inner"][1]
    if low > high:
        raise RuntimeError("Inconsistent temporal endpoint ordering")
    return {"outer": [low, high],
            "inner": [inner_low, inner_high] if inner_low <= inner_high else None,
            "time_interval": [start, end], "evaluated_time": at}


def transferred_decision(bracket, absolute, relative, rho, margin=2e-7):
    """Apply joint recovery to an outer hull and an optional physical inner."""
    if bracket["inner"] is not None:
        return decision(bracket, absolute, relative, rho, margin)
    # A dummy singleton cannot exclude a positive target for positive
    # tolerances. It is used only to invoke the OUTER sufficient test, and is
    # never exposed as a physical inner witness.
    dummy = {"outer": bracket["outer"], "inner": [0., 0.]}
    result = decision(dummy, absolute, relative, rho, margin)
    result["inner"] = None
    result["absolute_radius_bracket"][0] = None
    result["relative_radius_bracket"][0] = None
    if result["state"] == "insufficient":
        raise RuntimeError("An empty overlap cannot justify exclusion")
    return result


def adaptive_search(info, gamma, waits, absolute, relative, rho, levels, oracle,
                    feasible, use_secant=True, margin=2e-7, transport=None,
                    whole_decider=None):
    """Screen as v1, then solve a midpoint only when bracket transport fails.

    Whole-cell decisions cover a continuous interval. Secant decisions apply
    only at the explicitly listed times. Neither implies a continuum minimum.
    Costs count all invoked endpoint programs, including failed attempts.
    """
    if (not waits or len(set(waits)) != len(waits) or
            any(not math.isfinite(w) or w < 0 for w in waits)):
        raise ValueError("Unique nonnegative finite candidate times required")
    if (not levels or list(levels) != sorted(set(levels)) or
            absolute <= margin or not margin < relative < 1 or not 0 <= rho < 1):
        raise ValueError("Invalid mesh budget or tolerances")
    transfer = transport if transport is not None else transport_bracket
    ordered = sorted(waits)
    rows = {w: {"wait": w, "state": "unresolved", "route": "pending",
                "levels_used": [], "lp_calls": 0, "solver_calls": 0,
                "cell_variables": 0} for w in ordered}
    region = None
    if feasible is not None:
        for w in reversed(ordered):
            pair = rate_witness(info, gamma, w, feasible)
            if pair is not None and pair["target_increment"]/2 > absolute+margin:
                region = {"through_wait": w, **pair}
                break
    for w in ordered:
        row = rows[w]
        if feasible is None:
            row.update(route="strict_feasibility_unresolved")
        elif not bounded(info, gamma, w):
            row.update(state="unbounded", route="phase")
        elif rho > relative+margin:
            row.update(state="insufficient", route="input_floor")
        elif region is not None and w <= region["through_wait"]:
            row.update(state="insufficient", route="physical_pair", witness=region)
    cache, evaluated, cells, operations = {}, set(), [], 0

    def evaluate(w):
        evaluated.add(w)
        row = rows[w]
        for level in levels:
            row["levels_used"].append(level)
            try:
                bracket = oracle(w, level)
            except RuntimeError as error:
                row["lp_calls"] += getattr(error, "programs", 0)
                row["solver_calls"] += getattr(error, "solver_calls", 0)
                row["cell_variables"] += getattr(error, "programs", 0)*getattr(error, "cells", 0)
                row.update(route="inner_outer", numerical_failure=str(error))
                # Any earlier, replay-checked mesh remains usable for transport.
                return
            cache[w] = {**bracket, "lp_label": f"w{w}_c{level}"}
            row["lp_calls"] += bracket["programs"]
            row["solver_calls"] += bracket["solver_calls"]
            row["cell_variables"] += bracket["programs"]*bracket["cells"]
            row.update(decision(bracket, absolute, relative, rho, margin), route="inner_outer")
            if row["state"] != "unresolved":
                return

    def apply(row, result, route, source):
        if row["state"] not in ("unresolved", result["state"]):
            raise RuntimeError("Opposite decisive labels in temporal transport")
        if row["state"] == "unresolved":
            row.update(result, route=route, transport_source=source)

    active = [w for w in ordered if rows[w]["route"] == "pending"]
    if active:
        evaluate(active[0])
        if active[-1] != active[0]:
            evaluate(active[-1])
    while active:
        keys = sorted(cache)
        for start, end in zip(keys, keys[1:]):
            pending = [w for w in active if start < w < end and rows[w]["state"] == "unresolved"]
            if not pending:
                continue
            source = {"start": start, "end": end,
                      "left_level": cache[start]["requested_cells"],
                      "right_level": cache[end]["requested_cells"]}
            whole = transfer(cache[start], cache[end], start, end)
            result = transferred_decision(whole, absolute, relative, rho, margin)
            varying = False
            if result["state"] == "unresolved" and whole_decider is not None:
                varying = whole_decider(cache[start], cache[end], start, end,
                                         absolute, relative, rho, margin)
                if varying:
                    result = {"state": "sufficient", "joint_prediction": None}
            operations += 1
            if result["state"] != "unresolved":
                # The whole interval, not just the candidate midpoints, is covered.
                for w in pending:
                    if varying:
                        local = transferred_decision(transfer(cache[start], cache[end], start, end, at=w),
                                                     absolute, relative, rho, margin)
                        if local["state"] != "sufficient":
                            raise RuntimeError("Whole-cell predictor and pointwise bracket disagree")
                        apply(rows[w], local, "curvature_cell", source)
                    else:
                        apply(rows[w], result, "time_cell", source)
                cells.append({**source, "state": result["state"],
                              "joint_prediction": result["joint_prediction"],
                              "nominal_bracket": whole, "candidate_count": len(pending),
                              "varying_prediction": bool(varying),
                              "whole_cell_test": varying if varying else None})
                continue
            if use_secant:
                for w in pending:
                    bracket = transfer(cache[start], cache[end], start, end, at=w)
                    value = transferred_decision(bracket, absolute, relative, rho, margin)
                    operations += 1
                    if value["state"] != "unresolved":
                        apply(rows[w], value, "curvature_point" if transport is not None else "time_secant", source)
        remaining = [w for w in active if rows[w]["state"] == "unresolved" and w not in evaluated]
        if not remaining:
            break
        # Deterministic median of unresolved candidates, not bisection of an
        # assumed monotone diameter or of a pass/fail function.
        evaluate(remaining[len(remaining)//2])
    output = [rows[w] for w in ordered]
    passing = [r["wait"] for r in output if r["state"] == "sufficient"]
    first = min(passing) if passing else None
    return {"rows": output, "lp_calls": sum(r["lp_calls"] for r in output),
            "solver_calls": sum(r["solver_calls"] for r in output),
            "cell_variables": sum(r["cell_variables"] for r in output),
            "first_sufficient_candidate": first,
            "earliest_candidate_proved": first is not None and all(
                r["state"] in ("insufficient", "unbounded") for r in output if r["wait"] < first),
            "time_cells": cells, "transport_operations": operations,
            "evaluated_target_times": sorted(evaluated),
            "continuous_time_minimum_claimed": False}
