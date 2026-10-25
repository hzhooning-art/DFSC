"""Continuous dual correction for identical positive loading profiles.

For gamma=beta the normalized target is exp(-(wait-dead_time)*lambda).
Any signed combination of observed exponentials is therefore checked on the
ENTIRE half-line, including its tail. A residual bound times the anchor mass
bound turns a discrete LP dual into a continuous outer bound. This is classical
weak duality with a target-specific residual check, not a new LP optimizer.
Floating-point allowances below are not directed-rounding certificates.
"""
from __future__ import annotations

import heapq
import math
import numpy as np
from scipy.optimize import linprog
from prediction_time_search import (Information, bounded, representation, log_laplace,
                                    EnclosureFailure)
from observation_weighted_prediction import verify


def exponential_upper(rates, coefficients, *, residual_tolerance=1e-7,
                      max_cells=4096, initial_tail=128.):
    """Bound sup sum a_i exp(-d_i lambda), lambda>=0, d_i>=0.

    On [l,r], f <= max(f(l),f(r)) + max(0,-inf f'')*(r-l)^2/8.
    Signed exponential interval bounds enclose f''; a second natural interval
    bound is intersected. On [T,infinity), discard negative decaying terms.
    Stop at the budget even if the requested residual tolerance is unattained:
    the bound is still usable with its reported correction and gap.
    """
    d, a = np.asarray(rates, float), np.asarray(coefficients, float)
    if (d.ndim != 1 or a.shape != d.shape or len(d) == 0 or
            np.any(d < 0) or not np.isfinite(d).all() or not np.isfinite(a).all() or
            not isinstance(max_cells, int) or max_cells < 2 or
            not math.isfinite(residual_tolerance) or residual_tolerance <= 0 or
            not math.isfinite(initial_tail) or initial_tail <= 0):
        raise ValueError("Finite signed coefficients, nonnegative rates and positive budgets required")
    unique = np.unique(d)
    a = np.array([a[d == x].sum() for x in unique])
    d = unique
    constant = float(a[d == 0].sum())
    rounding = 2e-12*(1+float(np.abs(a).sum()))
    pos, neg = np.maximum(a, 0.), np.minimum(a, 0.)
    def value(x):
        return float(a@np.exp(-d*x))
    def cell(l, r):
        el, er = np.exp(-d*l), np.exp(-d*r)
        fl, fr = float(a@el), float(a@er)
        curvature = max(0., -float((pos*d*d)@er+(neg*d*d)@el))
        chord = max(fl, fr)+curvature*(r-l)**2/8
        natural = float(pos@el+neg@er)
        return min(chord, natural)+rounding, max(fl, fr)
    def tail(t):
        return constant+float(pos[d > 0]@np.exp(-d[d > 0]*t))+rounding
    horizon = float(initial_tail)
    heap, serial = [], 0
    upper, observed = cell(0., horizon)
    heapq.heappush(heap, (-upper, serial, 0., horizon))
    lower = max(observed, constant)
    splits = extensions = 0
    while len(heap)+1 < max_cells:
        tail_upper = tail(horizon)
        global_upper = max(-heap[0][0], tail_upper)
        if global_upper-lower <= residual_tolerance:
            break
        if tail_upper > -heap[0][0]:
            l, r = horizon, horizon*2
            if not np.isfinite(r):
                break
            horizon = r
            upper, observed = cell(l, r)
            serial += 1
            heapq.heappush(heap, (-upper, serial, l, r))
            lower = max(lower, observed)
            extensions += 1
        else:
            _, _, l, r = heapq.heappop(heap)
            midpoint = (l+r)/2
            for lo, hi in ((l, midpoint), (midpoint, r)):
                upper, observed = cell(lo, hi)
                serial += 1
                heapq.heappush(heap, (-upper, serial, lo, hi))
                lower = max(lower, observed)
            splits += 1
    leaves = sorted([[l, r, -u] for u, _, l, r in heap])
    tail_upper = tail(horizon)
    upper = max(-heap[0][0], tail_upper)
    return {"upper": upper, "sampled_lower": lower, "gap": upper-lower,
            "cells": len(heap)+1, "splits": splits, "tail_extensions": extensions,
            "tail_start": horizon, "tail_upper": tail_upper,
            "rounding_allowance": rounding, "leaves": leaves,
            "rates": d.tolist(), "coefficients": a.tolist(),
            "tolerance_attained": upper-lower <= residual_tolerance}


def correct_dual(info: Information, gamma, wait, low_y, high_y, *,
                 residual_tolerance=1e-7, max_cells=4096, margin=2e-7):
    """Convert two finite-node LP duals into full continuous outer endpoints.

    y follows scipy's min-LP convention for [E;-E] x <= [c+b;-c+b].
    Nonpositive y gives a dual objective after a uniform residual correction.
    The correction uses nu([0,infinity)) <= anchor center + anchor band.
    No generating spectrum is supplied or used.
    """
    if gamma != info.beta or not bounded(info, gamma, wait):
        raise ValueError("This accelerator requires a bounded identical-profile target")
    n = len(info.delays)
    lo, hi = np.asarray(low_y, float), np.asarray(high_y, float)
    if (lo.shape != (2*n,) or hi.shape != (2*n,) or
            not np.isfinite(lo).all() or not np.isfinite(hi).all()):
        raise ValueError("Two finite dual vectors of matching size required")
    # Projection preserves the correct sign. Continuous residual checking then
    # repairs ANY vector, even one from a failed/inaccurate discrete dual solve.
    lo, hi = np.minimum(lo, 0.), np.minimum(hi, 0.)
    rhs = np.r_[np.asarray(info.centers)+info.bands,
                -np.asarray(info.centers)+info.bands]
    mass_upper = float(info.centers[0]+info.bands[0])
    if mass_upper <= 0:
        raise ValueError("Positive anchor mass bound required")
    low_coeff = lo[:n]-lo[n:]
    high_coeff = -hi[:n]+hi[n:]
    rates = np.r_[wait-info.dead_time, info.delays]
    certificates = {}
    for name, coefficients in (("lower", np.r_[-1., low_coeff]),
                               ("upper", np.r_[1., -high_coeff])):
        certificates[name] = exponential_upper(rates, coefficients,
            residual_tolerance=residual_tolerance, max_cells=max_cells)
    corrections = {k: mass_upper*max(0., c["upper"]) for k, c in certificates.items()}
    return {"outer": [max(0., float(rhs@lo)-corrections["lower"]-margin),
                      float(-rhs@hi)+corrections["upper"]+margin],
            "dual_objectives": [float(rhs@lo), float(-rhs@hi)],
            "mass_upper": mass_upper, "corrections": corrections,
            "certificates": certificates, "numerical_margin": margin,
            "scope": "identical loading profiles; full nonnegative rate half-line",
            "outward_rounded": False}


def enclose_dual(info, gamma, wait, cells, label, archive, logs, *,
                 feasible, tail_exponent=20., residual_tolerance=1e-7,
                 max_certificate_cells=4096, margin=2e-7):
    """Two LPs on finite physical nodes, with a target-resolving fast atom.

    For eps>0, R=tail_exponent/eps makes the identical-profile normalized
    target <= exp(-tail_exponent). R is a witness node, NEVER a rate cap;
    continuous dual checking still includes all rates above R and infinity.
    Repair each floating primal by mixing with the supplied strict witness.
    """
    if gamma != info.beta or not bounded(info, gamma, wait):
        raise ValueError("Bounded identical-profile target required")
    if feasible is None or not math.isfinite(tail_exponent) or tail_exponent <= 0:
        raise ValueError("A strict physical witness and positive tail exponent required")
    rep = representation(info, gamma, wait, cells)
    excess = wait-info.dead_time
    fast_rate = tail_exponent/excess if excess > 0 else None
    nodes = np.unique(np.r_[rep['nodes'], [fast_rate] if fast_rate else []])
    if not np.isfinite(nodes).all():
        raise ValueError("Finite fast witness rate required")
    moments = np.exp(-np.array(info.delays)[:, None]*nodes)
    a = np.vstack((moments, -moments))
    rhs = np.r_[np.array(info.centers)+info.bands, -np.array(info.centers)+info.bands]
    target = np.exp(-excess*nodes)
    wnodes = np.asarray(feasible['nodes'])
    wx = np.asarray(feasible['weighted_masses'])
    wmoments = np.exp(-np.array(info.delays)[:, None]*wnodes)
    strict_margin = float(np.min(np.array(info.bands)-np.abs(wmoments@wx-info.centers)))
    if strict_margin <= 4*margin or np.min(wx) < 0:
        raise ValueError("Physical witness must have verified positive slack")
    objectives, duals, primal_records = {}, {}, {}
    programs = calls = 0
    for side, c in (('il', target), ('ih', -target)):
        programs += 1
        attempts = []
        for method in ('highs-ds', 'highs-ipm'):
            calls += 1
            answer = linprog(c, A_ub=a, b_ub=rhs, bounds=(0., None), method=method,
                options={'primal_feasibility_tolerance': 1e-10,
                         'dual_feasibility_tolerance': 1e-10})
            checks = verify(c, a, rhs, answer) if answer.success else None
            if answer.success and max(checks.values()) <= margin:
                break
            attempts.append({'method': method, 'status': int(answer.status),
                             'message': answer.message, 'checks': checks})
        else:
            raise EnclosureFailure(str(attempts), programs, calls, len(nodes))
        key = label+'_'+side
        archive[key+'_nodes'] = nodes
        for name, value in (('x', answer.x), ('y', answer.ineqlin.marginals),
                            ('z', answer.lower.marginals)):
            archive[key+'_'+name] = value
        duals[side] = answer.ineqlin.marginals
        # Clip, then mix into a strictly feasible physical spectrum. This
        # repairs primal floating residuals instead of using them as evidence.
        x = np.maximum(answer.x, 0.)
        violation = max(0., float(np.max(a@x-rhs)))
        theta = min(1., (violation+4e-12)/(strict_margin+violation))
        physical_nodes = np.r_[nodes, wnodes]
        physical_weights = np.r_[(1-theta)*x, theta*wx]
        keep = physical_weights > 0
        physical_nodes, physical_weights = physical_nodes[keep], physical_weights[keep]
        observation = np.exp(-np.array(info.delays)[:, None]*physical_nodes)@physical_weights
        slack = float(np.min(np.array(info.bands)-np.abs(observation-info.centers)))
        if slack < -1e-12:
            raise RuntimeError("Primal repair did not establish feasible bands")
        q = float(physical_weights@np.exp(-excess*physical_nodes))
        archive[key+'_physical_nodes'] = physical_nodes
        archive[key+'_physical_weights'] = physical_weights
        archive[key+'_log_physical_masses'] = (np.log(physical_weights)
            +info.dead_time*physical_nodes-log_laplace(physical_nodes, info.beta))
        objectives[side] = q
        primal_records[side] = {'label': key, 'mixing_fraction': theta,
                               'minimum_observation_slack': slack, 'target': q}
        logs.append({'label': key, 'checks': checks, 'method': method,
                     'failed_attempts': attempts, 'repair': primal_records[side]})
    correction = correct_dual(info, gamma, wait, duals['il'], duals['ih'],
        residual_tolerance=residual_tolerance, max_cells=max_certificate_cells, margin=margin)
    inner = [objectives['il']+margin, objectives['ih']-margin]
    if not correction['outer'][0] <= inner[0] <= inner[1] <= correction['outer'][1]:
        raise RuntimeError("Continuous outer / physical inner ordering failed")
    return {**correction, 'inner': inner, 'cells': len(nodes), 'requested_cells': cells,
            'endpoint_gap': max(inner[0]-correction['outer'][0], correction['outer'][1]-inner[1]),
            'programs': programs, 'solver_calls': calls, 'fast_rate': fast_rate,
            'tail_exponent': tail_exponent, 'physical_witnesses': primal_records}
