"""Two-sided target-time interpolation with an anchor-controlled remainder."""
import math
import numpy as np
from scipy.special import gammainc, gammaln

from prediction_time_search import representation
from prediction_time_transport import adaptive_search, transport_bracket


def curvature_bound(info, gamma, start, cells=128):
    """Upper bound U0 sup lambda^2 q_start/g over the WHOLE half-line.

    A strictly positive excess wait ensures finiteness. A zero gap does not
    obtain a finite curvature bound from this implementation.
    """
    excess = start-info.dead_time
    if excess <= 0:
        return None
    rep = representation(info, gamma, start, cells)
    right = rep["edges"][1:-1]
    compact = float(np.max(right**2*rep["q_high"][:-1]))
    tail = float(rep["edges"][-2])
    power = info.beta-gamma+2
    coefficient = math.exp((gammaln(gamma) if gamma else 0.)-gammaln(info.beta))/gammainc(info.beta, tail)
    critical = max(tail, power/excess) if power > 0 else tail
    high = coefficient*critical**power*math.exp(-excess*critical)
    weighted_mass = info.centers[0]+info.bands[0]
    return float(weighted_mass*max(compact, high)*(1+1e-12)+1e-12)


def quadratic_max(left, right, length, curvature):
    """max of endpoint-linear interpolation + curvature*h*(length-h)/2."""
    if length <= 0 or curvature < 0:
        raise ValueError("Positive length and nonnegative curvature required")
    if curvature == 0:
        return max(left, right)
    location = min(length, max(0., length/2+(right-left)/(length*curvature)))
    value = left+(right-left)*location/length+curvature*location*(length-location)/2
    return max(left, right, value)


def joint_cell_test(left, right, start, end, bound, absolute, relative, rho, margin=2e-7):
    """Four quadratic inequalities give a joint predictor for EVERY time.

    The predictor is time-varying; this does not claim one constant value meets
    tolerances across the entire cell. Inputs are nominal endpoint brackets.
    """
    if bound is None:
        return None
    e, tau = absolute-margin, relative-margin
    ls, us = left["outer"][0]/(1+rho), left["outer"][1]/(1-rho)
    lt, ut = right["outer"][0]/(1+rho), right["outer"][1]/(1-rho)
    effective = bound/(1+rho)
    forms = [(us-ls, ut-lt, 1., 2*e),
             ((1-tau)*us-(1+tau)*ls, (1-tau)*ut-(1+tau)*lt, 1+tau, 0.),
             (us-(1+tau)*ls, ut-(1+tau)*lt, 1+tau, e),
             ((1-tau)*us-ls, (1-tau)*ut-lt, 1., e)]
    maxima = [quadratic_max(a, b, end-start, factor*effective) for a, b, factor, limit in forms]
    if all(value <= form[3]-1e-12 for value, form in zip(maxima, forms)):
        return {"curvature_bound": bound, "maxima": maxima,
                "limits": [f[3] for f in forms],
                "predictor": "midpoint of the time-dependent joint estimate interval"}
    return None


def curvature_search(info, gamma, waits, absolute, relative, rho, levels, oracle,
                     feasible, margin=2e-7):
    bounds = {}
    def bound(start):
        if start not in bounds:
            bounds[start] = curvature_bound(info, gamma, start)
        return bounds[start]
    def transfer(left, right, start, end, at=None):
        base = transport_bracket(left, right, start, end, at)
        m = bound(start)
        if at is not None and m is not None:
            weight = (end-at)/(end-start)
            error = m*(at-start)*(end-at)/2
            lower = weight*left["outer"][0]+(1-weight)*right["outer"][0]-error
            base["outer"][0] = max(base["outer"][0], lower, 0.)
            base["outer"][1] = min(left["outer"][1], base["outer"][1])
            base["curvature_bound"] = m
        return base
    def whole(left, right, start, end, absolute, relative, rho, margin):
        return joint_cell_test(left, right, start, end, bound(start), absolute, relative, rho, margin)
    result = adaptive_search(info, gamma, waits, absolute, relative, rho, levels,
                             oracle, feasible, margin=margin, transport=transfer, whole_decider=whole)
    result["curvature_bound_evaluations"] = len(bounds)
    return result
