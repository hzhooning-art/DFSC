"""Stable public API for the P5 memory-identification protocol."""

from .core import (
    CurveRecord,
    GateConfig,
    ar1_profile_bic,
    decide,
    decide_transitions,
    evaluate,
    fit,
    fixed_grid_nnls_error,
    holm_adjust,
    identifiability_certificate,
    prony_error,
    report,
    residual_ar1_diagnostics,
)
from .extensions import (
    OscillatoryBounds,
    conformal_upper_pvalue,
    conformal_upper_quantile,
    continuous_spectrum_curves,
    fit_oscillatory_shared,
    fit_partially_shared,
    generalized_design,
    grouped_conformal_audit,
)

__all__ = [
    "CurveRecord",
    "GateConfig",
    "ar1_profile_bic",
    "fit",
    "fixed_grid_nnls_error",
    "prony_error",
    "evaluate",
    "decide",
    "decide_transitions",
    "holm_adjust",
    "identifiability_certificate",
    "report",
    "residual_ar1_diagnostics",
    "OscillatoryBounds",
    "generalized_design",
    "fit_oscillatory_shared",
    "fit_partially_shared",
    "continuous_spectrum_curves",
    "conformal_upper_quantile",
    "conformal_upper_pvalue",
    "grouped_conformal_audit",
]
__version__ = "0.1.0"


