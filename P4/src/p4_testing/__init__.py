"""Contract-compiled testing for differentiable numerical components."""

from .models import (
    CaseManifest,
    Layer,
    Outcome,
    PairResult,
    TestObservation,
    TestPlan,
    TestSpec,
)
from .contracts import (
    ComponentContract,
    ContractError,
    ContractValidationError,
    RiskProfile,
    parse_contract,
    validate_contract,
)
from .results import ConformanceResult

__all__ = [
    "CaseManifest",
    "Layer",
    "Outcome",
    "PairResult",
    "TestObservation",
    "TestPlan",
    "TestSpec",
    "ComponentContract",
    "ContractError",
    "ContractValidationError",
    "RiskProfile",
    "parse_contract",
    "validate_contract",
    "ConformanceResult",
]
