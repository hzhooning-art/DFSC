"""Matched testing-strategy adapters with explicit cost accounting.

The adapters deliberately share one outer :class:`Budget`.  A benchmark case
declares which baseline methods can express each relation; this prevents a
comparison from silently withholding an applicable relation from a baseline.
Optional property-generation engines are loaded by benchmark executors, never
at core-package import time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from .compiler import Budget


class StrategyError(ValueError):
    """Raised when a strategy case or executor violates the audit contract."""


class Strategy(Protocol):
    """A named adapter that decides whether it can express a relation."""

    name: str

    def supports(self, relation: Mapping[str, Any]) -> bool:
        """Return whether this method can use ``relation`` as declared."""


@dataclass(frozen=True)
class RelationObservation:
    relation_id: str
    outcome: str
    invocations: int
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    relations_available: tuple[str, ...]
    relations_eligible: tuple[str, ...]
    relations_used: tuple[str, ...]
    invocations: int
    elapsed_seconds: float
    termination_reason: str
    budget: Budget
    observations: tuple[RelationObservation, ...]


@dataclass(frozen=True)
class _DeclaredMethodStrategy:
    name: str

    def supports(self, relation: Mapping[str, Any]) -> bool:
        methods = _relation_methods(relation)
        return self.name in methods


class ExampleStrategy(_DeclaredMethodStrategy):
    def __init__(self) -> None:
        super().__init__("example")


class ToleranceStrategy(_DeclaredMethodStrategy):
    def __init__(self) -> None:
        super().__init__("tolerance")


class PropertyStrategy(_DeclaredMethodStrategy):
    def __init__(self) -> None:
        super().__init__("property")


class MetamorphicStrategy(_DeclaredMethodStrategy):
    def __init__(self) -> None:
        super().__init__("metamorphic")


class DifferentialStrategy(_DeclaredMethodStrategy):
    def __init__(self) -> None:
        super().__init__("differential")


class CompiledLayeredStrategy(_DeclaredMethodStrategy):
    def __init__(self) -> None:
        super().__init__("compiled_layered")

    def supports(self, relation: Mapping[str, Any]) -> bool:
        _relation_id(relation)
        _relation_methods(relation)
        return True


_STRATEGIES: tuple[Strategy, ...] = (
    ExampleStrategy(),
    ToleranceStrategy(),
    PropertyStrategy(),
    MetamorphicStrategy(),
    DifferentialStrategy(),
    CompiledLayeredStrategy(),
)


def all_strategies() -> tuple[Strategy, ...]:
    """Return the frozen display and execution order for all comparisons."""

    return _STRATEGIES


def allocate_case_budget(strategy: Strategy, budget: Budget) -> Budget:
    """Assign the same indivisible outer budget to every strategy.

    ``strategy`` is intentionally accepted to make allocation calls auditable;
    no method-specific multiplier or hidden reserve is permitted.
    """

    if strategy.name not in {item.name for item in _STRATEGIES}:
        raise StrategyError(f"unknown strategy: {strategy.name}")
    return Budget(budget.max_invocations, budget.wall_seconds)


def _relation_id(relation: Mapping[str, Any]) -> str:
    value = relation.get("id")
    if not isinstance(value, str) or not value.strip():
        raise StrategyError("each relation requires a non-empty id")
    return value


def _relation_methods(relation: Mapping[str, Any]) -> tuple[str, ...]:
    raw = relation.get("methods")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise StrategyError("each relation requires a methods sequence")
    methods = tuple(raw)
    if not all(isinstance(item, str) and item for item in methods):
        raise StrategyError("relation methods must be non-empty strings")
    if len(set(methods)) != len(methods):
        raise StrategyError("relation methods must be unique")
    known = {item.name for item in _STRATEGIES if item.name != "compiled_layered"}
    unknown = set(methods) - known
    if unknown:
        raise StrategyError(f"unknown relation methods: {sorted(unknown)}")
    return methods


def _case_parts(
    case: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], Callable[[Mapping[str, Any], int], Mapping[str, Any]]]:
    raw_relations = case.get("relations")
    executor = case.get("executor")
    if not isinstance(raw_relations, Sequence) or isinstance(raw_relations, (str, bytes)):
        raise StrategyError("case relations must be a sequence")
    if not callable(executor):
        raise StrategyError("case executor must be callable")
    relations: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_relations:
        if not isinstance(raw, Mapping):
            raise StrategyError("each relation must be a mapping")
        relation_id = _relation_id(raw)
        _relation_methods(raw)
        if relation_id in seen:
            raise StrategyError(f"duplicate relation id: {relation_id}")
        seen.add(relation_id)
        relations.append(raw)
    return tuple(relations), executor


def run_strategy(
    strategy: Strategy,
    case: Mapping[str, Any],
    budget: Budget,
) -> StrategyResult:
    """Run one strategy under a common invocation and wall-clock budget."""

    allocation = allocate_case_budget(strategy, budget)
    relations, executor = _case_parts(case)
    available = tuple(_relation_id(item) for item in relations)
    eligible_relations = tuple(item for item in relations if strategy.supports(item))
    eligible = tuple(_relation_id(item) for item in eligible_relations)
    used: list[str] = []
    observations: list[RelationObservation] = []
    invocations = 0
    termination = "completed"
    started = time.perf_counter()

    for relation in eligible_relations:
        elapsed = time.perf_counter() - started
        if elapsed >= allocation.wall_seconds:
            termination = "wall_budget"
            break
        remaining = allocation.max_invocations - invocations
        if remaining <= 0:
            termination = "invocation_budget"
            break
        relation_id = _relation_id(relation)
        try:
            raw = executor(relation, remaining)
        except Exception as error:  # executor failures remain auditable results
            used.append(relation_id)
            observations.append(
                RelationObservation(
                    relation_id,
                    "non_evaluable",
                    0,
                    {"reason": "executor_error", "error_type": type(error).__name__},
                )
            )
            termination = "executor_error"
            break
        if not isinstance(raw, Mapping):
            raise StrategyError("executor result must be a mapping")
        try:
            consumed = int(raw.get("invocations", 0))
        except (TypeError, ValueError) as error:
            raise StrategyError("executor invocations must be an integer") from error
        if isinstance(raw.get("invocations", 0), bool) or not 0 <= consumed <= remaining:
            raise StrategyError("executor invocations exceed the remaining budget")
        outcome = raw.get("outcome")
        if not isinstance(outcome, str) or not outcome:
            raise StrategyError("executor result requires a non-empty outcome")
        details = raw.get("details", {})
        if not isinstance(details, Mapping):
            raise StrategyError("executor details must be a mapping")
        used.append(relation_id)
        observations.append(RelationObservation(relation_id, outcome, consumed, details))
        invocations += consumed
        if invocations == allocation.max_invocations:
            termination = "invocation_budget"
            break
        if time.perf_counter() - started >= allocation.wall_seconds:
            termination = "wall_budget"
            break

    elapsed = time.perf_counter() - started
    return StrategyResult(
        strategy=strategy.name,
        relations_available=available,
        relations_eligible=eligible,
        relations_used=tuple(used),
        invocations=invocations,
        elapsed_seconds=elapsed,
        termination_reason=termination,
        budget=allocation,
        observations=tuple(observations),
    )
