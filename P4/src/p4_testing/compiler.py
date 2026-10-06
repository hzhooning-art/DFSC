"""Deterministic compilation of component contracts into layered test plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any

from .contracts import ComponentContract, RelationContract, RiskProfile, validate_contract
from .models import Layer, TestPlan, TestSpec


class CompileError(ValueError):
    """Raised when a valid contract cannot be compiled under a budget."""


@dataclass(frozen=True)
class Budget:
    max_invocations: int
    wall_seconds: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_invocations, int)
            or isinstance(self.max_invocations, bool)
            or self.max_invocations <= 0
        ):
            raise ValueError("max_invocations must be a positive integer")
        if self.wall_seconds <= 0:
            raise ValueError("wall_seconds must be positive")


_LAYER_ORDER = (
    Layer.REFERENCE,
    Layer.PROPERTY,
    Layer.EXECUTION,
    Layer.REPLAY,
)
_LAYER_INDEX = {layer: index for index, layer in enumerate(_LAYER_ORDER)}


def _test_seed(seed: int, contract_id: str, relation_id: str) -> int:
    material = f"{seed}\0{contract_id}\0{relation_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _ordered_relations(contract: ComponentContract) -> list[RelationContract]:
    return sorted(
        contract.relations.values(),
        key=lambda item: (_LAYER_INDEX[item.layer], item.relation_id),
    )


def _plan_payload(plan: TestPlan, include_digest: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": plan.schema,
        "contract_id": plan.contract_id,
        "seed": plan.seed,
        "layers": [layer.value for layer in plan.layers],
        "max_invocations": plan.max_invocations,
        "wall_seconds": plan.wall_seconds,
        "risk_mechanisms": list(plan.risk_mechanisms),
        "status": plan.status,
        "reason_code": plan.reason_code,
        "coverage_certificate": dict(plan.coverage_certificate),
        "tests": [
            {
                "test_id": test.test_id,
                "layer": test.layer.value,
                "relation": test.relation,
                "parameters": dict(test.parameters),
                "max_invocations": test.max_invocations,
                "marginal_mechanisms": list(test.marginal_mechanisms),
                "estimated_cost": test.estimated_cost,
            }
            for test in plan.tests
        ],
    }
    if include_digest:
        payload["digest"] = plan.digest
    return payload


def canonical_plan_bytes(plan: TestPlan, include_digest: bool = True) -> bytes:
    return json.dumps(
        _plan_payload(plan, include_digest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def compile_contract(
    contract: ComponentContract,
    seed: int,
    budget: Budget,
) -> TestPlan:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise CompileError("seed must be an integer")
    relations = _ordered_relations(contract)
    if budget.max_invocations < len(relations):
        raise CompileError("budget must allow at least one invocation per relation")

    remaining = budget.max_invocations
    tests: list[TestSpec] = []
    for index, relation in enumerate(relations):
        relations_after = len(relations) - index - 1
        allocated = min(
            relation.max_invocations,
            remaining - relations_after,
        )
        parameters = dict(relation.parameters)
        parameters.update(
            {
                "generator": relation.generator,
                "comparator": relation.comparator,
                "seed": _test_seed(seed, contract.contract_id, relation.relation_id),
            }
        )
        tests.append(
            TestSpec(
                test_id=f"{relation.layer.value}.{relation.relation_id}",
                layer=relation.layer,
                relation=f"{relation.generator} -> {relation.comparator}",
                parameters=parameters,
                max_invocations=allocated,
            )
        )
        remaining -= allocated

    layers = tuple(layer for layer in _LAYER_ORDER if any(test.layer is layer for test in tests))
    plan = TestPlan(
        schema="P4-Test-Plan-v1",
        contract_id=contract.contract_id,
        seed=seed,
        tests=tuple(tests),
        layers=layers,
        max_invocations=budget.max_invocations,
        wall_seconds=float(budget.wall_seconds),
        digest="",
    )
    digest = hashlib.sha256(canonical_plan_bytes(plan, include_digest=False)).hexdigest()
    return replace(plan, digest=digest)


def compile_plan(
    contract: ComponentContract,
    risk_profile: RiskProfile,
    budget: Budget,
    *,
    seed: int,
) -> TestPlan:
    """Compile a deterministic risk-aware plan without role-specific inputs."""

    validate_contract(contract)
    if not isinstance(risk_profile, RiskProfile):
        raise CompileError("risk_profile must be a RiskProfile")
    relations = _ordered_relations(contract)
    coverage = {
        relation.relation_id: tuple(
            item for item in relation.mechanisms
            if item in risk_profile.mechanisms
        )
        for relation in relations
    }
    if any(not any(mechanism in items for items in coverage.values()) for mechanism in risk_profile.mechanisms):
        reason = "UNSUPPORTED_RISK_MECHANISM"
    else:
        reason = "MANDATORY_COVERAGE_UNDER_BUDGET"
    if reason == "UNSUPPORTED_RISK_MECHANISM":
        payload = {
            "schema": "P4-Test-Plan-v1",
            "contract_id": contract.contract_id,
            "seed": seed,
            "risk_mechanisms": list(risk_profile.mechanisms),
            "status": "ABSTAIN",
            "reason_code": reason,
            "max_invocations": budget.max_invocations,
            "wall_seconds": budget.wall_seconds,
        }
        plan = TestPlan(
            schema="P4-Test-Plan-v1",
            contract_id=contract.contract_id,
            seed=seed,
            tests=(),
            layers=(),
            max_invocations=budget.max_invocations,
            wall_seconds=float(budget.wall_seconds),
            digest="",
            risk_mechanisms=risk_profile.mechanisms,
            status="ABSTAIN",
            reason_code=reason,
            coverage_certificate={
                "requested": list(risk_profile.mechanisms),
                "covered": [],
                "missing": list(risk_profile.mechanisms),
                "coverage_ratio": 0.0,
                "budget": budget.max_invocations,
            },
        )
        return replace(plan, digest=hashlib.sha256(canonical_plan_bytes(plan, include_digest=False)).hexdigest())

    uncovered = set(risk_profile.mechanisms)
    selected: list[RelationContract] = []
    remaining = budget.max_invocations
    candidates = list(relations)
    while uncovered and remaining > 0:
        ranked = sorted(
            candidates,
            key=lambda relation: (
                -len(uncovered.intersection(coverage[relation.relation_id])),
                relation.max_invocations,
                _LAYER_INDEX[relation.layer],
                relation.relation_id,
            ),
        )
        best = ranked[0]
        gain = uncovered.intersection(coverage[best.relation_id])
        if not gain:
            break
        selected.append(best)
        candidates.remove(best)
        uncovered -= gain
        remaining -= 1
    if uncovered:
        payload = {
            "schema": "P4-Test-Plan-v1",
            "contract_id": contract.contract_id,
            "seed": seed,
            "risk_mechanisms": list(risk_profile.mechanisms),
            "status": "ABSTAIN",
            "reason_code": "MANDATORY_COVERAGE_UNDER_BUDGET",
            "max_invocations": budget.max_invocations,
            "wall_seconds": budget.wall_seconds,
        }
        plan = TestPlan(
            schema="P4-Test-Plan-v1",
            contract_id=contract.contract_id,
            seed=seed,
            tests=(),
            layers=(),
            max_invocations=budget.max_invocations,
            wall_seconds=float(budget.wall_seconds),
            digest="",
            risk_mechanisms=risk_profile.mechanisms,
            status="ABSTAIN",
            reason_code="MANDATORY_COVERAGE_UNDER_BUDGET",
            coverage_certificate={
                "requested": list(risk_profile.mechanisms),
                "covered": [],
                "missing": list(risk_profile.mechanisms),
                "coverage_ratio": 0.0,
                "budget": budget.max_invocations,
            },
        )
        return replace(plan, digest=hashlib.sha256(canonical_plan_bytes(plan, include_digest=False)).hexdigest())
    while candidates and remaining > 0:
        selected.append(candidates.pop(0))
        remaining -= 1
    tests = tuple(
        TestSpec(
            test_id=f"{relation.layer.value}.{relation.relation_id}",
            layer=relation.layer,
            relation=f"{relation.generator} -> {relation.comparator}",
            parameters={
                **dict(relation.parameters),
                "generator": relation.generator,
                "comparator": relation.comparator,
                "seed": _test_seed(seed, contract.contract_id, relation.relation_id),
            },
            max_invocations=1,
            marginal_mechanisms=coverage[relation.relation_id] or (relation.layer.value,),
        )
        for relation in selected
    )
    layers = tuple(layer for layer in _LAYER_ORDER if any(test.layer is layer for test in tests))
    plan = TestPlan(
        schema="P4-Test-Plan-v1",
        contract_id=contract.contract_id,
        seed=seed,
        tests=tests,
        layers=layers,
        max_invocations=budget.max_invocations,
        wall_seconds=float(budget.wall_seconds),
        digest="",
    )
    enriched = replace(
        plan,
        tests=tests,
        risk_mechanisms=risk_profile.mechanisms,
        status="READY",
        reason_code=None,
        digest="",
        coverage_certificate={
            "requested": list(risk_profile.mechanisms),
            "covered": sorted(set(risk_profile.mechanisms) - uncovered),
            "missing": sorted(uncovered),
            "coverage_ratio": len(set(risk_profile.mechanisms) - uncovered) / len(risk_profile.mechanisms),
            "budget": budget.max_invocations,
            "used": len(tests),
        },
    )
    digest = hashlib.sha256(canonical_plan_bytes(enriched, include_digest=False)).hexdigest()
    return replace(enriched, digest=digest)
