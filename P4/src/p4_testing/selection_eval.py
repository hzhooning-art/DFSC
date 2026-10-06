"""Budget-matched evaluation of plan-selection policies."""

from __future__ import annotations

from typing import Any

from dataclasses import replace
import hashlib
import random

from .compiler import Budget, canonical_plan_bytes, compile_contract, compile_plan
from .contracts import ComponentContract, RiskProfile


def _covered(plan) -> list[str]:
    return sorted({mechanism for test in plan.tests for mechanism in test.marginal_mechanisms})


def summarize_selector_reports(reports: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate selector outcomes without treating mechanisms as defect families."""

    if not reports:
        raise ValueError("at least one selector report is required")
    strategies = ("risk_aware", "fixed_order", "random", "no_abstention", "no_replay")
    requested = set(reports[0]["risk_aware"].get("requested_mechanisms", ()))
    if not requested:
        # The report schema predates this field; infer the denominator from the
        # risk-aware/fixed comparison and keep the aggregation explicit.
        requested = set(reports[0].get("requested_mechanisms", ()))
    if not requested:
        requested = {"value_error", "replay"}
    result: dict[str, dict[str, float]] = {}
    fixed_mean = sum(
        len(set(report["fixed_order"]["covered_mechanisms"]) & requested) / len(requested)
        for report in reports
    ) / len(reports)
    for strategy in strategies:
        coverages = [
            len(set(report[strategy]["covered_mechanisms"]) & requested) / len(requested)
            for report in reports
        ]
        abstentions = [report[strategy]["status"] == "ABSTAIN" for report in reports]
        result[strategy] = {
            "mean_coverage": sum(coverages) / len(coverages),
            "abstention_rate": sum(abstentions) / len(abstentions),
            "coverage_increment_vs_fixed": sum(coverages) / len(coverages) - fixed_mean,
        }
    return result


def _fixed_order(contract: ComponentContract, profile: RiskProfile, budget: Budget, seed: int):
    raw = compile_contract(contract, seed=seed, budget=Budget(len(contract.relations), budget.wall_seconds))
    mechanism_by_test = {
        f"{relation.layer.value}.{relation.relation_id}": relation.mechanisms
        for relation in contract.relations.values()
    }
    annotated = tuple(
        replace(
            test,
            marginal_mechanisms=tuple(
                mechanism for mechanism in mechanism_by_test[test.test_id]
                if mechanism in profile.mechanisms
            ),
        )
        for test in raw.tests
    )
    plan = raw.__class__(
        schema=raw.schema,
        contract_id=raw.contract_id,
        seed=raw.seed,
        tests=annotated[: budget.max_invocations],
        layers=tuple(layer for layer in raw.layers if any(test.layer is layer for test in annotated[: budget.max_invocations])),
        max_invocations=budget.max_invocations,
        wall_seconds=raw.wall_seconds,
        digest="",
        risk_mechanisms=profile.mechanisms,
        status="READY",
        reason_code=None,
    )
    return replace(plan, digest=hashlib.sha256(canonical_plan_bytes(plan, include_digest=False)).hexdigest())


def compare_selectors(
    contract: ComponentContract,
    profile: RiskProfile,
    budget: Budget,
    *,
    seed: int,
) -> dict[str, Any]:
    risk = compile_plan(contract, profile, budget, seed=seed)
    fixed = _fixed_order(contract, profile, budget, seed)
    random_plan = _random_order(contract, profile, budget, seed)
    no_replay = _fixed_order(
        contract,
        RiskProfile.from_dict({"mechanisms": [item for item in profile.mechanisms if item not in {"replay", "historical_regression"}]} or {"mechanisms": ["value_error"]}),
        budget,
        seed,
    )
    no_abstention = fixed if fixed.status == "READY" else replace(fixed, status="READY", reason_code=None)
    return {
        "requested_mechanisms": list(profile.mechanisms),
        "risk_aware": {
            "status": risk.status,
            "covered_mechanisms": _covered(risk),
            "selected_tests": [test.test_id for test in risk.tests],
        },
        "fixed_order": {
            "status": fixed.status,
            "covered_mechanisms": _covered(fixed),
            "selected_tests": [test.test_id for test in fixed.tests],
        },
        "random": {
            "status": random_plan.status,
            "covered_mechanisms": _covered(random_plan),
            "selected_tests": [test.test_id for test in random_plan.tests],
        },
        "no_abstention": {
            "status": no_abstention.status,
            "covered_mechanisms": _covered(no_abstention),
            "selected_tests": [test.test_id for test in no_abstention.tests],
        },
        "no_replay": {
            "status": no_replay.status,
            "covered_mechanisms": _covered(no_replay),
            "selected_tests": [test.test_id for test in no_replay.tests],
        },
        "budget": budget.max_invocations,
    }


def _random_order(contract: ComponentContract, profile: RiskProfile, budget: Budget, seed: int):
    raw = _fixed_order(contract, profile, Budget(len(contract.relations), budget.wall_seconds), seed)
    ordered = list(raw.tests)
    random.Random(seed).shuffle(ordered)
    selected = tuple(ordered[: budget.max_invocations])
    plan = replace(raw, tests=selected, layers=tuple(layer for layer in raw.layers if any(test.layer is layer for test in selected)), risk_mechanisms=profile.mechanisms, digest="")
    return replace(plan, digest=hashlib.sha256(canonical_plan_bytes(plan, include_digest=False)).hexdigest())
