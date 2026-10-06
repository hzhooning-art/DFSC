"""Executable properties of the EMSE risk-aware testing method."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .models import TestPlan


def check_coverage_soundness(plan: TestPlan, mandatory: Iterable[str]) -> bool:
    required = set(mandatory)
    observed = {test.test_id for test in plan.tests}
    return plan.status != "CONFORMANT" or required <= observed


def check_abstention_safety(status: str, missing: Iterable[str]) -> bool:
    return not (status == "CONFORMANT" and tuple(missing))


def check_replay_determinism(first: TestPlan, second: TestPlan) -> bool:
    return first.digest == second.digest and [item.test_id for item in first.tests] == [
        item.test_id for item in second.tests
    ]


def check_role_isolation(plan_payload: Mapping[str, Any]) -> bool:
    forbidden = ("buggy", "fixed", "defect_label", "package_version")
    serialized = repr(plan_payload).casefold()
    return not any(token in serialized for token in forbidden)


def check_coverage_certificate(plan: TestPlan) -> bool:
    certificate = plan.coverage_certificate
    requested = set(certificate.get("requested", ()))
    covered = set(certificate.get("covered", ()))
    missing = set(certificate.get("missing", ()))
    ratio = certificate.get("coverage_ratio")
    if requested != covered | missing or covered & missing or not isinstance(ratio, (int, float)):
        return False
    observed = {
        mechanism
        for test in plan.tests
        for mechanism in test.marginal_mechanisms
    } & requested
    if plan.status == "READY" and observed != covered:
        return False
    expected = len(covered) / len(requested) if requested else 1.0
    if abs(float(ratio) - expected) > 1e-12:
        return False
    return plan.status != "READY" or not missing
