"""Parsing and semantic validation for P4 component contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, ClassVar

from .models import Layer


class ContractError(ValueError):
    """Raised when a component contract is incomplete or ambiguous."""


ContractValidationError = ContractError


@dataclass(frozen=True)
class RiskProfile:
    """Stable defect-mechanism vocabulary used by the plan compiler."""

    mechanisms: tuple[str, ...]

    ALLOWED: ClassVar[frozenset[str]] = frozenset(
        {
            "value_error",
            "derivative_error",
            "representation_mismatch",
            "batch_isolation",
            "dtype_device_mismatch",
            "execution_policy_mismatch",
            "historical_regression",
            "replay",
            "repeatability",
        }
    )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RiskProfile":
        if not isinstance(payload, Mapping):
            raise ContractValidationError("risk profile must be an object")
        mechanisms = payload.get("mechanisms")
        if not isinstance(mechanisms, (list, tuple)) or not mechanisms:
            raise ContractValidationError("risk profile requires mechanisms")
        normalized = tuple(sorted({str(item).strip() for item in mechanisms}))
        if not all(normalized):
            raise ContractValidationError("risk mechanisms must be non-empty strings")
        unknown = sorted(set(normalized) - cls.ALLOWED)
        if unknown:
            raise ContractValidationError(f"unknown risk mechanism: {unknown[0]}")
        return cls(mechanisms=normalized)


@dataclass(frozen=True)
class RelationContract:
    relation_id: str
    layer: Layer
    generator: str
    comparator: str
    parameters: Mapping[str, Any]
    max_invocations: int
    mechanisms: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentContract:
    schema: str
    contract_id: str
    component: Mapping[str, Any]
    domain: Mapping[str, Any]
    execution: Mapping[str, Any]
    relations: Mapping[str, RelationContract]
    references: Mapping[str, Any]
    rejection: Mapping[str, Any]
    provenance: Mapping[str, Any]


_REQUIRED = {"schema", "contract_id", "component", "domain", "execution", "relations"}
_OPTIONAL = {"references", "rejection", "provenance"}
_ALLOWED = _REQUIRED | _OPTIONAL


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping) or not value:
        raise ContractError(f"{name} must be a non-empty object")
    return MappingProxyType(dict(value))


def _validate_positive_numbers(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if ("tolerance" in key or key in {"rtol", "atol", "budget"}) and isinstance(item, (int, float)):
                if item <= 0:
                    raise ContractError(f"{child} must be positive")
            _validate_positive_numbers(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_positive_numbers(item, f"{path}[{index}]")


def parse_contract(payload: Mapping[str, Any]) -> ComponentContract:
    if not isinstance(payload, Mapping):
        raise ContractError("contract must be a JSON object")
    missing = sorted(_REQUIRED - set(payload))
    if missing:
        raise ContractError(f"missing required field: {missing[0]}")
    unknown = sorted(set(payload) - _ALLOWED)
    if unknown:
        raise ContractError(f"unknown top-level field: {unknown[0]}")
    if payload["schema"] != "P4-Component-Contract-v1":
        raise ContractError("unsupported contract schema")
    contract_id = payload["contract_id"]
    if not isinstance(contract_id, str) or not contract_id.strip():
        raise ContractError("contract_id must be a non-empty string")

    component = _mapping(payload, "component")
    domain = _mapping(payload, "domain")
    execution = _mapping(payload, "execution")
    raw_relations = _mapping(payload, "relations")
    relations: dict[str, RelationContract] = {}
    for relation_id, raw in raw_relations.items():
        if not isinstance(relation_id, str) or not relation_id.strip():
            raise ContractError("relation identifiers must be non-empty strings")
        if not isinstance(raw, Mapping):
            raise ContractError(f"relation {relation_id} must be an object")
        for required in ("layer", "generator", "comparator", "max_invocations"):
            if required not in raw:
                raise ContractError(f"relation {relation_id} requires {required}")
        try:
            layer = Layer(str(raw["layer"]))
        except ValueError as error:
            raise ContractError(f"relation {relation_id} has unknown layer") from error
        generator = raw["generator"]
        comparator = raw["comparator"]
        max_invocations = raw["max_invocations"]
        if not isinstance(generator, str) or not generator.strip():
            raise ContractError(f"relation {relation_id} requires generator")
        if not isinstance(comparator, str) or not comparator.strip():
            raise ContractError(f"relation {relation_id} requires comparator")
        if not isinstance(max_invocations, int) or isinstance(max_invocations, bool) or max_invocations <= 0:
            raise ContractError(f"relation {relation_id} max_invocations must be a positive integer")
        parameters = raw.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ContractError(f"relation {relation_id} parameters must be an object")
        raw_mechanisms = raw.get("mechanisms", ())
        if not isinstance(raw_mechanisms, (list, tuple)):
            raise ContractError(f"relation {relation_id} mechanisms must be an array")
        if any(not isinstance(item, str) or item not in RiskProfile.ALLOWED for item in raw_mechanisms):
            raise ContractError(f"relation {relation_id} has invalid mechanisms")
        if len(raw_mechanisms) != len(set(raw_mechanisms)):
            raise ContractError(f"relation {relation_id} has invalid mechanisms")
        _validate_positive_numbers(parameters, f"relations.{relation_id}.parameters")
        relations[relation_id] = RelationContract(
            relation_id=relation_id,
            layer=layer,
            generator=generator,
            comparator=comparator,
            parameters=MappingProxyType(dict(parameters)),
            max_invocations=max_invocations,
            mechanisms=tuple(sorted(raw_mechanisms)),
        )

    references = payload.get("references", {})
    rejection = payload.get("rejection", {})
    provenance = payload.get("provenance", {})
    for name, value in (("references", references), ("rejection", rejection), ("provenance", provenance)):
        if not isinstance(value, Mapping):
            raise ContractError(f"{name} must be an object")
    _validate_positive_numbers(payload, "contract")
    return ComponentContract(
        schema=payload["schema"],
        contract_id=contract_id,
        component=component,
        domain=domain,
        execution=execution,
        relations=MappingProxyType(relations),
        references=MappingProxyType(dict(references)),
        rejection=MappingProxyType(dict(rejection)),
        provenance=MappingProxyType(dict(provenance)),
    )


def validate_contract(payload: Mapping[str, Any] | ComponentContract) -> ComponentContract:
    """Validate semantic evidence prerequisites before compilation.

    ``parse_contract`` remains syntax-focused for backward compatibility;
    this stronger gate is used by the EMSE reconstruction route.
    """

    contract = payload if isinstance(payload, ComponentContract) else parse_contract(payload)
    if not contract.references:
        raise ContractValidationError("contract requires an independent reference")
    if not any(value is True for value in contract.references.values()):
        reference_keys = {
            "analytic_solution",
            "analytic_derivative",
            "analytic_expectation",
            "reference_dtype",
            "independent",
            "independent_path",
            "reference_evaluator",
        }
        if not reference_keys.intersection(contract.references):
            raise ContractValidationError("contract requires an independent reference")
    if not contract.relations:
        raise ContractValidationError("contract requires at least one test relation")
    return contract


def load_contract(path: Path) -> ComponentContract:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot load contract {path}: {error}") from error
    return parse_contract(payload)
