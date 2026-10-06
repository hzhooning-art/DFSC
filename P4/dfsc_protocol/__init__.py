"""Public API for the backend-independent differentiable primitive protocol."""

from .registry import (
    PROTOCOL_SCHEMA,
    PROFILE_SCHEMA,
    load_profile,
    load_registry,
    summarize_registry,
    validate_profile,
    validate_registry,
)
from .audit import (
    PrimitiveAudit,
    PrimitiveBackend,
    PrimitiveDomain,
    QualificationCriteria,
    audit_batch_and_device,
    audit_value_and_gradient,
    make_audit,
    qualify_audit,
)
from .conformance import (
    CONFORMANCE_SCHEMA,
    EARLIEST_LEGACY_SCHEMA,
    LEGACY_SCHEMA,
    PROFILE_COVERAGE_REQUIREMENTS,
    PROFILE_REQUIREMENTS,
    canonical_json,
    evaluate_conformance,
    migrate_record,
    record_digest,
)

__all__ = [
    "PROTOCOL_SCHEMA",
    "PROFILE_SCHEMA",
    "load_profile",
    "load_registry",
    "summarize_registry",
    "validate_profile",
    "validate_registry",
    "PrimitiveAudit",
    "PrimitiveBackend",
    "PrimitiveDomain",
    "QualificationCriteria",
    "audit_batch_and_device",
    "audit_value_and_gradient",
    "make_audit",
    "qualify_audit",
    "CONFORMANCE_SCHEMA",
    "EARLIEST_LEGACY_SCHEMA",
    "LEGACY_SCHEMA",
    "PROFILE_COVERAGE_REQUIREMENTS",
    "PROFILE_REQUIREMENTS",
    "canonical_json",
    "evaluate_conformance",
    "migrate_record",
    "record_digest",
]
