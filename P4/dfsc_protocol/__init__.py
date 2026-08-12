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
    audit_batch_and_device,
    audit_value_and_gradient,
    make_audit,
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
    "audit_batch_and_device",
    "audit_value_and_gradient",
    "make_audit",
]
