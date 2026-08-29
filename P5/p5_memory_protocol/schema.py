"""JSON schema for a frozen P5 reliability result."""

RESULT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "P5 shared-memory identification record",
    "type": "object",
    "required": ["schema_version", "experiment", "protocol_frozen_before_fit", "source", "evaluation", "decision"],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "experiment": {"type": "string", "minLength": 1},
        "protocol_frozen_before_fit": {"const": True},
        "source": {"type": "object"},
        "evaluation": {"type": "object"},
        "decision": {"type": "object"},
    },
}
