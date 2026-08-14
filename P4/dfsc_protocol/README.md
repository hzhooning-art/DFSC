# `dfsc_protocol`

This is the public access layer for auditable qualification of differentiable
engineering-computing components.

```python
from dfsc_protocol import load_profile, load_registry

registry = load_registry("P4/results/p4_primitive_protocol_registry.json")
profile = load_profile("P4/results/p4_primitive_profile.json")
```

The package validates schema version, required evidence dimensions, backend
records, and profile timing fields. It does not merge metrics from different
mathematical domains into a misleading universal score.

Numerical acceptance uses predeclared criteria rather than finite-value checks
alone:

```python
from dfsc_protocol import QualificationCriteria, qualify_audit

criteria = QualificationCriteria(
    value_max_abs_error=1e-8,
    gradient_directional_relative_error=1e-5,
)
qualified = qualify_audit(audit, criteria)
assert qualified.status in {"conformant", "nonconformant"}
```

Scope restrictions remain separate record fields and do not create a third
status. Failed gates are retained in `qualified.gates` and explained in
`qualified.warnings`, so CI jobs can refuse a component without discarding its
diagnostic record.

## Local installation

From the repository root:

```bash
python -m pip install ./P4
python -m unittest discover -s P4/tests -p "test_*.py" -v
```

The registry readers use the Python standard library; numerical audit helpers
use PyTorch for batched execution and automatic differentiation.
