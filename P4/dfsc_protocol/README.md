# `dfsc_protocol`

This is the public, dependency-free access layer for the differentiable
scientific-computing primitive protocol.

```python
from dfsc_protocol import load_profile, load_registry

registry = load_registry("P4/results/p4_primitive_protocol_registry.json")
profile = load_profile("P4/results/p4_primitive_profile.json")
```

The package validates schema version, required evidence dimensions, backend
records, and profile timing fields. It does not merge metrics from different
mathematical domains into a misleading universal score.

## Local installation

From the repository root:

```bash
python -m pip install --no-deps ./P4
python -m unittest discover -s P4/tests -p "test_*.py" -v
```

The package has no runtime dependency beyond the Python standard library.
