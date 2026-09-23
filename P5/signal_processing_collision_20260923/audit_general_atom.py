from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
result = json.loads((root / "general_atom_results.json").read_text(encoding="utf-8"))
assert all(result["gates"].values()), result["gates"]
for name, expected in result["hashes"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
assert result["summary"]["cells"] == 72
assert result["summary"]["max_small_gap_relative_error"] <= 0.05
print({"status": "smooth-atom audit passed", **result["summary"]})
