import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULT = HERE / "finite_training_calibration_results.json"
result = json.loads(RESULT.read_text(encoding="utf-8"))

assert result["settings"]["d"] == 48
assert result["settings"]["fixed_null_dimension"] == 9
assert result["settings"]["added_dimension"] == 4
assert len(result["rows"]) == 6
assert {row["covariance"] for row in result["rows"]} == {"identity", "colored"}
assert {row["K"] for row in result["rows"]} == {64, 128, 512}
assert all(result["gates"].values()), result["gates"]
for name, expected in result["hashes"].items():
    actual = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    assert actual == expected, name

print("finite-training calibration audit passed")
