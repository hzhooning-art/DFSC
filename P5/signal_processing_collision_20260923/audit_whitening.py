import json
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / "whitening_results.json").read_text(encoding="utf-8"))
assert all(r["gates"].values()), r["gates"]
assert [x["K"] for x in r["rows"]] == [64, 128, 512]
assert all(x[y]["minimum_rank"] == 4 for x in r["rows"] for y in ("null", "alternative"))
assert all(0 <= x[y][z] <= 1 for x in r["rows"] for y in ("null", "alternative")
           for z in ("whitened_rate", "unwhitened_rate"))
for name, expected in r["hashes"].items():
    assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name
print("independent-training whitening audit passed")
