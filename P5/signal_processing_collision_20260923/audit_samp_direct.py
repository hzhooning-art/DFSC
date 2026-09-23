import json
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / "samp_direct_results.json").read_text(encoding="utf-8"))
assert all(r["gates"].values()), r["gates"]
assert r["source"]["commit"] == "3f8cf58a32d7c7849da89ef911827fbfd57a11bd"
assert [x["target_nc"] for x in r["rows"]] == [0.0, 4.0, 8.0]
assert all(0 <= x["phase_rate"] <= 1 and 0 <= x["samp_decay_rate"] <= 1 for x in r["rows"])
assert r["official_SAMP"]["null"]["applicable"] is False
assert r["official_SAMP"]["two_rate"]["applicable"] is False
vendor = HERE / "vendor_samp"
for name, expected in r["source"]["hashes"].items():
    assert hashlib.sha256((vendor / name).read_bytes()).hexdigest() == expected, name
print("SAMP direct-compatibility audit passed")
