import hashlib
import json
import math
from pathlib import Path

import numpy as np

from finite_local_limit import (
    HERE,
    MS,
    TARGET_NC,
    condition,
    one_trial,
    prediction,
    profiled_rss,
)


stored = json.loads((HERE / "finite_local_limit_results.json").read_text(encoding="utf-8"))
assert stored["hashes"] == {
    name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    for name in ("FINITE_LOCAL_LIMIT_PROTOCOL.md", "finite_local_limit.py", "LOCAL_LIMIT_POWER_THEORY.md")
}
assert all(stored["gates"].values())
assert len(stored["rows"]) == 16

for row in stored["rows"]:
    assert row["m"] in MS
    assert 0 <= row["empirical_rejection"] <= 1
    assert 0 <= row["local_prediction_exact_distance"] <= 1
    assert 0 <= row["local_prediction_target"] <= 1
    assert math.isclose(
        row["local_prediction_exact_distance"],
        prediction(row["exact_noncentrality"], row["m"]),
        rel_tol=1e-13,
    )
    r, delta, mean, a, b = condition(row["path"], row["m"])
    distance2, rate = profiled_rss(mean)
    assert math.isclose(row["r"], r, rel_tol=1e-13, abs_tol=1e-15)
    assert math.isclose(row["delta"], delta, rel_tol=1e-13, abs_tol=1e-15)
    assert math.isclose(row["distance2"], distance2, rel_tol=1e-11, abs_tol=1e-12)
    assert math.isclose(row["nearest_null_rate"], rate, rel_tol=1e-11, abs_tol=1e-12)
    assert math.isclose(row["minimum_amplitude"], min(a.min(), b.min()), rel_tol=1e-13)
    for saved in row["replay"]:
        rebuilt = one_trial(row["path"], row["m"], saved["seed"])
        assert saved.keys() == rebuilt.keys()
        for key in saved:
            if isinstance(saved[key], float):
                assert math.isclose(saved[key], rebuilt[key], rel_tol=1e-12, abs_tol=1e-14)
            else:
                assert saved[key] == rebuilt[key]
    for replay in row["replay"]:
        fresh = one_trial(row["path"], row["m"], replay["seed"])
        assert replay.keys() == fresh.keys()
        for key in replay:
            if isinstance(replay[key], float):
                assert math.isclose(replay[key], fresh[key], rel_tol=1e-11, abs_tol=1e-12)
            else:
                assert replay[key] == fresh[key]

largest = [row for row in stored["rows"] if row["m"] == MS[-1] and row["path"] != "null"]
assert all(abs(row["exact_noncentrality"] / TARGET_NC - 1) <= 0.08 for row in largest)
assert all(row["absolute_power_error"] <= 0.06 for row in largest)
assert all(row["ks_distance"] <= 0.07 for row in largest)
assert all(
    row["empirical_rejection"] <= 0.07
    for row in stored["rows"]
    if row["path"] == "null"
)

print("finite local limit audit passed")
