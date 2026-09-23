import hashlib
import json

import numpy as np

from cone_matched_score import HERE, critical_value
from score_power_curve import MS_CURVE, TARGETS, target_condition, trial


def main():
    power = json.loads((HERE / "score_power_curve_results.json").read_text(encoding="utf-8"))
    cone = json.loads((HERE / "cone_matched_score_results.json").read_text(encoding="utf-8"))
    for result in (power, cone):
        for name, expected in result["hashes"].items():
            assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected
        assert all(result["gates"].values())

    assert power["settings"]["targets"] == list(TARGETS)
    assert power["settings"]["m_values"] == list(MS_CURVE)
    for row in power["rows"]:
        _, mean, _, _ = target_condition(row["path"], row["m"], row["target_nc"])
        for replay in row["replay"]:
            rebuilt = trial(mean, row["m"], replay["seed"])
            assert rebuilt["score"] == replay["score"]
            assert rebuilt["union"] == replay["union"]
        assert np.isclose(row["power_gain"], row["score_power"] - row["union_power"])

    assert np.isclose(cone["settings"]["critical"], critical_value(6 * 24 * (16384 - 1)))
    prop_low = [row for row in cone["rows"] if row["path"] == "proportional" and row["target_nc"] in (4.0, 8.0)]
    assert np.mean([row["gain"] for row in prop_low]) >= .015
    print("score innovation audits passed")


if __name__ == "__main__":
    main()
