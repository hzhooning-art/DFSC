import hashlib
import json

import numpy as np

from matched_glrt_benchmark import HERE, M, realization
from score_power_curve import target_condition


def main():
    result = json.loads((HERE / "matched_glrt_results.json").read_text(encoding="utf-8"))
    for name, expected in result["hashes"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected

    null = result["null"]
    assert null["rejections"] / result["settings"]["null_trials"] == null["rate"]
    assert .03 <= null["rate"] <= .07
    assert len(result["rows"]) == 6
    for row in result["rows"]:
        for key in ("glrt_power", "phase_power", "cone_power"):
            assert 0 <= row[key] <= 1
        assert np.isclose(row["paired_cone_minus_glrt"], row["cone_power"] - row["glrt_power"])
        for key in ("glrt_wilson", "phase_wilson", "cone_wilson"):
            assert 0 <= row[key][0] <= row[key][1] <= 1

    for replay in result["grid_replays"]:
        _, mean, _, _ = target_condition(replay["path"], M, replay["target"])
        coarse = realization(mean, replay["seed"], points=41)
        fine = realization(mean, replay["seed"], points=81)
        difference = abs(coarse["glr"] - fine["glr"])
        assert np.isclose(difference, replay["absolute_glr_difference"], atol=1e-9)
        assert difference <= .05

    assert all(result["gates"].values())
    print("matched GLRT audit passed")


if __name__ == "__main__":
    main()
