import hashlib
import json

import numpy as np

from score_design_stress import HERE, DESIGNS


def main():
    result=json.loads((HERE/"score_design_stress_results.json").read_text(encoding="utf-8"))
    for name,expected in result["hashes"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected
    assert len(result["designs"])==len(DESIGNS)==12
    assert all(result["gates"].values())
    for design in result["designs"]:
        assert set(design["paths"])=={"null","diversity","crossover","proportional"}
        for row in design["paths"].values():
            assert row["minimum_amplitude"]>0
            assert 0<=row["union"]<=1 and 0<=row["phase"]<=1 and 0<=row["cone"]<=1
            assert np.isfinite([row["distance_nc"],row["score_nc"],row["retention"]]).all()
    print("score design stress audit passed")


if __name__=="__main__":
    main()
