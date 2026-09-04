import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "probe_power_certificate_risk_coverage.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("probe_power_certificate_risk_coverage", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RiskCoverageCurveTests(unittest.TestCase):
    def test_pareto_frontier_removes_dominated_point(self):
        points = [
            {"coverage": 0.5, "selective_risk": 0.2, "name": "a"},
            {"coverage": 0.4, "selective_risk": 0.3, "name": "b"},
            {"coverage": 0.3, "selective_risk": 0.1, "name": "c"},
        ]
        names = {row["name"] for row in MODULE.pareto_frontier(points)}
        self.assertEqual(names, {"a", "c"})

    def test_threshold_grid_contains_frozen_operating_point(self):
        self.assertIn(MODULE.POWER_LOWER_BOUND, MODULE.POWER_THRESHOLDS)


if __name__ == "__main__":
    unittest.main()
