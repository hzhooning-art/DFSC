import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "probe_common_budget_subspace_baselines.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("probe_common_budget_subspace_baselines", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

from probe_common_budget_order_detection import make_curves


class CommonBudgetSubspaceBaselineTests(unittest.TestCase):
    def setUp(self):
        self.curves, _ = make_curves(2, 16.0, 48, 0.0001, "white", 0.32, 991)

    def test_block_hankel_has_grouped_columns(self):
        hankel = MODULE.block_hankel(self.curves)
        self.assertEqual(hankel.shape[1] % len(self.curves), 0)
        self.assertGreater(hankel.shape[0], 2)

    def test_subspace_criteria_return_candidate_rank(self):
        for criterion in ("aic", "mdl"):
            self.assertIn(MODULE.subspace_order(self.curves, criterion)["decision"], (1, 2))

    def test_prony_criteria_return_candidate_rank(self):
        for criterion in ("aicc", "bic"):
            self.assertIn(MODULE.shared_prony_order(self.curves, criterion)["decision"], (1, 2))

    def test_unknown_criteria_are_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.subspace_order(self.curves, "unknown")
        with self.assertRaises(ValueError):
            MODULE.shared_prony_order(self.curves, "unknown")


if __name__ == "__main__":
    unittest.main()
