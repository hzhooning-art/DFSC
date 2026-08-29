import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_cross_implementation_confirmation import (  # noqa: E402
    decision_replay,
    numerical_trials,
)


class CrossImplementationConfirmationTests(unittest.TestCase):
    def test_scipy_value_and_finite_difference_gradient_references(self):
        rows = numerical_trials()
        self.assertLessEqual(max(row["value_relative_error"] for row in rows), 1e-10)
        self.assertLessEqual(max(row["gradient_relative_error"] for row in rows), 1e-6)

    def test_independent_decision_replay_is_exact(self):
        replay = decision_replay()
        self.assertGreaterEqual(replay["count"], 70)
        self.assertEqual(replay["concordance"], 1.0)


if __name__ == "__main__":
    unittest.main()
