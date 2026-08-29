import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_external_scope_boundary_map import REGIMES, SOURCES, classify  # noqa: E402


class ExternalScopeBoundaryMapTests(unittest.TestCase):
    def test_scope_axes_and_sources_are_preregistered(self):
        self.assertEqual(set(REGIMES), {"noise_x2", "sparse_train", "short_window"})
        self.assertEqual(len(SOURCES), 3)

    def test_classification_uses_frozen_acceptance_checks(self):
        supported = {
            "coverage": 0.7,
            "selective_accuracy": 0.9,
            "false_refusal_fraction": 0.05,
            "severe_refusal_fraction": 0.8,
        }
        limited = dict(supported, coverage=0.5)
        self.assertEqual(classify(supported), "SUPPORTED")
        self.assertEqual(classify(limited), "SCOPE_LIMITED")


if __name__ == "__main__":
    unittest.main()
