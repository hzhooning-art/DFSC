import sys
import unittest
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_real_background_mechanism_audit import method_metrics, residual_segment  # noqa: E402


class RealBackgroundMechanismAuditTests(unittest.TestCase):
    def test_wrapped_residual_segment_is_standardized(self):
        segment = residual_segment(np.arange(10, dtype=float), 25, 7)
        self.assertEqual(len(segment), 25)
        self.assertAlmostEqual(float(np.mean(segment)), 0.0, places=12)
        self.assertAlmostEqual(float(np.std(segment)), 1.0, places=12)

    def test_method_metrics_keep_abstention_out_of_accuracy_denominator(self):
        metrics = method_metrics([("RETAIN", "RETAIN"), ("REFUSE", "INDETERMINATE")])
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["selective_accuracy"], 1.0)
        self.assertEqual(metrics["severe_refusal_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
