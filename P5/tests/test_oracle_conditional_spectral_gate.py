import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from probe_oracle_conditional_spectral_gate import (  # noqa: E402
    conditional_null_threshold,
    run_oracle_conditional_map,
)
from probe_spectral_long_memory_gate import make_prefix_case  # noqa: E402


class OracleConditionalSpectralGateTests(unittest.TestCase):
    def test_conditional_threshold_represents_deterministic_curvature(self):
        case = make_prefix_case(256, d=0.0, strength=0.085, repeat=0)
        threshold = conditional_null_threshold(
            case["clean"], draws=32, seed=1701, quantile=0.99
        )
        self.assertGreater(threshold, 0.0)

    def test_conditional_threshold_is_deterministic(self):
        case = make_prefix_case(512, d=0.0, strength=0.200, repeat=0)
        first = conditional_null_threshold(
            case["clean"], draws=32, seed=1702, quantile=0.99
        )
        second = conditional_null_threshold(
            case["clean"], draws=32, seed=1702, quantile=0.99
        )
        self.assertEqual(first, second)

    def test_small_oracle_matrix_is_complete(self):
        result = run_oracle_conditional_map(
            prefix_lengths=(78,),
            memory_orders=(0.0, 0.30),
            strengths=(0.085,),
            repeats=1,
            calibration_draws=8,
        )
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(result["assessment_by_length"]["78"]["expected_count"], 2)
        self.assertTrue(result["protocol"]["oracle_clean_trajectory"])


if __name__ == "__main__":
    unittest.main()
