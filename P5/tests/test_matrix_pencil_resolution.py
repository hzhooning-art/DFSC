import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p5_memory_protocol import (  # noqa: E402
    CurveRecord,
    matrix_pencil_consensus,
    matrix_pencil_order_selection,
    shared_matrix_pencil,
)


class MatrixPencilResolutionTests(unittest.TestCase):
    @staticmethod
    def curves(rates=(0.18, 0.72)):
        time = np.linspace(0.0, 12.0, 80)
        output = []
        for index, amplitudes in enumerate(((1.0, 0.5), (0.6, 1.1), (1.3, 0.4), (0.8, 0.9))):
            value = 0.2 + sum(a * np.exp(-rate * time) for a, rate in zip(amplitudes, rates))
            output.append(CurveRecord(str(index), f"g{index}", "signal", time, value))
        return output

    def test_noiseless_shared_rates_are_recovered(self):
        result = shared_matrix_pencil(self.curves(), rank=2)
        self.assertTrue(result["success"])
        np.testing.assert_allclose(result["rates"], (0.18, 0.72), rtol=1e-5, atol=1e-7)

    def test_bic_selects_two_modes_in_clear_case(self):
        result = matrix_pencil_order_selection(self.curves(), ranks=(1, 2))
        self.assertEqual(result["selected_rank"], 2)
        self.assertGreater(result["transitions"][0]["delta_bic"], 10.0)

    def test_information_criteria_are_reported_and_selectable(self):
        result = matrix_pencil_order_selection(
            self.curves(), ranks=(1, 2), criterion="aicc", minimum_improvement=0.0
        )
        self.assertEqual(result["criterion"], "aicc")
        self.assertEqual(result["selected_rank"], 2)
        for rank in ("1", "2"):
            record = result["rank_records"][rank]
            self.assertTrue(all(key in record for key in ("aic", "aicc", "bic")))

    def test_unknown_information_criterion_is_rejected(self):
        with self.assertRaises(ValueError):
            matrix_pencil_order_selection(self.curves(), criterion="hqic")

    def test_clear_case_is_stable_across_pencil_shapes(self):
        result = matrix_pencil_consensus(self.curves(), rank=2)
        self.assertTrue(result["passes_consensus"])
        self.assertLess(result["maximum_cross_pencil_log_rate_std"], 1e-4)

    def test_nonuniform_time_is_rejected(self):
        curves = self.curves()
        broken = CurveRecord(curves[0].unit, curves[0].group, curves[0].channel, curves[0].time**1.1, curves[0].value)
        with self.assertRaises(ValueError):
            shared_matrix_pencil([broken, *curves[1:]], rank=2)


if __name__ == "__main__":
    unittest.main()
