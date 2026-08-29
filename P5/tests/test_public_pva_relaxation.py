import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from probe_public_pva_relaxation import (  # noqa: E402
    DATA, EXPECTED_MD5, OUTPUT_JSON, file_md5, fit_coefficients, load_curves,
)


class PublicPVARelaxationTests(unittest.TestCase):
    def test_public_workbook_checksum_and_inventory(self):
        curves = load_curves()
        self.assertEqual(file_md5(DATA), EXPECTED_MD5)
        self.assertEqual(len(curves), 9)
        self.assertEqual({curve.sample for curve in curves}, {1, 2, 3})
        self.assertTrue(all(len(curve.time) > 200 for curve in curves))
        self.assertTrue(all(abs(curve.value[0] - 1.0) < 1e-12 for curve in curves))

    def test_nonnegative_conditional_fit_is_finite(self):
        curves = load_curves()
        coefficients, prediction = fit_coefficients(curves[0].time[:80], curves[0].value[:80], np.array([0.05, 0.5]))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((coefficients >= 0).all())

    def test_frozen_result_contract_when_present(self):
        if not OUTPUT_JSON.exists():
            self.skipTest("Stage 62 result has not been generated")
        payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        self.assertTrue(payload["protocol_frozen_before_fit"])
        self.assertTrue(payload["route_pass"])
        self.assertEqual(len(payload["boundary"]), 16)
        self.assertIn(payload["full_task"]["decision"], {
            "SUPPORTED_RANK_1", "SUPPORTED_RANK_2", "SUPPORTED_RANK_3", "INDETERMINATE",
        })


if __name__ == "__main__":
    unittest.main()
