import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "probe_external_power_certificate_transfer.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("probe_external_power_certificate_transfer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

from p5_memory_protocol import CurveRecord


class ExternalPowerCertificateTransferTests(unittest.TestCase):
    def curves(self, samples=48):
        time = np.linspace(2.0, 10.0, samples)
        return [
            CurveRecord(str(i), "g", str(i), time, 0.2 + (1.0 + 0.1 * i) * np.exp(-0.4 * time))
            for i in range(6)
        ]

    def test_standardization_has_frozen_shape_and_endpoints(self):
        rows, scope = MODULE.standardize_group("test", "g", self.curves())
        self.assertTrue(scope["eligible"])
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(rows[0].time), 24)
        self.assertAlmostEqual(float(rows[0].time[-1]), 16.0)
        self.assertAlmostEqual(float(rows[0].value[0]), 1.0)
        self.assertAlmostEqual(float(rows[0].value[-1]), 0.0)

    def test_short_source_is_scope_refused(self):
        rows, scope = MODULE.standardize_group("test", "g", self.curves(samples=10))
        self.assertIsNone(rows)
        self.assertEqual(scope["reason"], "fewer_than_24_observed_samples")

    def test_transfer_declares_no_external_retuning(self):
        self.assertEqual(MODULE.TARGET_CHANNELS, 6)
        self.assertEqual(MODULE.TARGET_SAMPLES, 24)


if __name__ == "__main__":
    unittest.main()
