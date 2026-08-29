import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from probe_public_uci_gas_recovery import DATA, EXPECTED_SHA256, OUTPUT, load_curves, sha256  # noqa: E402


class PublicUCIGasRecoveryTests(unittest.TestCase):
    def test_frozen_source_and_inventory(self):
        curves = load_curves()
        self.assertEqual(sha256(DATA), EXPECTED_SHA256)
        self.assertEqual(len({row.unit for row in curves}), 50)
        self.assertEqual(len({row.channel for row in curves}), 16)
        self.assertEqual(len({row.group for row in curves}), 5)
        self.assertTrue(all(len(row.time) == 10 for row in curves))

    def test_result_contract_when_present(self):
        if not OUTPUT.exists():
            self.skipTest("Stage 63 result has not been generated")
        payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
        self.assertTrue(payload["protocol_frozen_before_fit"])
        self.assertTrue(payload["route_pass"])
        self.assertEqual(len(payload["threshold_sensitivity"]), 27)
        self.assertGreaterEqual(len(payload["baselines"]), 4)


if __name__ == "__main__":
    unittest.main()
