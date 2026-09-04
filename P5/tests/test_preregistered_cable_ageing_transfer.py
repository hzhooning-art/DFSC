import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "probe_preregistered_cable_ageing_transfer.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("probe_preregistered_cable_ageing_transfer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PreregisteredCableAgeingTransferTests(unittest.TestCase):
    def test_contract_freezes_stage72_adapter(self):
        contract = json.loads(MODULE.CONTRACT_FILE.read_text(encoding="utf-8"))
        self.assertEqual(contract["frozen_stage72_adapter"]["target_channels"], 6)
        self.assertEqual(contract["frozen_stage72_adapter"]["target_samples"], 24)
        self.assertEqual(contract["frozen_stage72_adapter"]["maximum_calibrated_noise"], 0.005)

    def test_source_hash_and_six_curves_match_contract(self):
        contract = json.loads(MODULE.CONTRACT_FILE.read_text(encoding="utf-8"))
        rows, audit = MODULE.load_curves(contract)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(audit), 6)
        self.assertTrue(all(item["finite_unique_points"] >= 2500 for item in audit))

    def test_result_preserves_frozen_rule_and_claim_boundary(self):
        payload = MODULE.run()
        self.assertFalse(payload["thresholds_retuned_after_observing_outcome"])
        self.assertIn(payload["record"]["decision"], {
            "EVIDENCE_AGAINST_RANK_1",
            "SUPPORTED_RANK_1",
            "INDETERMINATE_EVIDENCE",
            "INDETERMINATE_SCOPE",
        })
        self.assertIn("not prospective acquisition", payload["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
