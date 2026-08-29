import sys
import unittest
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from freeze_stage61_evidence import build_manifest  # noqa: E402


class Stage61EvidenceFreezeTests(unittest.TestCase):
    def test_confirmatory_chain_is_complete_and_scope_bounded(self):
        manifest = build_manifest()
        self.assertTrue(manifest["freeze_pass"])
        self.assertIn("No independent external team has reproduced the artifact.", manifest["claims"]["not_claimed"])
        self.assertEqual(manifest["headline_metrics"]["scope_status_stage60"]["external_baseline"], "SUPPORTED")
        self.assertEqual(manifest["headline_metrics"]["scope_status_stage60"]["short_window"], "SCOPE_LIMITED")


if __name__ == "__main__":
    unittest.main()
