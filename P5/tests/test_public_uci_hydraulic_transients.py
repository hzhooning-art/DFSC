import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from probe_public_uci_hydraulic_transients import EXPECTED_SHA256, load_curves, sha256, ARCHIVE  # noqa: E402


class PublicUCIHydraulicTests(unittest.TestCase):
    def test_frozen_source_and_independent_units(self):
        self.assertEqual(sha256(ARCHIVE), EXPECTED_SHA256)
        curves = load_curves()
        self.assertEqual(len(curves), 120)
        self.assertEqual(len({row.unit for row in curves}), 30)
        self.assertEqual(len({row.group for row in curves}), 3)
        self.assertEqual(len({row.channel for row in curves}), 4)


if __name__ == "__main__":
    unittest.main()
