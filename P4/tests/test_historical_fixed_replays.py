import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "p4_historical_fixed_replays.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("p4_historical_fixed_replays", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HistoricalFixedReplayTests(unittest.TestCase):
    def test_capture_has_runner_digest_and_no_complete_pair(self):
        result = MODULE.run()
        self.assertEqual(len(result["runner_sha256"]), 64)
        self.assertEqual(result["summary"]["complete_pairs"], 0)
        self.assertEqual(result["summary"]["fixed_roles_confirmed"], result["summary"]["executed"])


if __name__ == "__main__":
    unittest.main()
