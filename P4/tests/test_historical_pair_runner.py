import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "pytorch_historical_pair_runner.py"
SPEC = importlib.util.spec_from_file_location("pytorch_historical_pair_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HistoricalPairRunnerTests(unittest.TestCase):
    def test_current_xlogy_confirms_fixed_role(self):
        result = MODULE.run_case("pytorch_80770", "fixed")
        self.assertTrue(result["role_confirmed"])
        self.assertFalse(result["observation"]["matches_reported_bug"])

    def test_current_xlogy_does_not_confirm_buggy_role(self):
        result = MODULE.run_case("pytorch_80770", "buggy")
        self.assertFalse(result["role_confirmed"])

    def test_every_case_has_upstream_issue_and_reported_version(self):
        for definition in MODULE.CASES.values():
            self.assertTrue(definition["upstream_issue"].startswith("https://github.com/"))
            self.assertTrue(definition["reported_buggy_version"])


if __name__ == "__main__":
    unittest.main()
