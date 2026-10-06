import importlib.util
import unittest
from pathlib import Path


class WorkflowControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("workflow_controls", root / "scripts/compare_workflows.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_nine_unique_prespecified_controls(self):
        self.assertEqual(len(self.module.SCENARIOS), 9)
        self.assertEqual(len(set(self.module.SCENARIOS)), 9)
        self.assertEqual(len(self.module.EXPECTED), 9)

    def test_strict_validator_rejects_empty_record(self):
        with self.assertRaises((KeyError, ValueError)):
            self.module.strict_plan_check({}, {})


if __name__ == "__main__":
    unittest.main()
