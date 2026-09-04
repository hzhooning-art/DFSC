import importlib.util
import math
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "probe_pva_group_composition_sensitivity.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("probe_pva_group_composition_sensitivity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PVAGroupCompositionSensitivityTests(unittest.TestCase):
    def test_all_six_of_nine_subsets_are_unique(self):
        groups = MODULE.enumerate_groups(MODULE.pva.load_curves())
        self.assertEqual(len(groups), math.comb(9, 6))
        self.assertEqual(len({group_id for group_id, _ in groups}), len(groups))
        self.assertTrue(all(len(rows) == 6 for _, rows in groups))

    def test_run_preserves_frozen_adapter(self):
        result = MODULE.run()
        self.assertEqual(result["summary"]["groups"], 84)
        self.assertFalse(result["design"]["adapter_or_certificate_retuned"])


if __name__ == "__main__":
    unittest.main()
