import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "stage62_boundary_factor_audit.json"


class Stage62BoundaryFactorTests(unittest.TestCase):
    def test_factor_audit_contract_when_present(self):
        if not OUTPUT.exists():
            self.skipTest("factor audit has not been generated")
        payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["tail_coverage_fixed_spacing"]), 4)
        self.assertEqual(len(payload["sampling_density_fixed_horizon"]), 4)
        self.assertEqual(len(payload["optimizer_start_budget"]), 4)
        for row in payload["sampling_density_fixed_horizon"]:
            self.assertIn("ar1_effective_sample_size_proxy", row["factor_diagnostics"])


if __name__ == "__main__":
    unittest.main()
