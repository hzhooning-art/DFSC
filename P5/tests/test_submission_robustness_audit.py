import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "submission_robustness_audit.json"


class SubmissionRobustnessAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not OUTPUT.exists():
            raise AssertionError(f"missing submission robustness artifact: {OUTPUT}")
        cls.payload = json.loads(OUTPUT.read_text(encoding="utf-8"))

    def test_all_public_tasks_declare_independent_units_and_leakage_controls(self):
        tasks = self.payload["public_task_audit"]
        self.assertEqual(set(tasks), {"pva", "gas_sensor", "hydraulic"})
        self.assertEqual(tasks["pva"]["independent_unit_count"], 3)
        self.assertEqual(tasks["gas_sensor"]["independent_unit_count"], 50)
        self.assertEqual(tasks["hydraulic"]["independent_unit_count"], 30)
        for task in tasks.values():
            self.assertTrue(task["independent_unit"])
            self.assertGreaterEqual(len(task["leakage_controls"]), 2)

    def test_noise_transfer_and_resampling_are_reported(self):
        noise = self.payload["noise_generator_transfer"]
        self.assertEqual(set(noise), {"iid_gaussian", "ar1", "ar2", "heteroscedastic"})
        for row in noise.values():
            self.assertIn("seed_cluster_bootstrap_95", row["separated_support"])
            self.assertIn("seed_cluster_bootstrap_95", row["coalesced_refusal"])

    def test_hydraulic_information_criterion_disagreement_is_preserved(self):
        sensitivity = self.payload["correlated_noise_model_sensitivity"]["hydraulic"]
        self.assertEqual(sensitivity["ordinary_bic_preferred_rank"], 2)
        self.assertEqual(sensitivity["ar1_profile_bic_preferred_rank"], 1)
        self.assertTrue(sensitivity["criterion_disagreement"])

    def test_audit_does_not_replace_primary_decisions(self):
        self.assertFalse(self.payload["changes_primary_public_decisions"])


if __name__ == "__main__":
    unittest.main()
