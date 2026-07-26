from __future__ import annotations

import unittest

import dfsc


class DfscDatasetContractTests(unittest.TestCase):
    def test_benchmark_targets_report_integration_status(self) -> None:
        targets = dfsc.benchmark_targets()
        self.assertGreaterEqual(len(targets), 2)
        statuses = {item["status"] for item in targets}
        self.assertIn("external-required", statuses)
        self.assertIn("integrated-local-license-review", statuses)

    def test_manifest_validation_reports_missing_fields(self) -> None:
        ok, missing = dfsc.validate_dataset_manifest({"name": "demo"})
        self.assertFalse(ok)
        self.assertIn("citation", missing)
        self.assertIn("tensors", missing)

    def test_manifest_validation_accepts_complete_schema(self) -> None:
        manifest = {
            "name": "demo",
            "domain": "fractional dynamics",
            "task": "inverse",
            "source": "local",
            "license": "test",
            "citation": "test",
            "splits": {"train": "all"},
            "tensors": {"inputs": "inputs.pt"},
        }
        ok, missing = dfsc.validate_dataset_manifest(manifest)
        self.assertTrue(ok)
        self.assertEqual(missing, ())


if __name__ == "__main__":
    unittest.main()
