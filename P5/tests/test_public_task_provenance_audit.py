import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "public_task_provenance_audit.json"

class PublicTaskProvenanceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not OUTPUT.exists():
            raise AssertionError(f"missing public provenance artifact: {OUTPUT}")
        cls.payload = json.loads(OUTPUT.read_text(encoding="utf-8"))

    def test_five_screened_sources_are_classified(self):
        sources = self.payload["sources"]
        self.assertEqual(set(sources), {"pva_gpe", "uci_gas", "uci_hydraulic", "kupferdigital", "hydrogel_candidate"})
        self.assertTrue(sources["kupferdigital"]["included"])
        self.assertFalse(sources["hydrogel_candidate"]["included"])

    def test_independent_units_and_evidence_tiers_are_explicit(self):
        sources = self.payload["sources"]
        self.assertEqual(sources["pva_gpe"]["independent_unit_count"], 3)
        self.assertEqual(sources["kupferdigital"]["independent_unit_count"], 17)
        self.assertEqual(sources["uci_gas"]["independent_unit_count"], 50)
        self.assertEqual(sources["uci_hydraulic"]["independent_unit_count"], 30)
        self.assertEqual(sources["pva_gpe"]["evidence_tier"], "small-sample support")
        self.assertEqual(sources["kupferdigital"]["evidence_tier"], "primary public positive task")

    def test_licenses_come_from_declared_metadata_fields(self):
        for name in ("pva_gpe", "kupferdigital", "hydrogel_candidate"):
            row = self.payload["sources"][name]
            self.assertEqual(row["license"], "CC-BY-4.0")
            self.assertEqual(row["license_metadata_field"], "metadata.license.id")
        for name in ("uci_gas", "uci_hydraulic"):
            self.assertEqual(self.payload["sources"][name]["license"], "CC-BY-4.0")

    def test_local_source_checksums_are_verified(self):
        for row in self.payload["sources"].values():
            if row.get("local_artifact"):
                self.assertTrue(row["checksum_verified"])

    def test_hydrogel_rejection_does_not_count_internal_trials_as_independent(self):
        row = self.payload["sources"]["hydrogel_candidate"]
        self.assertEqual(row["independent_unit_count_per_condition"], 2)
        self.assertIn("internal trial blocks", row["exclusion_reason"])

if __name__ == "__main__":
    unittest.main()