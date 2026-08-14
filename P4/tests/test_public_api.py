from pathlib import Path
import unittest

from dfsc_protocol import load_profile, load_registry


ROOT = Path(__file__).resolve().parents[1]


class PublicApiTests(unittest.TestCase):
    def test_registry_schema(self):
        record = load_registry(ROOT / "results" / "p4_primitive_protocol_registry.json")
        self.assertEqual(record["schema"], "DFSC-Primitive-Protocol-v1")
        self.assertGreaterEqual(len(record["records"]), 4)
        mlsl = next(row for row in record["records"] if row["backend"] == "MLSL")
        self.assertEqual(mlsl["validation_artifact"], "p4_mlsl_protocol_validation.json")
        self.assertTrue(all(mlsl[key] == "pass" for key in record["required_dimensions"]))
        self.assertEqual(record["real_data_evidence"]["status"], "conformant")
        self.assertGreaterEqual(len(record["real_data_evidence"]["datasets"]), 3)


    def test_profile_schema(self):
        record = load_profile(ROOT / "results" / "p4_primitive_profile.json")
        self.assertEqual(record["schema"], "DFSC-Primitive-Profile-v1")
        self.assertTrue(all(row["outputs_finite"] for row in record["rows"]))


if __name__ == "__main__":
    unittest.main()
