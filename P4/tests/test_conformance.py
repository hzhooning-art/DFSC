import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dfsc_protocol import CONFORMANCE_SCHEMA, canonical_json, evaluate_conformance, migrate_record


def valid_record(profile="application"):
    checks = {
        "value_accuracy": True,
        "gradient_accuracy": True,
        "batch_shape": True,
        "batch_independence": True,
        "repeatability": True,
        "ood_control": True,
        "long_horizon": True,
        "dtype_conformance": True,
        "device_local": True,
        "unit_consistency": True,
        "resource_reported": True,
        "calibration": True,
        "composition": True,
    }
    return {
        "schema": CONFORMANCE_SCHEMA,
        "component": {"name": "reference-propagator", "version": "1.0"},
        "profile": profile,
        "operating_domain": {"identifier": "stable-negative-spectrum-v1", "units": "dimensionless"},
        "coverage": {
            "scope_frozen": True,
            "sample_count": 24,
            "anchors": [
                "nominal",
                "boundary",
                "heterogeneous_batch",
                "perturbation",
                "execution_policy",
                "long_horizon",
                "application_composition",
            ],
        },
        "requested_execution": {"dtype": "float64", "device": "cpu"},
        "observed_execution": {"dtype": "float64", "device": "cpu"},
        "evidence": checks,
        "provenance": {"implementation": "python-api", "run_id": "unit-test"},
    }


class ConformanceTests(unittest.TestCase):
    def test_published_schema_matches_runtime_profiles(self):
        schema_path = Path(__file__).resolve().parents[1] / "spec" / "dfsc-dnc-conformance-v3.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema"]["const"], CONFORMANCE_SCHEMA)
        self.assertEqual(set(schema["properties"]["profile"]["enum"]), {"core", "extended", "application"})

    def test_application_profile_conforms(self):
        result = evaluate_conformance(valid_record())
        self.assertEqual(result["conformance"]["status"], "conformant")

    def test_dtype_downgrade_is_detected(self):
        record = valid_record("extended")
        record["observed_execution"]["dtype"] = "float32"
        result = evaluate_conformance(record)
        self.assertIn("dtype_conformance", result["conformance"]["failed_checks"])

    def test_missing_required_evidence_is_nonconformant(self):
        record = valid_record("core")
        del record["evidence"]["gradient_accuracy"]
        result = evaluate_conformance(record)
        self.assertEqual(result["conformance"]["missing_checks"], ["gradient_accuracy"])

    def test_v1_migration(self):
        record = valid_record("core")
        record["schema"] = "DFSC-DNC-Conformance-v1"
        del record["profile"]
        migrated = migrate_record(record)
        self.assertEqual(migrated["schema"], CONFORMANCE_SCHEMA)
        self.assertTrue(migrated["coverage"]["migration_requires_requalification"])
        self.assertEqual(evaluate_conformance(record)["conformance"]["status"], "nonconformant")

    def test_narrow_scope_cannot_pass_without_coverage_anchors(self):
        record = valid_record("core")
        record["coverage"] = {
            "scope_frozen": True,
            "sample_count": 1,
            "anchors": ["nominal"],
        }
        result = evaluate_conformance(record)
        self.assertEqual(result["conformance"]["status"], "nonconformant")
        self.assertIn("insufficient_samples", result["conformance"]["coverage_failures"])
        self.assertIn("missing_anchor:boundary", result["conformance"]["coverage_failures"])

    def test_cli_and_api_are_canonically_equal(self):
        record = valid_record()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            output = Path(directory) / "output.json"
            source.write_text(json.dumps(record, indent=2), encoding="utf-8")
            subprocess.run(
                [sys.executable, "-m", "dfsc_protocol.cli", str(source), str(output)],
                check=True,
            )
            self.assertEqual(output.read_text(encoding="utf-8").strip(), canonical_json(record))


if __name__ == "__main__":
    unittest.main()
