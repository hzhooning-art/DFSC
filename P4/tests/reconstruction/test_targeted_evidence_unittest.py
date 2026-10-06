"""Check saved claims against their records; this does not rerun subjects."""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "results/software_validation_20261006"


class TargetedEvidenceTests(unittest.TestCase):
    def test_workflow_protocol_binding_and_nine_rows(self):
        summary = json.loads((EVIDENCE / "workflows/summary.json").read_text())
        protocol = ROOT / "docs/SCP_TARGETED_VALIDATION_PROTOCOL.md"
        self.assertEqual(summary["protocol_sha256"], hashlib.sha256(protocol.read_bytes()).hexdigest())
        self.assertEqual(len(summary["rows"]), 9)
        self.assertEqual(len({row["scenario"] for row in summary["rows"]}), 9)
        for row in summary["rows"]:
            directory = EVIDENCE / "workflows" / row["scenario"]
            strict = json.loads((directory / "strict_scripted.json").read_text())
            self.assertEqual(strict["status"], row["expected"])
            self.assertEqual(row["p4"], row["expected"])
            self.assertEqual(row["strict_scripted"], row["expected"])
            process = json.loads((directory / "p4_process.json").read_text())
            if row["expected"] == "REFUSED":
                self.assertEqual(process["exit_code"], 2)
                self.assertFalse((directory / "p4.json").exists())
            else:
                self.assertEqual(process["exit_code"], 0)
                self.assertEqual(json.loads((directory / "p4.json").read_text())["status"], row["expected"])

    def test_workflow_worker_binding(self):
        summary = json.loads((EVIDENCE / "workflows/summary.json").read_text())
        worker = ROOT / "examples/expm1_relation_worker.py"
        self.assertEqual(summary["worker_file_sha256"], hashlib.sha256(worker.read_bytes()).hexdigest())

    def test_ode_positive_and_bias_records(self):
        for name, expected in (("positive", "CONFORMANT"), ("biased", "NONCONFORMANT")):
            report = json.loads((EVIDENCE / "ode" / (name + ".json")).read_text())
            self.assertEqual(report["status"], expected)
            self.assertTrue(report["selected_relations_executed"])
            self.assertEqual(len(report["observations"]), 3)
            self.assertEqual(report["reported_subject_invocations"], 10)
            hashes = {value for row in report["observations"] for key, value in row["command_file_sha256"].items()
                      if key.endswith("ode_decay_worker.py")}
            self.assertEqual(hashes, {hashlib.sha256((ROOT / "examples/ode_decay_worker.py").read_bytes()).hexdigest()})


if __name__ == "__main__":
    unittest.main()
