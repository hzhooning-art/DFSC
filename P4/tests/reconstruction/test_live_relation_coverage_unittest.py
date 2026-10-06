"""Ensure the seven-family live execution table does not overcount captures."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class LiveRelationCoverageTests(unittest.TestCase):
    def test_manifest_exactly_covers_admitted_families(self):
        benchmark = json.loads((ROOT / "benchmark/frozen.json").read_text(encoding="utf-8"))
        coverage = json.loads((ROOT / "results/reconstruction/live_relation_coverage.json")
                              .read_text(encoding="utf-8"))
        admitted = {row["case_id"] for row in benchmark["cases"]
                    if row["admission_state"] == "admitted_complete"}
        self.assertEqual(admitted, set(coverage["cases"]))
        self.assertEqual(coverage["admitted_family_count"], len(admitted))
        self.assertEqual(coverage["live_complete_family_count"], sum(
            row["status"] == "live_complete" for row in coverage["cases"].values()
        ))

    def test_live_rows_have_complete_plan_bound_reports(self):
        coverage = json.loads((ROOT / "results/reconstruction/live_relation_coverage.json")
                              .read_text(encoding="utf-8"))
        for case_id, row in coverage["cases"].items():
            with self.subTest(case_id=case_id):
                if row["status"] == "captured_pair_only":
                    self.assertNotIn("report", row)
                    continue
                self.assertEqual(row["status"], "live_complete")
                plan = json.loads((ROOT / row["plan"]).read_text(encoding="utf-8"))
                report = json.loads((ROOT / row["report"]).read_text(encoding="utf-8"))
                self.assertEqual(report["plan_digest"], plan["digest"])
                self.assertEqual(report["status"], "CONFORMANT")
                self.assertTrue(report["selected_relations_executed"])
                self.assertEqual(report["selected_relation_count"], len(plan["tests"]))
                self.assertEqual({item["test_id"] for item in plan["tests"]},
                                 {item["test_id"] for item in report["observations"]})
                self.assertTrue(all(item["outcome"] == "PASS" for item in report["observations"]))


if __name__ == "__main__":
    unittest.main()
