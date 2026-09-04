import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class JSSManuscriptIntegrationTests(unittest.TestCase):
    def test_bilingual_manuscripts_include_external_and_historical_results(self):
        result = json.loads((ROOT / "results" / "p4_historical_fault_derived_strategy.json").read_text(encoding="utf-8"))
        self.assertEqual(result["summary"]["equivalence_property_detections"], 32)
        for name in ("dfsc_primitive_protocol_en.tex", "dfsc_primitive_protocol_zh.tex"):
            text = (ROOT / "paper" / name).read_text(encoding="utf-8")
            self.assertIn("240", text)
            self.assertIn("32/32", text)
            self.assertIn("SciPy", text)
            self.assertIn("RQ3", text)

    def test_bilingual_manuscripts_include_cross_project_results(self):
        result = json.loads((ROOT / "results" / "p4_cross_project_external_subjects.json").read_text(encoding="utf-8"))
        complete = result["summary"]["execution_evidence_suite"]
        self.assertEqual(complete["injected_trials"], 252)
        self.assertEqual(complete["clean_trials"], 36)
        self.assertEqual(complete["project_subject_fault_clusters"], 21)
        for name in ("dfsc_primitive_protocol_en.tex", "dfsc_primitive_protocol_zh.tex"):
            text = (ROOT / "paper" / name).read_text(encoding="utf-8")
            for token in ("252", "28.57", "85.71", "36", "21"):
                self.assertIn(token, text)

    def test_bilingual_manuscripts_include_three_real_complete_pairs(self):
        pytorch = json.loads((ROOT / "results" / "p4_complete_historical_pair.json").read_text(encoding="utf-8"))
        scipy = json.loads((ROOT / "results" / "p4_scipy_complete_pair.json").read_text(encoding="utf-8"))
        resample = json.loads((ROOT / "results" / "p4_scipy_resample_poly_pair.json").read_text(encoding="utf-8"))
        self.assertEqual(pytorch["complete_pair_count"], 1)
        self.assertTrue(scipy["complete_pair"])
        self.assertTrue(resample["complete_pair"])
        self.assertEqual(resample["historical_family_ordinal"], 3)
        for name in ("dfsc_primitive_protocol_en.tex", "dfsc_primitive_protocol_zh.tex"):
            text = (ROOT / "paper" / name).read_text(encoding="utf-8")
            self.assertIn("1.11.0+cpu", text)
            self.assertIn("2.11.0+cu128", text)
            self.assertIn("fdb828...014", text)
            self.assertIn("1.14.1", text)
            self.assertIn("ce901...bc8", text)
            self.assertIn("#15620", text.replace("\\#", "#"))
            self.assertIn("3.001552", text)
            self.assertIn("0a7a72...f79", text)


if __name__ == "__main__":
    unittest.main()
