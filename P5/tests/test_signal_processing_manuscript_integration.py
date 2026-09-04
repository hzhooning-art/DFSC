import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SignalProcessingManuscriptIntegrationTests(unittest.TestCase):
    def test_bilingual_manuscripts_include_frozen_common_budget_metrics(self):
        baseline = json.loads((ROOT / "results" / "common_budget_subspace_baselines.json").read_text(encoding="utf-8"))
        selective = baseline["summary"]["power_certified_selective"]
        self.assertAlmostEqual(selective["coverage"], 0.2942708333333333)
        for name in ("manuscript_en.tex", "manuscript_zh.tex"):
            text = (ROOT / "paper" / name).read_text(encoding="utf-8")
            for token in ("1,152", "0.2943", "0.1386", "0.6771", "0.0612"):
                self.assertIn(token, text)

    def test_bilingual_manuscripts_preserve_external_scope_boundary(self):
        sensitivity = json.loads((ROOT / "results" / "pva_group_composition_sensitivity.json").read_text(encoding="utf-8"))
        self.assertEqual(sensitivity["summary"]["groups"], 84)
        for name in ("manuscript_en.tex", "manuscript_zh.tex"):
            text = (ROOT / "paper" / name).read_text(encoding="utf-8")
            self.assertIn("84", text)
            self.assertIn("1/56", text)

    def test_submission_sources_use_signal_processing_positioning_and_double_spacing(self):
        english = (ROOT / "paper" / "manuscript_en.tex").read_text(encoding="utf-8")
        chinese = (ROOT / "paper" / "manuscript_zh.tex").read_text(encoding="utf-8")
        for text in (english, chinese):
            self.assertIn(r"\doublespacing", text)
        for token in (
            "Power-Certified Model-Order Selection with Abstention",
            "statistical signal processing",
            "detection and estimation",
            "spectral resolution",
        ):
            self.assertIn(token, english)

    def test_bilingual_manuscripts_include_preregistered_cable_transfer(self):
        result = json.loads((ROOT / "results" / "preregistered_cable_ageing_transfer.json").read_text(encoding="utf-8"))
        self.assertEqual(result["record"]["decision"], "EVIDENCE_AGAINST_RANK_1")
        self.assertFalse(result["thresholds_retuned_after_observing_outcome"])
        for name in ("manuscript_en.tex", "manuscript_zh.tex"):
            text = (ROOT / "paper" / name).read_text(encoding="utf-8")
            for token in ("0.001810", "647.951", "2,757"):
                self.assertIn(token, text)

    def test_bilingual_manuscripts_include_cable_window_sensitivity(self):
        result = json.loads((ROOT / "results" / "cable_window_sensitivity.json").read_text(encoding="utf-8"))
        self.assertEqual(result["matching_parent_decision_count"], 12)
        self.assertTrue(result["success_rule_passes"])
        for name in ("manuscript_en.tex", "manuscript_zh.tex"):
            text = (ROOT / "paper" / name).read_text(encoding="utf-8")
            for token in ("12", "399.989", "714.271"):
                self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
