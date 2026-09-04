import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "submission_jss_20260903"


class JSSSubmissionReadinessTests(unittest.TestCase):
    def test_required_submission_documents_exist(self):
        expected = {"README.md", *(f"0{i}_{name}" for i, name in (
            (1, "highlights.txt"), (2, "title_page.txt"),
            (3, "cover_letter.txt"), (4, "declarations.txt"),
            (5, "data_and_code_statement.txt"),
            (6, "evidence_claim_matrix.md"),
            (7, "submission_readiness_checklist.md"),
            (8, "local_integrity_manifest.txt"),
        ))}
        self.assertEqual({p.name for p in PACK.iterdir()}, expected)

    def test_highlights_follow_elsevier_length_rule(self):
        lines = [x for x in (PACK / "01_highlights.txt").read_text(encoding="utf-8").splitlines() if x]
        self.assertGreaterEqual(len(lines), 3)
        self.assertLessEqual(len(lines), 5)
        self.assertTrue(all(len(x) <= 85 for x in lines), [(len(x), x) for x in lines])

    def test_metadata_and_claim_boundaries_match_active_paper(self):
        title = (PACK / "02_title_page.txt").read_text(encoding="utf-8")
        cover = (PACK / "03_cover_letter.txt").read_text(encoding="utf-8")
        matrix = (PACK / "06_evidence_claim_matrix.md").read_text(encoding="utf-8")
        self.assertIn("Executable Evidence for Testing Differentiable Numerical Components", title)
        self.assertIn("Corresponding author: Ning Hu", title)
        self.assertIn("Journal of Systems and Software", cover)
        for token in ("252", "36", "three complete pairs", "three families"):
            self.assertIn(token, matrix)

    def test_integrity_manifest_matches_local_artifacts(self):
        import hashlib

        lines = (PACK / "08_local_integrity_manifest.txt").read_text(encoding="utf-8").splitlines()
        entries = [line.split("  ", 1) for line in lines if len(line) > 66 and line[64:66] == "  "]
        self.assertEqual(len(entries), 8)
        workspace = ROOT.parent
        for expected, relative in entries:
            actual = hashlib.sha256((workspace / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
