import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from p4_external_subject_pilot import (  # noqa: E402
    LinearSolveAction,
    accepted,
    audit,
)


class ExternalSubjectPilotTests(unittest.TestCase):
    def test_clean_external_subject_passes_complete_suite(self):
        record = audit(LinearSolveAction, 0, None)
        self.assertTrue(accepted(record, "execution_evidence_suite"))

    def test_detached_gradient_escapes_value_only_test(self):
        record = audit(LinearSolveAction, 0, "detached_gradient")
        self.assertTrue(accepted(record, "full_batch_value_test"))
        self.assertFalse(accepted(record, "value_gradient_test"))

    def test_dtype_downgrade_is_explicitly_recorded(self):
        record = audit(LinearSolveAction, 0, "silent_dtype_downgrade")
        self.assertFalse(record["checks"]["dtype_conformance"])
        self.assertFalse(accepted(record, "execution_evidence_suite"))


if __name__ == "__main__":
    unittest.main()
