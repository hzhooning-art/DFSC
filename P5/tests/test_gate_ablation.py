import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p5_memory_protocol import decide_transitions


class GateAblationTests(unittest.TestCase):
    def _transition(self, failed_gate=None):
        checks = {
            "bic": True,
            "prediction": True,
            "stability": True,
            "separation": True,
            "finite": True,
        }
        if failed_gate is not None:
            checks[failed_gate] = False
        return [{"from_rank": 1, "to_rank": 2, "gates": checks}]

    def test_full_protocol_refuses_a_single_gate_failure(self):
        result = decide_transitions(self._transition("prediction"))
        self.assertEqual(result["decision"], "INDETERMINATE")

    def test_leaving_out_the_failed_gate_changes_the_decision(self):
        result = decide_transitions(
            self._transition("prediction"),
            active_gates={"information", "stability", "separation"},
        )
        self.assertEqual(result["decision"], "SUPPORTED_RANK_2")

    def test_finite_output_gate_is_never_ablatable(self):
        result = decide_transitions(self._transition("finite"), active_gates=set())
        self.assertEqual(result["decision"], "INDETERMINATE")

    def test_unknown_gate_name_is_rejected(self):
        with self.assertRaises(ValueError):
            decide_transitions(self._transition(), active_gates={"mystery"})


if __name__ == "__main__":
    unittest.main()

