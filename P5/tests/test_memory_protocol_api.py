import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from p5_memory_protocol import CurveRecord, decide, evaluate, fit, report  # noqa: E402


class MemoryProtocolAPITests(unittest.TestCase):
    def setUp(self):
        time = np.linspace(0, 8, 12)
        self.curves = []
        for group in ("a", "b", "c"):
            for channel, amplitude in enumerate((0.7, 1.1)):
                value = 0.1 + amplitude * np.exp(-0.35 * time)
                self.curves.append(CurveRecord(f"{group}-unit", group, str(channel), time, value))

    def test_stable_fit_evaluate_decide_report_surface(self):
        fitted = fit(self.curves, 1, starts=2)
        self.assertTrue(fitted["success"])
        evaluated = evaluate(self.curves, ranks=(1, 2), starts=2)
        outcome = decide(evaluated)
        self.assertIn(outcome["decision"], {"SUPPORTED_RANK_1", "SUPPORTED_RANK_2", "INDETERMINATE"})
        with tempfile.TemporaryDirectory() as directory:
            path = report({"ok": True}, Path(directory) / "record.json")
            self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["ok"])


if __name__ == "__main__":
    unittest.main()
