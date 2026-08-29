import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from probe_public_kupferdigital_relaxation import (  # noqa: E402
    ARCHIVE,
    EXPECTED_MD5,
    OUTPUT_JSON,
    file_md5,
    load_curves,
)


class PublicKupferDigitalRelaxationTests(unittest.TestCase):
    def test_experiment_module_bootstraps_project_import_path(self):
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(ROOT / 'experiments')!r}); "
            "import probe_public_kupferdigital_relaxation; "
            "import p5_memory_protocol"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
    def test_frozen_archive_checksum(self):
        self.assertEqual(file_md5(ARCHIVE), EXPECTED_MD5)

    def test_inventory_preserves_independent_units_and_material_groups(self):
        curves = load_curves()
        self.assertEqual(len(curves), 17)
        self.assertEqual(len({curve.unit for curve in curves}), 17)
        self.assertEqual({curve.group for curve in curves}, set("ABCDEFGHI"))

    def test_registered_curves_share_grid_and_are_normalized(self):
        curves = load_curves()
        reference_time = curves[0].time
        self.assertEqual(len(reference_time), 96)
        for curve in curves:
            self.assertTrue(np.allclose(curve.time, reference_time))
            self.assertAlmostEqual(float(curve.value[0]), 1.0, places=12)
            self.assertTrue(np.isfinite(curve.value).all())

    def test_frozen_result_contract_when_present(self):
        if not OUTPUT_JSON.exists():
            self.skipTest("KupferDigital result has not been generated")
        payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        self.assertTrue(payload["protocol_frozen_before_fit"])
        self.assertTrue(payload["checks"]["archive_checksum"])
        self.assertEqual(payload["source"]["independent_experiments"], 17)
        self.assertIn(payload["decision"]["decision"], {
            "SUPPORTED_RANK_1", "SUPPORTED_RANK_2", "SUPPORTED_RANK_3", "INDETERMINATE",
        })


if __name__ == "__main__":
    unittest.main()


