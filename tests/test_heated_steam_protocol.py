from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external" / "heated_steam"


class HeatedSteamProtocolTests(unittest.TestCase):
    def test_source_workbook_checksum(self) -> None:
        source = DATA / "2025_Garcia-Martinez_Summary_LabNumData.xlsx"
        self.assertTrue(source.exists())
        self.assertEqual(hashlib.md5(source.read_bytes()).hexdigest(), "d14aa1ee2a77890e75817ef0e185a363")

    @unittest.skipUnless((DATA / "manifest.json").exists(), "run prepare_heated_steam.py first")
    def test_standardized_csv_has_all_experiments(self) -> None:
        manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_doi"], "10.5281/zenodo.15064388")
        self.assertEqual(manifest["experiments"], 16)
        self.assertEqual(len(manifest["depth_coordinates_m"]), 10)
        with (DATA / "heated_steam_profiles.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreater(len(rows), 10000)
        self.assertEqual({int(row["experiment"]) for row in rows}, set(range(1, 17)))


if __name__ == "__main__":
    unittest.main()
