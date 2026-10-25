import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import freeze_protocol as f
from records import verify_bindings


class ProtocolTests(unittest.TestCase):
    def test_standalone_freeze_records_missing_git_without_fabricating_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(f.subprocess,'check_output',side_effect=subprocess.CalledProcessError(128,['git'])):
                receipt=f.freeze(Path(tmp),dependencies={},previous=set())
            self.assertIsNone(receipt['git_commit'])
            self.assertEqual(receipt['git_provenance_status'],'unavailable')
            verify_bindings(Path(tmp))

    def test_random_points_are_new_and_boundary_probes_are_not_replaced(self):
        rows = f.build_coordinates({(.6, 3.4)})
        random = [r for r in rows if r['stratum'] in ('uniform', 'log_uniform')]
        self.assertEqual(len(rows), 336)
        self.assertEqual(len(random), 288)
        self.assertEqual(len({(r['alpha'], r['radius']) for r in random}), 288)
        self.assertTrue(all((r['alpha'], r['radius']) != (.6, 3.4) for r in random))
        self.assertEqual(rows, f.build_coordinates({(.6, 3.4)}))
        self.assertEqual(sum(r['laplace_selected'] for r in rows), 16)
        self.assertEqual(sum(not r['within_domain'] for r in rows), 8)
        selected = sorted(r['id'] for r in rows if r['laplace_selected'])
        self.assertEqual(len(set(selected)), 16)

    def test_all_scheduled_ids_unique_and_method_cells_retained(self):
        jobs = f.build_schedule(f.build_protocol(), f.build_coordinates(set()))
        self.assertEqual(len({j['id'] for j in jobs}), len(jobs))
        self.assertEqual(sum(j['track']=='B' for j in jobs), 336)
        self.assertEqual(sum(j['track']=='C' for j in jobs), 72)
        self.assertEqual(sum(j['track']=='A' and j['kind']=='inverse' for j in jobs), 24)
        self.assertEqual(sum(j['track']=='A' and j['kind']=='forward' for j in jobs), 78)
        self.assertEqual(sum(j['track']=='A' and j['kind']=='gradient' for j in jobs), 72)

    def test_freeze_rejects_overwrite_and_source_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source.py'
            source.write_text('x=1\n')
            f.freeze(root, dependencies={'source.py':source}, previous=set())
            verify_bindings(root)
            with self.assertRaises(FileExistsError):
                f.freeze(root, dependencies={'source.py':source}, previous=set())
            source.write_text('x=2\n')
            with self.assertRaises(ValueError):
                verify_bindings(root)

    def test_budget_order_covers_operators_and_strata_before_repeat_seeds(self):
        jobs=f.build_schedule(f.build_protocol(),f.build_coordinates(set()))
        c=[j for j in jobs if j['track']=='C']
        self.assertEqual(len({(j['size'],j['regime']) for j in c[:24]}),6)
        b=[j for j in jobs if j['track']=='B']
        self.assertEqual({j['coordinate']['stratum'] for j in b[:3]}, {'uniform','log_uniform','branch'})


if __name__ == '__main__':
    unittest.main()
