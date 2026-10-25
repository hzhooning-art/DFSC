from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import inverse_training as inv
from freeze_protocol import build_protocol,build_schedule,build_coordinates


class InverseTests(unittest.TestCase):
    def test_paired_initial_conditions_and_full_fixed_plan(self):
        p=build_protocol()
        jobs=[j for j in build_schedule(p,build_coordinates(set())) if j['track']=='C']
        self.assertEqual(len(jobs),72)
        for method in ('guard','selective','full_hp','original'):
            self.assertEqual(sum(j['method']==method for j in jobs),18)
        first=inv.scenario(4,'hard',101)
        second=inv.scenario(4,'hard',101)
        np.testing.assert_array_equal(first['raw_initial'],second['raw_initial'])
        np.testing.assert_array_equal(first['operator'],second['operator'])
        for size in (4,8,16):
            for regime in ('mild','hard'):
                self.assertLess(inv.scenario(size,regime,101)['maximum_box_radius'],16)

    def test_partial_deadline_checkpoint_cannot_count_as_full_recovery(self):
        p=build_protocol(); p.update(reference_digits=[65,85],deadline_monotonic=0)
        with tempfile.TemporaryDirectory() as tmp:
            result=inv.train_joint({'size':4,'regime':'mild','seed':101,'method':'selective'},p,Path(tmp)/'cp.json')
        self.assertEqual(result['completed_updates'],0)
        self.assertEqual(result['status'],'partial')
        self.assertFalse(result['recovery'])

    def test_guard_refusal_leaves_parameters_unchanged(self):
        p=build_protocol(); p['reference_digits']=[65,85]; p['c_updates']=1
        job={'size':4,'regime':'hard','seed':101,'method':'guard'}
        with tempfile.TemporaryDirectory() as tmp:
            row=inv.train_joint(job,p,Path(tmp)/'cp.json')
        self.assertEqual(row['status'],'refused')
        self.assertEqual(row['completed_updates'],0)
        self.assertEqual(row['raw_final'],row['raw_initial'])

    def test_one_step_checkpoint_contains_real_update_not_best_replacement(self):
        p=build_protocol(); p.update(reference_digits=[65,85],c_updates=1)
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'cp.json'
            row=inv.train_joint({'size':4,'regime':'mild','seed':101,'method':'selective'},p,cp)
            self.assertTrue(cp.is_file())
        self.assertEqual(row['completed_updates'],1)
        self.assertEqual(row['raw_final'],row['trajectory'][-1]['raw_after'])
        self.assertFalse(row['is_hundred_step_run'])
        self.assertFalse(row['recovery'])

    def test_nonfinite_gradient_blocks_promotion_without_update(self):
        p=build_protocol(); p.update(reference_digits=[65,85],c_updates=1)
        with tempfile.TemporaryDirectory() as tmp, patch.object(inv.a,'connected_gradients',
                return_value=(torch.tensor([float('nan'),0.],dtype=torch.float64),)):
            row=inv.train_joint({'size':4,'regime':'mild','seed':101,'method':'selective'},p,Path(tmp)/'cp.json')
        self.assertEqual(row['status'],'error')
        self.assertEqual(row['completed_updates'],0)
        self.assertEqual(row['raw_final'],row['raw_initial'])
        self.assertTrue(row['promotion_blocked'])


if __name__=='__main__': unittest.main()
