from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import external_comparison as e
from freeze_protocol import build_protocol


class ExternalTests(unittest.TestCase):
    def test_accuracy_band_requires_both_errors_and_valid_reference(self):
        self.assertFalse(e.qualifies({'status':'returned','reference_status':'resolved',
            'relative_l2':1e-5,'max_abs':.1,'max_abs_reference':1},1e-3))
        for status,ref in [('not_run_budget','resolved'),('returned','reference_unresolved')]:
            self.assertFalse(e.qualifies({'status':status,'reference_status':ref},1e-3))
        self.assertTrue(e.qualifies({'status':'returned','reference_status':'resolved',
            'relative_l2':1e-5,'max_abs':1e-5,'max_abs_reference':1},1e-3))
        self.assertEqual(e.timing_summary([.3,.1,.2]),{'median_seconds':.2,'samples':[.3,.1,.2]})

    def test_inverse_initialization_is_paired_and_final_checkpoint_not_best(self):
        self.assertEqual(e.initial_alpha(101),e.initial_alpha(101))
        self.assertGreaterEqual(e.initial_alpha(101),.6)
        self.assertLess(e.initial_alpha(101),.8)
        p=build_protocol(); p['a_inverse_steps']=1; p['reference_digits']=[65,85]
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'cp.json'
            result=e.train_alpha({'method':'selective','seed':101,'truth_alpha':.55,'steps':None},p,cp)
            self.assertEqual(result['completed_updates'],1)
            self.assertEqual(result['final_alpha'],result['trajectory'][-1]['alpha_after'])
            self.assertTrue(cp.is_file())
            self.assertFalse(result['is_sixty_step_run'])

    def test_preflight_is_explicitly_excluded_from_scientific_counts(self):
        row=e.evaluate_external({'kind':'preflight'},build_protocol())
        self.assertTrue(row['excluded_from_scientific_counts'])
        self.assertEqual(len(row['forward_cases']),5)
        self.assertTrue(all(r['status']=='returned' for r in row['forward_cases']))


if __name__=='__main__': unittest.main()
