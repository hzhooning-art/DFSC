from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import stress_validation as s
from freeze_protocol import build_protocol,build_coordinates


class StressTests(unittest.TestCase):
    def test_inaccurate_return_blocks_promotion_but_refusal_does_not_pass_coverage(self):
        p=build_protocol(); ref={'status':'resolved','values':[1.,.5,.25]}
        bad=s.assess({'status':'returned','values':[2.,.5,.25]},ref,p,True)
        self.assertEqual(bad['outcome'],'inaccurate_return')
        self.assertTrue(bad['promotion_blocked'])
        refused=s.assess({'status':'refused'},ref,p,True)
        self.assertEqual(refused['outcome'],'refused')
        self.assertFalse(s.useful_repair_gate([refused]))
        unresolved=s.assess({'status':'returned','values':[2.,.5,.25]},
                            {'status':'reference_unresolved'},p,True)
        self.assertEqual(unresolved['outcome'],'reference_unresolved')
        self.assertFalse(unresolved['promotion_blocked'])

    def test_outside_domain_uses_actual_interface_exception(self):
        p=build_protocol()
        coordinate={'id':'test-outside','alpha':.45-1e-8,'radius':.2,
                    'within_domain':False,'laplace_selected':False,'stratum':'alpha_edge'}
        row=s.evaluate_stress({'coordinate':coordinate},p)
        repaired=next(r for r in row['methods'] if r['method']=='full_hp')
        self.assertEqual(repaired['status'],'refused')
        self.assertEqual(repaired['error_type'],'ValueError')
        self.assertFalse(row['promotion_blocked'])

    def test_boundary_overlaps_are_recorded_not_filtered(self):
        rows=build_coordinates({(.45,.2)})
        self.assertEqual(len(rows),336)
        self.assertTrue(next(r for r in rows if r['alpha']==.45 and r['radius']==.2)['overlaps_previous'])


if __name__=='__main__': unittest.main()
