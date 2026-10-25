from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import report_validation as r
from freeze_protocol import build_protocol


class ReportTests(unittest.TestCase):
    def test_missing_and_unresolved_remain_in_denominators(self):
        jobs=[{'id':'a','track':'A','kind':'forward','problem':'scalar','alpha':.5,'method':'fdeint','steps':64},
              {'id':'b','track':'A','kind':'forward','problem':'scalar','alpha':.5,'method':'pycaputo','steps':64}]
        rows=r.complete_rows(jobs,[dict(jobs[0],status='reference_unresolved')])
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[1]['status'],'not_run_budget')
        report=r.summarize(jobs,rows,build_protocol())
        self.assertEqual(report['tracks']['A']['planned'],2)
        self.assertEqual(report['tracks']['A']['status_counts']['reference_unresolved'],1)
        self.assertEqual(report['a_accuracy_matched'],[])

    def test_duplicate_terminal_ids_and_wrong_method_fail_audit(self):
        jobs=[{'id':'a','track':'A','method':'selective'}]
        with self.assertRaises(ValueError):
            r.complete_rows(jobs,[dict(jobs[0],status='returned'),dict(jobs[0],status='returned')])
        with self.assertRaises(ValueError):
            r.complete_rows(jobs,[{'id':'a','track':'A','method':'original','status':'returned'}])

    def test_partial_pair_is_not_hundred_step_match_or_speedup(self):
        rows=[{'id':'s','size':4,'regime':'hard','seed':101,'method':'selective','status':'partial',
               'trajectory':[{'step':1,'raw_after':[.4,-.1]}]},
              {'id':'h','size':4,'regime':'hard','seed':101,'method':'full_hp','status':'completed',
               'trajectory':[{'step':1,'raw_after':[.4,-.1]}]}]
        pairs=r.compare_pairs(rows,build_protocol())
        self.assertEqual(len(pairs),1)
        self.assertFalse(pairs[0]['full_hundred_step_pair'])
        self.assertNotIn('full_run_cost_ratio',pairs[0])
        self.assertEqual(r.compare_pairs([],build_protocol()),[])

    def test_mostly_refused_is_not_full_coverage_success(self):
        p=build_protocol()
        jobs=[{'id':'b','track':'B','kind':'stress','coordinate':{'stratum':'uniform','within_domain':True,'alpha':.7,'radius':1.}}]
        rows=[dict(jobs[0],status='completed',methods=[{'method':'full_hp','outcome':'refused','status':'refused'}])]
        report=r.summarize(jobs,rows,p)
        self.assertFalse(report['b_useful_observed_repair'])
        self.assertFalse(report['all_tracks_complete'])

    def test_zip_path_traversal_is_rejected_before_extraction(self):
        import tempfile
        import zipfile
        from package_evidence import verify_archive
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'bad.zip'
            with zipfile.ZipFile(archive,'w') as out:
                out.writestr('../escape.txt','bad')
            with self.assertRaises(ValueError):
                verify_archive(archive)


if __name__=='__main__': unittest.main()
