import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import recover_experiments as r


class RecoveryTests(unittest.TestCase):
    def test_legacy_contended_timings_not_exposed_as_eligible_cost_table(self):
        report={'a_accuracy_matched':[{'median_seconds':1.}], 'tracks':{}}
        event={'track_A':{'primary_uncontended_runtime_claim_supported':False}}
        qualified=r.qualify_report(report,event,source='original')
        self.assertEqual(qualified['a_accuracy_matched'],[])
        self.assertEqual(len(qualified['a_raw_accuracy_matched_costs_not_performance_evidence']),1)
        self.assertFalse(qualified['primary_uncontended_runtime_claim_supported'])

    def test_resume_cannot_replace_completed_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            terminal=root/'results/A/terminal/x.json'
            terminal.parent.mkdir(parents=True)
            terminal.write_text('{"status":"returned"}')
            before=terminal.read_bytes()
            self.assertEqual(r.next_action(root,'A'),'resume')
            self.assertEqual(terminal.read_bytes(),before)

    def test_all_old_b_rows_kept_and_no_new_b_job_is_scheduled_for_execution(self):
        jobs=[{'id':'a','track':'A'},{'id':'b','track':'B'},{'id':'c','track':'C'}]
        self.assertEqual([j['id'] for j in r.new_run_jobs(jobs)],['a','c'])

    def test_recovery_budget_is_bounded_and_does_not_loosen_numeric_targets(self):
        p=r.recovery_settings({'value_atol':1e-12,'gradient_atol':1e-10,'rtol':1e-9})
        self.assertEqual(p['value_atol'],1e-12)
        self.assertEqual(p['gradient_atol'],1e-10)
        self.assertEqual(p['rtol'],1e-9)
        self.assertEqual(p['track_budgets_seconds'],{'A':7200,'C':21600})
        self.assertEqual(p['training_limit_seconds'],900)


if __name__=='__main__': unittest.main()
