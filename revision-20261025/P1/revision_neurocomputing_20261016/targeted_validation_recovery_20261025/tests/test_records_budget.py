import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from records import write_checkpoint, write_terminal,read,sha,verify_bindings
from budget_runner import run_job, materialize_missing, remaining_budget,run_track


class RecordTests(unittest.TestCase):
    def test_environment_drift_is_not_silently_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_terminal(root/'freeze_receipt.json',{'bindings':{},'versions':{'numpy':'0.0-invalid'}})
            with self.assertRaisesRegex(ValueError,'version'):
                verify_bindings(root)

    def fixture_track(self,root):
        # Synthetic child only: deliberately truncate its result and exit 0.
        worker=root/'worker.py'
        worker.write_text('import sys\nfrom pathlib import Path\nPath(sys.argv[sys.argv.index("--output")+1]).write_text("{")\n')
        write_terminal(root/'protocol.json',{'track_budget_seconds':5,'worker_limit_seconds':2,'training_limit_seconds':2})
        write_terminal(root/'schedule.json',[{'id':'fake','track':'A','kind':'fake'}])
        write_terminal(root/'freeze_receipt.json',{'bindings':{p.name:{'path':p.name,'sha256':sha(p)}
                       for p in (worker,root/'protocol.json',root/'schedule.json')}})

    def test_partial_result_file_is_retained_as_error_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture_track(root)
            run_track('A',root)
            row=read(root/'results/A/terminal/fake.json')
            self.assertEqual(row['status'],'error')
            self.assertTrue(row['partial_result_preserved'])
            self.assertEqual(next((root/'results/A/attempts').glob('*/result.json')).read_text(),'{')

    def test_crashed_parent_charges_reserved_budget_and_never_overwrites_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture_track(root)
            write_checkpoint(root/'results/A/ledger.json',{'consumed_seconds':1.,'attempts':[],
                             'pending':{'id':'fake','limit_seconds':2.,'attempt':'previous'}})
            run_track('A',root,resume=True)
            self.assertGreaterEqual(read(root/'results/A/ledger.json')['consumed_seconds'],3.)
            terminal=root/'results/A/terminal/fake.json'; before=terminal.read_bytes()
            run_track('A',root,resume=True)
            self.assertEqual(terminal.read_bytes(),before)
            self.assertEqual(read(terminal)['status'],'interrupted')
            self.assertEqual(read(terminal)['kind'],'fake')

    def test_os_lock_prevents_concurrent_timed_tracks(self):
        from run_track import exclusive_track
        with tempfile.TemporaryDirectory() as tmp:
            with exclusive_track(Path(tmp)):
                with self.assertRaises(OSError):
                    with exclusive_track(Path(tmp)):
                        pass

    def test_nan_and_overwrite_rejected_without_corrupting_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint.json'
            write_checkpoint(p, {'completed_updates':1})
            with self.assertRaises(ValueError):
                write_checkpoint(p, {'bad':float('nan')})
            self.assertEqual(json.loads(p.read_text()), {'completed_updates':1})
            terminal=Path(tmp)/'terminal.json'
            write_terminal(terminal, {'status':'refused'})
            with self.assertRaises(FileExistsError):
                write_terminal(terminal, {'status':'completed'})
            self.assertEqual(json.loads(terminal.read_text())['status'], 'refused')

    def test_timeout_preserves_last_complete_checkpoint_and_kills_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint.json'
            write_checkpoint(p, {'completed_updates':2})
            result=run_job({'id':'sleep','command':[sys.executable,'-c','import time; time.sleep(10)']}, .2, p)
            self.assertEqual(result['status'], 'timeout')
            self.assertEqual(result['checkpoint']['completed_updates'], 2)
            self.assertIsNotNone(result['exit_code'])
            self.assertGreaterEqual(result['elapsed_seconds'], .2)

    def test_crash_and_budget_missing_are_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=run_job({'id':'crash','command':[sys.executable,'-c','raise RuntimeError("synthetic crash")']}, 3, root/'cp.json')
            self.assertEqual(result['status'], 'error')
            self.assertIn('synthetic crash', result['stderr'])
            materialize_missing([{'id':'a'},{'id':'b'}], root/'terminal')
            self.assertEqual(json.loads((root/'terminal/a.json').read_text())['status'], 'not_run_budget')
            self.assertEqual(remaining_budget({'consumed_seconds':12.5}, 15), 2.5)


if __name__ == '__main__':
    unittest.main()
