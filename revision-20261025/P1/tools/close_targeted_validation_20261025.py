"""Append-only closeout of the frozen shared-host experiment and one I/O retry.

Never rewrites either experiment, its receipts, statuses or terminal records.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

P1 = Path(__file__).resolve().parents[1]
REV = P1 / 'revision_neurocomputing_20261016'
RUN = REV / 'targeted_validation_recovery_20261025'
OLD = REV / 'targeted_validation_20261023'
OUT = REV / 'targeted_validation_closure_20261025'
sys.path.insert(0, str(RUN))
from records import read, sha, verify_bindings, write_terminal


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def reconcile():
    """Prove content identity via saved original receipts, not rewritten digests."""
    OUT.mkdir(exist_ok=True)
    verify_bindings(RUN)
    verify_bindings(OLD)
    archive = REV / 'archive/revise_delivery_20261022_before_visual_update_20261025'
    prior = read(RUN / 'prior_preservation.json')
    changes = []
    for rel, expected in prior['files'].items():
        actual = sha(OLD / rel)
        if actual != expected:
            if rel != 'freeze_receipt.json':
                raise ValueError('Scientific or prior evidence changed: ' + rel)
            changes.append(dict(path=rel, expected=expected, actual=actual))
    proofs = []
    for folder, expected in [
        (OLD, prior['files']['freeze_receipt.json']),
        (RUN, read(RUN / 'SHARED_HOST_AMENDMENT.json')['frozen_recovery_receipt_sha256']),
    ]:
        saved = P1 / 'docs/relocation_20261025' / (folder.name + '_freeze_receipt_original.json')
        assert sha(saved) == expected, ('Original receipt unavailable', str(saved))
        original, current = read(saved), read(folder / 'freeze_receipt.json')
        assert set(original) == set(current)
        for key in set(original) - {'bindings', 'preserved_artifacts'}:
            assert original[key] == current[key], key
        assert set(original['bindings']) == set(current['bindings'])
        moved = []
        for key, binding in original['bindings'].items():
            existing = current['bindings'][key]
            assert binding['sha256'] == existing['sha256'], key
            assert sha(folder / existing['path']) == binding['sha256'], key
            if binding['path'] != existing['path']:
                moved.append(dict(name=key, before=binding['path'], after=existing['path'], sha256=binding['sha256']))
        assert sorted(original['preserved_artifacts'].values()) == sorted(current['preserved_artifacts'].values())
        for rel, digest in current['preserved_artifacts'].items():
            assert sha(REV / rel) == digest, rel
        proofs.append(dict(experiment=folder.name, original_receipt=str(saved.relative_to(P1)),
                          original_sha256=expected, relocated_sha256=sha(folder / 'freeze_receipt.json'),
                          all_scientific_binding_hashes_unchanged=True, relocated_paths=moved))
    record = dict(created_utc=now(), reason='Concurrent manuscript archival relocated paths in two freeze receipts.',
                  authorization='User requested completion of four closeout items and replacement of current delivery.',
                  original_inventory_files=len(prior['files']), original_inventory_exceptions=changes,
                  exact_original_receipts_verified=True, proofs=proofs,
                  original_status_and_receipts_not_overwritten=True,
                  shared_host_runtime_not_qualified=True)
    target = OUT / 'PROVENANCE_RECONCILIATION.json'
    if target.exists():
        previous = read(target)
        assert previous['proofs'] == proofs
    else:
        write_terminal(target, record)
    return record


def retry():
    reconcile()
    directory = OUT / 'io_retry'
    directory.mkdir(exist_ok=False)
    identity = 'C-4-hard-307-full_hp'
    original = RUN / 'results/C/terminal' / (identity + '.json')
    bad = read(original)
    assert bad['status'] == 'error' and bad['error_type'] == 'PermissionError'
    assert bad['completed_updates'] == 64
    job = next(j for j in read(RUN / 'schedule.json') if j['id'] == identity)
    write_terminal(directory / 'job.json', job)
    amendment = dict(declared_utc=now(), original_terminal_sha256=sha(original),
                     original_id=identity, original_failure=bad['error'],
                     rationale='One checkpoint replace PermissionError; rerun entire original task from the same initialization.',
                     not_additional_independent_sample=True, overwrite_original=False,
                     numerical_settings_and_candidate_unchanged=True, protocol_sha256=sha(RUN / 'protocol.json'),
                     limit_seconds=900, shared_host_runtime_not_qualified=True,
                     monitoring_policy='No checkpoint content reads while worker is active; observe process and file metadata only.')
    write_terminal(directory / 'RETRY_AMENDMENT.json', amendment)
    from budget_runner import run_job
    command = [sys.executable, '-B', '-X', 'utf8', str(RUN / 'worker.py'), '--root', str(RUN),
               '--job', str(directory / 'job.json'), '--output', str(directory / 'result.json'),
               '--checkpoint', str(directory / 'checkpoint.json'), '--limit', '900']
    result = run_job(dict(id=identity, command=command), 900, directory / 'checkpoint.json')
    write_terminal(directory / 'process.json', result)
    payload = read(directory / 'result.json') if (directory / 'result.json').exists() else {}
    terminal = dict(job, **payload, retry_of=identity, original_terminal_sha256=sha(original))
    write_terminal(directory / 'terminal.json', terminal)
    print(json.dumps(dict(retry_finished=True, status=terminal.get('status'),
                          updates=terminal.get('completed_updates'), recovery=terminal.get('recovery'))), flush=True)


def audit():
    reconcile()
    from report_validation import audit as base_audit, summarize, complete_rows
    observed = base_audit(RUN)
    rows = [read(p) for p in sorted((RUN / 'results').glob('*/terminal/*.json'))]
    retried = read(OUT / 'io_retry/terminal.json')
    assert retried['status'] == 'completed' and retried['completed_updates'] == 100 and retried['recovery']
    jobs = read(RUN / 'schedule.json')
    effective = [retried if r['id'] == retried['id'] else r for r in rows]
    adjusted = summarize(jobs, effective, read(RUN / 'protocol.json'))
    # No performance rankings survive shared-host qualification.
    for report in (observed, adjusted):
        report['raw_accuracy_qualified_observations'] = report.pop('a_accuracy_matched')
        report['primary_uncontended_runtime_claim_supported'] = False
        for pair in report['c_pairs']:
            if 'full_run_cost_ratio' in pair:
                pair['raw_shared_host_cost_ratio_not_performance_evidence'] = pair.pop('full_run_cost_ratio')
    statistics = {}
    for method in ('selective', 'full_hp', 'original', 'fdeint', 'pycaputo'):
        forwards = [r for r in rows if r['track']=='A' and r.get('kind')=='forward' and r.get('method')==method]
        gradients = [r for r in rows if r['track']=='A' and r.get('kind')=='gradient' and r.get('method')==method]
        checks = [r['continuous_gradient_check'] for r in gradients if 'continuous_gradient_check' in r]
        statistics[method] = dict(forward_cells=len(forwards), max_forward_relative_l2=max(r['relative_l2'] for r in forwards),
                                 forward_pass_1e3=sum(r['qualified_bands']['0.001'] for r in forwards),
                                 forward_pass_1e4=sum(r['qualified_bands']['0.0001'] for r in forwards),
                                 gradient_cells=len(gradients), continuous_gradient_checks=len(checks),
                                 continuous_gradient_pass=sum(c['all_pass'] for c in checks),
                                 gradient_errors=sum(r['status']=='error' for r in gradients))
    document = dict(created_utc=now(), original_recovery=observed,
                    declared_io_retry_analysis=adjusted, a_method_statistics=statistics,
                    retry_is_not_extra_sample=True, B_reused_original_records=336,
                    raw_terminal_sha256={str(p.relative_to(REV)):sha(p) for p in sorted((RUN/'results').glob('*/terminal/*.json'))},
                    retry_terminal_sha256=sha(OUT/'io_retry/terminal.json'),
                    original_all_success=False, no_global_certificate=True,
                    no_uncontended_runtime_claim=True)
    write_terminal(OUT / 'AUDIT.json', document)
    print(json.dumps(dict(a=statistics, observed_C=observed['tracks']['C'],
                          retry_C=adjusted['tracks']['C'], pairs=adjusted['c_full_paired_matches'])), flush=True)


def regressions():
    from budget_runner import run_job
    # Preserve the restricted-sandbox attempt (including its timeout). This
    # separate attempt permits Windows process-tree and temporary-file tests.
    directory = OUT / 'verification_unrestricted'
    directory.mkdir(exist_ok=True)
    suites = [('recovery', RUN/'tests'), ('prior_core', REV/'tools'),
              ('prior_scalar', REV/'repair_feasibility'), ('prior_spectral', REV/'repair_spectral_integration'),
              ('prior_delivery', REV/'revise_delivery_20261022')]
    for name, location in suites:
        target = directory/(name+'.json')
        if target.exists():
            record=read(target)
            assert record['status']=='completed', ('Previous failure retained',name)
            continue
        command=[sys.executable,'-B','-X','utf8','-m','unittest','discover','-v','-s',str(location),'-p','test_*.py']
        record=run_job(dict(id=name,command=command),240,directory/'nonexistent.json')
        write_terminal(target,record)
        print(name,record['status'],record['stderr'][-300:],flush=True)
        if record['status']!='completed':
            raise RuntimeError('Regression failed; retained '+name)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['reconcile','retry','audit','regressions'])
    args=parser.parse_args()
    globals()[args.phase]()
