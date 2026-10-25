"""Explicit shared-host amendment. Frozen science is unchanged; timings unqualified.

The strict entry stopped before any scientific job because other Python services
exist. This separate entry permits numerical execution without a clean-cost claim.
It never kills those services, patches frozen modules or resets track budgets.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

import recover_experiments as r
from records import read,sha,write_terminal,write_checkpoint,verify_bindings
from report_validation import audit,report_text
from run_track import exclusive_track
from budget_runner import run_track

HERE=Path(__file__).resolve().parent


def qualified_shared_report(report,checks):
    result=r.qualify_report(report,{'track_A':{'primary_uncontended_runtime_claim_supported':False},
                                  'checks':checks},source='original')
    result.update(source_experiment='recovery-20261025-shared-host',
                  shared_host_observational_execution=True,
                  runtime_qualification_scope='Shared host: no uncontended performance claim even if sampled services appear idle',
                  B_reused_without_rerun=True)
    return result


def amendment(*,resume=False):
    path=HERE/'SHARED_HOST_AMENDMENT.json'
    if path.exists():
        if not resume:
            raise FileExistsError('Existing amendment; use --resume, never overwrite')
        document=read(path)
        for rel,digest in document['additional_source_bindings'].items():
            if sha(HERE/rel)!=digest:
                raise ValueError('Shared-host entry changed after freeze: '+rel)
        if document['frozen_recovery_receipt_sha256']!=sha(HERE/'freeze_receipt.json'):
            raise ValueError('Recovery freeze changed')
        return document
    if resume:
        raise FileNotFoundError('No existing shared-host experiment to resume')
    verify_bindings(HERE)
    if any((HERE/'results/A/terminal').glob('*.json')) or any((HERE/'results/C/terminal').glob('*.json')):
        raise ValueError('Shared-host policy must be declared before any A/C science')
    document={'version':'2.0.1-shared-host-entry','declared_utc':r.now(),
        'user_request':'尝试恢复完整实验运行',
        'reason':'Strict entry stopped before science on presence of independent RAG and P4 Python services; short CPU sample showed near-idle services.',
        'frozen_recovery_receipt_sha256':sha(HERE/'freeze_receipt.json'),
        'strict_start_attempt_status_sha256':sha(HERE/'RUN_STATUS.json'),
        'numeric_settings_and_candidates_unchanged':True,
        'budgets_unchanged':{'A':7200,'C':21600,'ordinary_worker':600,'training_worker':900},
        'primary_uncontended_runtime_claim_supported':False,
        'never_kill_unrelated_processes':True,'single_numerical_worker':True,
        'additional_source_bindings':{rel:sha(HERE/rel) for rel in ('run_observed_host.py','tests/test_observed_host.py')},
        'strict_start_attempts_preserved':True,'complete_schedule_not_all_methods_pass':True}
    write_terminal(path,document)
    return document


def execute(*,resume=False):
    amendment(resume=resume)
    path=HERE/'SHARED_HOST_STATUS.json'
    if path.exists() and not resume:
        raise FileExistsError('Existing execution; inspect then use --resume')
    state=read(path) if path.exists() else {'started_utc':r.now(),'tracks_done':[],
        'environment_checks':[],'status':'starting','primary_uncontended_runtime_claim_supported':False}
    with exclusive_track(r.OLD),exclusive_track(HERE):
        try:
            for track in ('A','C'):
                if track in state['tracks_done']: continue
                check=r.python_environment()
                state['environment_checks'].append(dict(check,track=track,boundary='before'))
                state.update(status='running',current_track=track,last_update_utc=r.now())
                write_checkpoint(path,state)
                print('Shared-host '+track+': '+str(len(check['unexpected_python_processes']))+
                      ' other Python processes observed; cost claims remain unqualified.',flush=True)
                run_track(track,HERE,resume=r.next_action(HERE,track)=='resume')
                state['tracks_done'].append(track)
                state['environment_checks'].append(dict(r.python_environment(),track=track,boundary='after'))
                state.update(last_update_utc=r.now())
                write_checkpoint(path,state)
            state['original_files_unchanged']=r.assert_old_preserved()
            state.update(status='science_finished_verifying_delivery',last_update_utc=r.now())
            write_checkpoint(path,state)
            if not (HERE/'results/audit.json').exists():
                report=qualified_shared_report(audit(HERE),state['environment_checks'])
                report.update(B_reuse_manifest_sha256=sha(HERE/'results/B/reuse.json'),
                    shared_host_amendment_sha256=sha(HERE/'SHARED_HOST_AMENDMENT.json'),
                    original_qualified_audit_sha256=sha(HERE/'results/original_qualified_audit.json'))
                write_terminal(HERE/'results/audit.json',report)
            else:
                report=read(HERE/'results/audit.json')
            if not (HERE/'RESULTS_REPORT.md').exists():
                text=report_text(report)+'\n## Recovery and timing qualification\n\n'
                text+='B reuses all original records without rerunning. Old A timings and new shared-host A timings are raw observations, not uncontended performance evidence. '
                text+='All original failed/partial results remain preserved. Resource limits changed, numerical tolerances and candidates did not. '
                text+='See SHARED_HOST_AMENDMENT.json and results/original_qualified_audit.json. '
                text+='Full execution means all planned jobs were attempted or explicitly censored; it does not mean every method passed.\n'
                with (HERE/'RESULTS_REPORT.md').open('x',encoding='utf-8') as handle: handle.write(text)
            if 'regressions' not in state:
                state['regressions']=r.verify_regressions()
                write_checkpoint(path,state)
            from package_evidence import build
            package=build()
            state.update(status='completed_with_recorded_outcomes',last_update_utc=r.now(),
                summary=report['tracks'],promotion_blocked=report['promotion_blocked'],
                package=package['archive'],package_sha256=package['sha256'],
                final_review_of_new_outcomes_still_required=True)
            write_checkpoint(path,state)
            print(json.dumps({'recovery_finished':True,'tracks':report['tracks'],
                  'primary_uncontended_runtime_claim_supported':False,'package':package['archive']}),flush=True)
        except BaseException as exc:
            state.update(status='needs_attention',last_update_utc=r.now(),error_type=type(exc).__name__,
                         error=str(exc),traceback=traceback.format_exc())
            write_checkpoint(path,state)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    execute(resume=args.resume)
