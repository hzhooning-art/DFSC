"""Versioned recovery: same candidates/science, complete A/C attempt, retained B.

The original 2026-10-24 directory is read-only. This new experiment is post-results,
not a retroactive extension of its budget or a blind preregistration.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from budget_runner import run_track,run_job
import freeze_protocol as f
from records import read,sha,write_checkpoint,write_terminal,verify_bindings
from report_validation import audit,report_text
from run_track import exclusive_track

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'targeted_validation_20261023'


def recovery_settings(original):
    p=dict(original)
    p.update(version='2.0.0-recovery-20261025',author_approved='2026-10-25',
             authorization='User requested attempt to restore full experiments',
             track_budgets_seconds={'A':7200,'C':21600},
             worker_limit_seconds=600,training_limit_seconds=900,
             post_existing_results_not_blind_preregistration=True,
             recovery_policy='Fresh complete A and C attempts; retain unchanged B numerical evidence with provenance',
             no_science_tolerance_or_candidate_change=True)
    return p


def new_run_jobs(jobs):
    return [j for j in jobs if j['track'] in ('A','C')]


def next_action(root,track):
    directory=Path(root)/'results'/track
    return 'resume' if (directory/'ledger.json').exists() or any((directory/'terminal').glob('*.json')) else 'start'


def qualify_report(report,event,*,source):
    report=dict(report)
    invalid=source=='original' and event.get('track_A',{}).get('primary_uncontended_runtime_claim_supported') is False
    report['source_experiment']=source
    report['primary_uncontended_runtime_claim_supported']=not invalid
    report['runtime_qualification_scope']=('No known competing Python process at checked boundaries; not a system-wide no-contention certificate'
                                           if not invalid else 'Disqualified: known competing CPU-consuming test process')
    if invalid:
        report['a_raw_accuracy_matched_costs_not_performance_evidence']=report.get('a_accuracy_matched',[])
        report['a_accuracy_matched']=[]
    report['environment_event']=event
    return report


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare():
    if (HERE/'freeze_receipt.json').exists():
        raise FileExistsError('Recovery is already frozen; use --run --resume')
    verify_bindings(OLD)
    # Hash EVERY old regular file except the OS-only lock, not just positive results.
    inventory={str(path.relative_to(OLD)):sha(path) for path in OLD.rglob('*')
               if path.is_file() and path.name!='execution.lock'}
    write_terminal(HERE/'prior_preservation.json',{'root':str(OLD),'files':inventory})
    base=f.build_protocol()
    settings=recovery_settings(base)
    coords=read(OLD/'coordinates.json')
    jobs=read(OLD/'schedule.json')
    original={r['id']:r for r in jobs}
    copied=[]
    for path in sorted((OLD/'results/B/terminal').glob('*.json')):
        row=read(path)
        if row['status']!='completed' or row['track']!='B' or row['id'] not in original:
            raise ValueError('B reuse requires exact completed original identities')
        for key,value in original[row['id']].items():
            if row.get(key)!=value:
                raise ValueError('B identity mismatch')
        target=HERE/'results/B/terminal'/path.name
        write_terminal(target,row)
        if sha(path)!=sha(target):
            raise ValueError('B copied bytes changed')
        copied.append({'id':row['id'],'original_path':str(path.relative_to(OLD)), 'sha256':sha(path)})
    if len(copied)!=336:
        raise ValueError('Require all 336 original B records, including refusals')
    write_terminal(HERE/'results/B/reuse.json',{'source_experiment':OLD.name,'records':copied,
        'new_scientific_B_jobs':0,'new_B_runtime_claimed':False,'original_environment_event_sha256':sha(OLD/'results/environment_event.json')})
    write_terminal(HERE/'results/B/ledger.json',{'reused_original_track':True,'consumed_seconds':0.,
        'original_ledger_sha256':sha(OLD/'results/B/ledger.json'),'all_planned_ids_materialized':True})
    # Freeze custom recovery settings without changing the original generator.
    dependencies=f.default_dependencies()
    dependencies['prior_preservation.json']=HERE/'prior_preservation.json'
    dependencies['original_freeze']=OLD/'freeze_receipt.json'
    dependencies['original_event']=OLD/'results/environment_event.json'
    dependencies['original_audit']=OLD/'results/audit.json'
    dependencies['B_reuse']=HERE/'results/B/reuse.json'
    for row in copied:
        dependencies['B/'+row['id']]=OLD/row['original_path']
        dependencies['B-copy/'+row['id']]=HERE/'results/B/terminal'/(row['id']+'.json')
    bindings={name:{'path':os.path.relpath(Path(path).resolve(),HERE),'sha256':sha(path)} for name,path in dependencies.items()}
    for name,payload in (('protocol.json',settings),('coordinates.json',coords),('schedule.json',jobs)):
        write_terminal(HERE/name,payload)
        bindings[name]={'path':name,'sha256':sha(HERE/name)}
    previous=read(OLD/'freeze_receipt.json')
    receipt={'bindings':bindings,'versions':previous['versions'],'python':sys.version,
             'installed_source_hashes':previous['installed_source_hashes'],
             'source_snapshot_execution':True,'prepared_utc':now(),
             'original_freeze_sha256':sha(OLD/'freeze_receipt.json'),
             'preserved_artifacts':previous['preserved_artifacts'],
             'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'only_expanded_resources_and_runner_correctness_changed':True,
             'new_scheduled_science_jobs':len(new_run_jobs(jobs))}
    write_terminal(HERE/'freeze_receipt.json',receipt)
    old_report=qualify_report(read(OLD/'results/audit.json'),read(OLD/'results/environment_event.json'),source='original')
    old_report['environment_event_sha256']=sha(OLD/'results/environment_event.json')
    write_terminal(HERE/'results/original_qualified_audit.json',old_report)
    print(json.dumps({'prepared':True,'new_A_C_jobs':len(new_run_jobs(jobs)),'reused_B':len(copied),
                      'protocol_sha256':sha(HERE/'protocol.json')}),flush=True)


def assert_old_preserved():
    manifest=read(HERE/'prior_preservation.json')
    for rel,digest in manifest['files'].items():
        if sha(OLD/rel)!=digest:
            raise ValueError('Original changed: '+rel)
    return len(manifest['files'])


def verify_regressions():
    suites=[('recovery',HERE/'tests','test_*.py'),
            ('prior_core',HERE.parent/'tools','test_*.py'),
            ('prior_scalar',HERE.parent/'repair_feasibility','test_*.py'),
            ('prior_spectral',HERE.parent/'repair_spectral_integration','test_*.py'),
            ('prior_delivery',HERE.parent/'revise_delivery_20261022','test_*.py')]
    results=[]
    for name,directory,pattern in suites:
        process=run_job({'id':name,'command':[sys.executable,'-B','-X','utf8','-m','unittest',
            'discover','-s',str(directory),'-p',pattern]},180,HERE/'tmp/nonexistent_checkpoint.json')
        write_terminal(HERE/'results/verification'/(name+'.json'),process)
        results.append({'suite':name,'status':process['status'],'exit_code':process['exit_code']})
        if process['status']!='completed':
            raise RuntimeError('Regression failed: '+name)
    return results


def python_environment():
    # No broad process termination: unexpected processes stop the new run instead.
    cmd="Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    proc=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',cmd],
                        capture_output=True,text=True,encoding='utf-8',timeout=30)
    if proc.returncode:
        raise RuntimeError('Cannot verify competing Python processes: '+proc.stderr)
    rows=json.loads(proc.stdout or '[]')
    if isinstance(rows,dict): rows=[rows]
    allowed={os.getpid()}
    by_id={r['ProcessId']:r for r in rows}
    current=os.getpid()
    while current in by_id:
        parent=by_id[current]['ParentProcessId']
        if parent in allowed: break
        allowed.add(parent); current=parent
    unexpected=[r for r in rows if r['ProcessId'] not in allowed]
    return {'checked_utc':now(),'observer_pid':os.getpid(),'python_processes':rows,
            'unexpected_python_processes':unexpected,'no_known_python_contention':not unexpected}


def execute(*,resume=False):
    verify_bindings(HERE)
    state_path=HERE/'RUN_STATUS.json'
    if state_path.exists() and not resume:
        raise FileExistsError('Existing run status; use --run --resume')
    state=read(state_path) if state_path.exists() else {'started_utc':now(),'tracks_done':[],
        'environment_checks':[],'status':'starting','new_run_all_planned_ids':True}
    with exclusive_track(OLD),exclusive_track(HERE):
        try:
            for track in ('A','C'):
                if track in state['tracks_done']: continue
                check=python_environment()
                state['environment_checks'].append(check)
                state.update(current_track=track,status='running',last_update_utc=now())
                write_checkpoint(state_path,state)
                if not check['no_known_python_contention']:
                    raise RuntimeError('Unexpected Python process; no process was killed. Inspect environment before resume.')
                run_track(track,HERE,resume=next_action(HERE,track)=='resume')
                state['tracks_done'].append(track)
                state['last_update_utc']=now()
                write_checkpoint(state_path,state)
            end=python_environment(); state['environment_checks'].append(end)
            state['original_files_unchanged']=assert_old_preserved()
            state['status']='science_attempts_finished_audit_pending'
            write_checkpoint(state_path,state)
            report=audit(HERE)
            event={'checks':state['environment_checks'],'original_A_timing_excluded':True,
                   'boundary_checks_only_not_continuous_host_monitoring':True}
            report=qualify_report(report,event,source='recovery-20261025')
            if not all(c['no_known_python_contention'] for c in state['environment_checks']):
                report=qualify_report(report,{'track_A':{'primary_uncontended_runtime_claim_supported':False},**event},source='original')
                report['source_experiment']='recovery-20261025-environment-qualified'
            report['B_reused_without_rerun']=True
            report['B_reuse_manifest_sha256']=sha(HERE/'results/B/reuse.json')
            report['original_qualified_audit_sha256']=sha(HERE/'results/original_qualified_audit.json')
            write_terminal(HERE/'results/audit.json',report)
            text=report_text(report)
            text+='\n## Recovery qualifications\n\nB is imported unchanged from the complete original experiment, not newly executed. '
            text+='Old A timings are excluded; see results/original_qualified_audit.json. '
            text+='New runtime observations have Python-process checks at track boundaries, not a system-wide contention certificate. '
            text+='A completed attempt or all materialized IDs is not proof that all scientific checks passed.\n'
            with (HERE/'RESULTS_REPORT.md').open('x',encoding='utf-8') as handle: handle.write(text)
            state.update(status='science_finished_verifying_delivery',last_update_utc=now(),
                         promotion_blocked=report['promotion_blocked'],summary=report['tracks'])
            write_checkpoint(state_path,state)
            state['regressions']=verify_regressions()
            from package_evidence import build
            package=build()
            state.update(status='completed_with_recorded_outcomes',last_update_utc=now(),
                         package=package['archive'],package_sha256=package['sha256'],
                         final_review_of_new_outcomes_still_required=True)
            write_checkpoint(state_path,state)
            print(json.dumps({'recovery_finished':True,'summary':report['tracks'],'promotion_blocked':report['promotion_blocked']}),flush=True)
        except BaseException as exc:
            state.update(status='needs_attention',last_update_utc=now(),error_type=type(exc).__name__,error=str(exc))
            write_checkpoint(state_path,state)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare()
    else: execute(resume=args.resume)
