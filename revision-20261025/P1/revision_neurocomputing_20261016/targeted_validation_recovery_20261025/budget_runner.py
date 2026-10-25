"""Process-isolated numerical jobs with resumable active-wall budget accounting."""
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

from records import read, write_checkpoint, write_terminal, verify_bindings


def remaining_budget(ledger, total):
    return max(0., total-ledger.get('consumed_seconds',0.))


def materialize_missing(jobs, directory):
    directory=Path(directory)
    for job in jobs:
        target=directory/(job['id']+'.json')
        if not target.exists():
            write_terminal(target,dict(job,status='not_run_budget'))


def run_job(job, timeout_seconds, checkpoint):
    start=time.perf_counter()
    command=job['command']
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',env=env)
    timed_out=False
    try:
        stdout,stderr=proc.communicate(timeout=max(.001,timeout_seconds-(time.perf_counter()-start)))
    except subprocess.TimeoutExpired:
        timed_out=True
        if os.name=='nt':
            # Kill our precise child process tree, including the Windows venv launcher child.
            subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,timeout=15)
        if proc.poll() is None:
            proc.kill()
        stdout,stderr=proc.communicate()
    result={'id':job['id'],'status':'timeout' if timed_out else ('completed' if proc.returncode==0 else 'error'),
            'exit_code':proc.returncode,'stdout':stdout,'stderr':stderr,
            'elapsed_seconds':time.perf_counter()-start,'command':command,'child_reaped':proc.poll() is not None}
    if Path(checkpoint).exists():
        try:
            result['checkpoint']=read(checkpoint)
        except Exception as exc:
            result['checkpoint_error']=str(exc)
    return result


def run_track(track, root, *, resume=False):
    root=Path(root); verify_bindings(root)
    p=read(root/'protocol.json')
    budget=p.get('track_budgets_seconds',{}).get(track,p['track_budget_seconds'])
    jobs=[j for j in read(root/'schedule.json') if j['track']==track]
    directory=root/'results'/track
    ledger_path=directory/'ledger.json'
    if ledger_path.exists() and not resume:
        raise FileExistsError('Use --resume; completed cells never rerun')
    ledger=read(ledger_path) if ledger_path.exists() else {'consumed_seconds':0.,'attempts':[],'pending':None}
    # A parent crash must not erase elapsed time. Conservatively charge the whole
    # reserved timeout; an attempt without terminal output stays explicitly interrupted.
    if ledger.get('pending'):
        pending=ledger['pending']
        ledger['consumed_seconds']+=pending['limit_seconds']
        target=directory/'terminal'/(pending['id']+'.json')
        if not target.exists():
            original=next(j for j in jobs if j['id']==pending['id'])
            cp=directory/'attempts'/pending['attempt']/'checkpoint.json'
            write_terminal(target,dict(original,status='interrupted',attempt=pending,
                checkpoint=read(cp) if cp.exists() else None,budget_charge='full_reserved_limit'))
        ledger['pending']=None
    ledger['session_started_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    session_start=time.perf_counter(); consumed_base=ledger['consumed_seconds']
    for job in jobs:
        target=directory/'terminal'/(job['id']+'.json')
        if target.exists():
            continue
        ledger['consumed_seconds']=consumed_base+time.perf_counter()-session_start
        remaining=remaining_budget(ledger,budget)
        if remaining<=0:
            break
        training=job['kind'] in ('inverse','joint')
        limit=min(remaining,p['training_limit_seconds'] if training else p['worker_limit_seconds'])
        attempt_id=f'{len(ledger["attempts"]):04}-{job["id"]}'
        attempt=directory/'attempts'/attempt_id
        attempt.mkdir(parents=True,exist_ok=False)
        write_terminal(attempt/'job.json',job)
        cp=attempt/'checkpoint.json'
        command=[sys.executable,'-B','-X','utf8',str(root/'worker.py'),'--root',str(root),
                 '--job',str(attempt/'job.json'),'--output',str(attempt/'result.json'),
                 '--checkpoint',str(cp),'--limit',str(limit)]
        ledger['pending']={'id':job['id'],'limit_seconds':limit,'attempt':attempt_id}
        write_checkpoint(ledger_path,ledger)
        process=run_job(dict(job,command=command),limit,cp)
        write_terminal(attempt/'process.json',process)
        try:
            payload=read(attempt/'result.json') if (attempt/'result.json').exists() else {}
        except (ValueError,OSError) as exc:
            payload={'status':'error','result_parse_error':str(exc),'partial_result_preserved':True}
        if process['status']!='completed':
            payload.update(status=process['status'],checkpoint=process.get('checkpoint'))
        if not payload:
            payload={'status':'error','error':'Worker produced no parseable result'}
        write_terminal(target,dict(job,**{k:v for k,v in payload.items() if k not in job},
                                  attempt=attempt_id,process_status=process['status']))
        ledger['attempts'].append({'id':job['id'],'attempt':attempt_id,'elapsed_seconds':process['elapsed_seconds']})
        ledger['pending']=None
        ledger['consumed_seconds']=consumed_base+time.perf_counter()-session_start
        write_checkpoint(ledger_path,ledger)
        print(f'{track}: {job["id"]} {payload.get("status",process["status"])}; active={ledger["consumed_seconds"]:.1f}s',flush=True)
    materialize_missing(jobs,directory/'terminal')
    ledger.update(consumed_seconds=consumed_base+time.perf_counter()-session_start,
                  all_planned_ids_materialized=True)
    write_checkpoint(ledger_path,ledger)
    return ledger
