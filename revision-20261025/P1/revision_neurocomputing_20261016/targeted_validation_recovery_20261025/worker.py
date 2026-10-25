"""Exactly one bounded scientific job; all exceptions become inspectable records."""
import argparse
from pathlib import Path
import time
import traceback

from records import read, write_terminal, exception_record


def main():
    start=time.perf_counter()
    parser=argparse.ArgumentParser()
    for arg in ('root','job','output','checkpoint'):
        parser.add_argument('--'+arg,required=True,type=Path)
    parser.add_argument('--limit',required=True,type=float)
    args=parser.parse_args()
    job=read(args.job); settings=read(args.root/'protocol.json')
    settings['deadline_monotonic']=start+args.limit
    settings['checkpoint_path']=str(args.checkpoint)
    try:
        import torch
        torch.set_num_threads(1)
        if job['track']=='A':
            from external_comparison import evaluate_external
            result=evaluate_external(job,settings)
        elif job['track']=='B':
            from stress_validation import evaluate_stress
            result=evaluate_stress(job,settings)
        else:
            from inverse_training import train_joint
            result=train_joint(job,settings,args.checkpoint)
    except Exception as exc:
        result=dict(status='error',traceback=traceback.format_exc(),**exception_record(exc))
    result['worker_elapsed_seconds']=time.perf_counter()-start
    write_terminal(args.output,result)


if __name__=='__main__':
    main()
