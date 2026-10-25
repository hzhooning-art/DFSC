"""CLI for one frozen track. OS lock prevents concurrent measured tracks."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path

from budget_runner import run_track


@contextmanager
def exclusive_track(root):
    path=Path(root)/'execution.lock'
    with path.open('a+b') as handle:
        if handle.tell()==0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name=='nt':
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(handle,fcntl.LOCK_UN)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--track',choices=['A','B','C'],required=True)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    with exclusive_track(args.root):
        run_track(args.track,args.root,resume=args.resume)
