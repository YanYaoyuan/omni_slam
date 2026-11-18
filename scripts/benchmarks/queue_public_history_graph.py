"""Bounded CPU execution of the already frozen public-history protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'build/benchmarks/evidence_retention_20261111/public_history_v1'
CG=ROOT/'build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python'
EVAL=ROOT/'build/benchmarks/ovo_full_20261110/venv311/bin/python'


def worker(key):
    began=time.monotonic()
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MPLCONFIGDIR='/tmp/omni-public-history-mpl',PYTHONUNBUFFERED='1')
    stages=[('mapping',CG,'run_public_history_graph.py',DEST/'runs'/key/'run.json'),
            ('evaluation',EVAL,'evaluate_public_history_graph.py',DEST/'evaluation'/key/'metrics.json')]
    for phase,python,script,done in stages:
        if done.exists():
            if json.loads(done.read_text()).get('complete'):
                continue
            raise RuntimeError(f'{key} has incomplete {phase}; inspect without overwriting')
        with (DEST/f'{key}_{phase}.log').open('x') as log:
            child=subprocess.Popen([str(python),str(ROOT/'scripts/benchmarks'/script),'--scene',key],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            status=DEST/f'{key}_worker.json'
            status.write_text(json.dumps(dict(key=key,phase=phase,pid=child.pid,complete=False)))
            rc=child.wait()
            if rc:
                status.write_text(json.dumps(dict(key=key,phase=phase,returncode=rc,complete=False)))
                raise RuntimeError(f'{key} {phase} failed with code {rc}')
    result=dict(key=key,complete=True,elapsed_s=time.monotonic()-began)
    (DEST/f'{key}_worker.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scenes',nargs='+',required=True);args=parser.parse_args()
    allowed={s['key'] for s in json.loads((DEST/'protocol.json').read_text())['scenes']}
    assert set(args.scenes)<=allowed and len(set(args.scenes))==len(args.scenes)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,key) for key in args.scenes]
        for future in as_completed(futures):
            print(json.dumps(future.result()),flush=True)
