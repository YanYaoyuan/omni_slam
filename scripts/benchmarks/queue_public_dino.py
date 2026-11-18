"""Two bounded CPU workers consume complete frozen feature caches."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'build/benchmarks/evidence_retention_20261111/public_dino_v1'
CG=ROOT/'build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python'
EVAL=ROOT/'build/benchmarks/ovo_full_20261110/venv311/bin/python'


def worker(key):
    status=DEST/(key+'_worker.json');began=time.monotonic()
    feature=DEST/'features'/key/'manifest.json'
    while not feature.exists():
        status.write_text(json.dumps(dict(key=key,phase='waiting_features',complete=False)))
        if time.monotonic()-began>3600:raise TimeoutError(key)
        time.sleep(5)
    assert json.loads(feature.read_text())['complete']
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/omni-public-dino-mpl',PYTHONUNBUFFERED='1')
    for phase,python,script in [('mapping',CG,'run_public_dino_graph.py'),('evaluation',EVAL,'evaluate_public_dino.py')]:
        with (DEST/f'{key}_{phase}.log').open('x') as log:
            child=subprocess.Popen([str(python),str(ROOT/'scripts/benchmarks'/script),'--scene',key],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            status.write_text(json.dumps(dict(key=key,phase=phase,complete=False,pid=child.pid)))
            rc=child.wait()
            if rc:
                status.write_text(json.dumps(dict(key=key,phase=phase,complete=False,returncode=rc)))
                raise RuntimeError(f'{key} {phase} returned {rc}')
    result=dict(key=key,complete=True,elapsed_s=time.monotonic()-began)
    status.write_text(json.dumps(result));return result


if __name__=='__main__':
    keys=[e['key'] for e in json.loads((DEST/'protocol.json').read_text())['scenes']]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in as_completed([pool.submit(worker,key) for key in keys]):print(json.dumps(result.result()),flush=True)
