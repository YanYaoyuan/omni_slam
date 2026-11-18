"""Finish the frozen new-scene experiment when a GPU is idle; never retune."""
import os
from pathlib import Path
import subprocess
import time
from audit_evidence_retention import ROOT,OUT,dump
from public_dino_protocol import read,verify


def main():
    dest=OUT/'category_holdout_v1';verify(read(dest/'selection.json'))
    assert not (dest/'queue_status.json').exists()
    jobs=[dest/'jobs'/(s+'.json') for s in ['025','030','032']]
    assert all(p.exists() for p in jobs)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',MPLCONFIGDIR='/tmp/omni-category-holdout-mpl')
    py=ROOT/'build/benchmarks/ovo_full_20261110/venv311/bin/python'
    cg=ROOT/'build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python'
    stages=[('frontend',[str(py),'scripts/benchmarks/run_split_queue.py','--gpu','0','--jobs',*map(str,jobs),'--status',str(dest/'gpu_queue_status.json')]),
            ('mapping',[str(cg),'scripts/benchmarks/category_holdout_finish.py','--mapping']),
            ('evaluation',[str(py),'scripts/benchmarks/category_holdout_annotation_adapter.py','--evaluate'])]
    state=dict(complete=False,pid=os.getpid(),scenes=['025','030','032'],scored_scenes=['025','032'],stages_completed=[])
    for name,cmd in stages:
        state.update(stage=name,updated_unix=time.time());dump(dest/'queue_status.json',state)
        with (dest/(name+'_queue.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;dump(dest/'queue_status.json',state)
            rc=child.wait()
        if rc:
            state.update(failed=True,returncode=rc,updated_unix=time.time());dump(dest/'queue_status.json',state)
            raise RuntimeError(f'{name} failed; see {name}_queue.log')
        state['stages_completed'].append(name)
    state.update(complete=True,stage='complete',child_pid=None,updated_unix=time.time());dump(dest/'queue_status.json',state)


if __name__=='__main__':main()
