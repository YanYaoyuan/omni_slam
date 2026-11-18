"""Wait for verified model assets and an idle GPU, then run the frozen probe."""
import os
from pathlib import Path
import subprocess
import time
from paper_final_common import ROOT, read, dump, PY, CGPY
from prepare_vggt_probe_assets import OUT


def main():
    state=OUT/'queue_state.json';log=OUT/'queue.log'
    if (OUT/'evaluation.json').exists():raise RuntimeError('Probe already scored')
    started=time.time();last=None;stable={0:0,1:0}
    while time.time()-started < 6*3600:
        available=[]
        if (OUT/'assets.json').exists() and read(OUT/'assets.json').get('complete'):
            output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
            for line in output.splitlines():
                gpu,free,util=map(int,line.split(','))
                stable[gpu]=stable.get(gpu,0)+1 if free>=10240 and util<=25 else 0
                if stable[gpu]>=3:available.append(gpu)
            phase='waiting_idle_gpu'
        else:phase='waiting_assets'
        dump(state,dict(pid=os.getpid(),phase=phase,updated_epoch=time.time(),started_epoch=started,
                        consecutive_idle_samples=stable,required_free_mib=10240,maximum_utilization=25))
        if phase!=last:print(phase,flush=True);last=phase
        if available:
            gpu=min(available);break
        time.sleep(20)
    else:
        dump(state,dict(pid=os.getpid(),phase='timeout',complete=False,updated_epoch=time.time()))
        raise TimeoutError('Six-hour readiness budget exhausted; no inference launched')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(ROOT/'omni_semantic_map'))
    with log.open('a') as f:
        for action,python in [('extract',PY),('evaluate',CGPY)]:
            dump(state,dict(pid=os.getpid(),phase=action,gpu=gpu,updated_epoch=time.time()))
            run=subprocess.run([str(python),'scripts/benchmarks/run_vggt_object_probe.py',action],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if run.returncode:
                dump(state,dict(pid=os.getpid(),phase='failed',action=action,returncode=run.returncode,updated_epoch=time.time()))
                raise RuntimeError(f'{action} failed; inspect {log}')
    dump(state,dict(pid=os.getpid(),phase='complete',complete=True,gpu=gpu,updated_epoch=time.time()))
    print('PROBE COMPLETE',flush=True)


if __name__=='__main__':main()
