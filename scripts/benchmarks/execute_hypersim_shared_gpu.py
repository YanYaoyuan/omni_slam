"""Resource-only execution amendment: one capped worker may share a busy GPU.

Children and scientific parameters are unchanged. Never modifies other jobs.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time
from paper_final_common import ROOT,read,dump,sha,CGPY
import run_hypersim_benchmark as benchmark

OUT=benchmark.OUT
OLD=ROOT/'build/benchmarks/ovo_full_20261110'


def freeze():
    p=benchmark.verify();assert not (OUT/'resource_amendment.json').exists()
    assert not (OUT/'predictions_complete.json').exists()
    dump(OUT/'resource_amendment.json',dict(created_epoch=time.time(),protocol_sha256=sha(OUT/'protocol.json'),
        source_sha256=sha(__file__),resource_only=True,
        change='Replace idle-utilization gating with >=10 GiB free on two samples 10 seconds apart. Highest-free eligible RTX 5090 used. Exactly one child at a time; unchanged staged frontend caps CUDA allocator at 6 GiB.',
        unchanged='Same frozen child scripts, model weights, SAM2/SigLIP precision/batching/seeds, mapping configuration, scene selection, reference and scoring. No scores have been decoded.',
        reason='Both GPUs are persistently busy but have sufficient free memory; public benchmark is the user priority. Utilization and selected physical GPU are logged.',
        superseded_queues=[str(benchmark.SHARED/'smoke_queue_status.json'),str(OUT/'queue_status.json')]))


def verify():
    p=benchmark.verify();a=read(OUT/'resource_amendment.json')
    assert a['source_sha256']==sha(__file__) and a['protocol_sha256']==sha(OUT/'protocol.json')
    return p


def samples():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    result=[]
    for line in raw.strip().splitlines():
        index,name,memory,util=line.split(',')
        result.append(dict(index=int(index),name=name.strip(),free_mib=int(memory),utilization=int(util)))
    return result


def run():
    protocol=verify();status=OUT/'execution_status.json';assert not status.exists()
    # Old controllers must be terminal before another driver is allowed.
    for path in read(OUT/'resource_amendment.json')['superseded_queues']:
        previous=read(path)
        if previous.get('complete'):continue
        try:os.kill(previous['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Prior controller is still alive: '+path)
    state=dict(pid=os.getpid(),complete=False,stage='starting',child_pid=None,completed=[])
    def publish():state['updated_epoch']=time.time();dump(status,state)
    publish();logs=OUT/'shared_gpu_logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,HF_HOME=str(OLD/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',
             MPLCONFIGDIR='/tmp/omni-hypersim-mpl',PYTHONPATH=str(ROOT/'omni_semantic_map'))
    def child(cmd,label,gpu=None):
        verify();log=logs/(label+'.log');child_env=dict(env)
        if gpu is not None:child_env['CUDA_VISIBLE_DEVICES']=str(gpu)
        with log.open('x') as f:
            process=subprocess.Popen(cmd,cwd=ROOT,env=child_env,stdout=f,stderr=subprocess.STDOUT)
            state.update(stage=label,child_pid=process.pid,log=str(log),gpu=gpu);publish()
            while process.poll() is None:publish();time.sleep(2)
            if process.returncode:
                state.update(stage='failed',failed_step=label,returncode=process.returncode,child_pid=None);publish()
                raise RuntimeError(label+' failed; inspect output rather than auto-retry')
        state['completed'].append(label);state['child_pid']=None;publish()
    def choose_gpu():
        last=None
        while True:
            rows=samples();state.update(stage='wait_capacity',gpu_samples=rows);publish()
            good=[r for r in rows if '5090' in r['name'] and r['free_mib']>=10240]
            pick=max(good,key=lambda r:r['free_mib'])['index'] if good else None
            if pick is not None and pick==last:return pick
            last=pick;time.sleep(10)
    def frontend(jobpath):
        job=read(jobpath);folder=Path(job['output'])
        for phase in ['masks','triplets','features','geometry']:
            existing=folder/(phase+'_run.json')
            if existing.exists():
                record=read(existing)
                assert record['complete'] and record['job_sha256']==sha(jobpath),'Incomplete phase needs inspection'
                for name,h in record['files_sha256'].items():assert sha(folder/name)==h
                state['completed'].append(job['scene']+'_'+phase+'_reused');publish();continue
            gpu=choose_gpu()
            child([str(OLD/'venv311/bin/python'),'scripts/benchmarks/split_strong_frontend.py',
                   '--job',str(jobpath),'--phase',phase],job['scene']+'_'+phase,gpu)
    frontend(benchmark.SHARED/'jobs/ai_001_001.json')
    for scene in protocol['scenes']:frontend(Path(protocol['jobs'][scene]))
    script=str(ROOT/'scripts/benchmarks/run_hypersim_benchmark.py')
    for scene in protocol['scenes']:
        for action in ['cg','mc','cleanup']:child([str(CGPY),script,action,'--scene',scene],scene+'_'+action)
    child([str(CGPY),script,'seal'],'seal_all_predictions')
    for scene in protocol['scenes']:child([str(CGPY),script,'reference','--scene',scene],scene+'_reference')
    child([str(CGPY),script,'evaluate'],'evaluate_once')
    state.update(complete=True,stage='complete',child_pid=None);publish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);args=p.parse_args()
    globals()[args.action]()
