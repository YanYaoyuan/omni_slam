"""Resource-bounded queue; never stops or changes other GPU processes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import resource
from audit_evidence_retention import ROOT, OUT, dump
from omni_semantic_map.flat_benchmark import sha256


def gpu_free(index):
    raw=subprocess.check_output(['nvidia-smi',f'--id={index}','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    memory,util=map(int,raw.strip().split(','));return dict(free_mib=memory,utilization=util)


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,nargs='+',required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--status',type=Path,default=OUT/'split_queue_status.json');args=p.parse_args()
    OUT.mkdir(exist_ok=True);logs=OUT/'logs';logs.mkdir(exist_ok=True)
    with (OUT/'split_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(complete=False,pid=os.getpid(),gpu=args.gpu,jobs=[str(p) for p in args.jobs],completed=[],
            policy='Only start below 10% utilization and >=7GiB free on two polls 10s apart; each child CUDA allocator capped at 6GiB.',
            source_sha256=sha256(Path(__file__)))
        status=args.status
        def publish():state['updated_epoch']=time.time();dump(status,state)
        env=os.environ.copy();old=ROOT/'build/benchmarks/ovo_full_20261110'
        env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),HF_HOME=str(old/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',MPLCONFIGDIR='/tmp/omni-retention-mpl')
        python=old/'venv311/bin/python';runner=ROOT/'scripts/benchmarks/split_strong_frontend.py'
        for path in args.jobs:
            job=json.loads(path.read_text());phases=(['masks','triplets','features'] if job['kind']=='livo_atomic' else ['masks','triplets','features','geometry'])+(['validate'] if job['kind']=='replica_smoke' else [])
            for phase in phases:
                state.update(job=str(path),phase=phase,state='waiting_for_gpu',child_pid=None);publish()
                previous=Path(job['output'])/(phase+'_run.json')
                if previous.exists():
                    prior=json.loads(previous.read_text())
                    if prior['complete'] and prior['job_sha256']==sha256(path):
                        state['completed'].append(dict(job=str(path),phase=phase,reused_complete=True));publish();continue
                    raise RuntimeError('Incomplete prior phase requires inspection; will not overwrite '+str(previous))
                if phase!='validate':
                    idle=0
                    while idle<2:
                        sample=gpu_free(args.gpu);state['gpu_sample']=sample
                        idle=idle+1 if sample['free_mib']>=7*1024 and sample['utilization']<10 else 0
                        publish()
                        if idle<2:time.sleep(10)
                logpath=logs/(path.parent.name+'_'+job['scene']+'_'+phase+'.log')
                with logpath.open('a') as log:
                    child=subprocess.Popen([str(python),str(runner),'--job',str(path),'--phase',phase],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    state.update(state='running',child_pid=child.pid,log=str(logpath));publish()
                    peak_rss=0
                    while child.poll() is None:
                        try:
                            lines=Path(f'/proc/{child.pid}/status').read_text().splitlines()
                            rss=next(int(r.split()[1])*1024 for r in lines if r.startswith('VmRSS:'));peak_rss=max(peak_rss,rss)
                        except (FileNotFoundError,ProcessLookupError,StopIteration):pass
                        state['sampled_child_peak_rss_bytes']=peak_rss;time.sleep(2);publish()
                    rc=child.returncode
                if rc:
                    state.update(state='failed',returncode=rc);publish();raise SystemExit(rc)
                usage=dict(phase=phase,sampled_peak_rss_bytes=peak_rss,completed_children_max_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
                    scope='Sampled RSS belongs to this worker; completed_children_max is cumulative maximum over workers of this queue process.')
                dump(Path(job['output'])/(phase+'_resources.json'),usage)
                state['completed'].append(dict(job=str(path),phase=phase,log_sha256=sha256(logpath),resources=usage));publish()
                print('FINISHED',job['scene'],phase,flush=True)
        state.update(complete=True,state='complete',child_pid=None);publish()


if __name__=='__main__':main()
