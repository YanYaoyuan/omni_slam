"""Bounded CPU execution; never retry a scored configuration or expand scenes."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import os,subprocess,time
from run_object_repair_v2 import ROOT,OUT,KEYS,VARIANTS,CGPY,verify,read,sha,dump


def main():
    verify();assert not (OUT/'queue_status.json').exists()
    jobs=[(k,v) for k in KEYS for v in VARIANTS];logs=OUT/'logs';logs.mkdir()
    state=dict(complete=False,stage='mapping',maximum_new_maps=12,total_jobs=len(jobs),completed=[],failed=[],cpu_workers=2,pid=os.getpid())
    dump(OUT/'queue_status.json',state);dump(OUT/'queue_execution.json',dict(source_sha256=sha(Path(__file__)),cpu_workers=2,
        threads_per_worker=1,no_new_gpu_model_inference=True,protocol_sha256=sha(OUT/'protocol.json')))
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/omni-v2-mpl',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='')
    def job(key,variant):
        log=logs/(key+'__'+variant+'.log')
        with log.open('x') as f:
            p=subprocess.run([str(CGPY),'scripts/benchmarks/run_object_repair_v2.py','--scene',key,'--variant',variant],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        path=OUT/'runs'/key/variant/'run.json'
        if not path.exists() or not read(path).get('complete'):
            data=read(path) if path.exists() else dict(complete=False,key=key,variant=variant,no_gt=True)
            if not data.get('failed'):dump(path,dict(data,failed=True,error='Process exited without completed map',returncode=p.returncode))
        return dict(key=key,variant=variant,returncode=p.returncode,complete=read(path)['complete'])
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(job,*j) for j in jobs]
        for future in as_completed(futures):
            result=future.result();state['completed'].append(result)
            if not result['complete']:state['failed'].append(result)
            state['updated_epoch']=time.time();dump(OUT/'queue_status.json',state)
            print('FINISHED',result,flush=True)
    paths=[OUT/'runs'/k/v/'run.json' for k,v in jobs]
    assert len(paths)==12 and all(p.exists() for p in paths)
    dump(OUT/'predictions_complete.json',dict(complete=True,runs={str(p.relative_to(OUT)):sha(p) for p in paths},
        all_12_attempts_accounted_for=True,failed_jobs=state['failed'],sealed_epoch=time.time(),protocol_sha256=sha(OUT/'protocol.json')))
    state['stage']='evaluation';dump(OUT/'queue_status.json',state)
    with (logs/'evaluation.log').open('x') as f:
        result=subprocess.run([str(CGPY),'scripts/benchmarks/evaluate_object_repair_v2.py'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    state.update(complete=result.returncode==0,stage='complete' if result.returncode==0 else 'evaluation_failed',evaluation_returncode=result.returncode,updated_epoch=time.time())
    dump(OUT/'queue_status.json',state);print('QUEUE',state['stage'],flush=True)


if __name__=='__main__':main()
