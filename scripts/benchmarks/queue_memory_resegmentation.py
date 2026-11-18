"""Wait for sealed validation, then finish the bounded delivery on GPU 0."""
import os
from pathlib import Path
import subprocess
import time
from memory_reseg_common import OUT,ROOT,PY,CGPY,SEEDS,read,dump,sha
from run_memory_resegmentation import KEYS

def select():
    v=read(OUT/'validation.json');assert v['complete']
    def pq(row):return row['pq_sum']/max(row['fg_observations'],1)
    wins=sum(pq(v['selected'][f'attention_{seed}']['metrics'])>pq(v['selected'][f'mean_{seed}']['metrics']) for seed in SEEDS)
    mode='attention' if wins>=2 else 'mean'
    records=[v['selected'][f'{mode}_{seed}'] for seed in SEEDS]
    import numpy as np
    checks=dict(median_pq_above_simple=float(np.median([pq(x['metrics']) for x in records]))>pq(v['metrics']['simple']),
        mixed_net_positive=sum(x['metrics']['mixed_gained']-x['metrics']['mixed_lost'] for x in records)>0,
        deployed_checkpoint_trained=records[0]['epoch']>0,
        history_beats_current_in_two_seeds=sum(pq(v['selected'][f'{mode}_{seed}']['metrics'])>pq(v['selected'][f'current_{seed}']['metrics']) for seed in SEEDS)>=2)
    result=dict(selected=records[0],mode=mode,seed=17,attention_wins_over_mean=wins,
        validation_gate_checks=checks,validation_gate_passed=all(checks.values()),
        validation_sha256=sha(OUT/'validation.json'),selection_uses_development=False,
        full_maps_if_failed='Diagnostic completion explicitly declared before validation selection; not promotion or lowered gate.')
    dump(OUT/'selection.json',result);print('SELECTED',mode,'gate',checks,flush=True)

def run(command,log,gpu=False):
    env=dict(os.environ,PYTHONPATH=str(ROOT/'omni_semantic_map'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/omni_memory_mpl')
    env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
    if gpu:env['PYTHONNOUSERSITE']='1'
    if gpu:
        stable=0
        while stable<2:
            free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            stable=stable+1 if free>=4096 else 0
            dump(OUT/'gpu_wait.json',dict(free_mib=free,stable_polls=stable,time=time.time(),required_free_mib=4096))
            if stable<2:time.sleep(10)
    with Path(log).open('w') as f:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{command} exited {result.returncode}; see {log}')

def main():
    while not (OUT/'validation.json').exists():time.sleep(10)
    if not (OUT/'selection.json').exists():select()
    source=ROOT/'scripts/benchmarks/run_memory_resegmentation.py'
    execution=OUT/'map_execution.json'
    if execution.exists():
        execution.rename(OUT/f'map_execution_before_resume_{int(time.time())}.json')
    dump(execution,dict(created=time.time(),source_sha256={str(x):sha(x) for x in
        [source,ROOT/'scripts/benchmarks/queue_memory_resegmentation.py',ROOT/'scripts/benchmarks/validate_memory_resegmentation.py',
         ROOT/'omni_semantic_map/omni_semantic_map/memory_resegmentation.py']},selection_sha256=sha(OUT/'selection.json'),maximum_maps=12))
    for key in KEYS:
        for arm in ['B1','B2']:
            regen=OUT/'regenerated'/key/arm; mapped=OUT/'maps'/key/arm
            if not (regen/'manifest.json').exists():
                if regen.exists():regen.rename(regen.with_name(arm+f'_interrupted_{int(time.time())}'))
                print('REGENERATE',key,arm,flush=True)
                run([str(PY),str(source),'regenerate','--scene',key,'--variant',arm],OUT/f'regenerate_{key}_{arm}.log',gpu=True)
            else:assert read(regen/'manifest.json')['complete']
            if not (mapped/'run.json').exists():
                print('INTEGRATE',key,arm,flush=True)
                run([str(CGPY),str(source),'map','--scene',key,'--variant',arm],OUT/f'map_{key}_{arm}.log')
            else:assert read(mapped/'run.json')['complete']
    runs={}
    for key in KEYS:
        for arm in ['B0','B1','B2']:
            path=OUT/'maps'/key/arm/'run.json';assert read(path)['complete'];runs[str(path.relative_to(OUT))]=sha(path)
    dump(OUT/'predictions_complete.json',dict(complete=True,maps=12,runs=runs,no_development_scoring_yet=True))
    print('ALL 12 MAPS SEALED',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        dump(OUT/'queue_failure.json',dict(error=repr(e),time=time.time()));raise
