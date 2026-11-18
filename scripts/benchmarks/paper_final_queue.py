"""Execute exactly the frozen matrix; wait for an idle GPU, retain all outputs."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from paper_final_common import ROOT,RUN,FROZEN,OLD,STORAGE,SCENES,PY,CGPY,read,sha,dump,verify_execution,run_scenes,scored_scenes


def freeze():
    assert not (RUN/'execution_manifest.json').exists()
    subprocess.run([sys.executable,'scripts/paper/verify_freeze.py'],cwd=ROOT,check=True)
    accepted=read(RUN/'validation/acceptance.json');assert accepted['complete']
    sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('paper_final_*.py')}
    # All original source files retain paper snapshot identity.
    for r in read(FROZEN/'manifest.json')['files']:
        if r['path'].endswith('.py') or r['path'].startswith('docs/paper/frozen_v1/config/'):
            sources[r['path']]=r['sha256']
    for name,want in accepted['source_sha256'].items():assert sha(Path(__file__).parent/name)==want,name
    revision=(RUN/'annotation_adapter_amendment.json').exists()
    if revision:
        parent=RUN/'execution_revisions/v2_before_annotation_adapter/execution_manifest.json'
        old=read(parent);amendment=read(RUN/'annotation_adapter_amendment.json')
        assert sha(RUN/'predictions_complete.json')==amendment['predictions_gate_sha256']
        changed=[name for name,want in old['source_sha256'].items() if sha(ROOT/name)!=want]
        assert set(changed)<=set(amendment['allowed_changed_sources']),changed
        gate=dict(old,source_sha256=sources,acceptance_sha256=sha(RUN/'validation/acceptance.json'),
            annotation_adapter_amendment_sha256=sha(RUN/'annotation_adapter_amendment.json'),
            prediction_execution_manifest_sha256=sha(parent),created_epoch=time.time(),
            revision_scope='Post-prediction GT adapter and evaluation accounting only; inference remains v2.',
            changed_sources=changed)
        dump(RUN/'execution_manifest.json',gate)
        verify_execution();print('SEALED EVALUATOR REVISION',len(sources),'hashes',flush=True);return
    inventory=[]
    for base in [ROOT/'build/benchmarks',STORAGE.parent]:
        for folder,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if d not in ['venv','venv311','cg_venv','.git','hf','node_modules','__pycache__','results']]
            path=Path(folder)
            if path.is_relative_to(RUN) or path.is_relative_to(STORAGE):dirs[:]=[];continue
            if any(s in path.parts or 'scenenn_'+s in path.parts for s in SCENES) and any(t in path.parts for t in ['runs','evaluation','frontend','cache','predictions']):
                inventory.extend(str(path/f) for f in files if f.endswith('.json'))
    dump(RUN/'preexecution_provenance.json',dict(prior_final_scene_result_paths=inventory,
        inspection='filenames only, no new RGB or semantic values; cannot prove other people never viewed public scenes',
        prior_results_found=bool(inventory),replacement_scenes=[]))
    assert not inventory,'Prior-scene results require explicit provenance classification; do not silently replace.'
    dump(RUN/'execution_manifest.json',dict(complete=True,source_sha256=sources,
        paper_manifest_sha256=sha(FROZEN/'manifest.json'),paper_protocol_sha256=sha(FROZEN/'protocol.json'),
        acceptance_sha256=sha(RUN/'validation/acceptance.json'),scenes=SCENES,
        data_amendment_sha256=sha(RUN/'data_amendment.json'),executable_scenes=run_scenes(),
        diagnostic_resolution={'observation_min_valid_supports':16,'dominant_purity':.7,
            'mixed_prediction_gt_fraction':.1,'fragment_gt_coverage':.1,
            'no_inference_effect':True,'raw_overlap_histograms_saved':True},
        gpu_policy='Select first GPU with >=28GiB free and <10% utilization twice; hold device choice for full round; wait before every model phase.',
        experiments_unchanged=True,created_epoch=time.time()))
    print('SEALED EXECUTION',len(sources),'source/config hashes',flush=True)


def gpu_samples():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [dict(zip(['index','free_mib','utilization'],map(int,line.split(',')))) for line in raw.strip().splitlines()]


def run():
    verify_execution();RUN.mkdir(exist_ok=True)
    scenes=run_scenes()
    lock=(RUN/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logs=RUN/'logs';logs.mkdir(exist_ok=True)
    state=dict(complete=False,pid=os.getpid(),stage='preparing',completed=[],gpu=None,child_pid=None)
    if (RUN/'queue_status.json').exists():
        prior=read(RUN/'queue_status.json')
        assert prior.get('failed'), 'Existing nonterminal queue state requires process inspection before resuming.'
        state['previous_failure']=prior;state['gpu']=prior.get('gpu')
    env=os.environ.copy();up=ROOT/'build/benchmarks/ovo_full_20261110'
    env.update(HF_HOME=str(up/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',MPLCONFIGDIR='/tmp/omni-paper-final-mpl')
    def publish():state['updated_epoch']=time.time();dump(RUN/'queue_status.json',state)
    def child(cmd,label,environment=env):
        log=logs/(label+'.log');assert not log.exists(),f'Inspect prior log before rerun: {log}'
        tic=time.monotonic();peak=0
        with log.open('x') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=environment,stdout=f,stderr=subprocess.STDOUT)
            state.update(stage=label,child_pid=proc.pid,log=str(log));publish()
            while proc.poll() is None:
                try:
                    rows=Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                    rss=next(int(r.split()[1])*1024 for r in rows if r.startswith('VmRSS:'));peak=max(peak,rss)
                except (FileNotFoundError,StopIteration,ProcessLookupError):pass
                state['child_peak_rss_bytes']=peak;publish();time.sleep(2)
            if proc.returncode:
                state.update(failed=True,returncode=proc.returncode,child_pid=None);publish()
                raise RuntimeError(f'{label} failed: {log}')
        usage=dict(process_wall_s=time.monotonic()-tic,sampled_peak_rss_bytes=peak,log_sha256=sha(log))
        state['completed'].append(label);state['child_pid']=None;publish();return usage
    publish()
    while not (RUN/'download.json').exists() or not read(RUN/'download.json')['complete']:
        state.update(stage='waiting_for_download',child_pid=None);publish();time.sleep(10)
    if not all((RUN/'jobs'/(s+'.json')).exists() for s in scenes):
        child([str(PY),'scripts/benchmarks/paper_final_data.py','prepare'],'prepare')
    for scene in scenes:
        job=RUN/'jobs'/(scene+'.json');front=RUN/'frontend'/scene
        for phase in ['masks','triplets','features','geometry']:
            previous=front/(phase+'_run.json')
            if previous.exists():
                p=read(previous);assert p['complete'] and p['job_sha256']==sha(job),'Incomplete phase requires audit'
                assert (front/(phase+'_resources.json')).exists();continue
            idle={0:0,1:0};state.update(stage='waiting_for_idle_gpu',scene=scene,phase=phase)
            while True:
                samples=gpu_samples();state['gpu_samples']=samples
                for s in samples:
                    i=s['index'];eligible=state['gpu'] in [None,i] and s['free_mib']>=28*1024 and s['utilization']<10
                    idle[i]=idle.get(i,0)+1 if eligible else 0
                    if idle[i]>=2:state['gpu']=i;break
                publish()
                if any(n>=2 for n in idle.values()):break
                time.sleep(10)
            jobenv=env.copy();jobenv['CUDA_VISIBLE_DEVICES']=str(state['gpu'])
            usage=child([str(PY),'scripts/benchmarks/split_strong_frontend.py','--job',str(job),'--phase',phase],scene+'_'+phase,jobenv)
            usage.update(gpu=state['gpu'],phase=phase,gpu_samples_after=gpu_samples())
            dump(front/(phase+'_resources.json'),usage)
        for action,python in [('cg',CGPY),('ovo',PY),('categories',PY)]:
            artifact=RUN/'maps'/scene/action/'run.json' if action!='categories' else RUN/'predictions'/(scene+'.json')
            if artifact.exists():assert read(artifact)['complete'];continue
            child([str(python),'scripts/benchmarks/paper_final_mapping.py',action,'--scene',scene],scene+'_'+action)
    # No final-test ground truth has been loaded before this all-scene barrier.
    files={s:sha(RUN/'predictions'/(s+'.json')) for s in scenes}
    if (RUN/'predictions_complete.json').exists():
        gate=read(RUN/'predictions_complete.json');assert gate['complete'] and gate['files']==files
    else:
        dump(RUN/'predictions_complete.json',dict(complete=True,files=files,execution_manifest_sha256=sha(RUN/'execution_manifest.json'),
            selected_scenes=SCENES,unavailable=read(RUN/'data_amendment.json')['unscored_scenes'],sealed_epoch=time.time()))
    for scene in scored_scenes():
        if (RUN/'evaluation'/scene/'metrics.json').exists():assert read(RUN/'evaluation'/scene/'metrics.json')['complete'];continue
        child([str(PY),'scripts/benchmarks/paper_final_evaluate.py','--scene',scene],scene+'_evaluate')
    child([str(PY),'scripts/benchmarks/paper_final_evaluate.py','--summarize'],'summarize')
    state.update(complete=True,stage='complete',child_pid=None,summary_sha256=sha(RUN/'summary.json'));publish()
    print('ALL AVAILABLE SCENES COMPLETE; MISSING DATA EXPLICITLY REPORTED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args()
    if a.action=='freeze':freeze()
    else:run()
