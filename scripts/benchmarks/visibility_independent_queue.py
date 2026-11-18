"""Frozen independent evaluation queue. Wait for data/GPU; never steal a device."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
from paper_final_common import ROOT,OLD,CGPY,PY,read,dump,sha

OUT=ROOT/'build/benchmarks/visibility_independent_20261112'


def freeze():
    assert not (OUT/'protocol.json').exists()
    selection=read(OUT/'selection.json')
    assert not selection['prior_prediction_training_or_evaluation_paths']
    for name in ['cleanup_acceptance.json','maskclustering_core_acceptance.json','evaluator_acceptance.json']:
        assert read(OUT/name)['complete']
    assert read(OUT/'acceptance/maskclustering_scenenn015_20frames/run.json')['complete']
    sources=[Path(__file__).resolve()]+list((ROOT/'scripts/benchmarks').glob('visibility_independent_*.py'))
    sources+=[ROOT/'scripts/benchmarks'/p for p in ['maskclustering_shared.py','paper_final_mapping.py','paper_final_metrics.py',
        'paper_final_common.py','category_holdout_data.py','split_strong_frontend.py','run_livo_history_graph.py','publish_visibility_cleanup.py',
        'prepare_strong_jobs.py','evaluate_retention_holdout.py','validate_visibility_independent.py']]
    sources+=[ROOT/'omni_semantic_map/omni_semantic_map'/p for p in ['visibility_cleanup.py','replay_fusion.py','fusion_diagnostics.py',
        'full_ovo_evaluation.py','mesh_instance_evaluation.py','instance_error_audit.py','shared_observations.py','history_graph.py','flat_surface_evaluation.py','scene_coordinate_basis.py']]
    sources+=list((OUT/'adapters/MaskClusteringCPU').rglob('*.py'))
    sources+=[OUT/'adapters/MaskClusteringCPU/configs/scannet.json',ROOT/'docs/paper/frozen_v1/config/cg_rgbd.json']
    files={str(p.relative_to(ROOT)):sha(p) for p in sources}
    config=read(ROOT/'build/benchmarks/visibility_cleanup_20261112/protocol.json')['config']
    protocol=dict(created_epoch=time.time(),scenes=selection['selected_scenes'],source_sha256=files,
        selection_sha256=sha(OUT/'selection.json'),availability_sha256=sha(OUT/'availability.json'),
        primary='visibility1',methods=['cg','history2','sor','visibility1','visibility3','visibility1_no_depth','maskclustering'],
        selection_note='visibility1 is exactly the previously measured single_refutation development arm. Chosen before any new RGB inference or label evaluation; all other thresholds unchanged. Original 3-frame development gate remains failed.',
        config=config,primary_minimum_negative_frames=1,
        sampling='400 uniform trajectory frames, 200 semantic frames per scene; same frozen SAM2.1/SigLIP384 stages and given poses.',
        independent_scope='New sequences within SceneNN. No prior project run paths found. All 8 local Replica scenes already exposed, not reused as independent tests.',
        maximum_scored_scenes=4,maximum_scored_geometry_outputs=28,
        incidental_unscored_outputs='Inherited CG replay helper also writes history1/history2_no_veto for compatibility; 8 extra outputs maximum, never used for selection or evaluation.',
        external=dict(name='MaskClustering shared-input CPU core',upstream_url='https://github.com/PKU-EPIC/MaskClustering',
            commit=subprocess.check_output(['git','-C',str(OUT/'upstream/MaskClustering'),'rev-parse','HEAD'],text=True).strip(),
            adaptation_sha256=sha(OUT/'adapters/MaskClusteringCPU/adaptation.json'),
            source_scope='Official view consensus, iterative graph clustering, DBSCAN splitting, detection-ratio filtering and duplicate suppression. CPU device placement only. Shared cached supports replace backprojection; SAM2/SigLIP replace CropFormer/OpenCLIP. Not original full-pipeline reproduction.',
            caution='Upstream already has multi-view point filtering. Filtering itself is not a novel contribution. Any advantage must be shown over that external mechanism.',
            overlap='Native overlapping masks saved; exclusive 5cm nearest projection uses deterministic export order for exact point ties. Report native overlap count.'),
        evaluation='Inherited fixed observed valid 5cm domain and strict IoU>0.5. No coordinate fitting. Exact coordinate mismatch => N/A, no scene replacement. All new predictions sealed before GT values decoded.',
        secondary='Freeze history2 nearest-point ownership before deleting points: rejected nearest owners become unassigned; do not reassign to second-nearest. Report separately from standard projection.',
        checks=['all four selected scenes score','positive mean delta vs history2','at least 3/4 nondecreasing','total TP not lower',
                'positive mean under locked nearest ownership','positive mean vs SOR','positive mean vs no-depth ablation'],
        external_comparison='Report every selected scene including failures and regressions. Do not claim best external rank unless directly supported.',
        gpu_policy=dict(minimum_free_mib=8192,maximum_utilization_percent=10,consecutive_samples=2,poll_seconds=10,per_process_allocator_limit_gib=6),
        background_policy='Wait for inputs and an eligible GPU without terminating or modifying other GPU jobs. Resume requires complete artifacts and source hashes; record failures instead of silently replacing partial output.',
        stop='One primary fixed version and fixed controls. No post-score tuning, scene replacement or threshold sweep. No automatic claim of novelty or paper readiness.',
        acceptances={n:sha(OUT/n) for n in ['cleanup_acceptance.json','maskclustering_core_acceptance.json','evaluator_acceptance.json']})
    dump(OUT/'protocol.json',protocol)
    for name in files:
        dest=OUT/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    print('INDEPENDENT PROTOCOL FROZEN',protocol['scenes'],protocol['methods'],flush=True)


def verify():
    p=read(OUT/'protocol.json')
    assert sha(OUT/'selection.json')==p['selection_sha256']
    for name,want in p['source_sha256'].items():assert sha(ROOT/name)==want,name
    for name,want in p['acceptances'].items():assert sha(OUT/name)==want,name
    return p


def samples():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return [dict(zip(['index','free_mib','utilization'],map(int,line.split(',')))) for line in raw.strip().splitlines()]


def run():
    protocol=verify();lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'queue_status.json').exists():raise FileExistsError('Inspect existing queue state before starting a second run.')
    env=dict(os.environ,HF_HOME=str(ROOT/'build/benchmarks/ovo_full_20261110/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',MPLCONFIGDIR='/tmp/omni-visibility-independent-mpl',
        PYTHONPATH=str(ROOT/'omni_semantic_map'))
    state=dict(complete=False,pid=os.getpid(),stage='waiting_for_download',completed=[],failures=[],child_pid=None)
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    def publish():
        state['updated_epoch']=time.time();dump(OUT/'queue_status.json',state)
    def child(cmd,label,environment=None,allow_failure=False):
        verify();path=logs/(label+'.log');assert not path.exists(),path
        with path.open('x') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=environment or env,stdout=f,stderr=subprocess.STDOUT)
            state.update(stage=label,child_pid=proc.pid,log=str(path));publish()
            while proc.poll() is None:publish();time.sleep(2)
        state['child_pid']=None
        if proc.returncode:
            state['failures'].append(dict(stage=label,returncode=proc.returncode,log=str(path)));publish()
            if not allow_failure:raise RuntimeError(label+' failed')
            return False
        state['completed'].append(label);publish();return True
    publish()
    try:
        for scene in protocol['scenes']:
            state['scene']=scene;job=OUT/'jobs'/(scene+'.json')
            required={f'main/{scene}/{scene}.ply',f'main/{scene}/{scene}.xml',
                      f'main/{scene}/trajectory.log',f'main/oni/{scene}.oni','main/intrinsic/asus.ini'}
            state['stage']='waiting_for_scene_download'
            while True:
                current=read(OUT/'download.json') if (OUT/'download.json').exists() else {'files':[]}
                present={str(Path(r['path']).relative_to(OUT/'data')) for r in current['files']}
                publish()
                if required<=present:break
                time.sleep(10)
            child([str(CGPY),'scripts/benchmarks/visibility_independent_data.py','prepare','--scene',scene],scene+'_prepare')
            for phase in ['masks','triplets','features','geometry']:
                good={};policy=protocol['gpu_policy'];state.update(stage='waiting_for_gpu_capacity',phase=phase)
                while True:
                    rows=samples();state['gpu_samples']=rows;chosen=None
                    for r in rows:
                        eligible=r['free_mib']>=policy['minimum_free_mib'] and r['utilization']<policy['maximum_utilization_percent']
                        good[r['index']]=good.get(r['index'],0)+1 if eligible else 0
                        if good[r['index']]>=policy['consecutive_samples'] and chosen is None:chosen=r['index']
                    publish()
                    if chosen is not None:break
                    time.sleep(policy['poll_seconds'])
                jobenv=dict(env,CUDA_VISIBLE_DEVICES=str(chosen));state['gpu']=chosen
                child([str(PY),'scripts/benchmarks/split_strong_frontend.py','--job',str(job),'--phase',phase],scene+'_'+phase,jobenv)
            for action in ['cg','cleanup','maskclustering','seal_scene']:
                ok=child([str(CGPY),'scripts/benchmarks/visibility_independent_mapping.py',action,'--scene',scene],scene+'_'+action,allow_failure=action!='seal_scene')
        files={s:sha(OUT/'predictions'/(s+'.json')) for s in protocol['scenes']}
        dump(OUT/'predictions_complete.json',dict(complete=True,files=files,protocol_sha256=sha(OUT/'protocol.json'),created_epoch=time.time()))
        state['stage']='waiting_for_annotation_download'
        while not read(OUT/'download.json')['complete']:
            publish();time.sleep(10)
        # The existing HDF5 reader is a CPython 3.11 wheel; mapping uses 3.10.
        child([str(PY),'scripts/benchmarks/visibility_independent_evaluate.py'],'evaluate')
        state.update(complete=True,stage='complete',summary_sha256=sha(OUT/'summary.json'));publish()
    except Exception as e:
        import traceback
        state.update(failed=True,stage='failed',error=repr(e),traceback=traceback.format_exc());publish();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);a=p.parse_args()
    freeze() if a.action=='freeze' else run()
