"""Resume the frozen run after its missing interpreter, never change parameters."""
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from paper_final_common import ROOT,FROZEN,CGPY,read,dump,sha
from run_kitti360_reserved import OUT,verify,job_path

PY=OUT/'runtime/bin/python'


def main():
    p=verify();assert not (OUT/'predictions_complete.json').exists()
    original=read(OUT/'status.json');assert original['failed']
    for w in p['windows']:
        assert job_path(w['scene']).exists()
        assert not Path(read(job_path(w['scene']))['output']).exists()
    versions=subprocess.check_output([str(PY),'-c',
        'import torch,torchvision,numpy,scipy,transformers,cv2; print(torch.__version__,torchvision.__version__,numpy.__version__,scipy.__version__,transformers.__version__,cv2.__version__)'],text=True)
    assert versions.startswith('2.9.1+cu128 0.24.1+cu128 ')
    record=OUT/('runtime_recovery_v2.json' if (OUT/'runtime_recovery.json').exists() else 'runtime_recovery.json')
    assert not record.exists()
    dump(record,dict(original_failure=original,protocol_sha256=sha(OUT/'protocol.json'),
        recovery_script_sha256=sha(__file__),python=str(PY),versions=versions.strip(),
        before_reserved_inference=True,before_reference_decoding=True,
        scope='Restore missing runtime dependencies in project-local venv; immutable method/input/scoring protocol remains unchanged.',
        smoke='One existing development frame: exact masks and feature cosine>=0.999, max absolute feature difference<=0.005, before reserved inference. No scene replacement or threshold tuning.'))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        HF_HOME=str(ROOT/'build/benchmarks/ovo_full_20261110/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    def call(python,script,args):
        dump(OUT/'status.json',dict(complete=False,pid=os.getpid(),stage=args,script=script,updated=time.time()))
        subprocess.run([str(python),script,*args],cwd=ROOT,env=env,check=True)
    driver=str(ROOT/'scripts/benchmarks/run_kitti360_reserved.py')
    try:
        old=ROOT/'build/benchmarks/kitti360_development_v1/jobs/2013_05_28_drive_0000_sync.json'
        job=read(old);prior=Path(job['output']);job['frames']=job['frames'][:1];job['output']=str(OUT/'runtime_smoke')
        smoke=OUT/'runtime_smoke_job.json';dump(smoke,job)
        for phase in ['masks','triplets','features']:
            folder=Path(job['output']);status=folder/(phase+'_run.json')
            if status.exists() and read(status)['complete']:
                previous=read(status);assert previous['job_sha256']==sha(smoke)
                for name,h in previous['files_sha256'].items():assert sha(folder/name)==h
                continue
            if (folder/phase).exists():
                archive=OUT/'recovery_revisions/v1'/('failed_'+phase)
                (folder/phase).rename(archive)
                if status.exists():status.rename(archive.with_suffix('.json'))
            call(PY,'scripts/benchmarks/split_strong_frontend.py',['--job',str(smoke),'--phase',phase])
        name=f"{job['frames'][0]['frame_id']:06d}.npz"
        oldm=np.load(prior/'masks'/name);newm=np.load(Path(job['output'])/'masks'/name)
        for field in ['segmentation','masks_packed']:np.testing.assert_array_equal(oldm[field],newm[field])
        oldf=np.load(prior/'features'/name)['features'];newf=np.load(Path(job['output'])/'features'/name)['features']
        assert oldf.shape==newf.shape and len(oldf)>0
        cosine=(oldf*newf).sum(1)/(np.linalg.norm(oldf,axis=1)*np.linalg.norm(newf,axis=1))
        error=float(np.abs(oldf-newf).max());assert cosine.min()>=.999 and error<=.005
        dump(OUT/'runtime_smoke_verified.json',dict(complete=True,exact_masks=True,
            minimum_feature_cosine=float(cosine.min()),max_abs_feature_difference=error,
            development_frame=job['frames'][0]['frame_id'],prior_features_sha256=sha(prior/'features'/name),
            recovered_features_sha256=sha(Path(job['output'])/'features'/name)))
        for w in p['windows']:
            scene=w['scene']
            for phase in ['masks','triplets','features']:
                call(PY,'scripts/benchmarks/split_strong_frontend.py',['--job',str(job_path(scene)),'--phase',phase])
            call(CGPY,driver,['geometry','--scene',scene])
            call(CGPY,'scripts/benchmarks/filter_livo_sparse_support.py',[
                '--cache',str(OUT/scene/'frontend'/scene/'cache'),'--output',str(OUT/scene/'connected/cache'),
                '--livo-config','build/experiments/semantic_20261106/managed_config/livo.yaml',
                '--cg-config',str(FROZEN/'config/cg_rgbd.json')])
            call(PY,driver,['extract','--scene',scene])
            for arm in ['base','vggt_single']:call(CGPY,driver,['mapping','--scene',scene,'--arm',arm])
        call(CGPY,driver,['seal']);call(CGPY,driver,['evaluate_all'])
        dump(OUT/'status.json',dict(complete=True,stage='evaluated',updated=time.time()))
    except Exception as e:
        old=read(OUT/'status.json');dump(OUT/'status.json',dict(old,failed=True,error=repr(e),updated=time.time()));raise


if __name__=='__main__':main()
