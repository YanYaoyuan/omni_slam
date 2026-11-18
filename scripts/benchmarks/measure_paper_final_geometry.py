"""One timing-only replay of frozen geometry; require exact cached arrays.

No detection, descriptor or association model is rerun. Outputs are temporary;
original predictions, scores, caches and execution sources remain immutable.
"""
import argparse
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,RUN,read,sha,dump,verify_execution,run_scenes
import split_strong_frontend as original


def measure(scene):
    verify_execution();assert scene in run_scenes()
    target=RUN/'timing_replay'/(scene+'.json');assert not target.exists()
    gpu=int(__import__('os').environ['CUDA_VISIBLE_DEVICES'])
    idle=0
    while idle<2:
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True).splitlines()
        values=[tuple(map(int,line.split(','))) for line in lines]
        selected=next(x for x in values if x[0]==gpu)
        idle=idle+1 if selected[1]>=28*1024 and selected[2]<10 else 0
        print('GPU IDLE CHECK',selected,'consecutive',idle,flush=True)
        if idle<2:time.sleep(5)
    torch=original.gpu_setup();path=RUN/'jobs'/(scene+'.json');job=read(path)
    for name,want in job['input_sha256'].items():assert sha(name)==want,name
    for name,want in job['source_sha256'].items():assert sha(ROOT/name)==want,name
    front=RUN/'frontend'/scene;manifest=read(front/'cache/manifest.json');records=[]
    class TimedFrames(list):
        def __iter__(self):
            for record in super().__iter__():
                torch.cuda.synchronize();start=time.monotonic()
                yield record
                torch.cuda.synchronize()
                records.append(dict(frame_id=record['frame_id'],semantic=record['semantic'],wall_s=time.monotonic()-start))
    job['frames']=TimedFrames(job['frames'])
    with tempfile.TemporaryDirectory(prefix='paper_final_geometry_',dir='/tmp') as temp:
        out=Path(temp)
        for name in ['masks','features']:(out/name).symlink_to(front/name,target_is_directory=True)
        status={'job_sha256':sha(path)};torch.cuda.synchronize();start=time.monotonic()
        original.geometry(job,out,torch,status,out/'geometry_run.json')
        torch.cuda.synchronize();wall=time.monotonic()-start
        replay=read(out/'cache/manifest.json');assert replay==manifest,'Original cache manifest changed'
        checked=0
        for p in (front/'cache').rglob('*'):
            if not p.is_file():continue
            q=out/'cache'/p.relative_to(front/'cache')
            if p.suffix=='.npz':
                with np.load(p) as a,np.load(q) as b:
                    assert a.files==b.files
                    for key in a.files:np.testing.assert_array_equal(a[key],b[key])
            elif p.name=='manifest.json':assert read(p)==read(q),p
            else:assert p.read_bytes()==q.read_bytes(),p
            checked+=1
        assert len(records)==400 and sum(x['semantic'] for x in records)==200
        times=[x['wall_s'] for x in records]
        result=dict(complete=True,scene=scene,timing_repetition=read(RUN/'timing_replay/plan.json')['attempt_counts'][scene],maximum_authorized_repetitions=3,gpu=gpu,
            source_sha256=sha(Path(__file__)),frozen_geometry_source_sha256=sha(Path(original.__file__)),
            job_sha256=sha(path),original_cache_manifest_sha256=sha(front/'cache/manifest.json'),
            prediction_sha256=sha(RUN/'predictions'/(scene+'.json')),records=records,
            frame_median_s=float(np.median(times)),frame_p95_s=float(np.quantile(times,.95)),
            total_geometry_wall_s=wall,initialization_and_final_export_s=wall-sum(times),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            exact_cached_arrays=True,exact_cached_manifest=True,checked_cache_files=checked,
            original_outputs_unchanged=True,no_detection_descriptor_or_association_inference=True,
            scope='Timing-only repeat; synchronize GPU at frame boundaries. Includes frame loading, native geometry and semantic support writes; final map export separately timed.')
        dump(target,result);print('TIMING VERIFIED',scene,'400 frames',result['frame_median_s'],result['frame_p95_s'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scene',required=True);args=parser.parse_args();measure(args.scene)
