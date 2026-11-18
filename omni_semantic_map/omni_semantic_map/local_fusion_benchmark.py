"""Isolated-process measurements of native full, cached full and local replay."""
import argparse
import gc
import gzip
import hashlib
import json
import os
import pickle
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .fusion_benchmark import load, _labels
from .replay_fusion import FusionObservation, ReplayableObjectMap, ConceptGraphsBackend
from .local_fusion import LocalReplayObjectMap
from .dependency_fusion import DependencyReplayObjectMap


def records_from_cache(cache, observations, frames, features):
    source=cache.parents[1]/'data'/cache.name
    records=[]
    for k,frame in enumerate(frames):
        fid=frame['source_frame']
        rgb=np.array(Image.open(source/'rgb'/f'rgb_{fid}.png').convert('RGB'))
        depth=np.array(Image.open(source/'depth'/f'depth_{fid}.png'))/1000.
        group=[]
        for oid in frame['observations']:
            obs=observations[oid];uv=obs['pixels'].astype(np.int64)
            uv=uv[np.argsort(uv[:,1]*depth.shape[1]+uv[:,0])]
            u,v=uv[:,0],uv[:,1];z=depth[v,u]
            xyz=np.column_stack([(u-599.5)*z/600.,(v-339.5)*z/600.,z])
            group.append(FusionObservation(oid,k,int(obs['label'].split('_')[-1]),xyz,
                rgb[v,u]/255.,features['image'][oid],features['text'][oid]))
        records.append(group)
    return records


class RssSampler:
    def __enter__(self):
        import psutil
        self.process=psutil.Process()
        self.before=self.peak=self.process.memory_info().rss
        self.stop=threading.Event()
        def poll():
            while not self.stop.wait(.005):
                self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start()
        return self

    def __exit__(self,*args):
        self.peak=max(self.peak,self.process.memory_info().rss)
        self.stop.set();self.thread.join()


def run(args):
    from omegaconf import OmegaConf
    import torch
    torch.set_num_threads(1)
    source=args.source;scene=args.scene
    out=args.output/scene/(args.case+'_'+args.scale)/args.method
    out.mkdir(parents=True,exist_ok=False)
    code={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(__file__).with_name('local_fusion.py'),Path(__file__).with_name('dependency_fusion.py'),Path(__file__).with_name('replay_fusion.py')]}
    meta=dict(complete=False,scene=scene,case=args.case,scale=args.scale,method=args.method,repeats=args.repeats,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        source_sha256=code,threads=1,peak_memory_scope='process RSS sampled every 5 ms during update; separate process per scene/case/method')
    (out/'run.json').write_text(json.dumps(meta,indent=2))
    cache,observations,frames,poses,features=load(source,scene)
    records=records_from_cache(cache,observations,frames,features)
    cfg=OmegaConf.create(json.loads((source/'conceptgraphs_runs'/scene/'mapping_config.json').read_text()))
    if args.method=='native_full':state=ReplayableObjectMap(lambda:ConceptGraphsBackend(cfg))
    elif args.method=='dependency':state=DependencyReplayObjectMap(cfg)
    else:state=LocalReplayObjectMap(cfg,mode='full' if args.method=='cached_full' else 'local')
    n=len(observations);first={'head10':0,'tail10':int(len(frames)*.9),'tail30':int(len(frames)*.7),'global':0}[args.case]
    last=int(len(frames)*.1) if args.case=='head10' else len(frames)
    before=poses.copy()
    amplitude=.05 if args.scale=='small' else .2
    for k in range(first,last):
        f=(k-first+1)/(last-first);angle=np.deg2rad(3)*f*amplitude/.2
        delta=np.eye(4);delta[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]];delta[0,3]=amplitude*f
        before[k]=delta@poses[k]
    start=time.perf_counter()
    for k,group in enumerate(records):state.add_frame(k,group,before[k])
    ingest_s=time.perf_counter()-start
    initial_labels=_labels(state.labels(),n);causal=_labels(state.causal_labels(),n)
    initial_backend=state._backend;initial_poses=state._poses
    with np.load(source/'conceptgraphs_runs'/scene/'assignments.npz') as f:expected_labels=f['final'].copy()
    reference=args.reference/'replay_runs'/scene/'objects.pkl.gz'
    with gzip.open(reference,'rb') as f:expected_objects=pickle.load(f)['objects']
    trials=[]
    for repeat in range(args.repeats):
        # Restore the same starting state outside timing; correction itself is
        # transactional and does not mutate this retained old backend.
        state._backend=initial_backend;state._poses=initial_poses;state.revision=0;state.events=[]
        gc.collect()
        with RssSampler() as memory:
            start=time.perf_counter();corrected=_labels(state.revise_poses(dict(enumerate(poses))),n)
            elapsed=time.perf_counter()-start
        np.testing.assert_array_equal(corrected,expected_labels)
        np.testing.assert_array_equal(_labels(state.causal_labels(),n),causal)
        objects=state.export_objects();assert len(objects)==len(expected_objects)
        for expected,actual in zip(expected_objects,objects):
            assert expected['observation_ids']==actual['observation_ids']
            for key in ['points','colors','box','image_feature','text_feature']:
                np.testing.assert_allclose(actual[key],expected[key],rtol=0,atol=1e-10)
        trial=dict(repeat=repeat,wall_s=elapsed,rss_before_bytes=memory.before,rss_peak_bytes=memory.peak,
            rss_increase_bytes=memory.peak-memory.before,exact_reference_labels=True,exact_reference_geometry=True,
            causal_preserved=True,stats=state.events[-1].get('update_stats',{}))
        trials.append(trial)
        (out/'trials.json').write_text(json.dumps(trials,indent=2))
        print(scene,args.case,args.method,repeat,round(elapsed,3),trial['stats'],flush=True)
        del objects
    np.savez_compressed(out/'assignments.npz',causal=causal,frozen=initial_labels,final=corrected)
    np.savez_compressed(out/'poses.npz',before=before,corrected=poses)
    for name,h in code.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,'code changed while running'
    meta.update(complete=True,frames=len(frames),changed_frames=last-first,observations=n,
        amplitude_translation_m=amplitude,amplitude_rotation_deg=3*amplitude/.2,
        reference_objects_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        ingest_s=ingest_s,median_update_s=float(np.median([t['wall_s'] for t in trials])),
        source_cache=str(cache),source_cache_manifest_sha256=hashlib.sha256((cache/'manifest.json').read_bytes()).hexdigest(),
        synthetic_correction='0.05m/0.75deg or 0.2m/3deg ramp in selected frame range; restore provided poses')
    (out/'run.json').write_text(json.dumps(meta,indent=2));print('COMPLETE',scene,args.case,args.method,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scene',required=True)
    p.add_argument('--case',choices=['head10','tail10','tail30','global'],required=True)
    p.add_argument('--method',choices=['native_full','cached_full','local','dependency'],required=True)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--scale',choices=['small','large'],default='large')
    a=p.parse_args()
    if a.repeats<1:p.error('--repeats must be positive')
    run(a)


if __name__=='__main__':main()
