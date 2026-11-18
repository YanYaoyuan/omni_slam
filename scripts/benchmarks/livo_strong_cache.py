"""Prepare frozen native frontend jobs and measured-support LIVO caches."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np
from audit_evidence_retention import ROOT, OUT, CACHE, dump
from prepare_strong_jobs import sources
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.sparse_semantic_support import sparse_support


# Frozen before inspecting live predictions. New IDs are live-only; background
# aliases use the existing frozen background IDs. Benchmark vocabulary unchanged.
LIVE_TERMS=[('car',1001),('person',1002),('bicycle',1003),('motorcycle',1004),
            ('truck',1005),('fire hydrant',1006),('fire extinguisher',1007),
            ('building',93),('road',40),('ground',40),('sky',31),('tree',1008)]


def prepare(capture_path,output,job_path,scene='rosbag2245_30s'):
    capture=json.loads(capture_path.read_text());assert capture['complete'] and len(capture['frames'])>=5
    output.mkdir(parents=True,exist_ok=False)
    reference=CACHE/'office_0';cfg=json.loads((reference/'manifest.json').read_text())['config']
    inputs={str(capture_path.resolve()):sha256(capture_path)}
    for record in capture['frames']:
        for name,h in record['sha256'].items():assert sha256(Path(name))==h;inputs[name]=h
    # Keep baseline text embeddings bit-for-bit and append a fixed deployment
    # ontology using the same frozen text encoder and baseline prompt template.
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(ROOT/'build/benchmarks/ovo_full_20261110/upstream'))
    from ovo.utils.clip_utils import load_clip_model
    model,tokenizer,_,_=load_clip_model('SigLIP-384',False)
    with torch.no_grad():
        added=model.encode_text(tokenizer(['This is a photo of a '+name for name,_ in LIVE_TERMS]),normalize=True).float().numpy()
    ref=output/'vocabulary';ref.mkdir()
    text=np.load(reference/'text_features.npy');vocab=json.loads((reference/'vocabulary.json').read_text())
    vocab['names'] += [name for name,_ in LIVE_TERMS];vocab['replica_class_ids'] += [i for _,i in LIVE_TERMS]
    vocab['scope']='Original 51 benchmark terms plus fixed live ontology; no benchmark scores use this extension.'
    vocab['added_terms_prompt']='This is a photo of a {name}'
    np.save(ref/'text_features.npy',np.concatenate([text,added]));dump(ref/'vocabulary.json',vocab)
    for name in ['text_features.npy','vocabulary.json']:inputs[str((ref/name).resolve())]=sha256(ref/name)
    source=sources();source[str(Path(__file__).relative_to(ROOT))]=sha256(Path(__file__))
    source['omni_semantic_map/omni_semantic_map/sparse_semantic_support.py']=sha256(ROOT/'omni_semantic_map/omni_semantic_map/sparse_semantic_support.py')
    job=dict(kind='livo_atomic',scene=scene,output=str(output.resolve()),config=cfg,
        frames=capture['frames'],reference_cache=str(ref.resolve()),input_sha256=inputs,source_sha256=source,
        capture=str(capture_path.resolve()),no_gt_in_inference=True,support='measured tracking-frame cloud, image-epoch transform; per-pixel 5cm z buffer',
        semantic_frontend='frozen SAM2/SigLIP/learned merger, batch8 BF16; unchanged split frontend phases',
        live_ontology_extension=LIVE_TERMS,live_ontology_text_precision='CPU FP32; original benchmark text features unchanged')
    assert not job_path.exists();dump(job_path,job);print('JOB',job_path,flush=True)


def geometry(job_path):
    from split_strong_frontend import verify_job
    job=json.loads(job_path.read_text());verify_job(job,job_path);out=Path(job['output'])
    for phase in ['masks','features']:
        status=json.loads((out/(phase+'_run.json')).read_text());assert status['complete'] and status['job_sha256']==sha256(job_path)
        for name,h in status['files_sha256'].items():assert sha256(out/name)==h
    cache=out/'cache';cache.mkdir(exist_ok=False);(cache/'frames').mkdir()
    manifest=dict(complete=False,scene=job['scene'],no_association=True,no_semantic_gt=True,
        geometry_scope=job['support'],job_sha256=sha256(job_path),config=job['config'],records=[])
    began=time.monotonic();xyzs=[];colors=[];poses=[];offset=0;point_offset=0
    for record in job['frames']:
        name=f"{record['frame_id']:06d}.npz";image=cv2.cvtColor(cv2.imread(record['rgb']),cv2.COLOR_BGR2RGB)
        points=np.load(record['cloud'])['points'];T=np.asarray(record['pose']);camera=np.asarray(record['camera_from_tracking'])
        with np.load(out/'masks'/name) as f:arrays={k:f[k] for k in f.files}
        arrays['features']=np.load(out/'features'/name)['features'];n=len(arrays['features'])
        projected=sparse_support(points,camera,record['intrinsic'],arrays['segmentation'])
        ids=projected['point_indices'];mid=projected['mask_ids'];assert not len(mid) or mid.max()<n
        order=np.argsort(mid,kind='stable')
        arrays.update(point_ids=(ids[order]+point_offset).astype(np.int64),offsets=np.r_[0,np.cumsum(np.bincount(mid,minlength=n))].astype(np.int64))
        colored=np.zeros((len(points),3),np.uint8);vis=projected['visible_indices'];uv=projected['visible_pixels'];colored[vis]=image[uv[:,1],uv[:,0]]
        xyzs.append((points@T[:3,:3].T+T[:3,3]).astype(np.float32));colors.append(colored)
        point_offset+=len(points);poses.append(record['camera_pose'])
        path=cache/'frames'/name;np.savez_compressed(path,**arrays)
        manifest['records'].append(dict(frame_id=record['frame_id'],file=name,masks=n,observation_offset=offset,
            matched_supports=len(ids),map_points=point_offset,visible_points=len(vis),occluded_points=projected['rejected_occluded'],
            masks_without_support=int(np.count_nonzero(np.diff(arrays['offsets'])==0)),sha256=sha256(path)))
        offset+=n
    np.savez_compressed(cache/'geometry.npz',xyz=np.concatenate(xyzs),colors=np.concatenate(colors))
    np.save(cache/'poses.npy',np.asarray(poses))
    for name in ['text_features.npy','vocabulary.json']:shutil.copyfile(Path(job['reference_cache'])/name,cache/name)
    manifest.update(complete=True,frames=len(poses),frames_cached=len(poses),observations=offset,map_points=point_offset,
        wall_s=time.monotonic()-began,files_sha256={p.name:sha256(p) for p in cache.iterdir() if p.is_file() and p.name!='manifest.json'})
    dump(cache/'manifest.json',manifest);print('CACHE',cache,'frames',len(poses),'observations',offset,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--capture',type=Path);p.add_argument('--output',type=Path);p.add_argument('--scene',default='rosbag2245_30s');p.add_argument('--geometry',action='store_true');a=p.parse_args()
    if a.geometry:geometry(a.job)
    else:prepare(a.capture,a.output,a.job,a.scene)
