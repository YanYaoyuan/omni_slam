"""Bounded real-image acquisition pilot, without reference-label access.

Preparation and actual SAM inference are separate commands. This pilot exports
candidate source partitions; it does not claim completed map revision or AP gain.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.hypothesis_acquisition import (
    AcquisitionConfig, choose_views, symmetric_prompts, source_partition_from_logits)
from omni_semantic_map.sparse_lidar_support import project_measured_visibility

BASE = ROOT/'build/benchmarks/kitti360_connected_development_v1'
OUT = ROOT/'build/benchmarks/object_hypothesis_acquisition_v1'
POLICIES = ['uniform', 'random', 'area', 'uncertainty', 'disagreement']


def source_paths():
    return [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/hypothesis_acquisition.py',
            ROOT/'omni_semantic_map/omni_semantic_map/sparse_lidar_support.py']


def prepare():
    assert not OUT.exists(), 'Do not overwrite the pilot or tune it in place'
    cfg = AcquisitionConfig()
    job = read(BASE/'job.json'); cache = BASE/'cache'; manifest = read(cache/'manifest.json')
    assert job['scene'] == '2013_05_28_drive_0000_sync' and manifest['complete']
    g = np.load(cache/'geometry.npz'); xyz = g['xyz']; poses = np.load(cache/'poses.npy')
    text = np.load(cache/'text_features.npy'); vocab = read(cache/'vocabulary.json')
    cam = manifest['config']['cam']
    roles = ['validate' if (i//5)%3 == 2 else 'acquire' for i in range(len(job['frames']))]
    OUT.mkdir(parents=True)
    inputs = {str(BASE/'job.json'):sha(BASE/'job.json'), str(cache/'manifest.json'):sha(cache/'manifest.json')}
    for name, expected in manifest['files_sha256'].items():
        assert sha(cache/name) == expected
        inputs[str(cache/name)] = expected
    for record, frame in zip(manifest['records'], job['frames']):
        assert record['frame_id'] == frame['frame_id']
        path = cache/'frames'/record['file']
        assert sha(path) == record['sha256']
        inputs[str(path)] = record['sha256']
        inputs[frame['measured']] = sha(frame['measured'])
        inputs[frame['rgb']] = sha(frame['rgb'])
    checkpoint = Path(job['config']['semantic']['sam']['sam_ckpt_path'])/'sam2.1_hiera_large.pt'
    inputs[str(checkpoint)] = sha(checkpoint)
    dump(OUT/'protocol.json', dict(created_epoch=time.time(), scene=job['scene'], config=asdict(cfg),
        source_sha256={str(p):sha(p) for p in source_paths()}, input_sha256=inputs,
        checkpoint=str(checkpoint), cache=str(cache), job=str(BASE/'job.json'), policies=POLICIES,
        frame_roles=[dict(frame_id=r['frame_id'], role=role) for r,role in zip(job['frames'], roles)],
        max_logical_decodes=cfg.max_events*cfg.views_per_event*3*len(POLICIES),
        max_encodings=cfg.max_events*cfg.views_per_event*len(POLICIES),
        inference='SAM2.1 Hiera large FP32 weights, BF16 CUDA; full original RGB at frozen native SAM resolution; three symmetric four-point prompt sets per action, one joint and two exclusive.',
        isolation='Five-frame blocks; every third block validation. Validation masks and features are not read for proposal or action selection. Geometry/poses remain observed input. Offline replay, not online robot exploration.',
        candidate_policy='Acquisition-frame observations classified car/building by frozen features. Each exclusive source fragment >=16; cosine>=.65 and nearest 3D distance<=.5m. Rank by cosine/(1+gap), then observation IDs. At most4 events with disjoint observation IDs and no >50% source overlap.',
        holdout_limit='View-group separation reduces leakage; no claim of statistically independent viewpoints. No GT in inference. Only exposed development sequence; reserved sequences untouched.',
        output_scope='Real-image masks, symmetric prompts and competing partitions of measured source points; no full map or new benchmark score yet.'))
    obs = []
    for i,(rec,role) in enumerate(zip(manifest['records'], roles)):
        if role != 'acquire': continue
        with np.load(cache/'frames'/rec['file']) as z:
            for j, (lo,hi) in enumerate(zip(z['offsets'][:-1], z['offsets'][1:])):
                ids = np.unique(z['point_ids'][lo:hi])
                if len(ids) < cfg.min_fragment_points: continue
                feature = z['features'][j].astype(float)
                feature /= max(np.linalg.norm(feature), 1e-12)
                semantic = vocab['semantic_ids'][int(np.argmax(text@feature))]
                if semantic not in (11,26): continue
                obs.append(dict(id=rec['observation_offset']+j, frame_index=i, mask=j,
                                feature=feature, ids=ids, semantic_id=semantic))
    proposals = []
    for ai,a in enumerate(obs):
        for b in obs[ai+1:]:
            cosine = float(a['feature']@b['feature'])
            if cosine < cfg.min_descriptor_cosine: continue
            pa, pb = np.setdiff1d(a['ids'],b['ids']), np.setdiff1d(b['ids'],a['ids'])
            if min(len(pa),len(pb)) < cfg.min_fragment_points: continue
            gap = float(cKDTree(xyz[pa]).query(xyz[pb],k=1,workers=1)[0].min())
            if gap > cfg.max_fragment_gap_m: continue
            proposals.append((cosine/(1+gap), a, b, pa, pb, cosine, gap))
    proposals.sort(key=lambda x:(-x[0],x[1]['id'],x[2]['id']))
    events, used_obs, used_sources, audits = [], set(), [], []
    for rank,(_,a,b,pa,pb,cosine,gap) in enumerate(proposals):
        if a['id'] in used_obs or b['id'] in used_obs: continue
        union = np.union1d(pa,pb)
        if any(len(np.intersect1d(union,old))/min(len(union),len(old)) > .5 for old in used_sources): continue
        rng = np.random.default_rng(cfg.seed+a['id']*1009+b['id'])
        samples = [np.sort(rng.choice(part,min(len(part),cfg.samples_per_fragment),replace=False)) for part in (pa,pb)]
        ids = np.concatenate(samples); fragment = np.repeat([0,1],list(map(len,samples)))
        partitions = np.stack([np.ones(len(ids),int),fragment+1])
        visible, pixels = [], []
        for frame, pose in zip(job['frames'],poses):
            depth = np.load(frame['measured'])['depth']
            x,y,_,v = project_measured_visibility(xyz[ids],pose,cam,depth,cfg)
            pixels.append(np.column_stack([x,y]));visible.append(v)
        pixels, visible = np.array(pixels),np.array(visible)
        # Exclude views where sparse returns project both labels onto one anchor.
        for fi in range(len(visible)):
            if roles[fi] != 'acquire': continue
            try: symmetric_prompts(pixels[fi],visible[fi],fragment)
            except ValueError: visible[fi] = False
        choices = {name:choose_views(partitions,visible,poses[:,:3,3],roles,fragment,name,cfg) for name in POLICIES}
        eligible_validation = [i for i,role in enumerate(roles) if role == 'validate'
                               and all(visible[i,fragment==k].sum()>=cfg.min_visible_per_fragment for k in (0,1))]
        audit = dict(pair=[a['id'],b['id']],proposal_rank=rank,
                     eligible_acquire=len(choices['disagreement']['eligible']),
                     eligible_validation=len(eligible_validation))
        audits.append(audit)
        if any(len(c['selected']) < cfg.views_per_event for c in choices.values()) or not eligible_validation: continue
        eid = len(events); dest = OUT/'events'/f'{eid:03d}';dest.mkdir(parents=True)
        np.savez_compressed(dest/'proposal.npz',sample_ids=ids,fragments=fragment,
            partitions=partitions,visibility=visible,pixels=pixels,
            source_a=pa,source_b=pb,shared_sources=np.intersect1d(a['ids'],b['ids']))
        event = dict(event_id=eid,observations=[a['id'],b['id']],source_frames=[job['frames'][o['frame_index']]['frame_id'] for o in (a,b)],
                     semantic_ids=[a['semantic_id'],b['semantic_id']],cosine=cosine,gap_m=gap,
                     source_counts=[len(pa),len(pb)],choices=choices,validation_indices=eligible_validation,
                     file=str(dest/'proposal.npz'),sha256=sha(dest/'proposal.npz'))
        dump(dest/'event.json',event);events.append(event)
        used_obs.update([a['id'],b['id']]);used_sources.append(union)
        if len(events) == cfg.max_events: break
    dump(OUT/'prepared.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),
        acquisition_observations=len(obs),raw_pairs=len(proposals),events=events,candidate_audit=audits,
        validation_masks_read=False,gt_accessed=False))
    print('PREPARED',len(obs),'observations',len(proposals),'pairs',len(events),'eligible events',flush=True)


def verify():
    p = read(OUT/'protocol.json'); ready = read(OUT/'prepared.json')
    assert ready['complete'] and ready['protocol_sha256'] == sha(OUT/'protocol.json')
    for key in ['source_sha256','input_sha256']:
        for path,expected in p[key].items(): assert sha(path) == expected,path
    for event in ready['events']: assert sha(event['file']) == event['sha256']
    return p,ready


def acquire():
    p,ready = verify()
    assert ready['events'], 'No eligible event; do not launch GPU inference'
    output = OUT/'acquired';output.mkdir(exist_ok=False)
    import torch
    import cv2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(2);cv2.setNumThreads(1)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(3*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(p['config']['seed'])
    model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',p['checkpoint'],device='cuda',apply_postprocessing=False)
    predictor = SAM2ImagePredictor(model.eval())
    cfg = AcquisitionConfig(**p['config']);job = read(p['job']);cam = job['config']['cam']
    xyz = np.load(Path(p['cache'])/'geometry.npz')['xyz']; poses=np.load(Path(p['cache'])/'poses.npy')
    records=[];started=time.monotonic();actual_calls=0;encodings=0;logical_calls=0
    for event in ready['events']:
        packet=np.load(event['file']);eid=event['event_id']
        # Reuse identical actions across policies. Report both logical and actual costs.
        requested=sorted({i for c in event['choices'].values() for i in c['selected']})
        for fi in requested:
            frame=job['frames'][fi];assert p['frame_roles'][fi]['role']=='acquire'
            image=cv2.cvtColor(cv2.imread(frame['rgb']),cv2.COLOR_BGR2RGB)
            coords,labels,anchors=symmetric_prompts(packet['pixels'][fi],packet['visibility'][fi],packet['fragments'])
            tic=time.monotonic()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                predictor.set_image(image)
                values,quality,_=predictor.predict(point_coords=coords,point_labels=labels,
                    multimask_output=False,return_logits=True)
            torch.cuda.synchronize();elapsed=time.monotonic()-tic
            values=np.asarray(values).reshape(3,*image.shape[:2]).astype(np.float32)
            assert np.isfinite(values).all()
            depth=np.load(frame['measured'])['depth']
            x,y,_,valid=project_measured_visibility(xyz,poses[fi],cam,depth,cfg)
            # Fixed local image envelope from both fragment projections, with 16px context.
            uv=packet['pixels'][fi][packet['visibility'][fi]]
            low=uv.min(0)-16;high=uv.max(0)+16
            valid &= (x>=low[0])&(x<=high[0])&(y>=low[1])&(y<=high[1])
            source_ids=np.flatnonzero(valid)
            proposed=source_partition_from_logits(values,np.column_stack([x[source_ids],y[source_ids]]))
            dest=output/f'event_{eid:03d}_frame_{fi:03d}.npz'
            np.savez_compressed(dest,source_ids=source_ids,partitions=proposed,
                masks_packed=np.packbits(values>0,axis=2),width=image.shape[1],
                prompt_coordinates=coords,prompt_labels=labels,prompt_source_ids=packet['sample_ids'][anchors],
                quality=np.asarray(quality).reshape(-1))
            panels=[image.copy()]
            for k,color in enumerate([[0,180,255],[255,80,80],[70,220,110]]):
                view=image.copy();mask=values[k]>0
                view[mask]=(view[mask]*.4+np.array(color)*.6).astype(np.uint8)
                for point,label in zip(coords[k],labels[k]):
                    cv2.circle(view,tuple(map(int,point)),5,(0,255,0) if label else (255,0,0),-1)
                panels.append(view)
            cv2.imwrite(str(dest.with_suffix('.jpg')),cv2.cvtColor(np.concatenate(panels,axis=0),cv2.COLOR_RGB2BGR))
            consumers=[name for name,c in event['choices'].items() if fi in c['selected']]
            logical_calls+=3*len(consumers);actual_calls+=3;encodings+=1
            records.append(dict(event_id=eid,frame_index=fi,frame_id=frame['frame_id'],policies=consumers,
                file=str(dest),sha256=sha(dest),wall_s=elapsed,source_points=len(source_ids),
                partition_counts=[dict((str(int(k)),int(n)) for k,n in zip(*np.unique(row,return_counts=True))) for row in proposed]))
            dump(output/'progress.json',dict(completed_actions=len(records),logical_calls=logical_calls,
                                           actual_calls=actual_calls,encodings=encodings))
            print('ACQUIRED',eid,frame['frame_id'],consumers,'sources',len(source_ids),flush=True)
    assert logical_calls <= p['max_logical_decodes'] and encodings <= p['max_encodings']
    dump(output/'manifest.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),
        prepared_sha256=sha(OUT/'prepared.json'),records=records,logical_calls=logical_calls,
        actual_calls=actual_calls,encodings=encodings,wall_s=time.monotonic()-started,
        gpu_name=torch.cuda.get_device_name(0),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        gt_accessed=False,validation_masks_read=False,
        scope='Candidate masks and source partitions only; acceptance and full historical-map edits remain pending.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','acquire','verify'])
    action=parser.parse_args().action
    {'prepare':prepare,'acquire':acquire,'verify':verify}[action]()
