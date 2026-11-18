"""Paths, immutable clean-input export and SAM2 adapter for the bounded pilot."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'omni_semantic_map'))
from omni_semantic_map.memory_resegmentation import CausalTokenMemory,choose_events
from omni_semantic_map.flat_benchmark import sha256 as sha
OUT=ROOT/'build/benchmarks/memory_resegmentation_20261112'
REPORT=ROOT/'docs/reports/memory_resegmentation_2026-11-12'
PRIOR=ROOT/'build/benchmarks/full_object_mainline_20261112'
DINO=ROOT/'build/benchmarks/evidence_retention_20261111/public_dino_v1'
HISTORY=ROOT/'build/benchmarks/evidence_retention_20261111/public_history_v1'
SAMROOT=ROOT/'build/benchmarks/ovo_full_20261110/upstream'
PY=ROOT/'build/benchmarks/ovo_full_20261110/venv311/bin/python'
CGPY=ROOT/'build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python'
SEEDS=[17,29,43]
MODES=['current','mean','attention']

def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def archive(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}
def spec():return read(OUT/'protocol.json')

def freeze():
    assert not (OUT/'protocol.json').exists()
    p=read(PRIOR/'protocol.json');images=read(DINO/'protocol.json')
    scenes=[]
    for s in p['scenes']:
        x=dict(s);x['images']=next(t['images'] for t in images['scenes'] if t['key']==s['key'])
        scenes.append(x)
    sample=read(Path(scenes[0]['cache'])/'manifest.json')['config']['semantic']['sam']
    ckpt=Path(sample['sam_ckpt_path'])/'sam2.1_hiera_large.pt'
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/memory_resegmentation.py',
             ROOT/'docs/paper/object_memory_experiment_plan_2026-11-12.md']
    dump(OUT/'protocol.json',dict(created=time.time(),scenes=scenes,splits=p['splits'],
        seeds=SEEDS,modes=MODES,sam_checkpoint=str(ckpt),sam_sha256=sha(ckpt),
        source_sha256={str(x):sha(x) for x in sources},parent_sha256=sha(PRIOR/'protocol.json'),
        events_per_frame=2,sam_calls_per_event=2,history=dict(max_tokens=8192,neighbors=8,radius=.20),
        training=dict(lr=.001,weight_decay=.0001,epochs=50,batch=512,pairs_per_scene_epoch=4096),
        maximum_training_runs=9,maximum_development_maps=12,
        selection='Validation measured-support per-observation instance PQ after actual SAM decoding; FP then earlier epoch ties. No evaluation labels.',
        implementation_scope='Causal raw local-evidence memory, not final CG object memberships. First implementation resegments current observations; it does not retroactively delete old fused points.',
        controlled_footprint='SAM produces new dense masks, then partitions the original measured mask footprint. Union of support stays fixed. Cannot recover geometry outside original support or claim mask-recall improvement.',
        semantic_control='New mask fragments inherit original frozen SigLIP descriptor; semantics and point budget remain fixed to isolate partition effects. Not freshly re-encoded semantic descriptors.',
        explicit_protocol_refinement='Before new fitting or scores: test identity partition only, not missing-geometry recovery. Shared raw-evidence memory and two anchor candidates; 2 positive + 2 negative prompts for learned reader, geometric control supplies identical call/point budget.',
        baseline='Native causal CG plus common final denoise/filter/guarded merge; no offline history graph. Archived original baseline may be reused only after equivalence check.',
        no_gt_in_inference=True,development_only=True))

def export_inputs():
    p=spec();tables={role:dict(features=[],xyz=[],memory=[],pairs=[],targets=[],weights=[],foreground=[],scenes=[]) for role in ['train','validation']}
    for s in p['scenes']:
        role=next(k for k,v in p['splits'].items() if s['key'] in v)
        dest=OUT/'inputs'/s['key'];dest.mkdir(parents=True,exist_ok=False)
        source=PRIOR/'supervision'/role/s['key'];sm=read(source/'manifest.json')
        c=Path(s['cache']);cm=read(c/'manifest.json');geo=archive(c/'geometry.npz');poses=np.load(c/'poses.npy')
        cam=cm['config']['cam'];K=np.array([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1.]])
        memory=CausalTokenMemory(**p['history']);records=[]
        global_offset=sum(len(a) for a in tables[role]['features']) if role in tables else 0
        frameoffset=0
        for i,(rec,sr) in enumerate(zip(cm['records'],sm['records'])):
            assert rec['frame_id']==sr['frame_id']
            # Evaluation labels are never read. Clean files contain input fields only.
            with np.load(source/sr['file']) as z:
                f=z['features'].astype(np.float32);f/=np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12)
                offsets=z['offsets'];points=z['point_ids'];po=z['point_offsets']
                xyz=np.array([geo['xyz'][points[a:b]].mean(0) for a,b in zip(po[:-1],po[1:])],np.float32).reshape(-1,3)
                T=np.linalg.inv(poses[i]);pc=xyz@T[:3,:3].T+T[:3,3];uv=pc@K.T
                pixels=uv[:,:2]/np.maximum(uv[:,2:],1e-8)
                pixels[:,0]=np.clip(pixels[:,0],0,cam['W']-1);pixels[:,1]=np.clip(pixels[:,1],0,cam['H']-1)
                _,valid,mi=memory.query(xyz,rec['frame_id'])
                token_ids=np.arange(len(f))+frameoffset+global_offset
                memory.add(xyz,f,rec['frame_id'],token_ids)
                event=choose_events(offsets,f,p['events_per_frame'])
                np.savez_compressed(dest/sr['file'],features=f.astype(np.float16),xyz=xyz,pixels=pixels,
                    offsets=offsets,point_ids=points,point_offsets=po)
                records.append(dict(frame_id=rec['frame_id'],file=sr['file'],original_file=rec['file'],
                    observation_offset=rec['observation_offset'],events=event,sha256=sha(dest/sr['file'])))
                if role in tables:
                    labels=z['target_instance_ids'];counts=z['target_counts'];to=z['target_offsets'];stats=z['target_stats']
                    distribution=[]
                    for a,b,n in zip(to[:-1],to[1:],stats[:,1]):
                        distribution.append({int(k):float(v)/max(int(n),1) for k,v in zip(labels[a:b],counts[a:b]) if k>0})
                    pairs=[];targets=[];weights=[]
                    for lo,hi in zip(offsets[:-1],offsets[1:]):
                        if hi-lo<2:continue
                        cosine=f[lo:hi]@f[lo:hi].T
                        a,b=np.unravel_index(np.argmin(cosine),cosine.shape)
                        for anchor in sorted(set([int(lo+a),int(lo+b)])):
                            for q in range(lo,hi):
                                if anchor==q or stats[q,1]==0 or stats[anchor,1]==0:continue
                                pairs.append([token_ids[anchor],token_ids[q]])
                                targets.append(sum(v*distribution[q].get(k,0.) for k,v in distribution[anchor].items()))
                                weights.append(min(stats[q,1]/max(stats[q,0],1),stats[anchor,1]/max(stats[anchor,0],1)))
                    t=tables[role];t['features'].append(f.astype(np.float16));t['xyz'].append(xyz);t['memory'].append(mi)
                    t['foreground'].append(stats[:,2]/np.maximum(stats[:,1],1))
                    t['pairs'].append(np.array(pairs,np.int64).reshape(-1,2));t['targets'].append(np.array(targets,np.float32))
                    t['weights'].append(np.array(weights,np.float32));t['scenes'].append(np.full(len(pairs),p['splits'][role].index(s['key']),np.int16))
            frameoffset+=len(f)
        dump(dest/'manifest.json',dict(complete=True,key=s['key'],role=role,records=records,gt_fields_absent=True))
        print('INPUTS',s['key'],len(records),'frames',frameoffset,'tokens',flush=True)
    for role,arrays in tables.items():
        folder=OUT/'training_data'/role;folder.mkdir(parents=True)
        for k,values in arrays.items():np.save(folder/(k+'.npy'),np.concatenate(values))
    dump(OUT/'inputs_complete.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json')))

def sam_predictor():
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    # Measured regeneration peak is 1.56 GiB; cap allocations on the shared GPU.
    torch.cuda.set_per_process_memory_fraction(3*1024**3/torch.cuda.get_device_properties(0).total_memory)
    model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',spec()['sam_checkpoint'],device='cuda',apply_postprocessing=False)
    return SAM2ImagePredictor(model.eval())

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','export']);a=ap.parse_args()
    freeze() if a.action=='freeze' else export_inputs()
