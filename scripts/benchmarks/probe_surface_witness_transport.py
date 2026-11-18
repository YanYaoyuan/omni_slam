"""Fixed mechanism screen, using existing video masks and unchanged maps.

Prediction is sealed before reference-label diagnostics. This is an exposed
development probe; no new full-map AP, held-out generalization or novelty claim.
"""
import argparse
import runpy
import time
from types import SimpleNamespace
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from probe_world_anchor_tracking import OUT as TRACK, INPUT, prepared
from audit_tracking_history import OUT as AUDIT, BASE, count, evidence_groups
from omni_semantic_map.surface_witness_transport import local_planes,surface_edges,frame_votes
from omni_semantic_map.sparse_lidar_support import project_measured_visibility
from evaluate_kitti360_development import transfer,read_static_ply

OUT=ROOT/'build/benchmarks/surface_witness_transport_probe_v1'
METHODS=['exact_point','world_cell','surface_transport']


def freeze():
    assert not OUT.exists()
    track=read(TRACK/'protocol.json'); inputs={}
    for name in ['summary','cell_summary']:
        f=AUDIT/(name+'.json'); d=read(f); assert d['complete']; inputs[str(f)]=sha(f)
        for path,h in d['files'].items(): assert sha(path)==h; inputs[path]=h
    for f in [TRACK/'prepared.json',TRACK/'predictions_complete.json',AUDIT/'protocol.json',AUDIT/'cell_protocol.json']:
        inputs[str(f)]=sha(f)
    for f,h in read(TRACK/'predictions_complete.json')['files'].items(): assert sha(f)==h; inputs[f]=h
    for f,h in read(AUDIT/'protocol.json')['input_hashes'].items(): assert sha(f)==h; inputs[f]=h
    for scene in track['scenes']:
        key=scene['key']; manifest=read(INPUT/key/'manifest.json')
        for f in [INPUT/key/'manifest.json',manifest['geometry']]+[r['file'] for r in manifest['records']]: inputs[str(f)]=sha(f)
    sources=[__file__,ROOT/'omni_semantic_map/omni_semantic_map/surface_witness_transport.py',ROOT/'omni_semantic_map/test/test_surface_witness_transport.py',ROOT/'omni_semantic_map/omni_semantic_map/sparse_lidar_support.py',ROOT/'scripts/benchmarks/evaluate_kitti360_development.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),input_hashes=inputs,source_hashes={str(f):sha(f) for f in sources},scenes=track['scenes'],references=track['references'],methods=METHODS,
        hypothesis='Transport temporal mask evidence between measurements on compatible local surfaces. Avoid exact-point revisit starvation and indiscriminate20cm cell transport. This is an observation operator, not a validated new mapping method.',
        masks='Existing video_seed only, all8previously selected cases, identical common nonseed frames. Never reuse failed world-feedback masks as Ours. Cases and all existing results are exposed; no independent-test claim.',
        geometry='Fit unoriented local planes in unchanged full measured-source cloud:16nearest within20cm,>=6,lambda0/trace<=.05,lambda1/trace>=.05,query-to-fitted-plane<=5cm. Radius and5cm from prior measured geometry conventions; planarity/normal thresholds are new fixed development choices, not optimized by a sweep.',
        transport='Current recorded measured-visible source witnesses only. Target is an original fused-map point in current image FOV. Radius<=20cm,absolute normal cosine>=sqrt(3)/2,both point-to-plane residuals<=5cm. One vote per target/frame; compatible witnesses allinside=support,alloutside=exclusion,mixed/missing=unknown. Normal fits use window-end geometry, not causal online processing. This is not exact-target visibility or proven occlusion handling.',
        controls='Unchanged exact-point5cm-depth and20cm unanimous-cell audit counters with identical video masks and frames. Common diagnostic rejection set:>=2excluded frames AND zero supporting frames; no maps are actually edited. A negative mask vote does not label a replacement owner/background.',
        screening='In BOTH exposed windows, surface_transport error-detection precision strictly exceeds world_cell and recall strictly exceeds exact_point, using known original-map points and fixed prior GT association; require rejected incorrect points>rejected correct points. Unknown/N/A counts retained. This necessary screen is not AP success, independent validation, statistical significance or novelty proof.',
        budget='One CPU geometry fit and transport pass over8fixed clips;no new model calls,training,parameter sweep,new masks,maps/AP,robot or new scenes. Failed screen closes this fixed version. Passing permits a separately frozen full-window/all-object comparison, including SOR and identical-mask exact/cell controls.',
        novelty='Local PCA,surfel association and multi-view rejection are established components. Only the sparse-observation transport hypothesis is under test; do not claim this alone resolves the paper novelty concern.'))
    print('FROZEN SURFACE TRANSPORT',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items(): assert sha(f)==h,f
    return p


def predict():
    p=verify(); assert not (OUT/'predictions_complete.json').exists(); cases=prepared()['cases']; files={};stats={};tick=time.monotonic()
    for scene in p['scenes']:
        key=scene['key']; selected=[c for c in cases if c['key']==key]; base=np.load(BASE/key/'geometry_graph/geometry.npz'); source=np.load(read(INPUT/key/'manifest.json')['geometry']);xyz=source['xyz'];job=read(scene['job']); jobframes={f['frame_id']:f for f in job['frames']}
        observations={r['frame_id']:dict(np.load(r['file'])) for c in selected for j,r in enumerate(c['records']) if j in c['common_frames']}
        witness_ids=np.unique(np.concatenate([d['visible_source_ids'] for d in observations.values()])); witness_index=np.full(len(xyz),-1,np.int64); witness_index[witness_ids]=np.arange(len(witness_ids));wn,wv=local_planes(xyz,xyz[witness_ids])
        used=np.flatnonzero(np.isin(base['instance'],[c['object_id'] for c in selected]));tn,tv=local_planes(xyz,base['xyz'][used]); target_index=np.full(len(base['xyz']),-1,np.int64);target_index[used]=np.arange(len(used));rows=[]
        for case in selected:
            mids=np.flatnonzero(base['instance']==case['object_id']); pos=np.zeros(len(mids),np.int16);neg=pos.copy(); normals=tn[target_index[mids]]; valid=tv[target_index[mids]]; frame_rows=[]; masks=np.load(TRACK/'cases'/case['name']/'video_seed.npz')
            for j in case['common_frames']:
                rec=case['records'][j]; obs=observations[rec['frame_id']]; sid=obs['visible_source_ids'];windex=witness_index[sid]; assert (windex>=0).all(); assert (source['source_frame'][sid]<=rec['frame_id']).all(); assert (sid<rec['map_points']).all()
                raw=jobframes[rec['frame_id']]; depth=np.load(raw['measured'])['depth'];_,_,fov,_=project_measured_visibility(base['xyz'][mids],np.asarray(raw['pose']),job['config']['cam'],depth,SimpleNamespace(depth_absolute_m=.05,depth_relative=0.))
                a,b=surface_edges(base['xyz'][mids],normals,valid&fov,xyz[sid],wn[windex],wv[windex]);xy=obs['pixel_xy'];packed=masks['masks_packed'][j];inside=((packed[xy[:,1],xy[:,0]//8]>>(7-xy[:,0]%8))&1).astype(bool)
                vote=frame_votes(len(mids),a,b,inside);pos+=vote==1;neg+=vote==-1;frame_rows.append(dict(frame=rec['frame_id'],measured_witnesses=len(sid),valid_witness_planes=int(wv[windex].sum()),valid_target_planes=int((valid&fov).sum()),edges=len(a),positive=int((vote==1).sum()),negative=int((vote==-1).sum()),ambiguous_or_unseen=int((vote==0).sum())))
            assert np.all(pos+neg<=len(case['common_frames']))
            f=OUT/'cases'/(case['name']+'.npz'); f.parent.mkdir(exist_ok=True);np.savez_compressed(f,map_ids=mids,positive=pos,negative=neg,valid_plane=valid);files[str(f)]=sha(f); rows.append(dict(case=case['name'],object_id=case['object_id'],map_points=len(mids),valid_planes=int(valid.sum()),frames=frame_rows))
            print('SURFACE PREDICTION',case['name'],'planes',int(valid.sum()),'of',len(mids),'repeated_exclusion',int(((pos==0)&(neg>=2)).sum()),flush=True)
        stats[key]=dict(witness_sources=len(witness_ids),valid_witness_planes=int(wv.sum()),cases=rows)
    dump(OUT/'predictions_complete.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),files=files,results=stats,seconds=time.monotonic()-tick,gt_read_in_predict=False,new_maps=0))
    print('SURFACE PREDICTIONS SEALED',flush=True)


def score():
    p=verify(); seal=read(OUT/'predictions_complete.json'); assert seal['complete'] and not (OUT/'summary.json').exists()
    for f,h in seal['files'].items(): assert sha(f)==h,f
    prior=read(AUDIT/'summary.json'); taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label'];results={};gate=True
    for scene in p['scenes']:
        key=scene['key']; base=np.load(BASE/key/'geometry_graph/geometry.npz');ref=np.load(scene['reference']);raw=read_static_ply(p['references'][key]['raw']);train=np.array([taxonomy[int(v)].trainId for v in raw['semantic']]);expected=train*1000+raw['instance'].astype(np.int64)%1000;expected[train==255]=0;np.testing.assert_array_equal(expected,ref['train_instance']);nn=transfer(ref['xyz'],np.arange(1,len(train)+1),base['xyz'],.2);ix=np.maximum(nn-1,0);known=(nn>0)&(train[ix]!=255)&(ref['confidence'][ix]>0);truth=ref['train_instance'][ix];w=ref['confidence'][ix];rows=[]
        for case in prior['results'][key]['cases']:
            name=case['case'];gid=case['gt_target']; data=np.load(OUT/'cases'/(name+'.npz'));mids=data['map_ids'];exact=np.load(AUDIT/'cases'/(name+'.npz'));cell=np.load(AUDIT/'cell_cases'/(name+'.npz'));np.testing.assert_array_equal(mids,exact['map_ids']);np.testing.assert_array_equal(mids,cell['map_ids']);valid=known[mids] if gid is not None else np.zeros(len(mids),bool);correct=truth[mids]==gid if gid is not None else np.zeros(len(mids),bool)
            counts={'exact_point':(exact['map_video_seed_positive'],exact['map_video_seed_negative']),'world_cell':(cell['video_seed_positive'],cell['video_seed_negative']),'surface_transport':(data['positive'],data['negative'])}
            row=dict(case=name,gt_target=gid,total=count(np.ones(len(mids),bool),valid,correct,w[mids]),methods={})
            for method,(pos,neg) in counts.items():row['methods'][method]={g:count(take,valid,correct,w[mids]) for g,take in evidence_groups(pos,neg).items()}
            rows.append(row)
        total_wrong=sum(r['total']['incorrect'] for r in rows);metrics={}
        for method in METHODS:
            entries=[r['methods'][method]['excluded_repeated'] for r in rows];counts={k:sum(e[k] for e in entries) for k in entries[0]};metrics[method]=dict(**counts,error_detection_precision=counts['incorrect']/max(1,counts['known']),error_detection_recall=counts['incorrect']/max(1,total_wrong))
        ours=metrics['surface_transport']; passed=ours['error_detection_precision']>metrics['world_cell']['error_detection_precision'] and ours['error_detection_recall']>metrics['exact_point']['error_detection_recall'] and ours['incorrect']>ours['correct'];gate &= passed
        results[key]=dict(cases=rows,total_known_incorrect=total_wrong,metrics=metrics,screen_pass=passed);print('SURFACE SCREEN',key,metrics,'PASS',passed,flush=True)
    dump(OUT/'summary.json',dict(complete=True,prediction_seal_sha256=sha(OUT/'predictions_complete.json'),results=results,screen_pass=bool(gate),new_maps=0,new_ap=0,scope='Post-seal exposed-development error-detection diagnostic. N/A retained; semantic/instance target fixed beforehand. This is not full-map benefit or validated novelty.'))
    print('OVERALL SCREEN',gate,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','predict','score']);globals()[a.parse_args().action]()
