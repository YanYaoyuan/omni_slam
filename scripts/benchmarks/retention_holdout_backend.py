"""Frozen ConceptGraphs + source guard; paired readouts; no GT access."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import numpy as np
from audit_evidence_retention import ROOT, BASE, OUT, dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.shared_observations import observation_table, select_views
from omni_semantic_map.evidence_readout import aggregate
from omni_semantic_map.full_ovo_evaluation import BACKGROUND, cg_geometry


def run(cache,output,expected=None):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    from omegaconf import OmegaConf
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend,FusionObservation
    from omni_semantic_map.fusion_diagnostics import guarded_merge,membership_audit
    selection=json.loads((OUT/'stage3_selection.json').read_text());cfg_path=Path(selection['cg_config'])
    assert sha256(cfg_path)==selection['cg_config_sha256']
    assert sha256(ROOT/'omni_semantic_map/omni_semantic_map/evidence_readout.py')==selection['frozen_readout_source_sha256']
    manifest=json.loads((cache/'manifest.json').read_text());assert manifest['complete'] and manifest['no_association']
    for name,h in manifest['files_sha256'].items():assert sha256(cache/name)==h
    for row in manifest['records']:assert sha256(cache/'frames'/row['file'])==row['sha256']
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();status=dict(complete=False,cache_manifest_sha256=sha256(cache/'manifest.json'),
        source_sha256=sha256(Path(__file__)),config_sha256=sha256(cfg_path),no_gt=True,frame_times=[])
    dump(output/'run.json',status)
    cfg=OmegaConf.create(json.loads(cfg_path.read_text()));engine=ConceptGraphsBackend(cfg,seed=0)
    geometry=np.load(cache/'geometry.npz');xyz,colors=geometry['xyz'],geometry['colors']
    features,frames,areas=observation_table(cache);text=np.load(cache/'text_features.npy');vocab=json.loads((cache/'vocabulary.json').read_text())
    poses=np.load(cache/'poses.npy');pose_by_frame={r['frame_id']:poses[i] for i,r in enumerate(manifest['records'])}
    for i,record in enumerate(manifest['records']):
        began=time.monotonic();f=np.load(cache/'frames'/record['file']);observations=[]
        for k in range(record['masks']):
            oid=record['observation_offset']+k;support=f['point_ids'][f['offsets'][k]:f['offsets'][k+1]]
            observations.append(FusionObservation(oid,record['frame_id'],0,xyz[support],colors[support].astype(np.float64)/255,features[oid],text[int((text@features[oid]).argmax())]))
        engine.integrate(observations,np.eye(4));status['frame_times'].append(time.monotonic()-began)
        if i%25==0:print('CG',manifest['scene'],i,len(engine._objects),flush=True)
    filtered=engine.utils.filter_objects(cfg,engine.utils.denoise_objects(cfg,deepcopy(engine._objects)))
    objects=guarded_merge(cfg,deepcopy(filtered));status['ownership']=membership_audit(objects)
    exported=[dict(object_id=min(o['observation_ids'])+1,points=np.asarray(o['pcd'].points)) for o in objects]
    points,labels,identity=cg_geometry(exported)
    memberships=[(r['evaluation_id'],objects[r['object_index']]['observation_ids']) for r in identity]
    np.savez_compressed(output/'geometry.npz',xyz=points,instance=labels)
    text=text/np.linalg.norm(text,axis=1,keepdims=True);class_ids=np.array(vocab['replica_class_ids']);rows=[]
    for oid,members in memberships:
        selected=np.array(select_views(members,frames,areas,top_k=0));center=points[labels==oid].mean(0)
        cameras=np.array([pose_by_frame[int(frames[i])][:3,3] for i in selected])
        row=dict(object_id=oid,observation_ids=sorted(set(map(int,members))),selected_observation_ids=selected.tolist(),
            center=center.tolist(),unique_frames=len(selected),predictions={})
        for method in ['mean','area_density']:
            readout=aggregate(features[selected],areas[selected],cameras,center,method=method,observation_ids=selected)
            scores=text@readout['descriptor'];order=np.argsort(-scores,kind='stable');best=order[0];cls=int(class_ids[best])
            row['predictions'][method]=dict(name=vocab['names'][best],replica_class_id=cls,score=float(scores[best]),
                margin=float(scores[best]-scores[order[1]]),retained=cls not in BACKGROUND,effective_views=readout['effective_views'])
        rows.append(row)
    dump(output/'classification.json',dict(objects=rows))
    if expected:
        reference=np.load(expected/'guarded_geometry.npz')
        np.testing.assert_array_equal(points,reference['xyz']);np.testing.assert_array_equal(labels,reference['instance'])
        original=json.loads((expected/'guarded_classification.json').read_text())['objects']
        for row,old in zip(rows,original):
            assert row['object_id']==old['object_id'] and row['observation_ids']==old['observation_ids']
            assert row['predictions']['mean']['replica_class_id']==old['predictions']['all_views_mean']['replica_class_id']
        assert len(rows)==len(original)
        status['original_backend_geometry_memberships_and_mean_classes_exact']=True
    status.update(complete=True,wall_s=time.monotonic()-start,objects=len(rows),points=len(points),
        files_sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='run.json'})
    dump(output/'run.json',status);print('COMPLETE CG',manifest['scene'],len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected',type=Path);args=p.parse_args()
    run(args.cache,args.output,args.expected)
