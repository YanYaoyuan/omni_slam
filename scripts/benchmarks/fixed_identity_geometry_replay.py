"""Trace native fusion, then replay geometry under frozen observation ownership.

This is a causal diagnostic, not a new object discovery mechanism. The full
membership, rejection, final persistence and semantic path is held fixed.
"""
from pathlib import Path
import sys,time,hashlib
import numpy as np
from paper_final_common import ROOT,read,dump


def setup():
    sys.path.insert(0,str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    import torch
    torch.set_num_threads(1)
    from conceptgraph.slam import mapping,utils
    from conceptgraph.slam.slam_classes import MapObjectList
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend
    return torch,mapping,utils,MapObjectList,ConceptGraphsBackend


def digest(obj):
    return hashlib.sha256(np.asarray(obj['pcd'].points).tobytes()).hexdigest()


def capture(folder,geometry,records,loader,features,frames,areas,classes,scores,text,vocab):
    from discovery_cg_backend import build_map
    torch,mapping,utils,_,backend=setup();events=[]
    integrate=backend.integrate;native_merge=utils.merge_obj2_into_obj1;mapping_merge=mapping.merge_obj2_into_obj1;denoise=utils.denoise_objects
    def frame(self,observations,pose):
        event=dict(kind='frame',ids=[int(o.observation_id) for o in observations]);events.append(event)
        result=integrate(self,observations,pose);rejected=set(self.rejected)
        event['accepted']=[i for i in event['ids'] if i not in rejected]
        return result
    def merge(config,a,b,run_dbscan=True):
        event=dict(kind='merge',a=list(map(int,a['observation_ids'])),b=list(map(int,b['observation_ids'])),run_dbscan=bool(run_dbscan));events.append(event)
        result=native_merge(config,a,b,run_dbscan=run_dbscan);event['xyz_sha256']=digest(result);return result
    def clean(config,objects):
        event=dict(kind='denoise',members=[list(map(int,o['observation_ids'])) for o in objects]);events.append(event)
        result=denoise(config,objects);event['xyz_sha256']=[digest(o) for o in result];return result
    backend.integrate=frame;utils.merge_obj2_into_obj1=merge;mapping.merge_obj2_into_obj1=merge;utils.denoise_objects=clean
    try:build_map(folder,geometry,records,loader,features,frames,areas,classes,scores,text,vocab)
    finally:
        backend.integrate=integrate;utils.merge_obj2_into_obj1=native_merge;mapping.merge_obj2_into_obj1=mapping_merge;utils.denoise_objects=denoise
    dump(folder/'trace.json',dict(events=events,scope='Exact native frame/merge/denoise schedule from largest-component baseline; no GT.'))
    return events


def replay(folder,geometry,records,loader,features,classes,text,vocab,trace_folder,*,assert_native=False):
    import open3d as o3d
    from omegaconf import OmegaConf
    torch,_,utils,MapObjectList,_=setup();run=read(trace_folder/'run.json');cfg=OmegaConf.create(run['effective_config'])
    events=read(trace_folder/'trace.json')['events'];classification=read(trace_folder/'classification.json')
    pool={};rng=np.random.RandomState(run['seed']);xyz=geometry['xyz'];oid_record={};sources={};start=time.monotonic();checked=0
    for r in records:
        data=loader(r)
        for j in range(r['masks']):
            oid=r['observation_offset']+j;oid_record[oid]=r;sources[oid]=data['point_ids'][data['offsets'][j]:data['offsets'][j+1]]
    key=lambda ids:tuple(sorted(map(int,ids)))
    for event in events:
        if event['kind']=='frame':
            accepted=set(event['accepted'])
            for oid in event['ids']:
                ids=sources[oid];points=np.array(xyz[ids],dtype=np.float64,copy=True);points+=rng.normal(0,4e-3,points.shape)
                if oid not in accepted:continue
                assert len(points)>=max(cfg.min_points_threshold,5)
                colors=geometry['colors'][ids].astype(float)/255 if 'colors' in geometry else np.zeros((len(ids),3))
                pcd=o3d.geometry.PointCloud();pcd.points=o3d.utility.Vector3dVector(points);pcd.colors=o3d.utility.Vector3dVector(np.array(colors,order='C',copy=True))
                pcd=utils.process_pcd(pcd.transform(np.eye(4)),cfg);box=utils.get_bounding_box(cfg,pcd);assert box.volume()>=1e-6
                semantic=int(classes[oid]);obj=dict(pcd=pcd,bbox=box,num_detections=1,observation_ids=[oid],class_id=[semantic],
                    clip_ft=torch.tensor(np.asarray(features[oid],np.float32)),text_ft=torch.tensor(np.asarray(text[vocab['semantic_ids'].index(semantic)],np.float32)))
                assert (oid,) not in pool;pool[(oid,)]=obj
        elif event['kind']=='merge':
            a,b=key(event['a']),key(event['b']);assert a!=b and not set(a)&set(b)
            obj=utils.merge_obj2_into_obj1(cfg,pool.pop(a),pool.pop(b),run_dbscan=event['run_dbscan']);pool[key(obj['observation_ids'])]=obj
            if assert_native:assert digest(obj)==event['xyz_sha256'];checked+=1
        elif event['kind']=='denoise':
            members=[key(v) for v in event['members']];objects=utils.denoise_objects(cfg,MapObjectList([pool[v] for v in members]))
            for k,obj,want in zip(members,objects,event['xyz_sha256']):
                pool[k]=obj
                if assert_native:assert digest(obj)==want;checked+=1
        else:raise ValueError('Unknown native trace operation')
    points=[];labels=[];retained=set()
    for row in classification['objects']:
        obj=pool[key(row['observation_ids'])];cloud=np.asarray(obj['pcd'].points);points.append(cloud);labels.append(np.full(len(cloud),row['object_id'],np.int64));retained.update(row['observation_ids'])
    points=np.concatenate(points) if points else np.empty((0,3));labels=np.concatenate(labels) if labels else np.empty(0,np.int64)
    if assert_native:
        old=np.load(trace_folder/'geometry.npz');assert np.array_equal(points,old['xyz']) and np.array_equal(labels,old['instance'])
    folder.mkdir(parents=True,exist_ok=False);np.savez_compressed(folder/'geometry.npz',xyz=points,instance=labels);dump(folder/'classification.json',classification)
    dump(folder/'run.json',dict(complete=True,objects=len(classification['objects']),map_points=len(points),replay_seconds=time.monotonic()-start,
        native_geometry_checks=checked,baseline_bitwise_equal=assert_native,ownership_and_semantics_fixed=True,
        accepted_observations=len({i for e in events if e['kind']=='frame' for i in e['accepted']}),retained_observations=len(retained),
        scope='Native ingestion/jitter/merge/denoise geometry replay with original membership/rejection/persistence/semantic path fixed. Cannot discover objects omitted by the baseline. This freezes the full identity path, not only one association threshold.'))
