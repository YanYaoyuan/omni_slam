"""Official MaskClustering core on the same cached masks/support as CG.

This is a shared-input adaptation, NOT the CropFormer/OpenCLIP full pipeline.
The upstream core and postprocessing are retained. CPU copies replace only
explicit .cuda() placement; cache construction replaces backprojection.
"""
import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from paper_final_common import ROOT,read,dump,sha

BASE=ROOT/'build/benchmarks/visibility_independent_20261112'
UP=BASE/'upstream/MaskClustering'
CPU=BASE/'adapters/MaskClusteringCPU'
FILES=['graph/node.py','graph/construction.py','graph/iterative_clustering.py',
       'utils/geometry.py','utils/post_process.py','configs/scannet.json']


def adapt():
    changed={}
    for name in FILES:
        src=UP/name;dest=CPU/name;dest.parent.mkdir(parents=True,exist_ok=True)
        raw=src.read_text();copy=raw.replace('.cuda()', '.cpu()')
        if name=='graph/construction.py':
            copy=copy.replace('from utils.mask_backprojection import frame_backprojection',
                '# Shared cached input replaces the unused upstream backprojection import.')
        if dest.exists():assert dest.read_text()==copy,name
        else:dest.write_text(copy)
        changed[name]=dict(upstream_sha256=sha(src),cpu_sha256=sha(dest),cuda_calls_replaced=raw.count('.cuda()'))
    dump(CPU/'adaptation.json',dict(files=changed,
        scope='Device-only CPU substitutions; skipped unused backprojection import. View consensus and postprocessing source otherwise unchanged.',
        shared_input='Original immutable per-frame cached point supports replace the RGB-D ball-query backprojection, to equalize observations with CG.',
        frontend_difference='SAM2.1/SigLIP shared frontend, not upstream CropFormer/OpenCLIP. Default ScanNet core parameters, no SceneNN fitting.'))
    return changed


def run(cache,target,frame_limit=None):
    import torch
    torch.set_num_threads(1);adapt();sys.path.insert(0,str(CPU))
    from graph.construction import process_masks,get_observer_num_thresholds,init_nodes
    from graph.iterative_clustering import iterative_clustering
    from utils.post_process import dbscan_process,filter_point,merge_overlapping_objects
    from omni_semantic_map.shared_observations import observation_table,select_views
    from omni_semantic_map.full_ovo_evaluation import BACKGROUND
    target.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    status=dict(complete=False,cache_manifest_sha256=sha(cache/'manifest.json'),no_gt=True,smoke_frame_limit=frame_limit)
    dump(target/'run.json',status)
    manifest=read(cache/'manifest.json')
    for name,h in manifest['files_sha256'].items():assert sha(cache/name)==h
    geometry=np.load(cache/'geometry.npz');xyz=geometry['xyz']
    records=manifest['records'][:frame_limit] if frame_limit else manifest['records']
    frames=[r['frame_id'] for r in records]
    point_masks=np.zeros((len(xyz),len(frames)),np.uint16)
    point_frames=np.zeros((len(xyz),len(frames)),bool)
    masks={};global_masks=[];boundary=set();observation_lookup={}
    for frame_index,rec in enumerate(records):
        assert sha(cache/'frames'/rec['file'])==rec['sha256']
        with np.load(cache/'frames'/rec['file']) as f:
            seen=set();conflict=set()
            for i in range(rec['masks']):
                ids=set(map(int,f['point_ids'][f['offsets'][i]:f['offsets'][i+1]]))
                if not ids:continue
                local=i+1;key=(rec['frame_id'],local)
                conflict.update(seen.intersection(ids));seen.update(ids)
                masks[f'{key[0]}_{local}']=ids;global_masks.append(key)
                point_masks[list(ids),frame_index]=local
                observation_lookup[key]=rec['observation_offset']+i
            point_frames[list(seen),frame_index]=True
            point_masks[list(conflict),frame_index]=0;boundary.update(conflict)
    print('MC cached graph input',len(xyz),len(frames),len(global_masks),flush=True)
    args=SimpleNamespace(**read(CPU/'configs/scannet.json'),debug=False)
    tic=time.monotonic()
    visible,contained,underseg=process_masks(frames,global_masks,point_masks,boundary,masks,args)
    thresholds=get_observer_num_thresholds(visible)
    nodes=init_nodes(global_masks,visible,contained,underseg,masks)
    if nodes:nodes=iterative_clustering(nodes,thresholds,args.view_consensus_threshold,False)
    graph_s=time.monotonic()-tic;del point_masks
    print('MC clusters',len(nodes),'underseg rejected',len(underseg),flush=True)
    tic=time.monotonic();parts=[];boxes=[];members=[]
    for node in nodes:
        if len(node.mask_list)<2:continue
        cloud,ids=node.get_point_cloud(xyz)
        pclds,point_ids=dbscan_process(cloud,ids)
        current,bounds,mask_lists=filter_point(point_frames,node,pclds,point_ids,masks,frames,args)
        parts.extend(current);boxes.extend(bounds);members.extend(mask_lists)
    parts,members=merge_overlapping_objects(parts,boxes,members,overlapping_ratio=.8)
    features,frame_ids,areas=observation_table(cache);text=np.load(cache/'text_features.npy')
    text=text/np.linalg.norm(text,axis=1,keepdims=True);vocab=read(cache/'vocabulary.json')
    rows=[];export_points=[];export_ids=[];coverage=np.zeros(len(xyz),np.uint16)
    for oid,(ids,entries) in enumerate(zip(parts,members),1):
        oids=sorted(set(observation_lookup[(f,m)] for f,m,_ in entries))
        selected=select_views(oids,frame_ids,areas,top_k=0)
        desc=features[selected].mean(0);scores=text@(desc/max(np.linalg.norm(desc),1e-12));best=int(scores.argmax())
        cls=int(vocab['replica_class_ids'][best]);ids=np.asarray(ids,np.int64)
        rows.append(dict(object_id=oid,observation_ids=oids,unique_frames=len(selected),
            predictions={'mean':dict(name=vocab['names'][best],replica_class_id=cls,score=float(scores[best]),retained=cls not in BACKGROUND)}))
        export_points.append(xyz[ids]);export_ids.append(np.full(len(ids),oid,np.int64));coverage[ids]+=1
    points=np.concatenate(export_points) if export_points else np.empty((0,3),np.float32)
    labels=np.concatenate(export_ids) if export_ids else np.empty(0,np.int64)
    np.savez_compressed(target/'geometry.npz',xyz=points,instance=labels)
    dump(target/'classification.json',dict(objects=rows))
    # Preserve original overlapping instance supports separately from the PQ adapter.
    np.savez_compressed(target/'native_instance_supports.npz',
        point_ids=np.concatenate(parts) if parts else np.empty(0,np.int64),offsets=np.r_[0,np.cumsum([len(p) for p in parts])])
    status.update(complete=True,frames=len(frames),input_masks=len(global_masks),undersegmented_masks=len(underseg),
        clusters=len(nodes),objects=len(rows),retained=sum(r['predictions']['mean']['retained'] for r in rows),
        overlap_points=int((coverage>1).sum()),graph_s=graph_s,postprocess_s=time.monotonic()-tic,total_s=time.monotonic()-begin,
        common_export='Duplicate raw supports retained across instances; same assigned-point 5cm nearest projector, stable object export order resolves exact ties. Native masks separately saved.',
        files_sha256={p.name:sha(p) for p in target.iterdir() if p.is_file() and p.name!='run.json'})
    dump(target/'run.json',status);print('MC COMPLETE',target,len(rows),status['total_s'],flush=True)


def test_core():
    import torch
    torch.set_num_threads(1);adapt();sys.path.insert(0,str(CPU))
    from graph.node import Node
    from graph.iterative_clustering import iterative_clustering
    from utils.post_process import merge_overlapping_objects
    nodes=[Node([(i,1)],torch.tensor([1.,1.,0.]),torch.tensor([1.,1.,0.]),{i},(0,i),None) for i in range(2)]
    nodes.append(Node([(2,1)],torch.tensor([0.,0.,1.]),torch.tensor([0.,0.,1.]),{2},(0,2),None))
    clusters=iterative_clustering(nodes,[1.],.9,False)
    assert sorted(sorted(n.point_ids) for n in clusters)==[[0,1],[2]]
    p,m=merge_overlapping_objects([np.array([0,1]),np.array([0,1,2]),np.array([3])],
        [[[0,0,0],[1,1,1]],[[0,0,0],[1,1,1]],[[5,5,5],[6,6,6]]],[['a'],['b'],['c']],.8)
    assert [x.tolist() for x in p]==[[0,1,2],[3]]
    dump(BASE/'maskclustering_core_acceptance.json',dict(complete=True,synthetic_clustering_and_duplicate_filter_passed=True,
        adaptation_sha256=sha(CPU/'adaptation.json'),source_sha256=sha(Path(__file__)),no_gt=True))
    print('MC CORE TEST PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');p.add_argument('--cache',type=Path);p.add_argument('--output',type=Path);p.add_argument('--smoke-frames',type=int)
    a=p.parse_args()
    if a.test:test_core()
    else:run(a.cache,a.output,a.smoke_frames)
