"""Pinned official MaskClustering core on already-exposed outdoor source tokens.

Shared observation adapter and closed-vocabulary query readout are explicit.
This is not the original CropFormer/OpenCLIP pipeline or an official score.
"""
import argparse,time,sys,resource
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from paper_final_common import ROOT,read,dump,sha
SRC=ROOT/'build/benchmarks/revision_assessment_v1'
OUT=ROOT/'build/benchmarks/reviewer_maskclustering_20261117'
CPU=ROOT/'build/benchmarks/visibility_independent_20261112/adapters/MaskClusteringCPU'
UP=ROOT/'build/benchmarks/visibility_independent_20261112/upstream/MaskClustering'
SCENES=['kitti360_0009','kitti360_0010']
FILES=['graph/node.py','graph/construction.py','graph/iterative_clustering.py','utils/geometry.py','utils/post_process.py','configs/scannet.json']

def freeze():
    assert not OUT.exists();inputs=[SRC/'raw/predictions_complete.json'];sources=[Path(__file__)];adapt={}
    for n in FILES:
        a=(UP/n).read_text();b=a.replace('.cuda()', '.cpu()')
        if n=='graph/construction.py':b=b.replace('from utils.mask_backprojection import frame_backprojection','# Shared cached input replaces the unused upstream backprojection import.')
        assert (CPU/n).read_text()==b,n;sources += [UP/n,CPU/n];adapt[n]=dict(upstream=sha(UP/n),cpu=sha(CPU/n))
    raw=read(SRC/'raw/predictions_complete.json')
    for k in SCENES:
        inputs += [SRC/'full_measured'/k/'geometry.npz',SRC/'observations/prepared'/k/'views.json',SRC/'evaluation'/k/'reference.npz']
        inputs += [Path(v['file']) for v in read(SRC/'observations/prepared'/k/'views.json')]+[Path(v['file']) for v in raw['results'][k]]
    dump(OUT/'protocol.json',dict(date='2026-11-17',created_epoch=time.time(),scenes=SCENES,inputs={str(p):sha(p) for p in inputs},sources={str(p):sha(p) for p in sources},adaptation=adapt,
       label='MaskClustering core / shared source-token adapter',scope='External algorithm-core comparison on two exposed windows. Not native full-pipeline reproduction, an independent assessment, or official leaderboard result.',
       observations='Exactly the fixed per-point highest-score same-class SAM3 query tokens and SegFormer semantics in the current observation cache; complete measured LiDAR sources and causal visibility. One winning mask per point/view; no source cell subsampling.',
       graph='Official view-consensus construction, iterative clustering and ScanNet default thresholds unchanged. CPU device replacements only. Empty/no-common-view graphs export no clusters rather than an undefined percentile.',
       postprocessing='Official DBSCAN(0.1m,min_points4), point filtering, >=2mask observations and .8overlap merge. Native overlapping supports retained separately; stable object-order nearest-map readout resolves source ties as in earlier shared-input adapter.',
       readout='Cluster class chosen from native SAM3 building/car query classes weighted by native score times mask coverage; mean supporting same-class score ranks instances. No new model, GT assignment or threshold tuning; differs from original OpenCLIP and current backbone semantic readout.',
       budget='Exactly2mapexports then6scores; no method/radius/confidence sweep. Full outputs sealed before loading references. Report zero-output outcomes.',
       limitations='Indoor-default postprocessing and point-token adapter can disadvantage this method outdoors; this comparison cannot establish state-of-the-art superiority.'))
    print('EXTERNAL CORE CONTROL FROZEN',flush=True)

def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['inputs'],**p['sources']}.items():assert sha(f)==h,f
    return p

def maps():
    import torch
    torch.set_num_threads(2);verify();assert not (OUT/'maps_complete.json').exists();sys.path.insert(0,str(CPU))
    from graph.construction import process_masks,get_observer_num_thresholds,init_nodes
    from graph.iterative_clustering import iterative_clustering
    from utils.post_process import dbscan_process,filter_point,merge_overlapping_objects
    raw=read(SRC/'raw/predictions_complete.json');results={};files={};args=SimpleNamespace(**read(CPU/'configs/scannet.json'),debug=False)
    for k in SCENES:
        start=time.monotonic();xyz=np.load(SRC/'full_measured'/k/'geometry.npz')['xyz'];views=read(SRC/'observations/prepared'/k/'views.json');frames=[v['frame_id'] for v in views];by={v['frame_id']:v for v in raw['results'][k]};pm=np.zeros((len(xyz),len(frames)),np.uint16);pf=np.zeros_like(pm,dtype=bool);masks={};global_masks=[];metadata={}
        for j,rec in enumerate(views):
            v=np.load(rec['file']);rows=v['query_rows'];ids=v['source_ids'];rd=np.load(by[rec['frame_id']]['file'])
            for q in np.unique(rows[rows>=0]):
                points=ids[rows==q];local=int(q)+1;key=(rec['frame_id'],local);assert np.all(pm[points,j]==0);pm[points,j]=local;pf[points,j]=True;masks[f'{key[0]}_{local}']=set(map(int,points));global_masks.append(key);metadata[key]=(int(rd['semantic_ids'][q]),float(rd['scores'][q]))
        print('MC INPUT',k,len(xyz),len(frames),len(global_masks),flush=True)
        t=time.monotonic();visible,contained,rejected=process_masks(frames,global_masks,pm,set(),masks,args);nodes=init_nodes(global_masks,visible,contained,rejected,masks)
        if nodes and bool((visible@visible.T>0).any()):nodes=iterative_clustering(nodes,get_observer_num_thresholds(visible),args.view_consensus_threshold,False)
        else:nodes=[]
        graph_s=time.monotonic()-t;del pm;t=time.monotonic();parts=[];boxes=[];members=[]
        for node in nodes:
            if len(node.mask_list)<2:continue
            cloud,ids=node.get_point_cloud(xyz);clouds,pointids=dbscan_process(cloud,ids);current,bounds,masklist=filter_point(pf,node,clouds,pointids,masks,frames,args);parts.extend(current);boxes.extend(bounds);members.extend(masklist)
        if parts:parts,members=merge_overlapping_objects(parts,boxes,members,overlapping_ratio=.8)
        post_s=time.monotonic()-t;records=[];coords=[];labels=[]
        for oid,(ids,entries) in enumerate(zip(parts,members),1):
            entries=sorted(set(entries));votes={11:0.,26:0.};scores={11:[],26:[]}
            for frame,mask,coverage in entries:
                cls,sc=metadata[(frame,mask)];votes[cls]+=sc*coverage;scores[cls].append(sc)
            cls=max(sorted(votes),key=lambda c:votes[c]);score=float(np.mean(scores[cls])) if scores[cls] else 0.;records.append(dict(object_id=oid,predictions={'mean':dict(semantic_id=cls,train_id={11:2,26:13}[cls],score=score,name={11:'building',26:'car'}[cls])}));ids=np.asarray(ids,np.int64);coords.append(xyz[ids]);labels.append(np.full(len(ids),oid,np.int64))
        dest=OUT/'maps'/k;dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'geometry.npz',xyz=np.concatenate(coords) if coords else np.empty((0,3)),instance=np.concatenate(labels) if labels else np.empty(0,np.int64));np.savez_compressed(dest/'native_supports.npz',point_ids=np.concatenate(parts) if parts else np.empty(0,np.int64),offsets=np.r_[0,np.cumsum([len(p) for p in parts])]);dump(dest/'classification.json',dict(objects=records));stats=dict(input_sources=len(xyz),frames=len(frames),input_masks=len(global_masks),rejected_masks=len(rejected),clusters=len(nodes),objects=len(records),graph_s=graph_s,postprocess_s=post_s,total_s=time.monotonic()-start,peak_driver_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);dump(dest/'run.json',stats);results[k]=stats
        for f in dest.iterdir():files[str(f)]=sha(f)
        print('MC COMPLETE',k,stats,flush=True)
    dump(OUT/'maps_complete.json',dict(complete=True,files=files,results=results,reference_labels_read=False,protocol_sha256=sha(OUT/'protocol.json')))

def score():
    from evaluate_kitti360_development import transfer,clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    verify();seal=read(OUT/'maps_complete.json');assert seal['complete'];assert not (OUT/'summary.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h
    results={}
    for k in SCENES:
        ref=np.load(SRC/'evaluation'/k/'reference.npz');g=np.load(OUT/'maps'/k/'geometry.npz');info={r['object_id']:r['predictions']['mean'] for r in read(OUT/'maps'/k/'classification.json')['objects']};pred=transfer(g['xyz'],g['instance'],ref['xyz'],.2);dest=OUT/'evaluation'/k;dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'reference_prediction.npz',instance=pred);row={}
        for mode,gt,uniform in [('primary',ref['train_instance'],False),('raw_targets',ref['raw_target_instance'],False),('uniform_scores',ref['train_instance'],True)]:
            cls={o:dict(v,score=1.) for o,v in info.items()} if uniform else info;z=evaluate(gt,ref['confidence'],pred,cls,ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py');row[mode]=dict(metrics=clean_json(z['averages']),matched_gt_ids=matched_ids(z['matches']))
        results[k]=row;print('MC SCORED',k,{m:v['metrics']['allAp'] for m,v in row.items()},flush=True)
    means={m:float(np.mean([x[m]['metrics']['allAp'] for x in results.values()])) for m in ['primary','raw_targets','uniform_scores']};dump(OUT/'summary.json',dict(complete=True,results=results,mean_ap=means,new_maps=2,new_scores=6,native_full_pipeline=False,independent_scenes=False,protocol_sha256=sha(OUT/'protocol.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','maps','score']);globals()[p.parse_args().action]()
