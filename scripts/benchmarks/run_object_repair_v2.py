"""One preregistered development pilot: four scenes, three new maps each."""
import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import resource
import sys
import time
import numpy as np
from paper_final_common import ROOT,OLD,read,sha,dump,CGPY
from omni_semantic_map.reversible_support_graph import SupportGraphConfig,ReversibleSupportGraph,atomize,frame_atom_labels

OUT=ROOT/'build/benchmarks/object_repair_v2_20261112'
HISTORY=OLD/'public_history_v1'
KEYS=['replica_office_0','replica_office_1','scenenn_014','scenenn_015']
VARIANTS=['constraint_control','repair','no_revoke']


def freeze():
    assert not (OUT/'protocol.json').exists()
    prior=read(HISTORY/'protocol.json');scenes=[next(s for s in prior['scenes'] if s['key']==k) for k in KEYS]
    sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'scripts/benchmarks/evaluate_object_repair_v2.py',
        ROOT/'omni_semantic_map/omni_semantic_map/reversible_support_graph.py',ROOT/'omni_semantic_map/test/test_reversible_support_graph.py',ROOT/'scripts/benchmarks/validate_object_repair_v2_adapters.py']}
    sources.update({str(Path(p).relative_to(ROOT)):h for p,h in prior['source_sha256'].items()})
    files={}
    for s in scenes:
        for name in ['baseline','native_bridge_2']:
            for file in ['geometry.npz','classification.json']:
                p=HISTORY/'runs'/s['key']/name/file;files[str(p)]=sha(p)
        for file in ['metrics.json','baseline.npz','native_bridge_2.npz']:
            p=HISTORY/'evaluation'/s['key']/file;files[str(p)]=sha(p)
    protocol=dict(complete=True,created_epoch=time.time(),scenes=scenes,variants=VARIANTS,maximum_new_maps=12,
        source_sha256=sources,prior_artifact_sha256=files,prior_protocol_sha256=sha(HISTORY/'protocol.json'),
        config=asdict(SupportGraphConfig()),semantic_readout='one max effective-area observation per distinct frame, unweighted mean; inherited foreground class set',
        retained_views=3,descriptor_networks_changed=False,new_training=False,
        positive='Adjacent fixed 5cm support cells assigned to the same mask in a frame.',
        negative='Adjacent cells with >=70% dominant, exclusive measured support from different masks in the same frame; unknown cells abstain.',
        repeated_exclusion='At least 2 negative frames and negative count >=2*positive count. Recomputed after every frame; no class labels.',
        partition='One deterministic signed greedy pass on affected old components and witnessed cells; 100000-atom hard repair bound, report failure if exceeded.',
        geometry='Same raw measured cache; repair variants partition supports, then inherited CG voxel downsample and DBSCAN cleaning, no unconstrained final merge.',
        ownership='Each (observation ID, measured point ID) token has one final owner or is unassigned; a single mask may split, whole-mask IDs are NOT duplicated geometry evidence.',
        control='Native CG with hard component coframe veto during online association, history2 merge and final guarded merge.',
        ablation='repair and no_revoke have identical observations, graph evidence, export and readout. Only no_revoke locks previous partitions against splitting.',
        givens='Original per-scene cached sampling, poses, sensors, configurations and evaluation domain; no new masks, descriptors or generated points.',
        evaluation='All 12 predictions (or explicit budget/runtime failures) sealed before evaluation; old shared valid 5cm domain, assigned-point 1NN<=5cm and strict IoU>.5.',
        gate=dict(dataset_mean_delta_pq100_min=2.,nondecreasing_scenes_min=3,total_tp_not_lower_each_dataset=True,
            positive_delta_vs_constraint_control_each_dataset=True,revocation_ablation_positive_each_dataset=True,
            p95_backend_ratio_max=2.,new_correct_matches_required=True),
        timing_limitation='Historical Replica CG/v1 exports have no per-frame timing. Do not fabricate historical P95 from total duration; report unavailable and withhold full cost-gate pass. No extra baseline maps outside the 12-map budget.',
        timing_scope='Report frame updates and offline final export separately; fixed atom preallocation and cache loading separately; no full-system FPS claim.',
        development_only=True,no_independent_generalization_claim=True,no_threshold_sweep=True,
        stop='One implementation/configuration and 12 maps maximum; preserve failures and negative results. No automatic expansion after gate failure.')
    dump(OUT/'protocol.json',protocol)
    for name in sources:
        p=ROOT/name;dest=OUT/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    print('FROZEN',len(sources),'sources; 4 scenes x 3 variants',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for name,want in p['source_sha256'].items():assert sha(ROOT/name)==want,name
    return p


def load(entry):
    from omni_semantic_map.shared_observations import observation_table
    cache=Path(entry['cache']);manifest=read(cache/'manifest.json');assert manifest['complete'] and manifest['no_association']
    for name,h in manifest['files_sha256'].items():assert sha(cache/name)==h,name
    for rec in manifest['records']:assert sha(cache/'frames'/rec['file'])==rec['sha256']
    features,frames,areas=observation_table(cache);geo=np.load(cache/'geometry.npz')
    text=np.load(cache/'text_features.npy');vocab=read(cache/'vocabulary.json');poses=np.load(cache/'poses.npy')
    return cache,manifest,geo,features,frames,areas,text,vocab,poses


def safe_final_merge(cfg,objects,frames):
    from conceptgraph.slam import utils
    from conceptgraph.slam.slam_classes import MapObjectList
    import torch.nn.functional as F
    if cfg.merge_overlap_thresh<=0 or not objects:return objects
    matrix=utils.compute_overlap_matrix(cfg,objects);x,y=matrix.nonzero();ratio=matrix[x,y];kept=np.ones(len(objects),bool)
    for k in np.argsort(ratio)[::-1]:
        i,j=x[k],y[k]
        if ratio[k]<=cfg.merge_overlap_thresh:break
        if i==j or not kept[i] or not kept[j]:continue
        if set(frames[objects[i]['observation_ids']])&set(frames[objects[j]['observation_ids']]):continue
        visual=F.cosine_similarity(utils.to_tensor(objects[i]['clip_ft']),utils.to_tensor(objects[j]['clip_ft']),dim=0)
        text=F.cosine_similarity(utils.to_tensor(objects[i]['text_ft']),utils.to_tensor(objects[j]['text_ft']),dim=0)
        if visual>cfg.merge_visual_sim_thresh and text>cfg.merge_text_sim_thresh:
            objects[j]=utils.merge_obj2_into_obj1(cfg,objects[j],objects[i],run_dbscan=True);kept[i]=False
    result=MapObjectList([o for o,k in zip(objects,kept) if k])
    for o in result:assert len(frames[o['observation_ids']])==len(set(frames[o['observation_ids']]))
    return result


def control(entry,target,data):
    import torch
    from conceptgraph.slam import mapping
    from omegaconf import OmegaConf
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend,FusionObservation
    from omni_semantic_map.history_graph import build_history_graph
    from run_livo_history_graph import finalize
    import omni_semantic_map.fusion_diagnostics as fd
    cache,m,g,f,frames,areas,text,vocab,poses=data;cfg=OmegaConf.create(read(entry['config']))
    engine=ConceptGraphsBackend(cfg,seed=0);events=[];times=[];original=mapping.merge_detections_to_objects
    def guarded(config,detections,objects,scores):
        for i,det in enumerate(detections):
            values=scores[i].clone();oid=int(det['observation_ids'][0]);frame=frames[oid]
            for j in range(len(values)):
                if frame in frames[objects[j]['observation_ids']]:values[j]=float('-inf')
            eligible=np.flatnonzero(np.isfinite(values.cpu().numpy()))
            if not len(eligible):objects.append(det);continue
            native=int(values.argmax())
            if len(eligible)>1:
                events.append(dict(query=oid,frame=int(frame),candidates=[list(map(int,objects[j]['observation_ids'])) for j in eligible]))
            objects[native]=mapping.merge_obj2_into_obj1(config,objects[native],det,run_dbscan=False)
        return objects
    mapping.merge_detections_to_objects=guarded
    try:
        for index,rec in enumerate(m['records']):
            tic=time.monotonic();z=np.load(cache/'frames'/rec['file']);obs=[]
            for j in range(rec['masks']):
                oid=rec['observation_offset']+j;ids=z['point_ids'][z['offsets'][j]:z['offsets'][j+1]]
                obs.append(FusionObservation(oid,rec['frame_id'],0,g['xyz'][ids],g['colors'][ids].astype(float)/255,f[oid],text[int((text@f[oid]).argmax())]))
            engine.integrate(obs,np.eye(4));times.append(time.monotonic()-tic)
            if index%25==0:print('CONTROL',entry['key'],index,flush=True)
    finally:mapping.merge_detections_to_objects=original
    nodes={min(o['observation_ids'])+1:o for o in engine._objects}
    owner={int(i):alias for alias,o in nodes.items() for i in o['observation_ids']}
    nodeframes={alias:sorted(set(map(int,frames[o['observation_ids']]))) for alias,o in nodes.items()}
    proposals=[]
    for e in events:
        for c in e['candidates']:
            for b in {owner[i] for i in c}:
                if owner[e['query']]!=b:proposals.append(dict(a=owner[e['query']],b=b,frame=e['frame'],dino_pass=False))
    graph=build_history_graph(nodeframes,proposals,min_votes=2)
    byframe={r['frame_id']:poses[i] for i,r in enumerate(m['records'])};norm=text/np.linalg.norm(text,axis=1,keepdims=True)
    tic=time.monotonic();old=fd.guarded_merge;fd.guarded_merge=lambda cfg,objects:safe_final_merge(cfg,objects,frames)
    try:result=finalize(engine,graph['groups'],target/'map',(f,frames,areas,norm,vocab,byframe))
    finally:fd.guarded_merge=old
    for row in result['objects']:assert len(row['observation_ids'])==row['unique_frames']
    dump(target/'graph.json',graph)
    return dict(frame_times_s=times,finalization_s=time.monotonic()-tic,confirmed_constraint_violations=0,geometry_path=str(target/'map/geometry.npz'),classification_path=str(target/'map/classification.json'))


def repair(entry,target,data,allow_revoke,config):
    from omegaconf import OmegaConf
    from conceptgraph.slam import utils
    import open3d as o3d
    from omni_semantic_map.full_ovo_evaluation import BACKGROUND,cg_geometry
    cache,m,g,features,frames,areas,text,vocab,poses=data;xyz=g['xyz'];tic=time.monotonic()
    keys,point_atom,edges=atomize(xyz,config.cell_size);preparation=time.monotonic()-tic
    engine=ReversibleSupportGraph(len(keys),edges,config,allow_revoke=allow_revoke)
    seen_points=np.zeros(len(xyz),bool);times=[];ledger=[]
    for index,rec in enumerate(m['records']):
        tic=time.monotonic();z=np.load(cache/'frames'/rec['file'])
        ids=z['point_ids'].astype(np.int64);oids=np.repeat(np.arange(rec['observation_offset'],rec['observation_offset']+rec['masks']),np.diff(z['offsets']))
        labels=frame_atom_labels(point_atom,ids,oids,len(keys),config.dominance)
        engine.update(rec['frame_id'],labels);seen_points[ids]=True
        # Each retained raw observation/support token is immutable and unique.
        for j in range(rec['masks']):ledger.append((rec['observation_offset']+j,ids[z['offsets'][j]:z['offsets'][j+1]].copy()))
        times.append(time.monotonic()-tic)
        if index%25==0:print('REPAIR' if allow_revoke else 'NO_REVOKE',entry['key'],index,'atoms',engine.events[-1]['repair_atoms'],flush=True)
    tic=time.monotonic();violations=engine.check_export();point_labels=np.where(seen_points,engine.labels[point_atom],0)
    evidence=defaultdict(list);assigned_tokens=0;unassigned_tokens=0
    for oid,ids in ledger:
        values,counts=np.unique(point_labels[ids],return_counts=True)
        for label,count in zip(values,counts):
            if label>0:
                evidence[int(label)].append((oid,int(count),int(len(ids))));assigned_tokens+=int(count)
            else:unassigned_tokens+=int(count)
    assert assigned_tokens+unassigned_tokens==sum(len(ids) for _,ids in ledger)
    cfg=OmegaConf.create(read(entry['config']));norm=text/np.linalg.norm(text,axis=1,keepdims=True)
    objects=[];rows=[];rejected=[]
    # Group all raw points once, avoiding per-object scans of the full cloud.
    order=np.argsort(point_labels,kind='stable');cuts=np.flatnonzero(np.diff(point_labels[order]))+1
    for selection in np.split(order,cuts):
        label=int(point_labels[selection[0]])
        if label==0:continue
        contributions=evidence[label];byframe={}
        for oid,count,total in contributions:
            effective=float(areas[oid])*count/total;frame=int(frames[oid]);record=(effective,-oid,oid)
            if frame not in byframe or record>byframe[frame]:byframe[frame]=record
        selected=[row[2] for _,row in sorted(byframe.items())]
        if len(selected)<cfg.obj_min_detections:
            rejected.append(dict(atom_object=label,reason='fewer_than_three_distinct_views',points=len(selection)));continue
        cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(xyz[selection]);cloud.colors=o3d.utility.Vector3dVector(g['colors'][selection].astype(float)/255)
        cloud=utils.process_pcd(cloud,cfg);points=np.asarray(cloud.points)
        if len(points)<cfg.obj_min_points or not len(points):
            rejected.append(dict(atom_object=label,reason='native_geometry_cleaning',points=len(selection)));continue
        desc=features[selected].mean(0);scores=norm@(desc/max(np.linalg.norm(desc),1e-12));best=int(np.argmax(scores));cls=int(vocab['replica_class_ids'][best])
        oid=len(rows)+1
        rows.append(dict(object_id=oid,atom_object=label,observation_ids=sorted(oid for oid,_,_ in contributions),unique_frames=len(selected),
            selected_observation_ids=selected,support_contributions=[dict(observation_id=o,assigned_points=n,original_support_points=t) for o,n,t in contributions],
            predictions={'mean':dict(name=vocab['names'][best],replica_class_id=cls,score=float(scores[best]),retained=cls not in BACKGROUND)}))
        objects.append(dict(object_id=oid,points=points.copy()))
    geometry,labels,_=cg_geometry(objects);folder=target/'map';folder.mkdir()
    np.savez_compressed(folder/'geometry.npz',xyz=geometry,instance=labels)
    dump(folder/'classification.json',dict(objects=rows));dump(target/'events.json',engine.events)
    np.savez_compressed(target/'support_graph.npz',keys=keys,edges=edges,positive=engine.positive,negative=engine.negative,labels=engine.labels,seen=engine.seen)
    dump(target/'ownership.json',dict(total_tokens=assigned_tokens+unassigned_tokens,assigned_tokens=assigned_tokens,unassigned_tokens=unassigned_tokens,
        point_geometry_owner_unique=True,raw_seen_points=int(seen_points.sum()),assigned_points_before_filter=int((point_labels>0).sum()),
        parent_masks_may_have_multiple_regions=True,partition_unit='(observation, measured support point), not whole masks',rejected=rejected))
    return dict(frame_times_s=times,preparation_s=preparation,finalization_s=time.monotonic()-tic,
        confirmed_constraint_violations=violations,split_events=sum(bool(e['split_lineage']) for e in engine.events),
        max_repair_atoms=max(e['repair_atoms'] for e in engine.events),atoms=len(keys),edges=len(edges),
        geometry_path=str(folder/'geometry.npz'),classification_path=str(folder/'classification.json'))


def run(key,variant):
    protocol=verify();assert variant in VARIANTS;entry=next(s for s in protocol['scenes'] if s['key']==key)
    target=OUT/'runs'/key/variant;target.mkdir(parents=True,exist_ok=False)
    status=dict(complete=False,key=key,variant=variant,no_gt=True,protocol_sha256=sha(OUT/'protocol.json'))
    dump(target/'run.json',status);begin=time.monotonic()
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    try:
        tic=time.monotonic();data=load(entry);loaded=time.monotonic()-tic
        if variant=='constraint_control':result=control(entry,target,data)
        else:result=repair(entry,target,data,variant=='repair',SupportGraphConfig(**protocol['config']))
        status.update(result,complete=True,cache_load_verify_s=loaded,wall_s=time.monotonic()-begin,
            frame_median_s=float(np.median(result['frame_times_s'])),frame_p95_s=float(np.quantile(result['frame_times_s'],.95)),
            cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            artifacts={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file() and p.name!='run.json'})
    except Exception as error:
        import traceback
        status.update(failed=True,error=repr(error),traceback=traceback.format_exc(),wall_s=time.monotonic()-begin);dump(target/'run.json',status);raise
    dump(target/'run.json',status);print('MAP COMPLETE',key,variant,status['wall_s'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--scene');p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    if a.freeze:freeze()
    else:run(a.scene,a.variant)
