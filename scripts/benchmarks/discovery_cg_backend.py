"""Existing sparse CG/history2 kernels, shared by every SAM3 query policy."""
from copy import deepcopy
from pathlib import Path
import sys,time
import numpy as np
from paper_final_common import ROOT,FROZEN,read,dump


def build_map(folder,geometry,records,frame_loader,features,frames,areas,classes,scores,text,vocab):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    from conceptgraph.slam import mapping,utils
    from conceptgraph.slam.slam_classes import MapObjectList
    from omegaconf import OmegaConf
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend,FusionObservation
    from omni_semantic_map.history_graph import build_history_graph
    from omni_semantic_map.fusion_diagnostics import guarded_merge,membership_audit
    from omni_semantic_map.full_ovo_evaluation import cg_geometry
    from omni_semantic_map.shared_observations import select_views
    cfg=OmegaConf.create(read(FROZEN/'config/cg_rgbd.json'));cfg.dbscan_remove_noise=False
    engine=ConceptGraphsBackend(cfg,seed=0);xyz=geometry['xyz'];events=[]
    old_merge=mapping.merge_detections_to_objects;old_overlap=mapping.compute_overlap_matrix_2set;old_final=utils.compute_overlap_matrix
    def scale(config):
        config=deepcopy(config);config.downsample_voxel_size=.2;return config
    mapping.compute_overlap_matrix_2set=lambda config,*args:old_overlap(scale(config),*args)
    utils.compute_overlap_matrix=lambda config,*args:old_final(scale(config),*args)
    def capture(config,detections,objects,similarities):
        for i,det in enumerate(detections):
            values=similarities[i].cpu().numpy();eligible=np.flatnonzero(np.isfinite(values))
            if len(eligible)<2:continue
            query=int(det['observation_ids'][0]);native=int(eligible[np.argmax(values[eligible])]);candidates=[]
            for j in eligible:
                members=list(map(int,objects[j]['observation_ids']));assert all(frames[k]<frames[query] for k in members)
                candidates.append(dict(members=members,is_native=int(j)==native,native_score=float(values[j])))
            events.append(dict(query=query,frame=int(frames[query]),candidates=candidates))
        return old_merge(config,detections,objects,similarities)
    mapping.merge_detections_to_objects=capture;started=time.monotonic()
    try:
        for rec in records:
            data=frame_loader(rec);points=data['point_ids'];offset=data['offsets'];observations=[]
            for j in range(rec['masks']):
                oid=rec['observation_offset']+j;ids=points[offset[j]:offset[j+1]]
                # RGB is not part of native association. Preserve available real colors.
                colors=geometry['colors'][ids].astype(float)/255 if 'colors' in geometry else np.zeros((len(ids),3))
                semantic=int(classes[oid]);text_index=vocab['semantic_ids'].index(semantic)
                observations.append(FusionObservation(oid,rec['frame_id'],semantic,xyz[ids],colors,features[oid],text[text_index]))
            engine.integrate(observations,np.eye(4))
        nodes=[];owner={};node_frames={}
        for obj in engine._objects:
            members=list(map(int,obj['observation_ids']));alias=min(members)+1;node_frames[alias]=sorted(set(map(int,frames[members])))
            nodes.append(dict(alias=alias,members=members,frames=node_frames[alias]))
            for oid in members:assert oid not in owner;owner[oid]=alias
        proposals=[]
        for event in events:
            a=owner[event['query']]
            assert {owner[i] for c in event['candidates'] if c['is_native'] for i in c['members']}=={a}
            for c in event['candidates']:
                mapped={owner[i] for i in c['members']};assert len(mapped)==1;b=mapped.pop()
                if a!=b:proposals.append(dict(a=a,b=b,query=event['query'],frame=event['frame'],dino_pass=False))
        graph=build_history_graph(node_frames,proposals,min_votes=2,use_dino=False)
        originals={min(o['observation_ids'])+1:o for o in engine._objects};objects=MapObjectList()
        for group in graph['groups']:
            obj=deepcopy(originals[group[0]])
            for key in group[1:]:obj=utils.merge_obj2_into_obj1(cfg,obj,deepcopy(originals[key]),run_dbscan=False)
            objects.append(obj)
        before=membership_audit(objects);assert before['repeated_memberships']==0
        assert before['unique_observations']==sum(len(o['observation_ids']) for o in engine._objects)
        objects=guarded_merge(cfg,utils.filter_objects(cfg,utils.denoise_objects(cfg,deepcopy(objects))))
        after=membership_audit(objects);assert after['repeated_memberships']==0
        points,labels,identity=cg_geometry([dict(object_id=min(o['observation_ids'])+1,points=np.asarray(o['pcd'].points)) for o in objects])
        rows=[]
        for item in identity:
            members=sorted(set(map(int,objects[item['object_index']]['observation_ids'])))
            chosen=np.asarray(select_views(members,frames,areas,top_k=0),int)
            # Detector semantics are common to all arms, not a new learned readout.
            per_class={c:float(np.sum(scores[chosen]*(classes[chosen]==c))) for c in [11,26]}
            semantic=max(per_class,key=lambda c:(per_class[c],-c));confidence=per_class[semantic]/len(chosen)
            rows.append(dict(object_id=item['evaluation_id'],observation_ids=members,selected_observation_ids=chosen.tolist(),
                predictions=dict(mean=dict(name='car' if semantic==26 else 'building',semantic_id=semantic,
                    train_id=13 if semantic==26 else 2,score=confidence,retained=True))))
    finally:
        mapping.merge_detections_to_objects=old_merge;mapping.compute_overlap_matrix_2set=old_overlap;utils.compute_overlap_matrix=old_final
    folder.mkdir(parents=True,exist_ok=False);np.savez_compressed(folder/'geometry.npz',xyz=points,instance=labels)
    dump(folder/'classification.json',dict(objects=rows,taxonomy='KITTI target prompts',readout='Distinct-frame native SAM3 confidence; identical for all policies.'))
    for name,value in [('nodes',nodes),('proposals',proposals),('graph',graph),('native_events',events)]:dump(folder/(name+'.json'),value)
    dump(folder/'run.json',dict(complete=True,frames=len(records),objects=len(rows),map_points=len(points),
        rejected_observations=engine.rejected,membership_before=before,membership_after=after,
        backend_seconds=time.monotonic()-started,effective_config=OmegaConf.to_container(cfg,resolve=True),
        overlap_radius_m=.2,history_votes=2,seed=0,geometry='Native CG denoise/filter/guarded final merge, not raw support relabeling.'))
    return read(folder/'run.json')
