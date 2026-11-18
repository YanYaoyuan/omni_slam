"""Frozen CG replay adapter: association-only feature intervention.

Copied from paper_final_mapping.run_cg; only descriptor input and emitted
variant count change. Original semantic table and final processing preserved.
"""
from paper_final_mapping import *

def run_cg(cache,target,association_features=None):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    from conceptgraph.slam import mapping
    from omegaconf import OmegaConf
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend,FusionObservation
    import omni_semantic_map.fusion_diagnostics as fd
    from run_livo_history_graph import finalize
    target.mkdir(parents=True,exist_ok=False)
    m,g,features,frames,areas,rawtext,text,vocab,byframe=cache_data(cache)
    xyz,colors=g['xyz'],g['colors'];table=features,frames,areas,text,vocab,byframe
    association=features if association_features is None else association_features
    assert len(association)==len(features) and np.isfinite(association).all()
    assert (np.linalg.norm(association,axis=1)>0).all()
    cfg=OmegaConf.create(read(FROZEN/'config/cg_rgbd.json'))
    engine=ConceptGraphsBackend(cfg,seed=0);events=[];frame_times=[]
    status=dict(complete=False,no_gt=True,cache_manifest_sha256=sha(cache/'manifest.json'),
                config_sha256=sha(FROZEN/'config/cg_rgbd.json'),cold_start_s=time.monotonic()-START)
    original=mapping.merge_detections_to_objects
    def capture(config,detections,objects,scores):
        for i,det in enumerate(detections):
            values=scores[i].cpu().numpy();eligible=np.flatnonzero(np.isfinite(values))
            if len(eligible)<2:continue
            query=int(det['observation_ids'][0]);native=int(eligible[np.argmax(values[eligible])]);candidates=[]
            for j in eligible:
                members=list(map(int,objects[j]['observation_ids']));assert all(frames[n]<frames[query] for n in members)
                candidates.append(dict(members=members,native_score=float(values[j]),is_native=int(j)==native))
            events.append(dict(query=query,frame=int(frames[query]),candidates=candidates))
        return original(config,detections,objects,scores)
    mapping.merge_detections_to_objects=capture;began=time.monotonic()
    try:
        for index,rec in enumerate(m['records']):
            tic=time.monotonic();f=np.load(cache/'frames'/rec['file']);observations=[]
            for i in range(rec['masks']):
                oid=rec['observation_offset']+i;support=f['point_ids'][f['offsets'][i]:f['offsets'][i+1]]
                observations.append(FusionObservation(oid,rec['frame_id'],0,xyz[support],colors[support].astype(np.float64)/255,
                    association[oid],rawtext[int((rawtext@features[oid]).argmax())]))
            engine.integrate(observations,np.eye(4));frame_times.append(time.monotonic()-tic)
            if index%25==0:
                dump(target/'run.json',dict(status,frame=index,nodes=len(engine._objects)))
                print('CG',target.parent.name,index,len(engine._objects),flush=True)
    finally:mapping.merge_detections_to_objects=original
    status.update(causal_mapping_s=time.monotonic()-began,frame_times_s=frame_times,
        frame_median_s=float(np.median(frame_times)),frame_p95_s=float(np.quantile(frame_times,.95)))
    nodes=[];owner={};node_frames={}
    for obj in engine._objects:
        ids=list(map(int,obj['observation_ids']));alias=min(ids)+1
        node_frames[alias]=sorted(set(map(int,frames[ids])));nodes.append(dict(alias=alias,members=ids,frames=node_frames[alias]))
        for oid in ids:assert oid not in owner;owner[oid]=alias
    proposals=[]
    for event in events:
        a=owner[event['query']]
        assert {owner[i] for c in event['candidates'] if c['is_native'] for i in c['members']}=={a}
        for c in event['candidates']:
            mapped={owner[i] for i in c['members']};assert len(mapped)==1;b=mapped.pop()
            if a!=b:proposals.append(dict(a=a,b=b,query=event['query'],frame=event['frame'],dino_pass=False))
    dump(target/'nodes.json',nodes);dump(target/'proposals.json',proposals);dump(target/'events.json',events)
    for name,votes in [('history2',2)]:
        tic=time.monotonic()
        graph=(dict(groups=[[n['alias']] for n in nodes],decisions=[]) if votes==0 else
               graph_without_veto(node_frames,proposals,votes) if name=='history2_no_veto' else
               build_history_graph(node_frames,proposals,min_votes=votes,use_dino=False))
        final_merges=[];native_guard=fd.guarded_merge
        def logged_guard(config,objects):
            before=[sorted(map(int,o['observation_ids'])) for o in objects]
            result=native_guard(config,objects)
            for obj in result:
                members=set(map(int,obj['observation_ids']));parts=[x for x in before if set(x)<=members]
                if len(parts)>1:
                    pairs=[]
                    for i,a in enumerate(parts):
                        for b in parts[i+1:]:
                            common=sorted(set(map(int,frames[a]))&set(map(int,frames[b])))
                            if common:pairs.append(dict(a=a,b=b,frames=common))
                    final_merges.append(dict(members=sorted(members),input_parts=parts,coframe_conflicts=pairs))
            return result
        fd.guarded_merge=logged_guard
        try:result=finalize(engine,graph['groups'],target/name,table)
        finally:fd.guarded_merge=native_guard
        dump(target/name/'graph.json',graph);dump(target/name/'final_native_merges.json',final_merges)
        dump(target/name/'run.json',dict(complete=True,no_gt=True,geometry=name,
            offline_finalization_s=time.monotonic()-tic,causal_mapping_s=status['causal_mapping_s'],
            finalization_scope='graph + native final processing + inherited mean/area_density audit readouts',
            membership_before=result['before_filter'],membership_after=result['after_final'],**resource_usage()))
        print('FINALIZED',target.parent.name,name,len(result['objects']),flush=True)
    status.update(complete=True,variants=['history2'],
                  rejected_observations=engine.rejected,**resource_usage())
    dump(target/'run.json',status)

