"""Fresh full-cache garage replay of frozen v1 plus archived LiDAR adaptation.

Reuses LIVO/SAM2/SigLIP observations; does not fit models or rerun odometry.
One causal pass, native reference and two-frame history finalization, frozen
consensus category readout. All outputs go to a new directory.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import sys
import time
import numpy as np
from memory_reseg_common import ROOT, read, dump, sha

OLD = ROOT/'build/benchmarks/evidence_retention_20261111'
CACHE = OLD/'livo_full_strong_v2/cache_supportcc'
OUT = ROOT/'build/experiments/garage_current_20261112'


def main():
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    from conceptgraph.slam import mapping, utils
    from omegaconf import OmegaConf
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend, FusionObservation
    from omni_semantic_map.history_graph import build_history_graph
    from omni_semantic_map.consensus_category import classify
    from paper_final_mapping import cache_data
    from run_livo_history_graph import finalize
    import omni_semantic_map.fusion_diagnostics as fd

    assert not OUT.exists(), 'Do not overwrite a previous run'
    OUT.mkdir(parents=True)
    diagnostics = read(OLD/'scale_ablation_v1/cache_supportcc/both/diagnostics.json')
    config = diagnostics['effective_config']
    assert not config['dbscan_remove_noise']
    assert config['downsample_voxel_size'] == .025
    radius = diagnostics['overlap_radius_m']
    assert radius == .2
    files = [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/replay_fusion.py',
        ROOT/'omni_semantic_map/omni_semantic_map/history_graph.py',
        ROOT/'omni_semantic_map/omni_semantic_map/consensus_category.py',
        ROOT/'omni_semantic_map/omni_semantic_map/category_evidence.py',
        ROOT/'omni_semantic_map/omni_semantic_map/fusion_diagnostics.py',
        ROOT/'scripts/benchmarks/run_livo_history_graph.py',ROOT/'docs/paper/frozen_v1/manifest.json']
    dump(OUT/'protocol.json', dict(scene='rosbag2245',method='omni_slam-paper-v1 + archived LiDAR scale adaptation',
        cache=str(CACHE),cache_sha256=sha(CACHE/'manifest.json'),config=config,overlap_m=radius,
        source_sha256={str(p):sha(p) for p in files},seed=0,
        category='consensus_sqrt_area; foreground retention fixed by original unique-frame mean',
        history='two distinct query frames; component co-frame veto; offline finalization',
        replay='All 330 cached semantic frames; LIVO poses and SAM2/SigLIP frontend reused, not recomputed',
        no_training=True,no_gt=True,no_attention_resegmentation=True,no_new_parameters=True))
    start=time.monotonic()
    m,g,features,frames,areas,rawtext,text,vocab,byframe=cache_data(CACHE)
    assert len(m['records'])==330
    table=features,frames,areas,text,vocab,byframe
    cfg=OmegaConf.create(config)
    engine=ConceptGraphsBackend(cfg,seed=0)
    events=[];frame_times=[]
    native_merge=mapping.merge_detections_to_objects
    original_overlap=mapping.compute_overlap_matrix_2set
    original_final_overlap=utils.compute_overlap_matrix

    def scaled(cfg):
        c=deepcopy(cfg);c.downsample_voxel_size=radius
        return c

    def capture(config,detections,objects,scores):
        for i,det in enumerate(detections):
            values=scores[i].cpu().numpy();eligible=np.flatnonzero(np.isfinite(values))
            if len(eligible)<2:continue
            query=int(det['observation_ids'][0]);native=int(eligible[np.argmax(values[eligible])]);candidates=[]
            for j in eligible:
                members=list(map(int,objects[j]['observation_ids']))
                assert all(frames[n]<frames[query] for n in members)
                candidates.append(dict(members=members,native_score=float(values[j]),is_native=int(j)==native))
            events.append(dict(query=query,frame=int(frames[query]),candidates=candidates))
        return native_merge(config,detections,objects,scores)

    mapping.merge_detections_to_objects=capture
    mapping.compute_overlap_matrix_2set=lambda c,*args:original_overlap(scaled(c),*args)
    utils.compute_overlap_matrix=lambda c,*args:original_final_overlap(scaled(c),*args)
    try:
        for i,rec in enumerate(m['records']):
            began=time.monotonic();observations=[]
            with np.load(CACHE/'frames'/rec['file']) as z:
                for j in range(rec['masks']):
                    oid=rec['observation_offset']+j;ids=z['point_ids'][z['offsets'][j]:z['offsets'][j+1]]
                    observations.append(FusionObservation(oid,rec['frame_id'],0,g['xyz'][ids],
                        g['colors'][ids].astype(float)/255,features[oid],rawtext[int((rawtext@features[oid]).argmax())]))
            engine.integrate(observations,np.eye(4));frame_times.append(time.monotonic()-began)
            if i%25==0:
                dump(OUT/'progress.json',dict(frames=i+1,total=len(m['records']),objects=len(engine._objects),elapsed_s=time.monotonic()-start))
                print('GARAGE REPLAY',i+1,'/',len(m['records']),flush=True)
        causal_s=sum(frame_times)
        nodes=[];owner={};node_frames={}
        for obj in engine._objects:
            ids=list(map(int,obj['observation_ids']));alias=min(ids)+1
            node_frames[alias]=sorted(set(map(int,frames[ids])))
            nodes.append(dict(alias=alias,members=ids,frames=node_frames[alias]))
            for oid in ids:
                assert oid not in owner
                owner[oid]=alias
        proposals=[]
        for event in events:
            a=owner[event['query']]
            assert {owner[i] for c in event['candidates'] if c['is_native'] for i in c['members']}=={a}
            for c in event['candidates']:
                targets={owner[i] for i in c['members']};assert len(targets)==1
                b=targets.pop()
                if a!=b:proposals.append(dict(a=a,b=b,query=event['query'],frame=event['frame'],dino_pass=False))
        dump(OUT/'nodes.json',nodes);dump(OUT/'events.json',events);dump(OUT/'proposals.json',proposals)
        statuses={}
        for name,votes,archived in [('native_reference',0,'baseline'),('current',2,'native_bridge_2')]:
            tic=time.monotonic()
            graph=build_history_graph(node_frames,proposals,2,False) if votes else dict(groups=[[n['alias']] for n in nodes],decisions=[])
            merges=[];original_guard=fd.guarded_merge
            def logged_guard(config,objects):
                before=[sorted(map(int,o['observation_ids'])) for o in objects]
                result=original_guard(config,objects)
                for obj in result:
                    members=set(map(int,obj['observation_ids']));parts=[x for x in before if set(x)<=members]
                    if len(parts)>1:merges.append(dict(members=sorted(members),input_parts=parts))
                return result
            fd.guarded_merge=logged_guard
            try:result=finalize(engine,graph['groups'],OUT/name,table)
            finally:fd.guarded_merge=original_guard
            dump(OUT/name/'graph.json',graph);dump(OUT/name/'final_merges.json',merges)
            actual=np.load(OUT/name/'geometry.npz');prior=np.load(OLD/'history_graph_v1'/archived/'geometry.npz')
            exact=all(np.array_equal(actual[k],prior[k]) for k in actual.files)
            assert exact,'Unexpected geometry drift against archived '+archived
            oldrows=read(OLD/'history_graph_v1'/archived/'classification.json')['objects']
            assert result['objects']==oldrows,'Unexpected membership/readout drift'
            rows=[]
            for r in result['objects']:
                prediction=classify(r['observation_ids'],features,frames,areas,text,vocab)
                points=actual['xyz'][actual['instance']==r['object_id']]
                low,high=points.min(0),points.max(0)
                rows.append(dict(object_id=r['object_id'],alias=min(r['observation_ids'])+1,
                    observation_ids=r['observation_ids'],unique_frames=r['unique_frames'],
                    retained=r['predictions']['mean']['retained'],mean_prediction=r['predictions']['mean'],prediction=prediction,
                    points=len(points),box=dict(min=low.tolist(),max=high.tolist(),center=((low+high)/2).tolist(),extent=(high-low).tolist())))
            dump(OUT/name/'objects.json',dict(objects=rows,coordinate_frame='omni_odom',units='m',box_scope='AABB of observed surface, not complete physical vehicle'))
            kept=[r for r in rows if r['retained']]
            statuses[name]=dict(regions=len(rows),retained=len(kept),foreground_members=sum(len(r['observation_ids']) for r in kept),
                classes=dict(Counter(r['prediction']['name'] for r in kept)),
                category_changed=[dict(alias=r['alias'],before=r['mean_prediction']['name'],after=r['prediction']['name']) for r in kept if r['mean_prediction']['name']!=r['prediction']['name']],
                joined_edges=sum(d['reason']=='joined' for d in graph['decisions']),
                archive_geometry_memberships_readout_exact=exact,finalization_s=time.monotonic()-tic)
            print('FINAL',name,statuses[name],flush=True)
    finally:
        mapping.merge_detections_to_objects=native_merge
        mapping.compute_overlap_matrix_2set=original_overlap
        utils.compute_overlap_matrix=original_final_overlap
    dump(OUT/'run.json',dict(complete=True,frames=len(frame_times),observations=len(features),
        backend_causal_s=causal_s,total_s=time.monotonic()-start,frame_p95_s=float(np.quantile(frame_times,.95)),
        native_ambiguous_events=len(events),proposals=len(proposals),nodes=len(nodes),methods=statuses,
        protocol_sha256=sha(OUT/'protocol.json'),no_gt=True,frontend_reused=True,
        output_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='progress.json'}))
    print('GARAGE COMPLETE',OUT,flush=True)


if __name__=='__main__':
    main()
