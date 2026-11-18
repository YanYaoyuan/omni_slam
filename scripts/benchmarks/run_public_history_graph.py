"""Transfer the frozen offline history-bridge rule to public RGB-D caches.

The mapper never opens instance references. Original per-scene RGB-D settings,
frontends, provided poses, final processing and semantic readout are retained.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

from audit_evidence_retention import ROOT, OUT, dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.history_graph import build_history_graph
from omni_semantic_map.shared_observations import observation_table
from run_livo_history_graph import finalize

DEST=OUT/'public_history_v1'
SHARED=ROOT/'build/benchmarks/shared_frontend_20261110'
VARIANTS=[('baseline',0),('native_bridge_1',1),('native_bridge_2',2)]


def read(path):
    return json.loads(path.read_text())


def freeze():
    assert not DEST.exists()
    original=read(SHARED/'protocol_backends.json')
    scenes=[]
    for scene in original['scenes']:
        base=SHARED/'runs/full'/scene/'cg'
        scenes.append(dict(key='replica_'+scene,dataset='Replica',scene=scene,
            cache=str(Path(original['cache_root'])/scene),
            config=str(ROOT/'build/benchmarks/expanded_semantic_20261109/conceptgraphs_runs'/scene/'mapping_config.json'),
            baseline_geometry=str(base/'guarded_geometry.npz'),baseline_classification=str(base/'guarded_classification.json'),
            baseline_readout='all_views_mean',
            reference=str(ROOT/'build/benchmarks/expanded_semantic_20261109/conceptgraphs_evaluation/evaluation'/scene/'reference.npz'),
            baseline_evaluation=str(SHARED/'evaluation/full'/scene/'metrics.json')))
    for scene in ['014','015','021']:
        base=OUT/'heldout_runs'/scene
        scenes.append(dict(key='scenenn_'+scene,dataset='SceneNN',scene=scene,
            cache=str(Path(read(OUT/'stage3_selection.json')['data_root'])/'strong_frontend/v1'/scene/'cache'),
            config=read(OUT/'stage3_selection.json')['cg_config'],
            baseline_geometry=str(base/'geometry.npz'),baseline_classification=str(base/'classification.json'),
            baseline_readout='mean',reference=str(OUT/'heldout_cvpr18_evaluation'/scene/'reference.npz'),
            baseline_evaluation=str(OUT/'heldout_cvpr18_evaluation'/scene/'metrics.json')))
    sources=[Path(__file__),ROOT/'scripts/benchmarks/evaluate_public_history_graph.py',
             ROOT/'scripts/benchmarks/run_livo_history_graph.py']
    sources += [ROOT/'omni_semantic_map/omni_semantic_map'/s for s in
                ['history_graph.py','replay_fusion.py','fusion_diagnostics.py','shared_observations.py',
                 'evidence_readout.py','full_ovo_evaluation.py','mesh_instance_evaluation.py','instance_error_audit.py','flat_benchmark.py']]
    sources += [ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs/conceptgraph/slam'/s for s in ['mapping.py','utils.py','slam_classes.py']]
    for s in scenes:
        paths=[Path(s[k]) for k in ['config','baseline_geometry','baseline_classification','reference','baseline_evaluation']]
        paths.append(Path(s['cache'])/'manifest.json')
        s['hashes']={str(p):sha256(p) for p in paths}
    dump(DEST/'protocol.json',dict(scope=__doc__,scenes=scenes,variants=VARIANTS,
        source_sha256={str(p):sha256(p) for p in sources},
        parent_history_protocol_sha256=sha256(OUT/'history_graph_v1/protocol.json'),
        original_public_protocol_sha256=sha256(SHARED/'protocol_backends.json'),
        intervention='Same history_graph.py; all native ambiguous decisions, 1 and 2 distinct query-frame votes, component-wide coframe veto, common native vote ordering. Merge before final filtering using original CG kernel.',
        original_config='Keep RGB-D DBSCAN and 2.5cm native overlap; do not transfer sparse-LiDAR scale controls.',
        baseline_acceptance='Exact geometry and memberships, identical mean class IDs and retained IDs, exact existing PQ. Replica score floats can differ across equivalent mean implementations; no class changes permitted.',
        evaluation='Frozen observed 5cm reference, assigned-point 1NN <=5cm, strict IoU>.5 class-agnostic PQ. SceneNN uses corrected CVPR18 valid mask including shared void exclusion.',
        identity_audit='Post-hoc node-point 1NN to reference <=5cm, at least 5 valid points and foreground dominant purity >=.7 including background. Compare dominant IDs on candidate edges; mixed/low-support/background are unresolved, not forced labels.',
        selection='All 8 Replica scenes plus all 3 previously evaluated SceneNN scenes; no result-based exclusion, tuning, or promotion.',
        primary='native_bridge_2',secondary='native_bridge_1',
        limitations='Previously examined datasets, development transfer only. Provided poses, cached frontend, offline graph; not LIVO localization or real-time evaluation. No DINO experiment in this round.',
        no_training=True,no_gt_in_mapping=True,no_parameter_sweep=True))
    for p in sources[:3]:
        target=DEST/'sources'/p.name;target.parent.mkdir(exist_ok=True);target.write_bytes(p.read_bytes())


def verify_sources(spec):
    for p,h in spec['source_sha256'].items():
        assert sha256(Path(p))==h,p


def run(key):
    spec=read(DEST/'protocol.json');verify_sources(spec)
    entry=next(s for s in spec['scenes'] if s['key']==key)
    # References are intentionally not read or hashed here: evaluation does that.
    for p,h in entry['hashes'].items():
        if p not in [entry['reference'],entry['baseline_evaluation']]:
            assert sha256(Path(p))==h,p
    target=DEST/'runs'/key;target.mkdir(parents=True,exist_ok=False)
    status=dict(complete=False,key=key,no_gt=True,protocol_sha256=sha256(DEST/'protocol.json'))
    dump(target/'run.json',status)
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    from conceptgraph.slam import mapping
    from omegaconf import OmegaConf
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend,FusionObservation
    cfg=OmegaConf.create(read(Path(entry['config'])))
    assert cfg.filter_interval<0 and cfg.merge_interval<0
    engine=ConceptGraphsBackend(cfg,seed=0)
    cache=Path(entry['cache']);manifest=read(cache/'manifest.json')
    assert manifest['complete'] and manifest['no_association']
    for name,h in manifest['files_sha256'].items():
        assert sha256(cache/name)==h,name
    for rec in manifest['records']:
        assert sha256(cache/'frames'/rec['file'])==rec['sha256']
    geo=np.load(cache/'geometry.npz');xyz,colors=geo['xyz'],geo['colors']
    features,frames,areas=observation_table(cache)
    rawtext=np.load(cache/'text_features.npy');vocab=read(cache/'vocabulary.json')
    poses=np.load(cache/'poses.npy');pose_by_frame={r['frame_id']:poses[i] for i,r in enumerate(manifest['records'])}
    text=rawtext/np.linalg.norm(rawtext,axis=1,keepdims=True)
    table=features,frames,areas,text,vocab,pose_by_frame
    original=mapping.merge_detections_to_objects;events=[]
    def capture(config,detections,objects,scores):
        for i,det in enumerate(detections):
            values=scores[i].cpu().numpy();eligible=np.flatnonzero(np.isfinite(values))
            if len(eligible)<2:
                continue
            query=int(det['observation_ids'][0]);native=int(eligible[np.argmax(values[eligible])])
            candidates=[]
            for j in eligible:
                members=list(map(int,objects[j]['observation_ids']))
                assert all(frames[m]<frames[query] for m in members)
                candidates.append(dict(members=members,native_score=float(values[j]),is_native=int(j)==native))
            events.append(dict(query=query,frame=int(frames[query]),candidates=candidates))
        return original(config,detections,objects,scores)
    began=time.monotonic();mapping.merge_detections_to_objects=capture
    try:
        for index,rec in enumerate(manifest['records']):
            f=np.load(cache/'frames'/rec['file']);obs=[]
            for i in range(rec['masks']):
                oid=rec['observation_offset']+i
                support=f['point_ids'][f['offsets'][i]:f['offsets'][i+1]]
                obs.append(FusionObservation(oid,rec['frame_id'],0,xyz[support],colors[support].astype(np.float64)/255,
                                             features[oid],rawtext[int((rawtext@features[oid]).argmax())]))
            engine.integrate(obs,np.eye(4))
            if index%25==0:
                status.update(frame=index,frames=len(manifest['records']),nodes=len(engine._objects),elapsed_s=time.monotonic()-began)
                dump(target/'run.json',status);print('MAPPING',key,index,len(engine._objects),flush=True)
    finally:
        mapping.merge_detections_to_objects=original
    nodes=[];owner={};points=[];labels=[];node_frames={}
    for obj in engine._objects:
        ids=list(map(int,obj['observation_ids']));alias=min(ids)+1
        node_frames[alias]=sorted(set(map(int,frames[ids])))
        nodes.append(dict(alias=alias,members=ids,frames=node_frames[alias]))
        for oid in ids:
            assert oid not in owner;owner[oid]=alias
        points.append(np.asarray(obj['pcd'].points));labels.append(np.full(len(obj['pcd'].points),alias,np.int64))
    dump(target/'nodes.json',nodes);dump(target/'events.json',events)
    np.savez_compressed(target/'snapshot.npz',xyz=np.concatenate(points),instance=np.concatenate(labels))
    proposals=[]
    for event in events:
        query=event['query'];a=owner[query]
        assert {owner[i] for c in event['candidates'] if c['is_native'] for i in c['members']}=={a}
        for c in event['candidates']:
            mapped={owner[i] for i in c['members']};assert len(mapped)==1
            b=mapped.pop()
            if a!=b:
                proposals.append(dict(a=a,b=b,query=query,frame=event['frame'],dino_pass=False))
    dump(target/'proposals.json',proposals)
    variants=[];baseline_members=None
    for name,votes in VARIANTS:
        graph=(dict(groups=[[n['alias']] for n in nodes],decisions=[]) if votes==0 else
               build_history_graph(node_frames,proposals,min_votes=votes,use_dino=False))
        result=finalize(engine,graph['groups'],target/name,table)
        dump(target/name/'graph.json',graph)
        rows=result['objects'];members={i for o in rows for i in o['observation_ids']}
        if votes==0:
            reference=np.load(entry['baseline_geometry']);actual=np.load(target/name/'geometry.npz')
            for k in actual.files:
                np.testing.assert_array_equal(actual[k],reference[k])
            old=read(Path(entry['baseline_classification']))['objects']
            assert len(rows)==len(old)
            for new,prior in zip(rows,old):
                assert new['object_id']==prior['object_id'] and new['observation_ids']==prior['observation_ids']
                assert new['predictions']['mean']['replica_class_id']==prior['predictions'][entry['baseline_readout']]['replica_class_id']
            baseline_members=members
        def extras(group):
            return sum(len(o['observation_ids'])-len(set(frames[o['observation_ids']])) for o in group)
        fg=[o for o in rows if o['predictions']['mean']['retained']]
        summary=dict(variant=name,regions=len(rows),foreground=len(fg),observations=len(members),
            foreground_observations=sum(len(o['observation_ids']) for o in fg),
            recovered_observations=sorted(members-baseline_members),lost_observations=sorted(baseline_members-members),
            graph_reasons=dict(Counter(d['reason'] for d in graph['decisions'])),
            same_frame_extras=extras(rows),foreground_same_frame_extras=extras(fg),
            before_filter=result['before_filter'],after_final=result['after_final'],
            hashes={p.name:sha256(p) for p in (target/name).iterdir()})
        dump(target/name/'summary.json',summary);variants.append(summary)
        print('FINALIZED',key,name,summary['graph_reasons'],flush=True)
    status.update(complete=True,baseline_exact_geometry_memberships_classes=True,elapsed_s=time.monotonic()-began,
                  ambiguous_decisions=len(events),proposals=len(proposals),nodes=len(nodes),variants=variants,
                  artifacts={p.name:sha256(p) for p in target.iterdir() if p.is_file() and p.name!='run.json'})
    dump(target/'run.json',status);print('COMPLETE MAPPING',key,status['elapsed_s'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--scene');a=p.parse_args()
    if a.freeze:freeze()
    if a.scene:run(a.scene)
