"""Five frozen geometry variants from shared observations, with no GT access."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time
START=time.monotonic()
import numpy as np
from paper_final_common import ROOT,RUN,FROZEN,read,sha,dump,verify_execution,resource_usage
from omni_semantic_map.history_graph import build_history_graph
from omni_semantic_map.shared_observations import observation_table,select_views,classify_objects
from omni_semantic_map.full_ovo_evaluation import BACKGROUND


def graph_without_veto(node_frames,proposals,min_votes=2):
    """Exact native ordering/support; sole intervention is allowing coframe unions."""
    parents={int(k):int(k) for k in node_frames};frames={int(k):set(map(int,v)) for k,v in node_frames.items()}
    edges=defaultdict(lambda:{'geometry_frames':set(),'dino_frames':set()})
    for p in proposals:
        a,b=sorted([int(p['a']),int(p['b'])]);assert a in parents and b in parents
        if a==b:continue
        edges[a,b]['geometry_frames'].add(int(p['frame']))
        if p['dino_pass']:edges[a,b]['dino_frames'].add(int(p['frame']))
    def root(k):
        while k!=parents[k]:k=parents[k]
        return k
    decisions=[]
    for (a,b),votes in sorted(edges.items(),key=lambda e:(-len(e[1]['geometry_frames']),e[0])):
        x,y=root(a),root(b);conflict=sorted(frames[x]&frames[y]) if x!=y else []
        if len(votes['geometry_frames'])<min_votes:reason='insufficient_votes'
        elif x==y:reason='already_connected'
        else:
            reason='joined';lo,hi=sorted([x,y]);parents[hi]=lo;frames[lo]|=frames[hi]
        decisions.append(dict(a=a,b=b,geometry_frames=sorted(votes['geometry_frames']),
                              dino_frames=sorted(votes['dino_frames']),reason=reason,conflict_frames=conflict))
    groups=defaultdict(list)
    for node in sorted(parents):groups[root(node)].append(node)
    return dict(groups=list(groups.values()),decisions=decisions)


def cache_data(cache):
    m=read(cache/'manifest.json');assert m['complete'] and m['no_association']
    for name,want in m['files_sha256'].items():assert sha(cache/name)==want,name
    for rec in m['records']:assert sha(cache/'frames'/rec['file'])==rec['sha256']
    g=np.load(cache/'geometry.npz');features,frames,areas=observation_table(cache)
    rawtext=np.load(cache/'text_features.npy');text=rawtext/np.linalg.norm(rawtext,axis=1,keepdims=True)
    vocab=read(cache/'vocabulary.json');poses=np.load(cache/'poses.npy')
    byframe={r['frame_id']:poses[i] for i,r in enumerate(m['records'])}
    return m,g,features,frames,areas,rawtext,text,vocab,byframe


def run_cg(cache,target):
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
                    features[oid],rawtext[int((rawtext@features[oid]).argmax())]))
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
    for name,votes in [('cg',0),('history1',1),('history2',2),('history2_no_veto',2)]:
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
    status.update(complete=True,variants=['cg','history1','history2','history2_no_veto'],
                  rejected_observations=engine.rejected,**resource_usage())
    dump(target/'run.json',status)


def run_ovo(cache,target):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'build/benchmarks/ovo_full_20261110/upstream'))
    from ovo.entities.ovo import OVO
    from ovo.entities.instance3d import Instance3D
    target.mkdir(parents=True,exist_ok=False)
    m,g,features,frames,areas,rawtext,text,vocab,byframe=cache_data(cache);xyz=g['xyz']
    engine=OVO.__new__(OVO);engine.objects={};engine.next_ins_id=0;Instance3D.n_top_kf=10
    ids=torch.arange(len(xyz)).reshape(-1,1);labels=torch.full((len(xyz),),-1,dtype=torch.int32);members={};times=[]
    cold=time.monotonic()-START;begin=time.monotonic()
    for rec in m['records']:
        tic=time.monotonic();f=np.load(cache/'frames'/rec['file']);support=torch.from_numpy(f['point_ids'].astype(np.int64))
        segids=torch.repeat_interleave(torch.arange(rec['masks']),torch.from_numpy(np.diff(f['offsets'])))
        labels,matched=engine._track_objects(ids,labels,support,segids,torch.from_numpy(f['segmentation'].astype(np.int32)),m['config']['semantic']['track_th'],rec['frame_id'])
        for obj,values in matched.items():members.setdefault(obj+1,[]).extend(rec['observation_offset']+int(v[0]) for v in values)
        times.append(time.monotonic()-tic)
    elapsed=time.monotonic()-begin;tic=time.monotonic()
    rows=classify_objects(sorted(members.items()),features,frames,areas,rawtext,vocab)
    for r in rows:
        r['predictions']['mean']=dict(r['predictions']['all_views_mean'],
            retained=r['predictions']['all_views_mean']['replica_class_id'] not in BACKGROUND and r['unique_frames']>=3)
    np.savez_compressed(target/'geometry.npz',xyz=xyz,instance=labels.numpy()+1)
    dump(target/'classification.json',dict(objects=rows))
    dump(target/'run.json',dict(complete=True,no_gt=True,geometry='ovo_3views',
        cache_manifest_sha256=sha(cache/'manifest.json'),causal_mapping_s=elapsed,
        offline_finalization_s=time.monotonic()-tic,cold_start_s=cold,frame_times_s=times,
        frame_median_s=float(np.median(times)),frame_p95_s=float(np.quantile(times,.95)),
        retention='original distinct-frame mean foreground AND >=3 unique frames',**resource_usage()))
    print('OVO DONE',target.parent.name,len(rows),flush=True)


def categories(cache,target,output):
    from omni_semantic_map.category_evidence import readout
    from omni_semantic_map.consensus_category import classify
    m,g,features,frames,areas,rawtext,text,vocab,byframe=cache_data(cache)
    protocol=read(FROZEN/'protocol.json');variants={}
    for variant in protocol['variants']:
        name=variant['geometry'];method=variant['category'];base=target/'ovo' if name=='ovo_3views' else target/'cg'/name
        rows=read(base/'classification.json')['objects'];updates=[];tic=time.monotonic()
        for r in rows:
            selected=np.array(select_views(r['observation_ids'],frames,areas,top_k=0))
            if method=='consensus_sqrt_area':p=classify(r['observation_ids'],features,frames,areas,text,vocab)
            else:
                mode='exclusive_quality' if method=='sqrt_area' else method
                result=readout(features[selected],areas[selected],areas[selected],text,method=mode)
                order=np.argsort(-result['scores'],kind='stable');best=int(order[0])
                p=dict(name=vocab['names'][best],replica_class_id=int(vocab['replica_class_ids'][best]),
                    score=float(result['scores'][best]),weights=result['weights'].tolist(),
                    selected_observation_ids=selected.tolist(),fallback=result['fallback'])
            if method=='mean':assert p['replica_class_id']==r['predictions']['mean']['replica_class_id']
            weights=np.array(p['weights'])
            updates.append(dict(object_id=r['object_id'],observation_ids=r['observation_ids'],
                base_retained=r['predictions']['mean']['retained'],prediction=p,
                effective_views=float(1/(weights@weights))))
        elapsed=time.monotonic()-tic
        variants[variant['id']]=dict(geometry=name,method=method,geometry_path=str(base/'geometry.npz'),
            geometry_sha256=sha(base/'geometry.npz'),category_s=elapsed,objects=updates)
    dump(output,dict(complete=True,no_gt=True,cache_manifest_sha256=sha(cache/'manifest.json'),variants=variants,
                     execution_manifest_sha256=sha(RUN/'execution_manifest.json') if (RUN/'execution_manifest.json').exists() else None))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['cg','ovo','categories']);p.add_argument('--scene',required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--target',type=Path);p.add_argument('--validation',action='store_true');a=p.parse_args()
    if not a.validation:verify_execution()
    cache=a.cache or RUN/'frontend'/a.scene/'cache';target=a.target or RUN/'maps'/a.scene
    if a.action=='cg':run_cg(cache,target/'cg')
    elif a.action=='ovo':run_ovo(cache,target/'ovo')
    else:categories(cache,target,target/'predictions.json' if a.validation else RUN/'predictions'/(a.scene+'.json'))
