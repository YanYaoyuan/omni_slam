"""Three frozen full-recording reranking controls; SigLIP readout unchanged."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from audit_evidence_retention import ROOT,OUT,REPORT,dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.evidence_rerank import rank_accepted,local_correspondence_score

DEST=OUT/'dino_association_v1'
VARIANTS=['baseline','dino_pooled','dino_local']


def freeze():
    path=DEST/'protocol.json';assert not path.exists()
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/evidence_rerank.py']
    dump(path,dict(scope=__doc__,variants=VARIANTS,base='scale_ablation_v1/cache_supportcc/both',
        base_run_sha256=sha256(OUT/'scale_ablation_v1/cache_supportcc/both/run.json'),
        scale_protocol_sha256=sha256(OUT/'scale_ablation_v1/protocol.json'),
        dino_manifest_sha256=sha256(OUT/'dino_livo_tokens_v2/manifest.json'),
        sources={str(p):sha256(p) for p in sources},
        intervention='Causal association only: rank among >=2 finite native accepted candidates. Never change accept/new-object threshold. Final native guard/merge unchanged.',
        pooled='For each historical object, maximum pooled DINO cosine over all its prior supported observation members.',
        local='For each historical object, maximum mean reciprocal appearance-token cosine among prior members with >=3 reciprocal pairs inside 0.40m; all others unknown.',
        radius_m=.4,min_pairs=3,tie='Native score then stable index',fallback='If any accepted candidate has unknown DINO evidence, retain native choice for whole decision.',
        selection='Every qualifying decision, all classes, all 330 frames. Inspect every changed choice and final membership consequences; no GT-based selection.',
        no_parameter_sweep=True,no_training=True,semantic_readout='Frozen SigLIP mean per distinct frame; DINO never replaces semantic class descriptors.',
        development_scope='Previously viewed recording and fixed six-pair diagnostics. This is mechanism development, not independent test or paper benchmark gain.'))


def run(variant):
    protocol=DEST/'protocol.json';spec=json.loads(protocol.read_text())
    for p,h in spec['sources'].items():assert sha256(Path(p))==h
    assert sha256(OUT/'scale_ablation_v1/protocol.json')==spec['scale_protocol_sha256']
    dino=OUT/'dino_livo_tokens_v2';dm=json.loads((dino/'manifest.json').read_text())
    assert sha256(dino/'manifest.json')==spec['dino_manifest_sha256'] and dm['complete']
    observations={}
    for r in dm['records']:
        path=dino/r['file'];assert sha256(path)==r['sha256'];f=dict(np.load(path))
        for i in range(r['masks']):
            lo,hi=f['offsets'][i:i+2];observations[r['observation_offset']+i]=dict(mean=f['mean'][i],xyz=f['local_xyz'][lo:hi],features=f['local_features'][lo:hi],frame=r['frame_id'])
    import run_livo_scale_ablation as scale
    sys.path.insert(0,str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    from conceptgraph.slam import mapping
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend
    replay=DEST/variant;replay.mkdir(exist_ok=False)
    (replay/'protocol.json').write_bytes((OUT/'scale_ablation_v1/protocol.json').read_bytes());scale.ROOT_OUT=replay
    original=mapping.merge_detections_to_objects;old_integrate=ConceptGraphsBackend.integrate
    events=[];history=[];counts=Counter();pair_cache={}
    def pair_score(query,past):
        key=(query,past)
        if key in pair_cache:return pair_cache[key]
        a,b=observations[query],observations[past]
        if not len(a['xyz']) or not len(b['xyz']):score=float('nan')
        elif variant=='dino_local':score,_=local_correspondence_score(a['xyz'],a['features'],b['xyz'],b['features'],spec['radius_m'],spec['min_pairs'])
        else:score=float(a['mean']@b['mean'])
        pair_cache[key]=score;return score
    def merge(cfg,detections,objects,sim):
        adjusted=sim.clone()
        for i,det in enumerate(detections):
            scores=sim[i].cpu().numpy();eligible=np.flatnonzero(np.isfinite(scores))
            if len(eligible)<2:
                counts['unique_candidate' if len(eligible) else 'new_object']+=1;continue
            query=int(det['observation_ids'][0]);evidence=np.full(len(scores),np.nan);candidates=[]
            for j in eligible:
                ids=list(map(int,objects[j]['observation_ids']));assert all(observations[t]['frame']<observations[query]['frame'] for t in ids)
                values=[(pair_score(query,t),t) for t in ids];valid=[x for x in values if np.isfinite(x[0])]
                best=max(valid,key=lambda x:(x[0],-x[1])) if valid else (float('nan'),None)
                evidence[j]=best[0];candidates.append(dict(index=int(j),alias=min(ids)+1,members=ids,native_score=float(scores[j]),dino_score=float(best[0]) if valid else None,best_evidence_observation=best[1]))
            native=int(eligible[np.argmax(scores[eligible])]);chosen,reason=rank_accepted(scores,evidence)
            actual=native if variant=='baseline' else chosen
            counts['ambiguous']+=1;counts[reason]+=1
            if actual!=native:
                adjusted[i,:]=float('-inf');adjusted[i,actual]=sim[i,actual];counts['changed']+=1
            events.append(dict(frame=observations[query]['frame'],observation_id=query,native_alias=min(objects[native]['observation_ids'])+1,
                chosen_alias=min(objects[actual]['observation_ids'])+1,changed=actual!=native,reason=reason,candidates=candidates))
        return original(cfg,detections,objects,adjusted)
    def observe(self,inputs,pose):
        labels=old_integrate(self,inputs,pose);all_labels=self.labels(finalize=False)
        history.append(dict(frame=inputs[0].frame_id if inputs else None,labels={str(o.observation_id):all_labels[o.observation_id] for o in inputs},return_alias=all_labels.get(3074)))
        return labels
    mapping.merge_detections_to_objects=merge;ConceptGraphsBackend.integrate=observe
    try:scale.execute('cache_supportcc','both')
    finally:mapping.merge_detections_to_objects=original;ConceptGraphsBackend.integrate=old_integrate
    output=replay/'cache_supportcc/both'
    if variant=='baseline':
        expected=OUT/spec['base'];assert sha256(expected/'run.json')==spec['base_run_sha256']
        g=np.load(output/'geometry.npz');old=np.load(expected/'geometry.npz')
        for k in g.files:np.testing.assert_array_equal(g[k],old[k])
        assert json.loads((output/'classification.json').read_text())==json.loads((expected/'classification.json').read_text())
    dump(replay/'association.json',dict(complete=True,variant=variant,protocol_sha256=sha256(protocol),baseline_exact=variant=='baseline',
        counts=dict(counts),events=events,causal_history=history,output=str(output),run_sha256=sha256(output/'run.json')))
    print('COMPLETE DINO ASSOCIATION',variant,dict(counts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    if a.freeze:freeze()
    if a.variant:run(a.variant)
