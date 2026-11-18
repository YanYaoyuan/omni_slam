"""Replay identical native history, then verify all proposals with frozen DINO."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import time
import numpy as np

from public_dino_protocol import DEST,PARENT,read,verify
from audit_evidence_retention import dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.dino_history_evidence import local_similarity,relative_vote
from omni_semantic_map.history_graph import build_history_graph
from run_livo_history_graph import finalize


def run(key):
    spec=read(DEST/'protocol.json');verify(spec)
    entry=next(s for s in spec['scenes'] if s['key']==key)
    assert sha256(PARENT/'runs'/key/'run.json')==entry['parent_run_sha256']
    features=DEST/'features'/key;fm=read(features/'manifest.json');assert fm['complete'] and fm['protocol_sha256']==sha256(DEST/'protocol.json')
    observations={};frames={};frame_obs={}
    for r in fm['records']:
        path=features/r['file'];assert sha256(path)==r['sha256'];f=dict(np.load(path));ids=[]
        for i in range(r['masks']):
            oid=r['observation_offset']+i;lo,hi=f['offsets'][i:i+2]
            observations[oid]=(f['mean'][i],f['local_features'][lo:hi]);frames[oid]=r['frame_id'];ids.append(oid)
        frame_obs[r['frame_id']]=ids
    # Run the frozen public runner in a separate output namespace; intercept its
    # finalizer solely to retain the exact in-memory pre-final CG objects.
    import run_public_history_graph as original
    replay=DEST/'replay';replay.mkdir(exist_ok=True)
    protocol_copy=replay/'protocol.json'
    if not protocol_copy.exists():
        # Atomic replacement prevents readers seeing a partially copied file.
        temp=replay/(key+'_protocol.tmp');temp.write_bytes((PARENT/'protocol.json').read_bytes());temp.replace(protocol_copy)
    assert sha256(protocol_copy)==sha256(PARENT/'protocol.json')
    original.DEST=replay;old_finalize=original.finalize;holder={}
    def capture(engine,groups,output,table):
        holder['engine']=engine;holder['table']=table
        return old_finalize(engine,groups,output,table)
    original.finalize=capture
    try:original.run(key)
    finally:original.finalize=old_finalize
    replay_run=replay/'runs'/key;parent=PARENT/'runs'/key
    assert read(replay_run/'events.json')==read(parent/'events.json')
    assert read(replay_run/'nodes.json')==read(parent/'nodes.json')
    for name in ['baseline','native_bridge_2']:
        assert read(replay_run/name/'classification.json')==read(parent/name/'classification.json')
        a,b=np.load(replay_run/name/'geometry.npz'),np.load(parent/name/'geometry.npz')
        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    target=DEST/'runs'/key;target.mkdir(parents=True,exist_ok=False)
    begun=time.monotonic();nodes=read(parent/'nodes.json');node_frames={n['alias']:n['frames'] for n in nodes}
    owner={i:n['alias'] for n in nodes for i in n['members']};events={e['query']:e for e in read(parent/'events.json')}
    raw=read(parent/'proposals.json');variants=[]
    for mode in ['pooled','local']:
        pair_cache={}
        def score(q,p):
            keypair=(q,p)
            if keypair not in pair_cache:
                a,b=observations[q],observations[p]
                if mode=='pooled':
                    den=float(np.linalg.norm(a[0])*np.linalg.norm(b[0]));v=float(a[0]@b[0])/den if den>1e-12 else float('nan')
                else:v=local_similarity(a[1],b[1])
                pair_cache[keypair]=v
            return pair_cache[keypair]
        proposals=[];diagnostics=[]
        for p in raw:
            q=p['query'];e=events[q]
            native=next(c['members'] for c in e['candidates'] if c['is_native'])
            alternate=next(c['members'] for c in e['candidates'] if owner[c['members'][0]]==p['b'])
            assert all(frames[i]<frames[q] for i in native+alternate)
            competitors=[i for i in frame_obs[frames[q]] if i!=q and np.linalg.norm(observations[i][0])>1e-12]
            result=relative_vote([score(q,i) for i in native],[score(q,i) for i in alternate],[score(q,i) for i in competitors])
            proposals.append(dict(p,dino_pass=result['pass_vote']))
            diagnostics.append(dict(**p,evidence=result,competitors=competitors))
        graph=build_history_graph(node_frames,proposals,min_votes=2,use_dino=True)
        name='dino_'+mode+'_2';output=target/name
        final=finalize(holder['engine'],graph['groups'],output,holder['table'])
        dump(output/'graph.json',graph);dump(output/'votes.json',diagnostics)
        summary=dict(name=name,vote_reasons=dict(Counter(d['evidence']['reason'] for d in diagnostics)),
            graph_reasons=dict(Counter(d['reason'] for d in graph['decisions'])),
            before_filter=final['before_filter'],after_final=final['after_final'],
            regions=len(final['objects']),foreground=sum(o['predictions']['mean']['retained'] for o in final['objects']),
            pair_scores=len(pair_cache),files_sha256={p.name:sha256(p) for p in output.iterdir()})
        dump(output/'summary.json',summary);variants.append(summary)
        print('COMPLETE DINO GRAPH',key,name,summary['graph_reasons'],summary['vote_reasons'],flush=True)
    dump(target/'run.json',dict(complete=True,no_gt=True,key=key,protocol_sha256=sha256(DEST/'protocol.json'),
        original_native_and_history_exact=True,feature_manifest_sha256=sha256(features/'manifest.json'),
        variants=variants,verification_and_finalization_s=time.monotonic()-begun))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();run(a.scene)
