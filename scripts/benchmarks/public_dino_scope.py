"""Post-v1 development control: identical local-eligible evidence for both kernels."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import os
import subprocess
import time
import numpy as np
from public_dino_protocol import DEST as V1,PARENT,read,verify
from audit_evidence_retention import ROOT,OUT,dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.dino_history_evidence import local_similarity,relative_vote
from omni_semantic_map.history_graph import build_history_graph

DEST=OUT/'public_dino_scope_v2'
NAMES=['pooled_matched_scope_2','local_partial_scope_2']


def freeze():
    assert not DEST.exists()
    spec=read(V1/'protocol.json');verify(spec)
    spec.update(variants=['baseline','native_bridge_2']+NAMES,
        extension='Post-v1 coverage-driven development control; not independent validation or original preregistration.',
        scope='For each query, retain only historical members and same-frame competitors with finite local similarity (>=3 reciprocal token pairs). Both kernels use exactly these same sets. Empty branch/competitor set falls back to native vote. Other missing competitors are excluded and their counts reported.',
        v1_protocol_sha256=sha256(V1/'protocol.json'))
    for name in ['public_dino_scope.py','restore_public_history.py']:
        path=ROOT/'scripts/benchmarks'/name;spec['source_sha256'][str(path)]=sha256(path)
    dump(DEST/'protocol.json',spec)
    for p in spec['source_sha256']:
        target=DEST/'sources'/os.path.basename(p);target.parent.mkdir(exist_ok=True);target.write_bytes(open(p,'rb').read())


def run(key):
    from restore_public_history import restore
    from run_livo_history_graph import finalize
    spec=read(DEST/'protocol.json');verify(spec)
    assert sha256(V1/'protocol.json')==spec['v1_protocol_sha256']
    entry=next(e for e in spec['scenes'] if e['key']==key)
    assert sha256(PARENT/'runs'/key/'run.json')==entry['parent_run_sha256']
    feature=V1/'features'/key;fm=read(feature/'manifest.json')
    assert fm['complete'] and fm['protocol_sha256']==sha256(V1/'protocol.json')
    obs={};frames={};frame_obs={}
    for r in fm['records']:
        assert sha256(feature/r['file'])==r['sha256']
        f=np.load(feature/r['file']);ids=[]
        for i in range(r['masks']):
            oid=r['observation_offset']+i;lo,hi=f['offsets'][i:i+2]
            obs[oid]=(f['mean'][i],f['local_features'][lo:hi]);frames[oid]=r['frame_id'];ids.append(oid)
        frame_obs[r['frame_id']]=ids
    target=DEST/'runs'/key;target.mkdir(parents=True,exist_ok=False)
    engine,table=restore(entry,target/'restoration_verification')
    parent=PARENT/'runs'/key;nodes=read(parent/'nodes.json');node_frames={n['alias']:n['frames'] for n in nodes}
    owner={i:n['alias'] for n in nodes for i in n['members']};events={e['query']:e for e in read(parent/'events.json')}
    raw=read(parent/'proposals.json');cache={}
    def scores(q,i):
        if (q,i) not in cache:
            a,b=obs[q],obs[i];den=float(np.linalg.norm(a[0])*np.linalg.norm(b[0]))
            cache[q,i]=(float(a[0]@b[0])/den if den>1e-12 else float('nan'),local_similarity(a[1],b[1]))
        return cache[q,i]
    proposals=[[],[]];diagnostics=[[],[]];scopes=[]
    for p in raw:
        q=p['query'];e=events[q]
        native=next(c['members'] for c in e['candidates'] if c['is_native'])
        alternate=next(c['members'] for c in e['candidates'] if owner[c['members'][0]]==p['b'])
        assert all(frames[i]<frames[q] for i in native+alternate)
        competitors=[i for i in frame_obs[frames[q]] if i!=q and np.linalg.norm(obs[i][0])>1e-12]
        sets=[native,alternate,competitors]
        eligible=[[i for i in group if np.isfinite(scores(q,i)[1])] for group in sets]
        # A local-eligible pooled pair must also be usable; fail rather than silently alter scope.
        assert all(np.isfinite(scores(q,i)[0]) for group in eligible for i in group)
        scopes.append(dict(query=q,a=p['a'],b=p['b'],original_counts=list(map(len,sets)),eligible=eligible))
        for mode in range(2):
            result=relative_vote(*[[scores(q,i)[mode] for i in group] for group in eligible])
            proposals[mode].append(dict(p,dino_pass=result['pass_vote']))
            diagnostics[mode].append(dict(**p,evidence=result,scope_index=len(scopes)-1))
    dump(target/'scopes.json',scopes);variants=[]
    for mode,name in enumerate(NAMES):
        graph=build_history_graph(node_frames,proposals[mode],min_votes=2,use_dino=True)
        output=target/name;final=finalize(engine,graph['groups'],output,table)
        dump(output/'graph.json',graph);dump(output/'votes.json',diagnostics[mode])
        summary=dict(name=name,vote_reasons=dict(Counter(d['evidence']['reason'] for d in diagnostics[mode])),
            graph_reasons=dict(Counter(d['reason'] for d in graph['decisions'])),
            files_sha256={p.name:sha256(p) for p in output.iterdir()})
        dump(output/'summary.json',summary);variants.append(summary)
        print(key,name,summary['vote_reasons'],flush=True)
    dump(target/'run.json',dict(complete=True,no_gt=True,key=key,protocol_sha256=sha256(DEST/'protocol.json'),
        original_native_and_history_exact=True,shared_scope_sha256=sha256(target/'scopes.json'),
        feature_manifest_sha256=sha256(feature/'manifest.json'),variants=variants))


def queue():
    cg=ROOT/'build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python'
    ev=ROOT/'build/benchmarks/ovo_full_20261110/venv311/bin/python'
    def worker(key):
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        for phase,python in [('run',cg),('evaluate',ev)]:
            with (DEST/(key+'_'+phase+'.log')).open('x') as log:
                subprocess.run([str(python),__file__,'--'+phase,key],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        dump(DEST/(key+'_complete.json'),dict(complete=True));return key
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(worker,e['key']) for e in read(DEST/'protocol.json')['scenes']]):print('COMPLETE',f.result(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--queue',action='store_true')
    p.add_argument('--run');p.add_argument('--evaluate');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    if a.queue:queue()
    if a.run:run(a.run)
    if a.evaluate or a.summarize:
        import evaluate_public_dino as evaluation
        evaluation.DEST=DEST
        if a.evaluate:evaluation.evaluate(a.evaluate)
        if a.summarize:evaluation.summarize()
