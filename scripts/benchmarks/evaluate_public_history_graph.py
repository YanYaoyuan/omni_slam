"""Post-hoc fixed-domain evaluation; labels never enter graph construction."""
import argparse
from collections import Counter
import json

import numpy as np
from scipy.spatial import cKDTree

from audit_evidence_retention import OUT, dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.instance_error_audit import audit_instances, filter_transition
from run_public_history_graph import DEST, VARIANTS, read, verify_sources
from pathlib import Path


def evaluate(key):
    spec=read(DEST/'protocol.json');verify_sources(spec)
    entry=next(s for s in spec['scenes'] if s['key']==key)
    run=DEST/'runs'/key;finished=read(run/'run.json')
    assert finished['complete'] and finished['no_gt'] and finished['protocol_sha256']==sha256(DEST/'protocol.json')
    for name,h in finished['artifacts'].items():
        assert sha256(run/name)==h
    for variant in finished['variants']:
        for name,h in variant['hashes'].items():
            assert sha256(run/variant['variant']/name)==h
    for field in ['reference','baseline_evaluation']:
        assert sha256(entry[field])==entry['hashes'][entry[field]]
    # Open GT only after completed prediction and provenance checks.
    reference=np.load(entry['reference']);vertices,truth=reference['xyz'],reference['instance']
    valid=reference['valid'] if 'valid' in reference.files else np.ones(len(truth),bool)
    assert (truth[valid]>=0).all()
    output=DEST/'evaluation'/key;output.mkdir(parents=True,exist_ok=False)
    old=read(Path(entry['baseline_evaluation']))
    expected=(old['methods']['cg_guarded']['all_views_mean_things'] if entry['dataset']=='Replica' else old['methods']['mean'])
    snapshot=np.load(run/'snapshot.npz');dist,idx=cKDTree(vertices).query(snapshot['xyz'],workers=1)
    supported=(dist<=.05)&valid[idx]
    node_signatures={}
    for n in read(run/'nodes.json'):
        mask=(snapshot['instance']==n['alias'])&supported
        labels=truth[idx[mask]]
        foreground=labels[labels>0];counts=Counter(map(int,foreground))
        dominant,count=max(counts.items(),key=lambda x:(x[1],-x[0])) if counts else (None,0)
        purity=count/max(len(labels),1)
        node_signatures[n['alias']]=dict(valid_points=int(len(labels)),dominant_gt=dominant,
            dominant_purity=float(purity),usable=bool(len(labels)>=5 and dominant is not None and purity>=.7))
    dump(output/'node_gt_signatures.json',{str(k):v for k,v in node_signatures.items()})
    result=dict(key=key,dataset=entry['dataset'],scene=entry['scene'],complete=False,
                run_sha256=sha256(run/'run.json'),reference_sha256=sha256(entry['reference']),
                valid_voxels=int(valid.sum()),reference_voxels=len(valid),methods={})
    base_audit=None
    for name,_ in VARIANTS:
        folder=run/name;geometry=np.load(folder/'geometry.npz');rows=read(folder/'classification.json')['objects']
        kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
        labels=np.where(np.isin(geometry['instance'],kept),geometry['instance'],0)
        projected,_=transfer_labels(geometry['xyz'],labels,vertices,mode='nearest_5cm')
        score=voxel_metrics(truth[valid],projected[valid]);audit=audit_instances(truth[valid],projected[valid])
        if name=='baseline':
            for metric in ['observed_voxel_instance_pq','tp','fp','fn','sq','rq']:
                assert score[metric]==expected[metric],(key,metric,score[metric],expected[metric])
            oldpred=(Path(entry['baseline_evaluation']).parent /
                     ('cg_guarded__all_views_mean_things.npz' if entry['dataset']=='Replica' else 'mean.npz'))
            np.testing.assert_array_equal(projected,np.load(oldpred)['instance'])
            base_audit=audit
        score.update(predicted_error_counts=audit['predicted_counts'],gt_error_counts=audit['gt_counts'],
                     matched_gt_transition=filter_transition(base_audit,audit))
        edges=[]
        for decision in read(folder/'graph.json')['decisions']:
            a,b=node_signatures[decision['a']],node_signatures[decision['b']]
            identity=('same_dominant_gt' if a['dominant_gt']==b['dominant_gt'] else 'different_dominant_gt') if a['usable'] and b['usable'] else 'unresolved'
            edges.append(dict(**decision,identity_signature=identity))
        score['joined_edge_signatures']=dict(Counter(e['identity_signature'] for e in edges if e['reason']=='joined'))
        score['all_edge_signatures']=dict(Counter(e['identity_signature'] for e in edges))
        score['evaluated_prediction_instances']=len(audit['predicted'])
        dump(output/(name+'_edges.json'),edges);dump(output/(name+'_audit.json'),audit)
        np.savez_compressed(output/(name+'.npz'),instance=projected)
        result['methods'][name]=score
    result.update(complete=True,baseline_exact=True,files_sha256={p.name:sha256(p) for p in output.iterdir()})
    dump(output/'metrics.json',result)
    print('EVALUATED',key,{k:round(v['observed_voxel_instance_pq']*100,6) for k,v in result['methods'].items()},flush=True)


def summarize():
    spec=read(DEST/'protocol.json');results=[]
    for entry in spec['scenes']:
        row=read(DEST/'evaluation'/entry['key']/'metrics.json');assert row['complete'] and row['baseline_exact']
        results.append(row)
    result=dict(complete=True,protocol_sha256=sha256(DEST/'protocol.json'),datasets={},scenes=results,
        scope='Previously viewed public datasets, frozen development transfer. Provided poses and shared cached frontend. Class-agnostic observed-voxel PQ, not official leaderboard.')
    for dataset in ['Replica','SceneNN']:
        subset=[r for r in results if r['dataset']==dataset];group={}
        for name,_ in VARIANTS:
            scores=[r['methods'][name] for r in subset]
            delta=np.array([r['methods'][name]['observed_voxel_instance_pq']-r['methods']['baseline']['observed_voxel_instance_pq'] for r in subset])*100
            group[name]=dict(macro_pq100=float(np.mean([s['observed_voxel_instance_pq'] for s in scores])*100),
                delta_pq100=float(delta.mean()),wins=int((delta>1e-8).sum()),ties=int((abs(delta)<=1e-8).sum()),losses=int((delta < -1e-8).sum()),
                totals={k:sum(s[k] for s in scores) for k in ['tp','fp','fn']},
                predicted_error_counts=dict(sum((Counter(s['predicted_error_counts']) for s in scores),Counter())),
                gt_error_counts=dict(sum((Counter(s['gt_error_counts']) for s in scores),Counter())),
                joined_edge_signatures=dict(sum((Counter(s['joined_edge_signatures']) for s in scores),Counter())),
                gained_gt=sum(len(s['matched_gt_transition']['gained_matched_gt']) for s in scores),
                lost_gt=sum(len(s['matched_gt_transition']['lost_matched_gt']) for s in scores))
        result['datasets'][dataset]=group
    dump(DEST/'summary.json',result)
    print(json.dumps(result['datasets'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.scene:evaluate(a.scene)
    if a.summarize:summarize()
