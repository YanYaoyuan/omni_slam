"""Frozen, scene-crossfit operation acceptance experiment on measured maps.

Teacher is training-only. All target predictions are sealed before evaluation.
No source point is deleted and no target GT is exposed to either fitted model.
"""
import argparse
import time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.operation_revision import OP_FEATURES,gated_revision
from omni_semantic_map.learned_observation_revision import signed_weights,rollback
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.instance_error_audit import audit_instances,filter_transition
from omni_semantic_map.kitti360_evaluation import evaluate
from evaluate_kitti360_development import transfer,clean_json

OLD=ROOT/'build/benchmarks/learned_observation_revision_v1'
OUT=ROOT/'build/benchmarks/object_operation_gate_v1'
METHODS=['gated_merge_only','utility_only','gated_revision']
CONTROLS=['native_replay','merge_only','reassign_history']


def freeze():
    assert not OUT.exists()
    parent=read(OLD/'protocol.json')
    paths=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/operation_revision.py',
        ROOT/'omni_semantic_map/omni_semantic_map/operation_teacher.py',ROOT/'omni_semantic_map/test/test_operation_revision.py']
    paths+=list(map(Path,parent['source_hashes']))
    for f,h in parent['source_hashes'].items():assert sha(f)==h,f
    seals=['protocol.json','prepared_complete.json','trained.json','predictions_complete.json','summary.json']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=parent['scenes'],
        source_hashes={str(f):sha(f) for f in paths},parent_seals={str(OLD/f):sha(OLD/f) for f in seals},
        reference_hashes=parent['reference_hashes'],methods=METHODS,controls=CONTROLS,
        features=list(OP_FEATURES),edge_classifier=parent['classifier'],
        operation_model=dict(max_iter=100,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.,learning_rate=.1,early_stopping=False,random_state=0),
        crossfit='Four folds by scene. Edge model excludes the scene whose operation rewards are generated. Ungated native trajectory with up to7 additional merges and3 moves per decision. No target-scene reward used for fitting.',
        reward='Exact counterfactual deltaPQ on entire training reference; scale by pre-edit PQ denominator. Separate classifier predicts deltaTP<0. Equal total scene sample weights; no class rebalancing.',
        acceptance='Predicted normalized deltaPQ>0; risk arms additionally require predicted TP-loss probability<=0.1. Threshold fixed before rewards; no calibrated risk or confidence guarantee.',
        support=parent['retained_support'],primary=parent['primary'],causality=parent['causality'],
        gate='Gated revision must exceed native AND previous merge-only AND gated merge-only by >=2 PQ points on EACH indoor scene without TP loss against any. Outdoor AP50 must not fall below native or either merge control. Otherwise freeze; no parameter search.',
        budget='Four crossfit edge fits for training data; one utility regressor and one loss-risk classifier; 3 arms times4 targets=12 new maps. Reuse frozen pair probabilities on targets. Predictions sealed before target GT scoring.',
        limitation='Previously exposed development scenes; not independent tests. All retained observations kept; semantic readout unchanged. Robot has no independent annotations. Training geometry differs by domain; no novelty claim from training alone.'))
    print('FROZEN',flush=True)


def verify(inputs=False):
    p=read(OUT/'protocol.json')
    for f,h in {**p['source_hashes'],**p['parent_seals']}.items():assert sha(f)==h,f
    if inputs:
        for seal in ['prepared_complete.json','predictions_complete.json']:
            for f,h in read(OLD/seal)['files'].items():assert sha(f)==h,f
    return p


def arrays(key):
    with np.load(OLD/key/'observations.npz') as z:
        return {k:z[k] for k in ['features','frames','native','semantic','source_owner','source_xyz','active_source_indices','observation_ids']}


def graph(a,pairs,prob,kind):
    w=signed_weights(len(a['native']),pairs,prob,a['frames'])
    if kind!='indoor':w=np.where(a['semantic'][:,None]==a['semantic'][None,:],w,0)
    return w


def generate():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from omni_semantic_map.operation_teacher import OperationTeacher
    import joblib
    p=verify(True);assert not (OUT/'training_complete.json').exists()
    scenes=[s for s in p['scenes'] if s['role']=='train'];edges={};files={};stats=[]
    params=dict(p['edge_classifier']);params.pop('type')
    for s in scenes:
        with np.load(OLD/s['key']/'pairs.npz') as pairs,np.load(OLD/s['key']/'TRAINING_LABELS_ONLY.npz') as labels:
            edges[s['key']]=(pairs['features'][labels['edge_indices']],labels['labels'])
    for s in scenes:
        key=s['key'];dest=OUT/'training'/key;dest.mkdir(parents=True,exist_ok=False)
        training=[k for k in edges if k!=key];x=np.concatenate([edges[k][0] for k in training]);y=np.concatenate([edges[k][1] for k in training])
        weight=np.concatenate([np.full(len(edges[k][1]),1/len(edges[k][1])) for k in training]);weight*=len(weight)/weight.sum()
        edge=HistGradientBoostingClassifier(**params).fit(x,y,sample_weight=weight);joblib.dump(edge,dest/'crossfit_pair_model.joblib')
        a=arrays(key)
        with np.load(OLD/key/'pairs.npz') as pairs:
            probability=edge.predict_proba(pairs['features'])[:,1];w=graph(a,pairs['pairs'],probability,'indoor')
        assert sha(s['reference'])==p['reference_hashes'][s['reference']]
        with np.load(s['reference']) as ref:
            valid=ref['valid'] if 'valid' in ref else np.ones(len(ref['instance']),bool)
            truth=ref['instance'][valid];owner,_=transfer_labels(a['source_xyz'],a['source_owner']+1,ref['xyz'][valid],mode='nearest_5cm')
        unique=np.unique(np.r_[0,truth]);gt=np.searchsorted(unique,truth);g=len(unique);known=owner>0
        counts=np.bincount((owner[known]-1)*g+gt[known],minlength=len(a['native'])*g).reshape(len(a['native']),g)
        teacher=OperationTeacher(counts,np.bincount(gt,minlength=g));xs=[];rewards=[];types=[];checks=[]
        def full(labels):
            pred=np.zeros(len(truth),np.int64);pred[known]=labels[owner[known]-1]
            return voxel_metrics(truth,pred)
        baseline=full(a['native']);np.testing.assert_allclose(teacher.metrics(a['native'])['pq'],baseline['observed_voxel_instance_pq'],atol=1e-12)
        def record(labels,indices,target,features,gain,kind):
            r=teacher.reward(labels,indices,target);xs.append(features);rewards.append(r);types.append(kind)
            # Bound independent full-evaluator validation, covering early merges and later moves.
            if len(checks)<12 and (len(checks)<6 or kind=='reassign'):
                old=full(labels);new=labels.copy();new[indices]=target;after=full(new)
                np.testing.assert_allclose(r['delta_pq'],after['observed_voxel_instance_pq']-old['observed_voxel_instance_pq'],atol=1e-12)
                assert r['delta_tp']==after['tp']-old['tp'];checks.append(dict(kind=kind,**r))
        started=time.monotonic();result=gated_revision(w,a['native'],a['features'],a['frames'],np.bincount(a['source_owner'],minlength=len(a['native'])),recorder=record)
        final=full(result['labels']);np.testing.assert_allclose(teacher.metrics(result['labels'])['pq'],final['observed_voxel_instance_pq'],atol=1e-12)
        assert len(xs)>0
        np.savez_compressed(dest/'TRAINING_OPERATIONS_ONLY.npz',features=np.asarray(xs),normalized_gain=[r['normalized_gain'] for r in rewards],
            delta_pq=[r['delta_pq'] for r in rewards],delta_tp=[r['delta_tp'] for r in rewards],kind=np.asarray(types))
        np.save(dest/'crossfit_final_identity.npy',result['labels']);dump(dest/'crossfit_events.json',result['events'])
        stat=dict(scene=key,edge_training_scenes=training,operations=len(xs),merges=types.count('merge'),moves=types.count('reassign'),
            improve=sum(r['delta_pq']>0 for r in rewards),harm=sum(r['delta_pq']<0 for r in rewards),lose_tp=sum(r['delta_tp']<0 for r in rewards),
            teacher_checks=checks,baseline=baseline,final=final,converged=result['converged'],wall_seconds=time.monotonic()-started)
        dump(dest/'summary.json',stat);stats.append(stat)
        for f in dest.iterdir():files[str(f)]=sha(f)
        print('GENERATED',key,len(xs),'beneficial',stat['improve'],'harmful',stat['harm'],'TP-loss',stat['lose_tp'],flush=True)
    dump(OUT/'training_complete.json',dict(complete=True,files=files,scenes=stats,protocol_sha256=sha(OUT/'protocol.json')))


def train():
    from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
    from sklearn.dummy import DummyClassifier
    import joblib
    p=verify();seal=read(OUT/'training_complete.json');assert not (OUT/'trained.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    xs=[];ys=[];risks=[];weights=[]
    for s in p['scenes']:
        if s['role']!='train':continue
        with np.load(OUT/'training'/s['key']/'TRAINING_OPERATIONS_ONLY.npz') as d:
            xs.append(d['features']);ys.append(d['normalized_gain']);risks.append(d['delta_tp']<0);weights.append(np.full(len(ys[-1]),1/len(ys[-1])))
    x=np.concatenate(xs);y=np.concatenate(ys);risk=np.concatenate(risks);weight=np.concatenate(weights);weight*=len(weight)/weight.sum()
    started=time.monotonic();utility=HistGradientBoostingRegressor(**p['operation_model']).fit(x,y,sample_weight=weight)
    danger=(HistGradientBoostingClassifier(**p['operation_model']) if len(np.unique(risk))>1 else DummyClassifier(strategy='prior')).fit(x,risk,sample_weight=weight)
    joblib.dump(dict(utility=utility,risk=danger),OUT/'operation_models.joblib')
    dump(OUT/'trained.json',dict(complete=True,operations=len(y),positive_reward=int((y>0).sum()),negative_reward=int((y<0).sum()),
        risk_labels=int(risk.sum()),training_only_mse=float(np.average((utility.predict(x)-y)**2,weights=weight)),
        model_sha256=sha(OUT/'operation_models.joblib'),training_seal_sha256=sha(OUT/'training_complete.json'),wall_seconds=time.monotonic()-started,
        heldout_performance_claim=False))
    print('TRAINED',len(y),flush=True)


def predict():
    import joblib
    p=verify(True);trained=read(OUT/'trained.json');assert sha(OUT/'operation_models.joblib')==trained['model_sha256']
    assert not (OUT/'predictions_complete.json').exists();models=joblib.load(OUT/'operation_models.joblib');files={}
    risk_classes=list(models['risk'].classes_)
    def gate(x,risk_gate):
        utility=float(models['utility'].predict(x[None])[0]);prob=models['risk'].predict_proba(x[None])[0]
        risk=float(prob[risk_classes.index(True)]) if True in risk_classes else 0.
        return dict(accept=utility>0 and (not risk_gate or risk<=.1),predicted_utility=utility,predicted_tp_loss_probability=risk)
    for s in p['scenes']:
        if s['role']=='train':continue
        key=s['key'];a=arrays(key);n=len(a['native']);meta=read(OLD/key/'metadata.json')
        with np.load(OLD/key/'pairs.npz') as pairs,np.load(OLD/key/'pair_predictions.npz') as prediction:
            w=graph(a,pairs['pairs'],prediction['probability'],s['kind'])
        mass=np.bincount(a['source_owner'],minlength=n)
        # On actual maps, confirm the new proposal engine has not changed the prior baseline.
        control=gated_revision(w,a['native'],a['features'],a['frames'],mass)
        np.testing.assert_array_equal(control['labels'],np.load(OLD/key/'reassign_history/observation_identity.npy'))
        assert control['events']==read(OLD/key/'reassign_history/events.json')
        for method in METHODS:
            started=time.monotonic();r=gated_revision(w,a['native'],a['features'],a['frames'],mass,
                gate=lambda x:gate(x,method!='utility_only'),allow_reassignment=method!='gated_merge_only')
            labels=r.pop('labels');folder=OUT/key/method;folder.mkdir(parents=True,exist_ok=False)
            np.testing.assert_array_equal(rollback(labels,r['events']),a['native']);render=labels[a['source_owner']]
            np.save(folder/'observation_identity.npy',labels);np.savez_compressed(folder/'geometry.npz',xyz=a['source_xyz'],instance=render,source_indices=a['active_source_indices'])
            rows=[]
            for oid in np.unique(labels):
                members=np.flatnonzero(labels==oid);classes=a['semantic'][members]
                if s['kind']!='indoor':assert len(np.unique(classes))==1
                inherited=[meta['native_semantics'][str(int(k))] for k in a['native'][members]]
                votes={int(c):int(np.count_nonzero(classes==c)) for c in np.unique(classes)};winner=max(votes,key=lambda k:(votes[k],-k))
                base=next(v for v in inherited if v.get('semantic_id',v.get('replica_class_id'))==winner)
                classification=dict(base,retained=True,score=float(np.mean([v['score'] for v in inherited])))
                rows.append(dict(object_id=int(oid),observation_ids=a['observation_ids'][members].tolist(),predictions=dict(mean=classification)))
            dump(folder/'classification.json',dict(objects=rows,foreground_support_fixed=True,indoor_class_is_display_only=s['kind']=='indoor'))
            decisions=r.pop('decisions');events=r.pop('events');dump(folder/'decisions.json',decisions);dump(folder/'events.json',events)
            dump(folder/'run.json',dict(r,complete=True,objects=len(rows),observations=n,source_points=len(render),
                decisions=len(decisions),accepted=len(events),merges=sum(e['kind']=='merge' for e in events),moves=sum(e['kind']=='reassign' for e in events),
                proposal_control_exact=True,wall_seconds=time.monotonic()-started))
            for f in folder.iterdir():files[str(f)]=sha(f)
            print('PREDICTED',key,method,len(rows),'accepted',len(events),'considered',len(decisions),flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,model_sha256=trained['model_sha256'],protocol_sha256=sha(OUT/'protocol.json')))


def score():
    p=verify(True);seal=read(OUT/'predictions_complete.json');assert not (OUT/'summary.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    old=read(OLD/'summary.json');results={};gates=[]
    for s in p['scenes']:
        if s['role']=='train':continue
        key=s['key'];a=arrays(key);metrics=dict(old['results'][key]['methods']);audits={};changes={};support=None
        if s['reference']:
            assert sha(s['reference'])==p['reference_hashes'][s['reference']];ref=np.load(s['reference'])
        if s['kind']=='outdoor':
            original=np.load(Path(s['fixed_map'])/'geometry.npz');rows=read(Path(s['fixed_map'])/'classification.json')['objects']
            fg=[r['object_id'] for r in rows if r['predictions']['mean']['retained']];blockers=original['xyz'][~np.isin(original['instance'],fg)]
        for method in CONTROLS+METHODS:
            folder=(OLD if method in CONTROLS else OUT)/key/method
            with np.load(folder/'geometry.npz') as g:
                xyz=g['xyz'];labels=g['instance'];np.testing.assert_array_equal(xyz,a['source_xyz']);np.testing.assert_array_equal(g['source_indices'],a['active_source_indices'])
            np.testing.assert_array_equal(labels,np.load(folder/'observation_identity.npy')[a['source_owner']])
            assert (labels>0).all()
            if s['kind']=='robot':metrics[method]=read(folder/'run.json');continue
            if s['kind']=='outdoor':
                projected=transfer(np.concatenate([xyz,blockers]),np.r_[labels,np.zeros(len(blockers),np.int64)],ref['xyz'],.2)
                rows=read(folder/'classification.json')['objects'];pred={r['object_id']:r['predictions']['mean'] for r in rows}
                result=evaluate(ref['train_instance'],ref['confidence'],projected,pred,ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
                m=clean_json(result['averages'])
                if method in METHODS:dump(folder/'matches.json',clean_json(result['matches']))
            else:
                valid=ref['valid'] if 'valid' in ref else np.ones(len(ref['instance']),bool)
                projected,_=transfer_labels(xyz,labels,ref['xyz'][valid],mode='nearest_5cm');m=voxel_metrics(ref['instance'][valid],projected)
                audits[method]=audit_instances(ref['instance'][valid],projected)
                if method in METHODS:dump(folder/'object_audit.json',audits[method])
            if method in CONTROLS:assert m==metrics[method],(key,method,'control score changed')
            metrics[method]=m
            if support is None:support=projected>0
            np.testing.assert_array_equal(projected>0,support)
        compare=['native_replay','merge_only','gated_merge_only']
        if s['kind']=='indoor':
            changes={m:filter_transition(audits['native_replay'],audits[m]) for m in CONTROLS+METHODS}
            gate=all(metrics['gated_revision']['observed_voxel_instance_pq']-metrics[m]['observed_voxel_instance_pq']>=.02 and metrics['gated_revision']['tp']>=metrics[m]['tp'] for m in compare);gates.append(gate)
        elif s['kind']=='outdoor':gate=all(metrics['gated_revision']['allAp50%']>=metrics[m]['allAp50%'] for m in compare);gates.append(gate)
        else:gate=None
        results[key]=dict(kind=s['kind'],methods=metrics,changes_vs_native=changes,gate=gate,identical_source_and_reference_support=True)
        print('SCORED',key,{m:v.get('observed_voxel_instance_pq',v.get('allAp50%',v.get('objects'))) for m,v in metrics.items()},flush=True)
    dump(OUT/'summary.json',clean_json(dict(complete=True,results=results,advance_gate=all(gates),development_only=True,robot_accuracy_available=False,
        protocol_sha256=sha(OUT/'protocol.json'),prediction_seal_sha256=sha(OUT/'predictions_complete.json'))))
    print('GATE',all(gates),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','generate','train','predict','score'])
    globals()[parser.parse_args().action]()
