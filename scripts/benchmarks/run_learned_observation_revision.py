"""One scene-disjoint training run and bounded observation revision evaluation."""
import argparse
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.learned_observation_revision import FEATURE_NAMES,pair_features,signed_weights,revise,source_ownership
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.instance_error_audit import audit_instances,filter_transition
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.kitti360_evaluation import evaluate
from evaluate_kitti360_development import transfer,clean_json

OUT=ROOT/'build/benchmarks/learned_observation_revision_v1'
METHODS=['native_replay','merge_only','reassign_history']


def freeze():
    assert not OUT.exists()
    parent=read(ROOT/'build/benchmarks/evidence_retention_20261111/public_dino_v1/protocol.json')
    spec={s['key']:s for s in parent['scenes']};scenes=[]
    train=['replica_office_1','replica_room_0','scenenn_015','scenenn_021']
    for key in train:
        s=spec[key];scenes.append(dict(key=key,role='train',kind='indoor',cache=s['cache'],reference=s['reference'],
            fixed_map=str(ROOT/'build/benchmarks/evidence_retention_20261111/public_history_v1/runs'/key/'native_bridge_2')))
    for key in ['replica_office_0','scenenn_014']:
        s=spec[key];scenes.append(dict(key=key,role='development',kind='indoor',cache=s['cache'],reference=s['reference'],
            fixed_map=str(ROOT/'build/benchmarks/vggt_fullmap_v1/maps'/key/'vggt_single/history2')))
    key='2013_05_28_drive_0000_sync'
    scenes.append(dict(key=key,role='development',kind='outdoor',cache=str(ROOT/'build/benchmarks/kitti360_connected_development_v1/cache'),
        fixed_map=str(ROOT/'build/benchmarks/context_residual_memory_v1'/key/'exported_maps/fixed_context'),
        reference=str(ROOT/'build/benchmarks/kitti360_connected_development_v1/evaluation/reference.npz')))
    scenes.append(dict(key='rosbag2245',role='robot',kind='robot',cache=str(ROOT/'build/benchmarks/evidence_retention_20261111/livo_full_strong_v2/cache_supportcc'),
        fixed_map=str(ROOT/'build/experiments/garage_current_20261112/current'),reference=None))
    hashes={};refs={}
    for s in scenes:
        cache=Path(s['cache']);m=read(cache/'manifest.json');assert m['complete'] and m['no_association']
        paths=[cache/'manifest.json',cache/'geometry.npz']+[Path(s['fixed_map'])/f for f in ['geometry.npz','classification.json']]
        for rec in m['records']:
            f=cache/'frames'/rec['file'];assert sha(f)==rec['sha256'];paths.append(f)
        for f in paths:hashes[str(f)]=sha(f)
        if s['reference']:refs[s['reference']]=sha(s['reference'])
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/learned_observation_revision.py',
        ROOT/'omni_semantic_map/test/test_learned_observation_revision.py',
        ROOT/'omni_semantic_map/omni_semantic_map/full_ovo_evaluation.py',
        ROOT/'omni_semantic_map/omni_semantic_map/mesh_instance_evaluation.py',
        ROOT/'omni_semantic_map/omni_semantic_map/instance_error_audit.py',
        ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',
        ROOT/'scripts/benchmarks/evaluate_kitti360_development.py']
    import sklearn
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=scenes,input_hashes=hashes,reference_hashes=refs,
        source_hashes={str(f):sha(f) for f in sources},methods=METHODS,sklearn=sklearn.__version__,
        features=list(FEATURE_NAMES),candidate_radius_m=.2,
        retained_support='Only observations retained as foreground by frozen parent maps. Preserve every supported source point; fixed earliest-frame then original observation-ID ownership. No new geometry, deletion, within-observation split, denoising or semantic reclassification.',
        training='Four specified scenes, no target-scene GT. Nearest reference within5cm at measured source points; observation plurality purity>=0.8, labelled support coverage>=0.5, >=16 labelled points. GT0 never a shared foreground identity. Up to100000 natural-distribution edges per scene, seeded0; equal total scene weight.',
        classifier=dict(type='HistGradientBoostingClassifier',max_iter=100,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.,learning_rate=.1,early_stopping=False,random_state=0),
        inference='Only measured pair features and trained pair probabilities. Probability clipped[.01,.99] to finite signed log odds; normalize duplicate mask edges per target frame. Unknown edges zero. Outside indoor class-agnostic diagnostics, only same-parent-semantic-class edges are active.',
        partition='Start native identities. Merge when total cross evidence>0. Same signed objective and log(1+N) per-identity coding cost for both arms. Joint arm also permits whole-observation moves to existing or new identities; max10 sweeps, report convergence. Exact inverse event replay mandatory.',
        causality='Offline retrospective correction using complete retained sequence and final predicted memberships; not online-causal tracking. Historical observations change membership, not merely label text.',
        budget='One classifier fit, no sweep; three public development scenes plus one330-frame robot, three partitions each. No GPU neural inference. Predictions for all targets sealed before GT scoring.',
        primary='Indoor PQ: common raw source support, nearest5cm. Outdoor: common raw foreground support plus fixed parent nontarget blockers, nearest20cm official adapted kernel. Report matched native replay, merge-only, revision; never subtract from a different CG-denoised representation.',
        gate='Joint must beat BOTH paired native replay and merge-only by >=2 PQ points on EACH indoor scene; no TP loss against either; outdoor AP50 must not fall below merge-only. Otherwise freeze this branch without additional parameter search. Robot accuracy is not asserted without independent annotations.',
        limitation='All development scenes previously exposed in research. Training scenes disjoint from evaluated scene IDs, but not a claim of fresh independent validation or novel method by itself. Does not recover objects removed before retained-observation stage.'))
    print('FROZEN',[(s['key'],s['role']) for s in scenes],flush=True)


def verify(inputs=False):
    p=read(OUT/'protocol.json')
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    if inputs:
        for f,h in p['input_hashes'].items():assert sha(f)==h,f
    return p


def prepare():
    p=verify(True);files={}
    assert not (OUT/'prepared_complete.json').exists()
    for s in p['scenes']:
        dest=OUT/s['key'];dest.mkdir(exist_ok=False);cache=Path(s['cache']);m=read(cache/'manifest.json')
        xyz=np.load(cache/'geometry.npz')['xyz'];rows=read(Path(s['fixed_map'])/'classification.json')['objects']
        fg=[r for r in rows if r['predictions']['mean']['retained']]
        owners={int(o):r['object_id'] for r in fg for o in r['observation_ids']}
        assert len(owners)==sum(len(r['observation_ids']) for r in fg)
        semantics={r['object_id']:r['predictions']['mean'] for r in fg}
        supports=[];features=[];frames=[];obs=[];native=[];semantic=[]
        for rank,rec in enumerate(m['records']):
            f=np.load(cache/'frames'/rec['file'])
            for j in range(rec['masks']):
                oid=rec['observation_offset']+j
                if oid not in owners:continue
                ids=np.unique(f['point_ids'][f['offsets'][j]:f['offsets'][j+1]])
                if not len(ids):raise ValueError('Retained observation has no measured support')
                supports.append(ids);features.append(f['features'][j]);frames.append(rank);obs.append(oid);native.append(owners[oid])
                row=semantics[owners[oid]];semantic.append(row.get('semantic_id',row.get('replica_class_id')))
        arrays=dict(features=np.asarray(features),frames=np.array(frames),observation_ids=np.array(obs),native=np.array(native),semantic=np.array(semantic))
        pairs,x=pair_features(xyz,supports,arrays['features'],arrays['frames'],arrays['native'],p['candidate_radius_m'])
        owner=source_ownership(supports,arrays['observation_ids'],arrays['frames'],len(xyz));active=np.flatnonzero(owner>=0)
        np.savez_compressed(dest/'observations.npz',**arrays,source_point_ids=np.concatenate(supports),offsets=np.r_[0,np.cumsum([len(v) for v in supports])],
            active_source_indices=active,source_owner=owner[active],source_xyz=xyz[active])
        np.savez_compressed(dest/'pairs.npz',pairs=pairs,features=x)
        dump(dest/'metadata.json',dict(scene=s['key'],role=s['role'],observations=len(obs),candidate_pairs=len(pairs),source_points=len(active),
            native_semantics={str(k):v for k,v in semantics.items()},frames=len(m['records'])))
        if s['role']=='train':
            assert sha(s['reference'])==p['reference_hashes'][s['reference']];ref=np.load(s['reference'])
            valid=ref['valid'] if 'valid' in ref else np.ones(len(ref['instance']),bool)
            tree=cKDTree(ref['xyz'][valid]);gt=ref['instance'][valid];point_truth=np.full(len(xyz),-1,np.int64)
            for start in range(0,len(active),200000):
                ids=active[start:start+200000];distance,nearest=tree.query(xyz[ids],distance_upper_bound=.05,workers=1);known=np.isfinite(distance)
                point_truth[ids[known]]=gt[nearest[known]]
            target=np.full(len(obs),-1,np.int64);purity=np.zeros(len(obs));coverage=np.zeros(len(obs))
            for i,ids in enumerate(supports):
                labels=point_truth[ids];known=labels>=0;coverage[i]=known.mean()
                if known.sum()<16 or coverage[i]<.5:continue
                u,c=np.unique(labels[known],return_counts=True);winner=int(c.argmax());purity[i]=c[winner]/known.sum()
                if purity[i]>=.8:target[i]=u[winner]
            eligible=(target[pairs[:,0]]>=0)&(target[pairs[:,1]]>=0)
            y=(target[pairs[:,0]]==target[pairs[:,1]])&(target[pairs[:,0]]>0)
            chosen=np.flatnonzero(eligible)
            if len(chosen)>100000:chosen=np.sort(np.random.default_rng(0).choice(chosen,100000,replace=False))
            np.savez_compressed(dest/'TRAINING_LABELS_ONLY.npz',observation_target=target,purity=purity,coverage=coverage,edge_indices=chosen,labels=y[chosen].astype(np.uint8))
        for f in dest.iterdir():files[str(f)]=sha(f)
        print('PREPARED',s['key'],len(obs),len(pairs),'active',len(active),flush=True)
    dump(OUT/'prepared_complete.json',dict(complete=True,files=files,protocol_sha256=sha(OUT/'protocol.json')))


def train():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score,log_loss
    import joblib,sklearn
    p=verify();assert sklearn.__version__==p['sklearn'];seal=read(OUT/'prepared_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    assert not (OUT/'trained.json').exists();xs=[];ys=[];weights=[];stats=[]
    for s in p['scenes']:
        if s['role']!='train':continue
        pairs=np.load(OUT/s['key']/'pairs.npz');labels=np.load(OUT/s['key']/'TRAINING_LABELS_ONLY.npz')
        x=pairs['features'][labels['edge_indices']];y=labels['labels'];assert len(y)>=64 and min(np.bincount(y,minlength=2))>=16
        xs.append(x);ys.append(y);weights.append(np.full(len(y),1./len(y)));stats.append(dict(scene=s['key'],pairs=len(y),positives=int(y.sum())))
    x=np.concatenate(xs);y=np.concatenate(ys);weight=np.concatenate(weights);weight*=len(weight)/weight.sum()
    params=dict(p['classifier']);params.pop('type');model=HistGradientBoostingClassifier(**params)
    start=time.monotonic();model.fit(x,y,sample_weight=weight);prob=model.predict_proba(x)[:,1]
    joblib.dump(model,OUT/'pair_model.joblib')
    dump(OUT/'trained.json',dict(complete=True,training_scenes=stats,training_pairs=len(y),training_only_auc=float(roc_auc_score(y,prob)),
        training_only_log_loss=float(log_loss(y,prob)),wall_seconds=time.monotonic()-start,
        model_sha256=sha(OUT/'pair_model.joblib'),prepared_seal_sha256=sha(OUT/'prepared_complete.json'),
        protocol_sha256=sha(OUT/'protocol.json'),heldout_performance_claim=False))
    print('TRAINED',len(y),'pairs',stats,flush=True)


def predict():
    import joblib
    p=verify();train_record=read(OUT/'trained.json');assert sha(OUT/'pair_model.joblib')==train_record['model_sha256']
    for f,h in read(OUT/'prepared_complete.json')['files'].items():assert sha(f)==h,f
    assert not (OUT/'predictions_complete.json').exists();model=joblib.load(OUT/'pair_model.joblib');files={}
    for s in p['scenes']:
        if s['role']=='train':continue
        dest=OUT/s['key'];a=np.load(dest/'observations.npz');data=np.load(dest/'pairs.npz');n=len(a['native'])
        probability=model.predict_proba(data['features'])[:,1] if len(data['pairs']) else np.empty(0)
        weights=signed_weights(n,data['pairs'],probability,a['frames'])
        if s['kind']!='indoor':weights=np.where(a['semantic'][:,None]==a['semantic'][None,:],weights,0)
        np.savez_compressed(dest/'pair_predictions.npz',probability=probability)
        for method in METHODS:
            started=time.monotonic()
            if method=='native_replay':r=dict(labels=a['native'],events=[],converged=True,exact_rollback_verified=True)
            else:r=revise(weights,a['native'],allow_reassignment=method=='reassign_history')
            labels=r.pop('labels');folder=dest/method;folder.mkdir(exist_ok=False)
            render=labels[a['source_owner']];assert (render>0).all()
            np.save(folder/'observation_identity.npy',labels);np.savez_compressed(folder/'geometry.npz',xyz=a['source_xyz'],instance=render,source_indices=a['active_source_indices'])
            meta=read(dest/'metadata.json');rows=[]
            for oid in np.unique(labels):
                members=np.flatnonzero(labels==oid);classes=a['semantic'][members]
                if s['kind']!='indoor':assert len(np.unique(classes))==1
                inherited=[meta['native_semantics'][str(int(k))] for k in a['native'][members]]
                # Foreground remains fixed; indoor category is display-only.
                votes={int(c):int(np.count_nonzero(classes==c)) for c in np.unique(classes)}
                winner=max(votes,key=lambda k:(votes[k],-k));base=next(v for v in inherited if v.get('semantic_id',v.get('replica_class_id'))==winner)
                classification=dict(base,retained=True,score=float(np.mean([v['score'] for v in inherited])))
                rows.append(dict(object_id=int(oid),observation_ids=a['observation_ids'][members].tolist(),predictions=dict(mean=classification)))
            dump(folder/'classification.json',dict(objects=rows,foreground_support_fixed=True,indoor_class_is_display_only=s['kind']=='indoor'))
            dump(folder/'events.json',r.pop('events'));dump(folder/'run.json',dict(r,complete=True,objects=len(rows),observations=n,source_points=len(render),wall_seconds=time.monotonic()-started))
            for f in folder.iterdir():files[str(f)]=sha(f)
            print('PREDICTED',s['key'],method,len(rows),flush=True)
        files[str(dest/'pair_predictions.npz')]=sha(dest/'pair_predictions.npz')
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,model_sha256=train_record['model_sha256'],protocol_sha256=sha(OUT/'protocol.json')))


def score():
    p=verify();seal=read(OUT/'predictions_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    assert not (OUT/'summary.json').exists();results={};gates=[]
    for s in p['scenes']:
        if s['role']=='train':continue
        dest=OUT/s['key'];metrics={};audits={};preds={};checks=[]
        if s['reference']:
            assert sha(s['reference'])==p['reference_hashes'][s['reference']];ref=np.load(s['reference'])
        if s['kind']=='outdoor':
            original=np.load(Path(s['fixed_map'])/'geometry.npz');rows=read(Path(s['fixed_map'])/'classification.json')['objects']
            fg=[r['object_id'] for r in rows if r['predictions']['mean']['retained']];blockers=original['xyz'][~np.isin(original['instance'],fg)]
        for method in METHODS:
            folder=dest/method;g=np.load(folder/'geometry.npz');a=np.load(dest/'observations.npz')
            np.testing.assert_array_equal(g['xyz'],a['source_xyz']);np.testing.assert_array_equal(g['source_indices'],a['active_source_indices'])
            assert np.all(g['instance']>0)
            if s['kind']=='robot':metrics[method]=read(folder/'run.json');continue
            if s['kind']=='outdoor':
                xyz=np.concatenate([g['xyz'],blockers]);labels=np.r_[g['instance'],np.zeros(len(blockers),np.int64)]
                projected=transfer(xyz,labels,ref['xyz'],.2);rows=read(folder/'classification.json')['objects']
                pred={r['object_id']:r['predictions']['mean'] for r in rows}
                result=evaluate(ref['train_instance'],ref['confidence'],projected,pred,ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
                metrics[method]=clean_json(result['averages']);dump(folder/'matches.json',clean_json(result['matches']))
            else:
                valid=ref['valid'] if 'valid' in ref else np.ones(len(ref['instance']),bool)
                projected,_=transfer_labels(g['xyz'],g['instance'],ref['xyz'][valid],mode='nearest_5cm')
                metrics[method]=voxel_metrics(ref['instance'][valid],projected);audits[method]=audit_instances(ref['instance'][valid],projected)
                dump(folder/'object_audit.json',audits[method])
            preds[method]=projected
        if preds:
            for method in METHODS:np.testing.assert_array_equal(preds[method]>0,preds['native_replay']>0)
        if s['kind']=='indoor':
            old=read(ROOT/'build/benchmarks/association_ceiling_v1'/s['key']/'results.json')['methods']['earliest_native']
            assert abs(metrics['native_replay']['observed_voxel_instance_pq']-old['observed_voxel_instance_pq'])<1e-12
            changes={m:filter_transition(audits['native_replay'],audits[m]) for m in METHODS}
            gate=all(metrics['reassign_history']['observed_voxel_instance_pq']-metrics[m]['observed_voxel_instance_pq']>=.02 and metrics['reassign_history']['tp']>=metrics[m]['tp'] for m in ['native_replay','merge_only']);gates.append(gate)
        elif s['kind']=='outdoor':
            changes={};gate=metrics['reassign_history']['allAp50%']>=metrics['merge_only']['allAp50%'];gates.append(gate)
        else:changes={};gate=None
        results[s['key']]=dict(kind=s['kind'],methods=metrics,changes_vs_native=changes,gate=gate,identical_source_and_reference_support=True)
        print('SCORED',s['key'],{m:v.get('observed_voxel_instance_pq',v.get('allAp50%',v.get('objects'))) for m,v in metrics.items()},flush=True)
    dump(OUT/'summary.json',clean_json(dict(complete=True,results=results,advance_gate=all(gates),
        protocol_sha256=sha(OUT/'protocol.json'),prediction_seal_sha256=sha(OUT/'predictions_complete.json'),
        development_only=True,robot_accuracy_available=False)))
    print('GATE',all(gates),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','train','predict','score']);a=ap.parse_args();globals()[a.action]()
