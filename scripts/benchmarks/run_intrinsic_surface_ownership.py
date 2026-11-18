"""Single representation correction, fixed crossfit/robot controls and budget."""
import argparse,time,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from paper_final_common import ROOT,read,dump,sha
from run_surface_ownership import OUT as PARENT,PRIOR,INPUT,OUTDOOR,verify as verify_parent,sealed as parent_sealed
from omni_semantic_map.surface_ownership import FEATURES,OwnershipScorer,allocate,append_owned
from omni_semantic_map.intrinsic_surface_geometry import GEOMETRY_FEATURES,geometry_features

OUT=ROOT/'build/benchmarks/intrinsic_surface_ownership_v1'
METHODS=['baseline','nearest_core','axis_geometry','axis_evidence','intrinsic_geometry','ours_intrinsic']
TRAINED=['axis_geometry','intrinsic_geometry','ours_intrinsic']
COPIED=dict(baseline='baseline',nearest_core='nearest_core',axis_evidence='independent')


def freeze():
    assert not OUT.exists();p=verify_parent(True)
    seals={}
    for name in ['features_complete','training_labels_complete','trained','predictions_complete']:
        parent_sealed(name);seals[str(PARENT/(name+'.json'))]=sha(PARENT/(name+'.json'))
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/intrinsic_surface_geometry.py']
    rotations=[dict(name='yaw90',rotation=Rotation.from_euler('z',90,degrees=True).as_matrix().tolist(),translation=[0.,0.,0.]),
               dict(name='rigid_xyz',rotation=Rotation.from_euler('xyz',[21,37,83],degrees=True).as_matrix().tolist(),translation=[100.,-300.,12.])]
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=p['scenes'],methods=METHODS,sources={str(f):sha(f) for f in sources},
        parent_protocol_sha256=sha(PARENT/'protocol.json'),parent_seals=seals,intrinsic_features=list(GEOMETRY_FEATURES)+list(FEATURES[12:]),
        hypothesis='Can coordinate-invariant point-to-core shape relations remove the observed arbitrary-frame dependence while retaining or improving actual object-map gains?',
        representation='Replace original first12 geometric/semantic channels with nearest distances, centroid/RMS distance, covariance-regularized Mahalanobis distance, radial parallel/perpendicular standard deviations and sorted covariance eigenvalue square roots. No PCA eigenvectors or canonical axes. Ridge0.05m is existing measured voxel scale. All temporal evidence channels/candidate owners/supervision identical.',
        scope='Conditional window-end restoration of unowned measurements to existing objects. Core geometry, identities, scores immutable. No depth synthesis, object birth, old-point deletion or online SLAM invariance claim. Physical evidence/cell memberships held fixed under world-frame transform; upstream masks/association/grid construction not rerun.',
        controls='Exact copied original baseline, nearest-core and old axis-dependent independent-evidence maps. New axis_geometry and intrinsic_geometry both use12features and independent BCE; ours_intrinsic uses20features and same BCE. Same MLP32/32, seed0,40epochs,Adam1e-3,batch2048,training-only standardization. No hyperparameter/threshold sweep or checkpoint selection.',
        splits=p['splits'],supervision=p['supervision'],rotations=rotations,
        invariance='Recompute geometric features from all actual core/query/competitor points after both fixed rigid transforms. Check feature agreement rtol/atol2e-5 and compare learned owner decisions/probabilities using the same model. Full-map metrics are not used to choose transforms or select a rotated output.',
        budget='9fits (3models x3targets),9new maps plus9copied controls;2 exposed outdoor windows and robot2245. Two fixed transformed-feature checks; no extra maps or AP scoring for transforms. New0004/0005 remain unused unless advance gate passes.',
        gate='Invariant feature checks and zero changed owner decisions under both transforms for intrinsic models. Ours mean AP50:90>=baseline+0.02 and>=nearest_core+0.02; no per-window AP50:90/AP50 regression versus baseline; no lost AP50 GT and >=1 gained GT. To substantiate representation/full-evidence claims, ours mean must exceed axis_evidence and intrinsic_geometry. Otherwise seal this version without tuning or heldout expansion.',
        novelty='Rigid invariance is a required representation property, not by itself a new algorithmic contribution. Learned support recovery and full-map improvements require controls; no SOTA/generalization claim from development crossfit.'))
    print('FROZEN',flush=True)


def verify(data=False):
    p=read(OUT/'protocol.json');verify_parent(data);assert sha(PARENT/'protocol.json')==p['parent_protocol_sha256']
    for f,h in {**p['sources'],**p['parent_seals']}.items():assert sha(f)==h,f
    if data:
        for name in ['features_complete','training_labels_complete','trained','predictions_complete']:parent_sealed(name)
    return p


def sealed(name):
    d=read(OUT/(name+'.json'))
    for f,h in d['files'].items():assert sha(f)==h,f
    return d


def features():
    p=verify(True);assert not (OUT/'features_complete.json').exists();files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];start=time.monotonic();d=np.load(PARENT/'features'/key/'candidates.npz');tensor=d['features'].copy();valid=d['valid'];owner=d['owners'];ids=d['source_ids']
        xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];core=read(PARENT/'features'/key/'core_sources.json');allcore=set().union(*(set(v) for v in core.values()))
        rows=read(PRIOR/'maps'/key/'largest'/'classification.json')['objects'];semantic={r['object_id']:r['predictions']['mean']['semantic_id'] for r in rows}
        transformed=np.repeat(tensor[None],len(p['rotations']),axis=0)
        for oid,source_ids in core.items():
            oid=int(oid);g,j=np.where(owner==oid)
            if not len(g):continue
            own=xyz[np.asarray(source_ids,np.int64)];other=xyz[np.asarray(sorted(allcore-set(source_ids)),np.int64)];points=xyz[ids[g]]
            tensor[g,j,:12]=geometry_features(points,own,other,semantic[oid])
            for k,transform in enumerate(p['rotations']):
                rotation=np.asarray(transform['rotation']);translation=np.asarray(transform['translation'])
                transformed[k,g,j,:12]=geometry_features(points@rotation.T+translation,own@rotation.T+translation,other@rotation.T+translation,semantic[oid])
        np.testing.assert_allclose(tensor[valid,:3],d['features'][valid,:3],rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(tensor[:,:,10:],d['features'][:,:,10:])
        checks={}
        for k,t in enumerate(p['rotations']):
            np.testing.assert_allclose(tensor[valid],transformed[k][valid],rtol=2e-5,atol=2e-5)
            checks[t['name']]=dict(max_feature_absolute_error=float(np.max(np.abs(tensor[valid]-transformed[k][valid]))),allclose=True)
        folder=OUT/'features'/key;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'candidates.npz',features=tensor,valid=valid,owners=owner,source_ids=ids)
        np.savez_compressed(folder/'rigid_features.npz',features=transformed)
        for f in folder.iterdir():files[str(f)]=sha(f)
        stats[key]=dict(candidate_sources=len(ids),candidate_pairs=int(valid.sum()),core_sources=len(allcore),checks=checks,seconds=time.monotonic()-start)
        print('FEATURES',key,stats[key],flush=True)
    dump(OUT/'features_complete.json',dict(complete=True,files=files,stats=stats))


def train():
    import torch
    p=verify(True);sealed('features_complete');assert not (OUT/'trained.json').exists();files={};stats={};torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for scene in p['scenes']:
        key=scene['key'];training=[k for k in OUTDOOR if k!=key]
        for method in TRAINED:
            torch.manual_seed(0);rng=np.random.RandomState(0);dim=20 if method=='ours_intrinsic' else 12;root=PARENT if method=='axis_geometry' else OUT;parts=[];width=0
            for k in training:
                d=np.load(root/'features'/k/'candidates.npz');y=np.load(PARENT/'training_labels'/k/'TRAINING_ONLY.npz');keep=y['eligible'];width=max(width,d['valid'].shape[1]);parts.append((d['features'][keep,:,:dim],d['valid'][keep],y['positive'][keep],y['weight'][keep]))
            arrays=[]
            for i in range(4):
                values=[]
                for part in parts:
                    a=part[i]
                    if i<3:a=np.pad(a,[(0,0),(0,width-a.shape[1])]+([(0,0)] if i==0 else []))
                    values.append(a)
                arrays.append(np.concatenate(values))
            xx,v,positive,w=arrays;mean=xx[v].mean(0);std=np.maximum(xx[v].std(0),1e-6);x=torch.tensor((xx-mean)/std);valid=torch.tensor(v);pos=torch.tensor(positive);weight=torch.tensor(w/w.mean(),dtype=torch.float32)
            model=OwnershipScorer(dim);optimizer=torch.optim.Adam(model.parameters(),lr=.001);curve=[];start=time.monotonic()
            for epoch in range(40):
                order=rng.permutation(len(x));total=0.
                for lo in range(0,len(order),2048):
                    ix=order[lo:lo+2048];z=model(x[ix]);loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,pos[ix].float(),reduction='none')*valid[ix]).sum(1)/valid[ix].sum(1)
                    objective=(loss*weight[ix]).mean();optimizer.zero_grad();objective.backward();optimizer.step();total+=float(objective.detach())*len(ix)
                curve.append(total/len(x))
            folder=OUT/'models'/key/method;folder.mkdir(parents=True,exist_ok=False);torch.save(dict(state=model.state_dict(),mean=mean,std=std,dimensions=dim,training=training,method=method),folder/'model.pt')
            stat=dict(training=training,groups=len(x),loss=curve,seconds=time.monotonic()-start);dump(folder/'training.json',stat)
            for f in folder.iterdir():files[str(f)]=sha(f)
            stats[key+'/'+method]=stat;print('TRAINED',key,method,curve[0],curve[-1],flush=True)
    dump(OUT/'trained.json',dict(complete=True,files=files,stats=stats))


def inference(model_data,features,valid,owners):
    import torch
    m=model_data;model=OwnershipScorer(m['dimensions']);model.load_state_dict(m['state']);model.eval();logits=[]
    with torch.no_grad():
        for lo in range(0,len(features),4096):logits.append(model(torch.tensor((features[lo:lo+4096,:,:m['dimensions']]-m['mean'])/m['std'])).numpy())
    z=np.concatenate(logits) if logits else np.empty(valid.shape)
    return allocate(z,valid,owners,independent=True)


def predict():
    import torch
    p=verify(True);sealed('features_complete');sealed('trained');assert not (OUT/'predictions_complete.json').exists();files={};stats={};invariance={};torch.set_num_threads(1)
    for scene in p['scenes']:
        key=scene['key'];d=np.load(OUT/'features'/key/'candidates.npz');axis=np.load(PARENT/'features'/key/'candidates.npz');rotated=np.load(OUT/'features'/key/'rigid_features.npz')['features'];xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];core=np.load(PRIOR/'maps'/key/'largest'/'geometry.npz');classification=read(PRIOR/'maps'/key/'largest'/'classification.json')
        for method in METHODS:
            folder=OUT/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False);start=time.monotonic()
            if method in COPIED:
                source=PARENT/'maps'/key/COPIED[method]
                for name in ['geometry.npz','decisions.npz','classification.json']:shutil.copyfile(source/name,folder/name);assert sha(source/name)==sha(folder/name)
                stat=read(source/'run.json');stat.update(copied_from=str(source),copied_files_exact=True)
            else:
                path=OUT/'models'/key/method/'model.pt';m=torch.load(path,map_location='cpu',weights_only=False);assert key not in m['training'];x=axis['features'] if method=='axis_geometry' else d['features'];owner,prob=inference(m,x,d['valid'],d['owners'])
                result=append_owned(core['xyz'],core['instance'],xyz,d['source_ids'],owner);np.savez_compressed(folder/'geometry.npz',**result);np.savez_compressed(folder/'decisions.npz',source_ids=d['source_ids'],owners=owner,confidence=prob);dump(folder/'classification.json',classification)
                stat=dict(complete=True,objects=len(classification['objects']),old_points=len(core['xyz']),added_points=int((owner>0).sum()),new_map_points=len(result['xyz']),core_bitwise_equal=True,exclusive_source_assignment=True,model_sha256=sha(path),seconds=time.monotonic()-start)
                if method!='axis_geometry':
                    checks={}
                    for k,t in enumerate(p['rotations']):
                        alternate,confidence=inference(m,rotated[k],d['valid'],d['owners']);checks[t['name']]=dict(changed_decisions=int((alternate!=owner).sum()),max_probability_absolute_error=float(np.max(np.abs(prob-confidence))))
                    invariance[key+'/'+method]=checks
            dump(folder/'run.json',stat)
            for f in folder.iterdir():files[str(f)]=sha(f)
            stats[key+'/'+method]=stat;print('PREDICTED',key,method,stat['added_points'],flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,rigid_invariance=invariance,protocol_sha256=sha(OUT/'protocol.json'),trained_seal_sha256=sha(OUT/'trained.json')))


def score():
    from omni_semantic_map.kitti360_evaluation import evaluate
    from evaluate_kitti360_development import transfer,clean_json
    from run_discovery_query_benchmark import matched_ids
    p=verify();seal=sealed('predictions_complete');assert not (OUT/'summary.json').exists();parent=read(PARENT/'summary.json');results={};means={m:[] for m in METHODS};no_harm=True;gains=0
    for scene in p['scenes']:
        key=scene['key'];methods={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;row=dict(run=read(folder/'run.json'))
            if scene['reference']:
                ref=np.load(scene['reference']);g=np.load(folder/'geometry.npz');projected=transfer(g['xyz'],g['instance'],ref['xyz'],.2);cls=read(folder/'classification.json')['objects'];result=evaluate(ref['train_instance'],ref['confidence'],projected,{r['object_id']:r['predictions']['mean'] for r in cls},ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
                row.update(metrics=clean_json(result['averages']),matched_gt_ids=sorted(matched_ids(result['matches'])));dump(folder/'matches.json',clean_json(result['matches']));means[method].append(result['averages']['allAp'])
                if method in COPIED:assert row['metrics']==parent['results'][key]['methods'][COPIED[method]]['metrics']
                base=set(parent['results'][key]['methods']['baseline']['matched_gt_ids']);current=set(row['matched_gt_ids']);row['changes_vs_baseline']=dict(gained=sorted(current-base),lost=sorted(base-current));print('SCORED',key,method,row['metrics']['allAp'],row['metrics']['allAp50%'],row['changes_vs_baseline'],flush=True)
            methods[method]=row
        if scene['reference']:
            a=methods['ours_intrinsic'];b=methods['baseline'];no_harm&=not a['changes_vs_baseline']['lost'] and all(a['metrics'][m]>=b['metrics'][m]-1e-12 for m in ['allAp','allAp50%']);gains+=len(a['changes_vs_baseline']['gained'])
        results[key]=dict(methods=methods)
    mean={m:float(np.mean(v)) for m,v in means.items()};invariant=all(r['changed_decisions']==0 for checks in seal['rigid_invariance'].values() for r in checks.values());gate=bool(invariant and no_harm and gains>0 and all(mean['ours_intrinsic']>=mean[c]+.02 for c in ['baseline','nearest_core']) and all(mean['ours_intrinsic']>mean[c] for c in ['axis_evidence','intrinsic_geometry']))
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,advance_gate=gate,no_harm=bool(no_harm),gained_gt=gains,rigid_invariance_pass=invariant,development_crossfit=True,robot_accuracy_available=False,prediction_seal_sha256=sha(OUT/'predictions_complete.json')));print('ADVANCE',gate,mean,'invariant',invariant,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','features','train','predict','score']);globals()[parser.parse_args().action]()
