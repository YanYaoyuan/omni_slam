"""Fixed five-arm crossfit test of object-conditioned semantic support acceptance.

Only labels()/score() read reference arrays. Full maps retain the strong old
geometry/IDs/scores; new measurements compete for an existing object or null.
"""
import argparse,copy,time,shutil,runpy
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha
from capture_semantic_context import OUT as CONTEXT,verify as verify_context
from audit_mask_observation_code import source_anchors
from omni_semantic_map.semantic_context_evidence import equal_cell_mean
from omni_semantic_map.intrinsic_surface_geometry import geometry_features
from omni_semantic_map.surface_ownership import OwnershipScorer,allocate,append_owned

OUT=ROOT/'build/benchmarks/object_context_acceptance_v1'
PARENT=ROOT/'build/benchmarks/signed_query_association_v1'
BASE=ROOT/'build/benchmarks/intrinsic_surface_ownership_v1'
OWN=ROOT/'build/benchmarks/surface_ownership_v1'
INPUT=ROOT/'build/benchmarks/discovery_measured_inputs_v1'
FIELD=ROOT/'build/benchmarks/surface_evidence_field_v1'
METHODS=['baseline','expanded_geometry','simple_context','category_context','ours_object_context']
TRAINED=['category_context','ours_object_context']
KEYS=['kitti360_0000','kitti360_0002']
EVALUATOR=ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'


def sealed(name):
    d=read(OUT/(name+'.json'));assert d['complete']
    for f,h in d['files'].items():assert sha(f)==h,f
    return d


def freeze():
    assert not OUT.exists();cp=verify_context();cs=read(CONTEXT/'predictions_complete.json');assert cs['complete'] and cs['all_target_raw_outputs_exact']
    for f,h in cs['files'].items():assert sha(f)==h,f
    parent=read(PARENT/'protocol.json');refs=read(OWN/'protocol.json')['references'];inputs={}
    for f in [CONTEXT/'protocol.json',CONTEXT/'predictions_complete.json',PARENT/'predictions_complete.json',FIELD/'features_complete.json',EVALUATOR,ROOT/'build/datasets/kitti360_prepare/labels.py']:
        inputs[str(f)]=sha(f)
    for scene in parent['scenes']:
        key=scene['key'];manifest=read(INPUT/key/'manifest.json')
        paths=[manifest['geometry'],INPUT/key/'manifest.json',INPUT/'query_cells'/key/'source_to_cell.npy',
               OWN/'features'/key/'core_sources.json',BASE/'maps'/key/'intrinsic_geometry/geometry.npz',BASE/'models'/key/'intrinsic_geometry/model.pt',
               FIELD/'features'/key/'retrieval.json',scene['reference'],refs[key]['raw']]
        paths += [PARENT/'maps'/key/'geometry_graph'/n for n in ['geometry.npz','classification.json','groups.json']]
        for f in paths:inputs[str(f)]=sha(f)
    sources=[Path(__file__),ROOT/'scripts/benchmarks/capture_semantic_context.py',ROOT/'scripts/benchmarks/audit_mask_observation_code.py',
             ROOT/'omni_semantic_map/omni_semantic_map/semantic_context_evidence.py',ROOT/'omni_semantic_map/omni_semantic_map/intrinsic_surface_geometry.py',
             ROOT/'omni_semantic_map/omni_semantic_map/surface_ownership.py',ROOT/'scripts/benchmarks/evaluate_kitti360_development.py',
             ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',ROOT/'omni_semantic_map/test/test_object_context_acceptance.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=parent['scenes'],references=refs,methods=METHODS,trained=TRAINED,
         input_hashes=inputs,source_hashes={str(f):sha(f) for f in sources},vocabulary=cp['vocabulary'],
         hypothesis='Object-conditioned semantic residuals can distinguish missing object surface from context intrusions better than category-conditioned calibration or simple native background rejection, while restoring measured geometry.',
         candidates='All weak raw queries already selected by the frozen surface-field retrieval, now used ONLY as unfiltered proposals. Per object/frame require16distinct exact measured source-anchor cells. All measured source points inside that selected raw query may propose. Exclude any old source provenance and sources<=5cm from strong old geometry. No field probabilities/labels/selected added points or GT-based candidate subset. Shared pool for all non-baseline methods.',
         anchors='Strong geometry_graph original-source provenance plus its and intrinsic additions, conflicting source claims ignored. Context mean per object/frame uses all exact visible source anchors, equal20cm-cell weight. Category mean is equal average of all eligible same-class object means in that frame. Both are measured, transductive window-end statistics, not causal online mapping.',
         evidence='All19standard native category point responses from sealed new capture. Point vector is mean over exact measured-visible frames. Pair anchor vector averages object-frame means over its proposal frames; comparator replaces each mean with pooled same-class mean. Absent observations excluded, zero response remains distinct from a positive alternative class. Scores are features, not calibrated semantic posteriors.',
         features='Common50channels:12intrinsic geometry features relative to full source anchors;19mean point semantic responses;19point-minus-anchor context residuals. category_context uses category anchor; Ours uses object anchor. Same dimensions,architecture,training data/loss/seed/epochs/export. Relational residual cancels common additive category bias in its19channels; raw channels retained. This property alone is not novelty.',
         controls='baseline exact frozen geometry_graph. expanded_geometry applies already frozen cross-window intrinsic_geometry scorer to new candidate pool, no retraining. simple_context uses same pair logits but vetoes when any other class mean response strictly exceeds target; all-zero unknown responses do not veto. category_context and Ours independently learn pair acceptance with explicit null logit0 via independent BCE, then unique best probability>0.5; ties abstain.',
         training='Two development crossfit folds, train other window only. Native geometry scorer already follows same crossfit. Learned models MLP50->32->32->1,seed0,40epochs,Adam1e-3,batch2048,training-only standardization. Per-source mean BCE across valid candidate owners, confidence weighting. No class rebalance/checkpoint choice/threshold or feature sweep.',
         supervision='Direct training labels only from other window: exact frozen raw taxonomy, ignore train255, retain road train0. Nearest reference<=20cm with confidence>0. Source-anchor majority same-class instance must exceed0.5 of all valid anchor mass; otherwise owner unknown. Exclude source groups with any unknown owner. Positive iff source GT equals owner target, null if all candidates negative. No target-window reference array is loaded by the new fit/predict functions.',
         inherited_split_limitation='This is NOT leakage-free nested cross-validation. Parent maps use an earlier two-window crossfit model; the training-window parent can therefore have inherited supervision from the opposite target window. Both windows are already exposed development data. Report two training directions only, never out-of-fold generalization. Any later genuine independent test must be entirely absent from every parent model and all development inputs.',
         prefeature_clarification='Earlier protocol-only draft preserved at object_context_acceptance_prefeature_draft_v1. Before feature extraction/training, replace overly broad no-test-GT wording with the explicit inherited stacking limitation. Algorithms,inputs,hyperparameters and gate unchanged; no outputs or scores existed when clarified.',
         export='Old XYZ,point IDs,classification ranking preserved exactly. Append only unique original source measurements assigned to one existing object. No synthesized coordinates,new object IDs,old deletions,relabeling,score reranking,NMS or cleanup. Not a full SLAM innovation or discovery algorithm.',
         budget='108encodes/2052decodes already shared in context capture. Four fixed CPU fits and two windows x five complete maps including baseline copies. All predictions sealed before new AP scoring. No extra query/prompt/parameter runs to rescue failure.',
         gate='Ours AP50:90 no lower than frozen strong baseline in either window; mean>=baseline+0.01 and strictly>all other controls; >=1gained AP50 GT across windows and zero lost; category vs object conditioning actually changes assignments in both windows. Failure closes version without tuning or robot/independent expansion.',
         scope='All0000/0002previously heavily exposed. Development only, not official leaderboard or independent generalization. No effective new-method success presumed. Robot and truly unused scenes only after this gate. Earlier failed field/code/association branches remain sealed; this adds new19-class evidence and a different acceptance mechanism.'))
    print('FROZEN OBJECT CONTEXT ACCEPTANCE',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    return p


def features():
    p=verify();assert not (OUT/'features_complete.json').exists();context=read(CONTEXT/'predictions_complete.json')
    for f,h in context['files'].items():assert sha(f)==h,f
    files={};stats={}
    semantic_columns={v['semantic_id']:i for i,v in enumerate(p['vocabulary'])}
    for scene in p['scenes']:
        start=time.monotonic();key=scene['key'];manifest=read(INPUT/key/'manifest.json');xyz=np.load(manifest['geometry'])['xyz'];cell=np.load(INPUT/'query_cells'/key/'source_to_cell.npy')
        old=dict(np.load(PARENT/'maps'/key/'geometry_graph/geometry.npz'));cls=read(PARENT/'maps'/key/'geometry_graph/classification.json');sem={o['object_id']:o['predictions']['mean']['semantic_id'] for o in cls['objects']}
        anchors=source_anchors(key,len(xyz));core={o:np.flatnonzero(anchors==o) for o in sem};assert all(len(ids) for ids in core.values())
        eligible=(anchors==0)&(cKDTree(old['xyz']).query(xyz,workers=1)[0]>.05)
        total=np.zeros((len(xyz),19),float);count=np.zeros(len(xyz),np.int16);observations={o:[] for o in core};events=[]
        selection={e['frame_id']:e for e in read(FIELD/'features'/key/'retrieval.json')}
        for rec in context['results'][key]:
            frame=rec['frame_id'];d=dict(np.load(rec['file']));ids=d['source_ids'];s=d['point_scores'];total[ids]+=s;count[ids]+=1
            mu={}
            for oid in core:
                own=anchors[ids]==oid
                if len(np.unique(cell[ids[own]]))>=16:mu[oid]=equal_cell_mean(s[own],cell[ids[own]])
            pooled={semantic:np.mean([mu[o] for o in mu if sem[o]==semantic],axis=0) for semantic in set(sem[o] for o in mu)}
            used=0;points=0
            for row in selection[frame]['selections']:
                if row['mode']!='weak' or row['object_id'] not in mu:continue
                oid=row['object_id'];column=semantic_columns[sem[oid]]
                q=row['query']-(0 if sem[oid]==26 else 200)
                assert 0<=q<200 and d['semantic_ids'][column]==sem[oid]
                inside=np.unpackbits(d['membership_packed'][column,q])[:len(ids)].astype(bool)
                selected=ids[inside&eligible[ids]]
                if not len(selected):continue
                observations[oid].append((selected,mu[oid],pooled[sem[oid]],frame));used+=1;points+=len(selected)
            events.append(dict(frame_id=frame,eligible_anchor_objects=len(mu),proposal_objects=used,proposal_points=points))
        semantic_mean=np.divide(total,count[:,None],out=np.zeros_like(total),where=count[:,None]>0).astype(np.float32)
        parts=[];allcore=np.flatnonzero(anchors>0);core_all=xyz[allcore]
        for oid,obs in observations.items():
            if not obs:continue
            ids=np.unique(np.concatenate([v[0] for v in obs]));ownsum=np.zeros((len(ids),19));catsum=np.zeros_like(ownsum);n=np.zeros(len(ids),np.int16)
            for sid,mu,pooled,frame in obs:
                ix=np.searchsorted(ids,sid);ownsum[ix]+=mu;catsum[ix]+=pooled;n[ix]+=1
            assert np.all(n>0) and np.all(count[ids]>0)
            geo=geometry_features(xyz[ids],xyz[core[oid]],core_all[anchors[allcore]!=oid],sem[oid])
            raw=semantic_mean[ids];object_x=np.column_stack([geo,raw,raw-ownsum/n[:,None]]).astype(np.float32)
            class_x=np.column_stack([geo,raw,raw-catsum/n[:,None]]).astype(np.float32)
            parts.append(dict(ids=ids,owner=oid,object_x=object_x,class_x=class_x,frames=n))
        assert parts
        sid=np.concatenate([r['ids'] for r in parts]);owners=np.concatenate([np.full(len(r['ids']),r['owner'],np.int64) for r in parts]);xx=np.concatenate([r['object_x'] for r in parts]);cx=np.concatenate([r['class_x'] for r in parts]);nf=np.concatenate([r['frames'] for r in parts])
        order=np.lexsort((owners,sid));sid,owners,xx,cx,nf=[a[order] for a in [sid,owners,xx,cx,nf]]
        unique,offset,counts=np.unique(sid,return_index=True,return_counts=True);width=int(counts.max());row=np.repeat(np.arange(len(unique)),counts);col=np.arange(len(sid))-np.repeat(offset,counts)
        x=np.zeros((len(unique),width,50),np.float32);c=np.zeros_like(x);valid=np.zeros((len(unique),width),bool);oid=np.zeros(valid.shape,np.int64);frames=np.zeros(valid.shape,np.int16)
        x[row,col]=xx;c[row,col]=cx;valid[row,col]=True;oid[row,col]=owners;frames[row,col]=nf
        np.testing.assert_array_equal(x[:,:,:31],c[:,:,:31]);assert np.isfinite(x).all() and np.isfinite(c).all()
        folder=OUT/'features'/key;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'candidates.npz',source_ids=unique,owners=oid,valid=valid,object_features=x,category_features=c,proposal_frames=frames)
        dump(folder/'core_sources.json',{str(o):v.tolist() for o,v in core.items()});dump(folder/'observations.json',events)
        stat=dict(candidate_sources=len(unique),pairs=len(sid),max_owners=width,objects_proposed=len(parts),frames=len(events),
                  distinct_context_pairs=int(np.any(xx[:,31:]!=cx[:,31:],axis=1).sum()),
                  raw_class_response_nonzero=int((xx[:,12:31]>0).sum()),seconds=time.monotonic()-start)
        dump(folder/'run.json',stat)
        for f in folder.iterdir():files[str(f)]=sha(f)
        stats[key]=stat;print('FEATURES',key,stat,flush=True)
    dump(OUT/'features_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json')))


def labels():
    from evaluate_kitti360_development import transfer,read_static_ply
    p=verify();sealed('features_complete');assert not (OUT/'training_labels_complete.json').exists();files={};stats={}
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    for scene in p['scenes']:
        key=scene['key'];ref=dict(np.load(scene['reference']));raw=read_static_ply(p['references'][key]['raw'])
        train=np.array([taxonomy[int(v)].trainId for v in raw['semantic']],np.int64);expected=train*1000+raw['instance'].astype(np.int64)%1000;expected[train==255]=0
        np.testing.assert_array_equal(expected,ref['train_instance']);xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz']
        nearest=transfer(ref['xyz'],np.arange(1,len(raw)+1),xyz,.2);ix=np.maximum(nearest-1,0);known=(nearest>0)&(train[ix]!=255)&(ref['confidence'][ix]>0);truth=ref['train_instance'][ix];w=ref['confidence'][ix]
        core=read(OUT/'features'/key/'core_sources.json');objects=read(PARENT/'maps'/key/'geometry_graph/classification.json')['objects'];target={};purities={}
        for obj in objects:
            oid=obj['object_id'];ids=np.array(core[str(oid)],np.int64);ids=ids[known[ids]];best=-1;purity=0.
            if len(ids):
                mass=np.bincount(truth[ids],weights=w[ids]);allowed=np.flatnonzero(np.arange(len(mass))//1000==obj['predictions']['mean']['train_id'])
                if len(allowed):
                    candidate=int(allowed[np.argmax(mass[allowed])]);purity=float(mass[candidate]/mass.sum())
                    if purity>.5:best=candidate
            target[oid]=best;purities[oid]=purity
        d=dict(np.load(OUT/'features'/key/'candidates.npz'));ids=d['source_ids'];v=d['valid'];at=np.full(v.shape,-1,np.int64)
        for oid,gt in target.items():at[d['owners']==oid]=gt
        eligible=known[ids]&np.all((at>=0)|~v,axis=1);positive=(at==truth[ids,None])&v
        folder=OUT/'training_labels'/key;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'TRAINING_ONLY.npz',eligible=eligible,positive=positive,weight=np.where(eligible,w[ids],0))
        dump(folder/'anchors.json',dict(gt_by_owner=target,purity_by_owner=purities,training_only=True))
        stats[key]=dict(groups=int(eligible.sum()),positive_sources=int((positive.any(1)&eligible).sum()),null_sources=int((~positive.any(1)&eligible).sum()),unknown_owners=sum(v<0 for v in target.values()))
        for f in folder.iterdir():files[str(f)]=sha(f)
        print('TRAINING LABELS',key,stats[key],flush=True)
    dump(OUT/'training_labels_complete.json',dict(complete=True,files=files,stats=stats))


def train():
    import torch
    p=verify();sealed('features_complete');sealed('training_labels_complete');assert not (OUT/'trained.json').exists();files={};stats={}
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for target in KEYS:
        key=next(k for k in KEYS if k!=target);d=dict(np.load(OUT/'features'/key/'candidates.npz'));y=dict(np.load(OUT/'training_labels'/key/'TRAINING_ONLY.npz'));keep=y['eligible'];v=d['valid'][keep];positive=y['positive'][keep];weights=y['weight'][keep]
        for method in TRAINED:
            torch.manual_seed(0);rng=np.random.RandomState(0);xx=d['object_features' if method=='ours_object_context' else 'category_features'][keep]
            mean=xx[v].mean(0);std=np.maximum(xx[v].std(0),1e-6);x=torch.tensor((xx-mean)/std);valid=torch.tensor(v);pos=torch.tensor(positive);weight=torch.tensor(weights/weights.mean(),dtype=torch.float32)
            model=OwnershipScorer(50);optimizer=torch.optim.Adam(model.parameters(),lr=.001);curve=[];start=time.monotonic()
            for epoch in range(40):
                order=rng.permutation(len(x));total=0.
                for lo in range(0,len(order),2048):
                    ii=order[lo:lo+2048];z=model(x[ii]);loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,pos[ii].float(),reduction='none')*valid[ii]).sum(1)/valid[ii].sum(1)
                    objective=(loss*weight[ii]).mean();optimizer.zero_grad();objective.backward();optimizer.step();total+=float(objective.detach())*len(ii)
                curve.append(total/len(x))
            folder=OUT/'models'/target/method;folder.mkdir(parents=True,exist_ok=False)
            torch.save(dict(state=model.state_dict(),mean=mean,std=std,dimensions=50,training=[key],method=method),folder/'model.pt')
            stat=dict(training=[key],groups=len(x),loss=curve,seconds=time.monotonic()-start);dump(folder/'training.json',stat)
            for f in folder.iterdir():files[str(f)]=sha(f)
            stats[target+'/'+method]=stat;print('TRAINED',target,method,curve[0],curve[-1],flush=True)
    dump(OUT/'trained.json',dict(complete=True,files=files,stats=stats))


def logits(model_data,features):
    import torch
    m=model_data;model=OwnershipScorer(m['dimensions']);model.load_state_dict(m['state']);model.eval();out=[]
    with torch.no_grad():
        for lo in range(0,len(features),4096):out.append(model(torch.tensor((features[lo:lo+4096,:,:m['dimensions']]-m['mean'])/m['std'])).numpy())
    return np.concatenate(out)


def simple_veto(features,owners,valid,semantics,columns):
    response=features[:,:,12:31];allow=valid.copy()
    for oid,semantic in semantics.items():
        take=valid&(owners==oid);j=columns[semantic]
        other=np.max(np.delete(response,j,axis=2),axis=2)
        allow[take]&=response[:,:,j][take]>=other[take]
    return allow


def predict():
    import torch
    p=verify();sealed('features_complete');sealed('trained');assert not (OUT/'predictions_complete.json').exists();torch.set_num_threads(1);files={};stats={};checks={}
    columns={v['semantic_id']:i for i,v in enumerate(p['vocabulary'])}
    for scene in p['scenes']:
        key=scene['key'];d=dict(np.load(OUT/'features'/key/'candidates.npz'));source=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];base=dict(np.load(PARENT/'maps'/key/'geometry_graph/geometry.npz'));classification=read(PARENT/'maps'/key/'geometry_graph/classification.json');sem={o['object_id']:o['predictions']['mean']['semantic_id'] for o in classification['objects']}
        gm=torch.load(BASE/'models'/key/'intrinsic_geometry/model.pt',map_location='cpu',weights_only=False);assert key not in gm['training'];geometry_logits=logits(gm,d['object_features']);predictions={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False)
            if method=='baseline':
                for name in ['geometry.npz','classification.json']:shutil.copyfile(PARENT/'maps'/key/'geometry_graph'/name,folder/name)
                stat=dict(exact_copy=True,added_points=0,old_point_ids_changed=0)
            else:
                valid=d['valid'];x=d['object_features'];z=geometry_logits
                if method=='simple_context':valid=simple_veto(x,d['owners'],valid,sem,columns)
                elif method in TRAINED:
                    m=torch.load(OUT/'models'/key/method/'model.pt',map_location='cpu',weights_only=False);assert m['training']==[k for k in KEYS if k!=key]
                    x=d['object_features' if method=='ours_object_context' else 'category_features'];z=logits(m,x)
                owner,prob=allocate(z,valid,d['owners'],independent=True);geometry=append_owned(base['xyz'],base['instance'],source,d['source_ids'],owner)
                np.testing.assert_array_equal(geometry['xyz'][:len(base['xyz'])],base['xyz']);np.testing.assert_array_equal(geometry['instance'][:len(base['xyz'])],base['instance'])
                np.savez_compressed(folder/'geometry.npz',**geometry);np.savez_compressed(folder/'decisions.npz',source_ids=d['source_ids'],owners=owner,confidence=prob);dump(folder/'classification.json',classification);predictions[method]=owner
                stat=dict(added_points=int((owner>0).sum()),old_point_ids_changed=0,new_objects=0,old_coordinates_exact=True)
            dump(folder/'run.json',stat);stats[key+'/'+method]=stat
            for f in folder.iterdir():files[str(f)]=sha(f)
            print('PREDICT',key,method,stat,flush=True)
        checks[key]=dict(object_vs_category_changed=int((predictions['ours_object_context']!=predictions['category_context']).sum()),simple_veto_changed=int((predictions['simple_context']!=predictions['expanded_geometry']).sum()))
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,identifiability=checks,protocol_sha256=sha(OUT/'protocol.json'),sealed_before_scoring=True))


def score():
    from omni_semantic_map.kitti360_evaluation import evaluate
    from evaluate_kitti360_development import transfer,clean_json
    from run_discovery_query_benchmark import matched_ids
    p=verify();seal=sealed('predictions_complete');assert not (OUT/'summary.json').exists();results={};means={m:[] for m in METHODS};gate=True;gained=lost=0
    for scene in p['scenes']:
        key=scene['key'];ref=dict(np.load(scene['reference']));rows={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=dict(np.load(folder/'geometry.npz'));objects=read(folder/'classification.json')['objects'];projection=transfer(g['xyz'],g['instance'],ref['xyz'],.2)
            r=evaluate(ref['train_instance'],ref['confidence'],projection,{o['object_id']:o['predictions']['mean'] for o in objects},EVALUATOR)
            dump(folder/'matches.json',clean_json(r['matches']));rows[method]=dict(metrics=clean_json(r['averages']),matched_gt_ids=sorted(matched_ids(r['matches'])),run=read(folder/'run.json'));means[method].append(float(r['averages']['allAp']))
            print('SCORE',key,method,rows[method]['metrics'],flush=True)
        expected=read(PARENT/'summary.json')['results'][key]['methods']['geometry_graph']['metrics']
        for metric in ['allAp','allAp50%']:assert abs(rows['baseline']['metrics'][metric]-expected[metric])<1e-12
        base=set(rows['baseline']['matched_gt_ids'])
        for row in rows.values():actual=set(row['matched_gt_ids']);row['changes_vs_baseline']=dict(gained=sorted(actual-base),lost=sorted(base-actual))
        change=rows['ours_object_context']['changes_vs_baseline'];gained+=len(change['gained']);lost+=len(change['lost']);gate&=rows['ours_object_context']['metrics']['allAp']>=rows['baseline']['metrics']['allAp'];gate&=seal['identifiability'][key]['object_vs_category_changed']>0
        results[key]=dict(methods=rows)
    mean={m:float(np.mean(v)) for m,v in means.items()};gate&=mean['ours_object_context']>=mean['baseline']+.01 and mean['ours_object_context']>max(v for m,v in mean.items() if m!='ours_object_context') and gained>0 and lost==0
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,advance_gate=bool(gate),gained=gained,lost=lost,development_crossfit=True,official_leaderboard=False,prediction_seal_sha256=sha(OUT/'predictions_complete.json')))
    print('GATE',bool(gate),mean,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','features','labels','train','predict','score']);globals()[parser.parse_args().action]()
