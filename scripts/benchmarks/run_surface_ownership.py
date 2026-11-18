"""Finite scene-crossfit experiment for unowned measured-point allocation.

Feature extraction and prediction never read references. Separate training-only
labels and post-seal full-map evaluation are the only reference consumers.
"""
import argparse,time,runpy
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha
from run_fixed_identity_recovery_control import OUT as PRIOR,verify as verify_prior
from run_deferred_surface_support import OUT as SUPPORT
from run_discovery_query_benchmark import OUT as FRONT,INPUT,matched_ids
from omni_semantic_map.surface_ownership import FEATURES,GEOMETRY_FEATURES,OwnershipScorer,grouped_loss,allocate,append_owned

OUT=ROOT/'build/benchmarks/surface_ownership_v1'
METHODS=['baseline','nearest_core','learned_geometry','independent','ours_competitive']
OUTDOOR=['kitti360_0000','kitti360_0002']


def freeze():
    assert not OUT.exists();p=verify_prior(True)
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/surface_ownership.py']
    refs={}
    for s in p['scenes']:
        if not s['reference']:continue
        ep=Path(s['reference']).parent/'protocol.json';raw=Path(read(ep)['reference'])
        refs[s['key']]=dict(evaluation_protocol=str(ep),protocol_sha256=sha(ep),raw=str(raw),sha256=sha(raw))
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=p['scenes'],methods=METHODS,
        sources={str(f):sha(f) for f in sources},prior_protocol_sha256=sha(PRIOR/'protocol.json'),
        parent_prediction_seal_sha256=sha(PRIOR/'predictions_complete.json'),references=refs,
        features=list(FEATURES),geometry_dimensions=GEOMETRY_FEATURES,
        mechanism='Window-end conditional allocation of previously unowned measured sources. Original core map, memberships and semantic scores immutable. Candidate owner only if one of its original retained masks contains that actual source. Explicit null competes with candidate object scores; commit only posterior>0.5, ties abstain. Same source at most one owner. No synthetic depth, new mask or object birth.',
        learning='Shared MLP20(or12)->32->32->1; null logit0; set-target negative log likelihood, treating multiple baseline IDs matching one training GT as valid alternatives. Independent ablation uses per-pair BCE with mean loss per source. Training source-confidence weights; no positive oversampling. Train-only standardization; Adam1e-3,40epochs,batch2048,seed0,CPU. No tuning or validation-driven stopping.',
        supervision='Training only: nearest reference within20cm, raw semantic trainId!=255 and positive confidence. Road trainId0 is valid null, ignored trainId255 excluded. Baseline owner target is its measured-core confidence-majority same-class instance with >0.5 of valid mass; otherwise unknown. Exclude groups containing an unknown owner. Null iff no candidate owner matches source GT. Labels/IDs never input features.',
        splits='Two scene-crossfit folds: train0000->evaluate0002; train0002->evaluate0000. Both were already exposed for design, so this is development, not independent validation. Robot trains on both; no robot labels. New0004/0005 untouched.',
        controls='Baseline exact original map; nearest_core uses nearest candidate core<=0.4m (existing component radius), no learning; learned_geometry uses first12features and joint loss; independent uses all20features/BCE; ours_competitive uses all20features/joint loss. All additions unique measured points appended unchanged with same output semantics. No further cleanup in any arm.',
        budget='Nine fixed fits (three models x two scene folds and one robot fit),12 revised maps plus3 exact baseline copies; no threshold/model/hyperparameter sweep.',
        gate='Mean crossfit AP50:90 must improve>=0.02 over baseline and nearest_core; neither scene AP50:90/AP50 may decrease versus baseline; no lost AP50 GT and at least one gained GT across windows. Cross-view/competition contribution requires mean primary greater than learned_geometry and independent. Failure seals this candidate; do not tune on results or expand to held-out data.',
        scope='Candidate mechanism, no established novelty or generalization. Conditional restoration only; does not discover new object IDs, fix old core errors or produce online updates. Native full-reference adapted KITTI AP50:90/AP50; not official leaderboard. Robot map execution without accuracy labels.'))
    print('FROZEN',flush=True)


def verify(data=False):
    p=read(OUT/'protocol.json');verify_prior(data)
    assert sha(PRIOR/'protocol.json')==p['prior_protocol_sha256']
    assert sha(PRIOR/'predictions_complete.json')==p['parent_prediction_seal_sha256']
    for f,h in p['sources'].items():assert sha(f)==h,f
    if data:
        for f,h in read(PRIOR/'predictions_complete.json')['files'].items():assert sha(f)==h,f
    return p


def sealed(name):
    d=read(OUT/(name+'.json'))
    for f,h in d['files'].items():assert sha(f)==h,f
    return d


def features():
    p=verify(True);assert not (OUT/'features_complete.json').exists();files={};stats={}
    for s in p['scenes']:
        start=time.monotonic();key=s['key'];manifest=read(INPUT/key/'manifest.json');xyz=np.load(manifest['geometry'])['xyz']
        rows=read(PRIOR/'maps'/key/'largest'/'classification.json')['objects'];byobs={o:r['object_id'] for r in rows for o in r['observation_ids']}
        core={r['object_id']:set() for r in rows};raw={r['object_id']:defaultdict(dict) for r in rows};cell_frames=defaultdict(lambda:defaultdict(set))
        own_cells={r['object_id']:defaultdict(set) for r in rows};cells=np.load(INPUT/'query_cells'/key/'source_to_cell.npy')
        visible=np.zeros(len(xyz),np.int32);frames={r['frame_id']:r for r in manifest['records']}
        records=read(FRONT/'masks_complete.json')['scenes'][key]['records']
        for r in records:
            frame=r['frame_id'];measured=np.load(frames[frame]['file']);visible[measured['visible_source_ids']]+=1
            a=np.load(SUPPORT/'supports'/key/'all'/f'{frame:06d}.npz');b=np.load(SUPPORT/'supports'/key/'largest'/f'{frame:06d}.npz');scores=np.load(r['file'])['scores']
            for j in range(r['masks']):
                obs=r['observation_offset']+j
                if obs not in byobs:continue
                owner=byobs[obs];ids=a['point_ids'][a['offsets'][j]:a['offsets'][j+1]]
                core[owner].update(map(int,b['point_ids'][b['offsets'][j]:b['offsets'][j+1]]))
                for sid in ids:raw[owner][int(sid)][frame]=max(float(scores[j]),raw[owner][int(sid)].get(frame,0.))
                for c in np.unique(cells[ids]):own_cells[owner][int(c)].add(frame);cell_frames[int(c)][frame].add(owner)
        core_ids=np.asarray(sorted(set().union(*core.values())),np.int64);iscore=np.zeros(len(xyz),bool);iscore[core_ids]=True
        pair_ids=[];pair_owner=[];vectors=[];core_table={};rowby={r['object_id']:r for r in rows}
        for owner in sorted(raw):
            ids=np.asarray(sorted(sid for sid in raw[owner] if not iscore[sid]),np.int64)
            core_table[str(owner)]=sorted(core[owner])
            if not len(ids):continue
            own=xyz[np.asarray(sorted(core[owner]),np.int64)];other=xyz[np.asarray(sorted(set(core_ids)-core[owner]),np.int64)]
            d=cKDTree(own).query(xyz[ids],workers=1)[0];od=cKDTree(other).query(xyz[ids],workers=1)[0] if len(other) else np.full(len(ids),100.)
            lo=own.min(0);hi=own.max(0);extent=np.maximum(hi-lo,.05);center=(lo+hi)/2
            outside=np.maximum(np.maximum(lo-xyz[ids],xyz[ids]-hi),0)/extent
            semantic=rowby[owner]['predictions']['mean']['semantic_id'];object_frames=len({next(rr['frame_id'] for rr in records if rr['observation_offset']<=o<rr['observation_offset']+rr['masks']) for o in rowby[owner]['observation_ids']})
            geom=np.column_stack([np.log1p(d),np.log1p(od),d/(d+od+1e-6),np.log1p(np.linalg.norm((xyz[ids]-center)/extent,axis=1)),outside,
                                  np.broadcast_to(np.log1p(extent),(len(ids),3)),np.full(len(ids),semantic==26),np.full(len(ids),semantic==11)])
            temporal=[]
            for sid in ids:
                evidence=raw[owner][int(sid)];cell=int(cells[sid]);own_count=len(own_cells[owner][cell]);others=sum(bool(owners-{owner}) for owners in cell_frames[cell].values());seen=int(visible[sid]);assert seen>=len(evidence)>0
                temporal.append([np.log1p(len(evidence)),np.log1p(own_count),np.log1p(others),np.log1p(seen),len(evidence)/seen,own_count/(own_count+others),np.mean(list(evidence.values())),np.log1p(object_frames)])
            pair_ids.extend(ids);pair_owner.extend([owner]*len(ids));vectors.extend(np.column_stack([geom,np.asarray(temporal)]))
        sid=np.asarray(pair_ids,np.int64);owner=np.asarray(pair_owner,np.int64);x=np.asarray(vectors,np.float32).reshape(-1,len(FEATURES))
        order=np.lexsort((owner,sid));sid,owner,x=sid[order],owner[order],x[order];sources,inverse,count=np.unique(sid,return_inverse=True,return_counts=True)
        width=int(max(count,default=1));shape=(len(sources),width);valid=np.zeros(shape,bool);owners=np.zeros(shape,np.int64);tensor=np.zeros(shape+(len(FEATURES),),np.float32)
        slot=np.arange(len(sid))-np.repeat(np.r_[0,np.cumsum(count)[:-1]],count)
        valid[inverse,slot]=True;owners[inverse,slot]=owner;tensor[inverse,slot]=x
        assert np.isfinite(tensor).all() and not iscore[sources].any()
        folder=OUT/'features'/key;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'candidates.npz',features=tensor,valid=valid,owners=owners,source_ids=sources)
        dump(folder/'core_sources.json',core_table)
        for f in folder.iterdir():files[str(f)]=sha(f)
        stats[key]=dict(candidate_sources=len(sources),candidate_pairs=len(sid),multi_owner_sources=int((count>1).sum()),max_candidates=width,core_sources=len(core_ids),seconds=time.monotonic()-start)
        print('FEATURES',key,stats[key],flush=True)
    dump(OUT/'features_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json')))


def labels():
    from evaluate_kitti360_development import transfer,read_static_ply
    p=verify();sealed('features_complete');assert not (OUT/'training_labels_complete.json').exists();files={};stats={}
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    for s in p['scenes']:
        key=s['key']
        if not s['reference']:continue
        info=p['references'][key];assert sha(info['raw'])==info['sha256'] and sha(info['evaluation_protocol'])==info['protocol_sha256']
        ref=np.load(s['reference']);raw=read_static_ply(info['raw']);assert len(raw)==len(ref['xyz'])
        train=np.array([taxonomy[int(v)].trainId for v in raw['semantic']],np.int64)
        expected=train*1000+raw['instance'].astype(np.int64)%1000;expected[train==255]=0
        np.testing.assert_array_equal(expected,ref['train_instance'])
        xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];nearest=transfer(ref['xyz'],np.arange(1,len(raw)+1),xyz,.2);known=nearest>0;index=np.maximum(nearest-1,0)
        known&=(train[index]!=255)&(ref['confidence'][index]>0);truth=ref['train_instance'][index];weight=ref['confidence'][index]
        core=read(OUT/'features'/key/'core_sources.json');rows=read(PRIOR/'maps'/key/'largest'/'classification.json')['objects'];anchors={};purities={}
        for row in rows:
            oid=row['object_id'];ids=np.asarray(core[str(oid)],np.int64);ids=ids[known[ids]];target=-1;purity=0.
            if len(ids):
                mass=np.bincount(truth[ids],weights=weight[ids]);candidate=np.flatnonzero((np.arange(len(mass))//1000==row['predictions']['mean']['train_id'])&(np.arange(len(mass))>=1000))
                if len(candidate):
                    best=int(candidate[np.argmax(mass[candidate])]);purity=float(mass[best]/mass.sum())
                    if purity>.5:target=best
            anchors[oid]=target;purities[oid]=purity
        d=np.load(OUT/'features'/key/'candidates.npz');ids=d['source_ids'];valid=d['valid'];owners=d['owners'];at=np.full(owners.shape,-1,np.int64)
        for oid,target in anchors.items():at[owners==oid]=target
        eligible=known[ids]&np.all((at>=0)|~valid,axis=1);positive=(at==truth[ids,None])&valid
        folder=OUT/'training_labels'/key;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'TRAINING_ONLY.npz',eligible=eligible,positive=positive,weight=np.where(eligible,weight[ids],0.))
        dump(folder/'anchors.json',dict(gt_by_owner=anchors,purity_by_owner=purities,training_only=True))
        for f in folder.iterdir():files[str(f)]=sha(f)
        stats[key]=dict(eligible=int(eligible.sum()),positive_sources=int((eligible&positive.any(1)).sum()),null_sources=int((eligible&~positive.any(1)).sum()),unreliable_or_unknown_sources=int((~eligible).sum()),known_anchors=sum(v>=0 for v in anchors.values()),total_anchors=len(anchors))
        print('TRAINING LABELS',key,stats[key],flush=True)
    dump(OUT/'training_labels_complete.json',dict(complete=True,files=files,stats=stats))


def train():
    import torch
    p=verify();sealed('features_complete');sealed('training_labels_complete');assert not (OUT/'trained.json').exists();files={};stats={}
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for scene in p['scenes']:
        key=scene['key'];training=[k for k in OUTDOOR if k!=key];assert key not in training
        for method in ['learned_geometry','independent','ours_competitive']:
            torch.manual_seed(0);rng=np.random.RandomState(0);dimensions=GEOMETRY_FEATURES if method=='learned_geometry' else len(FEATURES)
            parts=[];max_width=0
            for k in training:
                d=np.load(OUT/'features'/k/'candidates.npz');y=np.load(OUT/'training_labels'/k/'TRAINING_ONLY.npz');keep=y['eligible'];max_width=max(max_width,d['valid'].shape[1])
                parts.append((d['features'][keep,:,:dimensions],d['valid'][keep],y['positive'][keep],y['weight'][keep]))
            arrays=[]
            for i in range(4):
                values=[]
                for part in parts:
                    a=part[i]
                    if i<3:a=np.pad(a,[(0,0),(0,max_width-a.shape[1])]+([(0,0)] if i==0 else []))
                    values.append(a)
                arrays.append(np.concatenate(values))
            xx,v,positive,w=arrays;assert len(xx)>0 and positive.any() and (~positive.any(1)).any()
            mean=xx[v].mean(0);std=np.maximum(xx[v].std(0),1e-6);xx=(xx-mean)/std;w=w/w.mean()
            x=torch.tensor(xx);valid=torch.tensor(v);pos=torch.tensor(positive);weight=torch.tensor(w,dtype=torch.float32)
            model=OwnershipScorer(dimensions);optimizer=torch.optim.Adam(model.parameters(),lr=.001);curve=[];start=time.monotonic()
            for epoch in range(40):
                order=rng.permutation(len(x));loss_sum=0.
                for lo in range(0,len(order),2048):
                    ix=order[lo:lo+2048];z=model(x[ix]);vb=valid[ix];pb=pos[ix]
                    if method=='independent':loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,pb.float(),reduction='none')*vb).sum(1)/vb.sum(1)
                    else:loss=grouped_loss(z,vb,pb)
                    objective=(loss*weight[ix]).mean();optimizer.zero_grad();objective.backward();optimizer.step();loss_sum+=float(objective.detach())*len(ix)
                curve.append(loss_sum/len(x))
            folder=OUT/'models'/key/method;folder.mkdir(parents=True,exist_ok=False)
            torch.save(dict(state=model.state_dict(),mean=mean,std=std,dimensions=dimensions,training=training,method=method),folder/'model.pt')
            stat=dict(training=training,groups=len(x),positive_groups=int(positive.any(1).sum()),epochs=40,loss=curve,seconds=time.monotonic()-start)
            dump(folder/'training.json',stat)
            for f in folder.iterdir():files[str(f)]=sha(f)
            stats[key+'/'+method]=stat;print('TRAINED',key,method,'n',len(x),'loss',curve[0],curve[-1],'sec',stat['seconds'],flush=True)
    dump(OUT/'trained.json',dict(complete=True,files=files,stats=stats))


def predict():
    import torch
    p=verify(True);sealed('features_complete');sealed('trained');assert not (OUT/'predictions_complete.json').exists();files={};stats={};torch.set_num_threads(1)
    for scene in p['scenes']:
        key=scene['key'];d=np.load(OUT/'features'/key/'candidates.npz');features=d['features'];valid=d['valid'];ids=d['source_ids'];owners=d['owners']
        core=np.load(PRIOR/'maps'/key/'largest'/'geometry.npz');classification=read(PRIOR/'maps'/key/'largest'/'classification.json');xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz']
        for method in METHODS:
            start=time.monotonic();chosen=np.zeros(len(ids),np.int64);prob=np.zeros(len(ids));model_hash=None
            if method=='nearest_core':
                distance=np.where(valid,np.expm1(features[:,:,0]),np.inf);j=distance.argmin(1);best=distance[np.arange(len(j)),j];unique=(distance==best[:,None]).sum(1)==1
                chosen=np.where((best<=.4)&unique,owners[np.arange(len(j)),j],0)
            elif method!='baseline':
                path=OUT/'models'/key/method/'model.pt';m=torch.load(path,map_location='cpu',weights_only=False);assert key not in m['training'];model=OwnershipScorer(m['dimensions']);model.load_state_dict(m['state']);model.eval();model_hash=sha(path)
                z=[]
                with torch.no_grad():
                    for lo in range(0,len(ids),4096):z.append(model(torch.tensor((features[lo:lo+4096,:,:m['dimensions']]-m['mean'])/m['std'])).numpy())
                logits=np.concatenate(z) if z else np.empty(valid.shape);chosen,prob=allocate(logits,valid,owners,independent=method=='independent')
            result=append_owned(core['xyz'],core['instance'],xyz,ids,chosen);n=len(core['xyz'])
            assert np.array_equal(result['xyz'][:n],core['xyz']) and np.array_equal(result['instance'][:n],core['instance'])
            folder=OUT/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False);np.savez_compressed(folder/'geometry.npz',**result);np.savez_compressed(folder/'decisions.npz',source_ids=ids,owners=chosen,confidence=prob)
            dump(folder/'classification.json',classification);stat=dict(complete=True,objects=len(classification['objects']),old_points=n,added_points=int((chosen>0).sum()),new_map_points=len(result['xyz']),core_bitwise_equal=True,exclusive_source_assignment=True,model_sha256=model_hash,seconds=time.monotonic()-start)
            dump(folder/'run.json',stat)
            for f in folder.iterdir():files[str(f)]=sha(f)
            stats[key+'/'+method]=stat;print('PREDICTED',key,method,'added',stat['added_points'],flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json'),trained_seal_sha256=sha(OUT/'trained.json')))


def score():
    from omni_semantic_map.kitti360_evaluation import evaluate
    from evaluate_kitti360_development import transfer,clean_json
    p=verify();sealed('predictions_complete');assert not (OUT/'summary.json').exists();results={};means={m:[] for m in METHODS};no_harm=True;gains=0
    parent=read(PRIOR/'summary.json')
    for scene in p['scenes']:
        key=scene['key'];methods={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;row=dict(run=read(folder/'run.json'))
            if scene['reference']:
                ref=np.load(scene['reference']);g=np.load(folder/'geometry.npz');pred=transfer(g['xyz'],g['instance'],ref['xyz'],.2);cls=read(folder/'classification.json')['objects']
                result=evaluate(ref['train_instance'],ref['confidence'],pred,{r['object_id']:r['predictions']['mean'] for r in cls},ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
                row.update(metrics=clean_json(result['averages']),matched_gt_ids=sorted(matched_ids(result['matches'])));dump(folder/'matches.json',clean_json(result['matches']));means[method].append(result['averages']['allAp'])
                if method=='baseline':assert row['metrics']==parent['results'][key]['methods']['largest']['metrics']
                base=set(parent['results'][key]['methods']['largest']['matched_gt_ids']);current=set(row['matched_gt_ids']);row['changes_vs_baseline']=dict(gained=sorted(current-base),lost=sorted(base-current))
                print('SCORED',key,method,'AP',row['metrics']['allAp'],'AP50',row['metrics']['allAp50%'],'changes',row['changes_vs_baseline'],flush=True)
            methods[method]=row
        if scene['reference']:
            a=methods['ours_competitive'];b=methods['baseline'];no_harm&=not a['changes_vs_baseline']['lost'] and all(a['metrics'][m]>=b['metrics'][m]-1e-12 for m in ['allAp','allAp50%']);gains+=len(a['changes_vs_baseline']['gained'])
        results[key]=dict(methods=methods)
    mean={m:float(np.mean(v)) for m,v in means.items()};gate=bool(no_harm and gains>0 and all(mean['ours_competitive']>=mean[c]+.02 for c in ['baseline','nearest_core']) and all(mean['ours_competitive']>mean[c] for c in ['learned_geometry','independent']))
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,advance_gate=gate,no_harm=bool(no_harm),gained_gt=gains,development_crossfit=True,robot_accuracy_available=False,prediction_seal_sha256=sha(OUT/'predictions_complete.json')))
    print('ADVANCE',gate,mean,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','features','labels','train','predict','score']);globals()[parser.parse_args().action]()
