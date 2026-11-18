"""Bounded training comparison: point credit versus full-map spatial credit.

Uses the CURRENT strong map, not the weaker historical parent. This stage
tests geometry admission, not new-ID discovery or a completed SLAM method.
"""
import argparse
from pathlib import Path
import runpy
import time
import numpy as np
from paper_final_common import ROOT,read,dump,sha

B=ROOT/'build/benchmarks'
OUT=B/'map_credit_training_v1'
POOL=B/'object_context_acceptance_v1'
METHODS=['point','object_balanced','mean_neighbor','ordered_map']
EVALUATOR=ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['inputs'],**p['sources']}.items():assert sha(f)==h,f
    return p


def freeze():
    assert not OUT.exists()
    contexts=[c for c in read(B/'joint_semantic_revision_v1/protocol.json')['contexts'] if c['kind']=='outdoor']
    old=read(POOL/'protocol.json');inputs={}
    for c in contexts:
        c['raw_reference']=old['references'][c['key']]['raw']
        c['features']=str(POOL/'features'/c['key']/'candidates.npz')
        paths=[c['features'],c['reference'],c['raw_reference'],Path(c['ours'])/'geometry.npz',
               Path(c['ours'])/'classification.json',Path(c['full'])/'geometry.npz',Path(c['full'])/'old_to_full.npy']
        for f in paths:inputs[str(f)]=sha(f)
    paths=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/map_acceptance_objective.py',
           ROOT/'omni_semantic_map/omni_semantic_map/spatial_acceptance_readout.py',
           ROOT/'omni_semantic_map/omni_semantic_map/surface_ownership.py',
           ROOT/'omni_semantic_map/test/test_map_acceptance_objective.py',
           ROOT/'scripts/benchmarks/evaluate_kitti360_development.py',
           ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',EVALUATOR,
           ROOT/'build/datasets/kitti360_prepare/labels.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),contexts=contexts,methods=METHODS,inputs=inputs,
        sources={str(f):sha(f) for f in paths},
        hypothesis='The effect of accepting a measured source depends on closer competing accepted sources and the retained map. Training through the actual spatial readout can improve object geometry beyond source classification or equal-supervision mean-neighbor readout.',
        baseline='Current frozen batch-stable-transport ours_frame (AP about28.012 mean), not historical23.134 reference. Preserve all baseline coordinates,IDs,classes,scores. Shared raw proposal/50D features from earlier exposed object-context candidate pool; remove sources<=5cm from CURRENT baseline. No GT-dependent candidate choice.',
        network='All four arms same50->32->32->1 MLP, categorical null-logit0,training-only valid-feature mean/std,seed0,Adam1e-3,120full-batch updates,last step. Same initialization per training window. No checkpoint/threshold/seed/epoch search. GPUFP32network,FP64categorical probabilities and map moments; seed does not promise bitwise CUDA scatter reproducibility.',
        train_split='Two EXPOSED development directions: train0000 predict0002,train0002 predict0000. All training labels/positions restricted to named training window. Inherited upstream crossfit stacking is NOT leakage-free nested validation; neither window is independent. No robot/test reference used for fitting.',
        target_matching='TRAINING ONLY: class-constrained maximum-total-IoU Hungarian assignment between current baseline predicted IDs and eligible building/car GT (confidence mass>=100), retain positive-IoU pairs only. Freeze once before any fit. Unmatched predictions remain unmatched, not omitted; unusedGT cannot be discovered by this fixed-ID candidate pool.',
        labels='Original raw taxonomy checks reference encoding. Ignore train255 and zero confidence; preserve roadtrain0. Point targets use nearest TRAINING reference<20cm and matching target owner or null. Macro-object control balances desired owner including null. Unknown source labels excluded from these two point losses, never made negative.',
        map_loss='Same full known TRAINING reference for mean_neighbor and ordered_map. Equal-predicted-object loss: matched1-I/(GTmass+M-I); unmatchedM/(M+s),s=mean eligible trainingGTmass of that class (1if no eligibleGT). Fixed GT matching, no test-time labels or reference positions. This is a ratio-of-expected-moments surrogate, NOT exact expected IoU/AP. Unmatched GT contributes no learnable discovery objective in this stage.',
        mechanism_control='ordered_map exact first-accepted-source marginal with baseline fallback. mean_neighbor uses uniform mixture over all nearer source choices and their baseline fallback, removing nearest-source competition while using IDENTICAL labels,coordinates,network,candidates and loss.',
        export='Joint candidate posterior>0.5,exact ties abstain,at mostoneowner per source. Reject means absent,not a spatialnull point. Append ONLY original measured coordinates; baseline intact. Scores and classes never reranked. This stage does not revise old IDs,split/merge or create newIDs; those mainline requirements remain open.',
        budget='Eight fits (two directions x four losses),eight new maps,two baseline re-evaluations;zero new foundation-model calls. No other parameters,postprocessors or scenes to rescue results. Robot/new independent assessment remains a later required stage,not counted complete here.',
        gate='Ordered candidate must not reduce AP/AP50 in either public window,meanAP>=currentbaseline+0.001 and strictly>all threecontrols,at least1new AP50match and0lost,actual ownership difference from mean_neighbor. Passing is development evidence only. Failing closes this configuration without tuning.',
        scope='Complete-map geometry-admission experiment within the larger persistent object-mapping goal. Does NOT replace original objective with this fixed-ID subproblem. Differentiable readout/IoU losses themselves not claimed novel; effective object-map mechanism and independent benefit remain unproven.'))
    print('MAP CREDIT FROZEN',flush=True)


def prepare():
    from scipy.spatial import cKDTree
    from scipy.optimize import linear_sum_assignment
    from evaluate_kitti360_development import read_static_ply
    from omni_semantic_map.spatial_acceptance_readout import build_readout
    p=verify();assert not (OUT/'prepared.json').exists();files={};results={}
    tax=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    for c in p['contexts']:
        tick=time.monotonic();key=c['key'];features=np.load(c['features']);base=np.load(Path(c['ours'])/'geometry.npz')
        mapping=np.load(Path(c['full'])/'old_to_full.npy');full=np.load(Path(c['full'])/'geometry.npz')
        fullids=mapping[features['source_ids']];xyz=full['xyz'][fullids]
        keep=cKDTree(base['xyz']).query(xyz,workers=1)[0]>.05
        src=fullids[keep];xyz=xyz[keep];owners=features['owners'][keep];valid=features['valid'][keep]
        records=read(Path(c['ours'])/'classification.json')['objects'];oids=np.array([r['object_id'] for r in records],np.int64)
        assert np.isin(owners,np.r_[0,oids]).all()
        ref=np.load(c['reference']);raw=read_static_ply(c['raw_reference'])
        train=np.array([tax[int(v)].trainId for v in raw['semantic']],np.int64)
        encoded=train*1000+raw['instance'].astype(np.int64)%1000;encoded[train==255]=0
        np.testing.assert_array_equal(encoded,ref['train_instance'])
        truth=ref['train_instance'];weight=ref['confidence'].astype(float)*(train!=255)
        adj=build_readout(base['xyz'],base['instance'],xyz,ref['xyz'],.2)
        gt_mass=np.bincount(truth,weights=weight);pred_mass=np.bincount(adj['fallback'],weights=weight,minlength=int(oids.max())+1)
        target=np.full(len(pred_mass),-1,np.int64);targetmass=np.zeros(len(target));scale=np.ones(len(target));matching=[]
        for cls in [2,13]:
            old=np.array([r['object_id'] for r in records if r['predictions']['mean']['train_id']==cls],np.int64)
            gt=np.flatnonzero((np.arange(len(gt_mass))//1000==cls)&(gt_mass>=100))
            scale[old]=float(np.mean(gt_mass[gt])) if len(gt) else 1.
            score=np.zeros((len(old),len(gt)))
            for j,oid in enumerate(old):
                inter=np.bincount(truth,weights=weight*(adj['fallback']==oid),minlength=len(gt_mass))[gt]
                score[j]=inter/np.maximum(1e-12,gt_mass[gt]+pred_mass[oid]-inter)
            if len(old) and len(gt):
                ii,jj=linear_sum_assignment(-score)
                for i,j in zip(ii,jj):
                    if score[i,j]<=0:continue
                    target[old[i]]=gt[j];targetmass[old[i]]=gt_mass[gt[j]]
                    matching.append(dict(owner=int(old[i]),gt=int(gt[j]),baseline_iou=float(score[i,j])))
        dist,nearest=cKDTree(ref['xyz']).query(xyz,workers=1)
        eligible=(dist<.2)&(weight[nearest]>0);positive=(target[owners]==truth[nearest,None])&valid&(target[owners]>=0)
        assert np.all(positive.sum(1)<=1)
        y=np.where(positive.any(1),1+positive.argmax(1),0).astype(np.int64)
        desired=np.where(y>0,owners[np.arange(len(y)),np.maximum(y-1,0)],0)
        pointweight=np.where(eligible,weight[nearest],0.)
        groupmass=np.bincount(desired,weights=pointweight,minlength=len(target))
        balanced=np.divide(pointweight,groupmass[desired],out=np.zeros_like(pointweight),where=groupmass[desired]>0)
        assert pointweight.sum()>0 and balanced.sum()>0
        folder=OUT/'inputs'/key;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'candidates.npz',source_ids=src,xyz=xyz,owners=owners,valid=valid,features=features['object_features'][keep])
        np.savez_compressed(folder/'TRAINING_ONLY.npz',truth=truth,weight=weight,target=target,target_mass=targetmass,
                            unmatched_scale=scale,object_ids=oids,y=y,point_weight=pointweight,balanced_weight=balanced)
        np.savez_compressed(folder/'adjacency.npz',**adj)
        dump(folder/'matching_TRAINING_ONLY.json',dict(matches=matching,unmatched=sorted(set(oids.tolist())-{r['owner'] for r in matching})))
        stats=dict(candidate_sources=len(src),candidate_pairs=int(valid.sum()),reference_positions=len(truth),
                   affected_reference_positions=int((np.diff(adj['offsets'])>0).sum()),adjacency_entries=len(adj['sources']),
                   point_labeled_sources=int(eligible.sum()),matched_objects=len(matching),unmatched_objects=len(oids)-len(matching),
                   seconds=time.monotonic()-tick)
        dump(folder/'run.json',stats)
        for f in folder.iterdir():files[str(f)]=sha(f)
        results[key]=stats;print('MAP CREDIT PREPARED',key,stats,flush=True)
    dump(OUT/'prepared.json',dict(complete=True,files=files,results=results,reference_read_for_training_only=True))


def train(target_key):
    import torch
    from omni_semantic_map.surface_ownership import OwnershipScorer
    from omni_semantic_map.map_acceptance_objective import compile_moments,moments,object_loss
    p=verify();prep=read(OUT/'prepared.json');train_key=next(c['key'] for c in p['contexts'] if c['key']!=target_key)
    folder=OUT/'models'/target_key;folder.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    free,total=torch.cuda.mem_get_info();assert free>=16*1024**3
    torch.cuda.set_per_process_memory_fraction(14*1024**3/total)
    dump(folder/'execution.json',dict(started_epoch=time.time(),training=[train_key],target=target_key,free_bytes=free,torch=torch.__version__))
    root=OUT/'inputs'/train_key
    for name in ['candidates.npz','TRAINING_ONLY.npz','adjacency.npz']:
        f=root/name;assert sha(f)==prep['files'][str(f)]
    d=np.load(root/'candidates.npz');y=np.load(root/'TRAINING_ONLY.npz');adj=dict(np.load(root/'adjacency.npz'))
    mean=d['features'][d['valid']].mean(0);std=np.maximum(d['features'][d['valid']].std(0),1e-6)
    x=torch.as_tensor((d['features']-mean)/std,device='cuda');valid=torch.as_tensor(d['valid'],device='cuda')
    labels=torch.as_tensor(y['y'],device='cuda');pw=torch.as_tensor(y['point_weight'],device='cuda');bw=torch.as_tensor(y['balanced_weight'],device='cuda')
    compiled=compile_moments(adj,y['truth'],y['weight'],y['target'],'cuda')
    files={};results={}
    for method in METHODS:
        torch.manual_seed(0);model=OwnershipScorer(50).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        curve=[];tick=time.monotonic()
        for step in range(120):
            z=model(x).masked_fill(~valid,-torch.inf).double()
            z=torch.cat([torch.zeros_like(z[:,:1]),z],1);logp=z.log_softmax(1)
            if method in ['point','object_balanced']:
                weight=pw if method=='point' else bw
                loss=-(logp[torch.arange(len(labels),device='cuda'),labels]*weight).sum()/weight.sum()
            else:
                prob=logp.exp()[:,1:]
                mass,inter=moments(compiled,prob,d['owners'],'ordered' if method=='ordered_map' else 'mean_neighbor')
                loss=object_loss(mass,inter,y['object_ids'],y['target_mass'],y['unmatched_scale'])
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward()
            assert all(torch.isfinite(v.grad).all() for v in model.parameters())
            optimizer.step();curve.append(float(loss.detach()))
            if (step+1)%40==0:print('MAP CREDIT TRAIN',target_key,method,step+1,curve[-1],flush=True)
        dest=folder/method;dest.mkdir()
        torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=mean,std=std,
                        dimensions=50,training=[train_key],method=method),dest/'model.pt')
        stats=dict(training=[train_key],steps=120,loss=curve,seconds=time.monotonic()-tick,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
        dump(dest/'training.json',stats);results[method]=stats
        for f in dest.iterdir():files[str(f)]=sha(f)
    dump(folder/'complete.json',dict(complete=True,files=files,results=results,target=target_key,training=[train_key]))


def predict():
    import torch
    from omni_semantic_map.surface_ownership import OwnershipScorer,allocate
    p=verify();assert not (OUT/'predictions_complete.json').exists();files={};results={}
    torch.set_num_threads(2)
    for c in p['contexts']:
        key=c['key'];f=OUT/'models'/key/'complete.json';trained=read(f);assert trained['complete']
        for path,h in trained['files'].items():assert sha(path)==h
        d=np.load(OUT/'inputs'/key/'candidates.npz');base=np.load(Path(c['ours'])/'geometry.npz')
        results[key]={}
        for method in METHODS:
            m=torch.load(OUT/'models'/key/method/'model.pt',map_location='cpu',weights_only=False)
            assert m['training']==[k['key'] for k in p['contexts'] if k['key']!=key]
            model=OwnershipScorer(50);model.load_state_dict(m['state'],strict=True);model.eval()
            with torch.no_grad():z=model(torch.tensor((d['features']-m['mean'])/m['std'])).numpy()
            owner,confidence=allocate(z,d['valid'],d['owners']);take=owner>0
            folder=OUT/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(folder/'geometry.npz',xyz=np.concatenate([base['xyz'],d['xyz'][take]]),
                                instance=np.r_[base['instance'],owner[take]],added_full_source_ids=d['source_ids'][take])
            (folder/'classification.json').write_bytes((Path(c['ours'])/'classification.json').read_bytes())
            np.savez_compressed(folder/'decisions.npz',owners=owner,confidence=confidence,source_ids=d['source_ids'],logits=z)
            results[key][method]=dict(added_sources=int(take.sum()),new_objects=0,changed_old_owners=0)
            dump(folder/'run.json',results[key][method])
            for path in folder.iterdir():files[str(path)]=sha(path)
            print('MAP CREDIT EXPORTED',key,method,results[key][method],flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,results=results,sealed_before_new_ap=True,
        new_maps=8,new_objects=0,scope='Fixed-ID geometry admission only; full object formation still pending'))


def score():
    from evaluate_kitti360_development import transfer,clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p=verify();seal=read(OUT/'predictions_complete.json');assert seal['complete'] and not (OUT/'summary.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h
    results={};means={m:[] for m in ['baseline']+METHODS};gate=True;gained=lost=0
    for c in p['contexts']:
        key=c['key'];ref=np.load(c['reference']);rows={}
        for method in ['baseline']+METHODS:
            folder=Path(c['ours']) if method=='baseline' else OUT/'maps'/key/method
            g=np.load(folder/'geometry.npz');records=read(folder/'classification.json')['objects']
            pred=transfer(g['xyz'],g['instance'],ref['xyz'],.2)
            e=evaluate(ref['train_instance'],ref['confidence'],pred,{o['object_id']:o['predictions']['mean'] for o in records},EVALUATOR)
            dest=OUT/'evaluation'/key/method;dest.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(dest/'reference_prediction.npz',instance=pred)
            dump(dest/'matches.json',clean_json(e['matches']))
            rows[method]=dict(metrics=clean_json(e['averages']),matched_gt_ids=sorted(matched_ids(e['matches'])))
            means[method].append(float(e['averages']['allAp']));print('MAP CREDIT SCORE',key,method,rows[method]['metrics'],flush=True)
        old=read(B/'joint_semantic_revision_v1/summary.json')['results'][key]['methods']['baseline']['metrics']
        assert abs(old['allAp']-rows['baseline']['metrics']['allAp'])<1e-12
        before=set(rows['baseline']['matched_gt_ids'])
        for v in rows.values():
            after=set(v['matched_gt_ids']);v['changes_vs_baseline']=dict(gained=sorted(after-before),lost=sorted(before-after))
        change=rows['ordered_map']['changes_vs_baseline'];gained+=len(change['gained']);lost+=len(change['lost'])
        for metric in ['allAp','allAp50%']:gate &= rows['ordered_map']['metrics'][metric]>=rows['baseline']['metrics'][metric]
        a=np.load(OUT/'maps'/key/'ordered_map/decisions.npz')['owners'];b=np.load(OUT/'maps'/key/'mean_neighbor/decisions.npz')['owners']
        different=int((a!=b).sum());gate &= different>0
        results[key]=dict(methods=rows,ownership_changes_vs_mean_neighbor=different)
    mean={m:float(np.mean(v)) for m,v in means.items()}
    gate &= mean['ordered_map']>=mean['baseline']+.001 and mean['ordered_map']>max(v for m,v in mean.items() if m!='ordered_map') and gained>=1 and lost==0
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,development_gate=bool(gate),gained=gained,lost=lost,
        new_fits=8,new_maps=8,official_leaderboard=False,independent_test=False,robot=False,goal_complete=False))
    print('MAP CREDIT DEVELOPMENT GATE',bool(gate),mean,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','train','predict','score']);ap.add_argument('--target')
    args=ap.parse_args();train(args.target) if args.action=='train' else globals()[args.action]()
