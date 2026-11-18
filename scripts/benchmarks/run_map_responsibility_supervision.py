"""Finite same-input loss comparison for measured-point object mapping.

Three losses, six fits, eight complete maps. Training references are confined
to prepare(); train(target) only loads the other window's prepared labels.
Both windows are exposed development data with inherited parent supervision.
"""
import argparse, time, runpy, shutil
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from paper_final_common import ROOT, read, dump, sha
from evaluate_kitti360_development import transfer, read_static_ply, clean_json
from omni_semantic_map.surface_ownership import OwnershipScorer, allocate, append_owned
from omni_semantic_map.map_responsibility_supervision import categorical, source_labels, marginal_nll, normalized_weights, CompiledReadout, select_queries

OUT=ROOT/'build/benchmarks/map_responsibility_supervision_v1'
PARENT=ROOT/'build/benchmarks/object_context_acceptance_v1'
AUDIT=ROOT/'build/benchmarks/spatial_readout_audit_v1'
INPUT=ROOT/'build/benchmarks/discovery_measured_inputs_v1'
METHODS=['baseline','source_nll','balanced_source_nll','ours_spatial_nll']
KEYS=['kitti360_0000','kitti360_0002']
EVALUATOR=ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'


def freeze():
    assert not OUT.exists(); p=read(PARENT/'protocol.json'); inputs={}
    for root, names in [(PARENT,['protocol.json','features_complete.json','predictions_complete.json']),
                        (AUDIT,['protocol.json','replay_complete.json','attribution_complete.json'])]:
        for name in names:
            f=root/name; d=read(f); inputs[str(f)]=sha(f)
            for k,h in d.get('files',{}).items(): assert sha(k)==h; inputs[k]=h
    for scene in p['scenes']:
        key=scene['key']; manifest=read(INPUT/key/'manifest.json')
        for f in [scene['reference'],p['references'][key]['raw'],INPUT/key/'manifest.json',manifest['geometry']]:inputs[str(f)]=sha(f)
    for f in [EVALUATOR,ROOT/'build/datasets/kitti360_prepare/labels.py']:inputs[str(f)]=sha(f)
    paths=[Path(__file__),ROOT/'scripts/benchmarks/run_object_context_acceptance.py',ROOT/'scripts/benchmarks/evaluate_kitti360_development.py']
    paths += [ROOT/'omni_semantic_map/omni_semantic_map'/n for n in ['map_responsibility_supervision.py','spatial_acceptance_readout.py','surface_ownership.py','kitti360_evaluation.py']]
    paths += [ROOT/'omni_semantic_map/test'/n for n in ['test_map_responsibility_supervision.py','test_map_responsibility_driver.py']]
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=p['scenes'],references=p['references'],methods=METHODS,
         input_hashes=inputs,source_hashes={str(f):sha(f) for f in paths},
         hypothesis='Supervise the expected spatial responsibility of accepted measurements, rather than treating every accepted measurement as an isolated classification. Test whether this improves complete object maps beyond source weighting alone.',
         shared='Exactly the frozen object_context50Dfeatures and candidates, old geometry_graph maps/IDs/classes/scores, MLP50->32->32->1, initialization seed0, Adam1e-3,200 full-batch updates, training-window-only standardization over all valid candidate feature rows. Three independent fits per direction; no pretrained-context initialization, warmup, scheduler, tuning or checkpoint selection.',
         target_assignment='Training only: confidence-weighted whole-baseline projection on nonignored reference. Maximum total baseline IoU one-to-one Hungarian matching to same-class GT with >=100known confidence mass, positive overlap only, dummy columns for unmatched predictions. Deterministic sortedIDs, no IoU threshold. Same fixed matching for all three losses. An unmatched prediction has no positive training target; its new responsibility at known locations is penalized through categorical competition. Existing unmatched core cannot be deleted by this method.',
         unknown='Read raw taxonomy:train255ignored,train0road remains known. Reference confidence<=0ignored. SourceNLL uses nearest known-reference label within strict20cm; missing/ignored/zero-confidence source labels receive zero weight. SpatialNLL uses known reference locations; a source without a nearest valid point label may receive indirect supervision if it affects known locations. Same complete annotated training scene, but different supervision domains by design; not identical labeled-example counts.',
         targets='Reference GT matched to a baseline prediction maps to that owner ID; all other known classes/GT map to null0. Source target additionally becomes null when its GT-matched owner is absent from its proposal set, since rejection is its only correct available action. Training one-to-one identity assignment is shared and is not inference-time relabeling or new-object discovery.',
         losses='source_nll:confidence-weighted categoricalNLL of source object/null probabilities. balanced_source_nll:same, equal total weight per represented source target including null. ours_spatial_nll:equal total weight per represented reference target including null; exact marginal labels under independent categorical source choices and nearest-map readout. Reference target with zero possible probability contributes fixed -log(1e-8),zero gradient; avoids rewarding diffuse wrong labels. Fixed1e-8 numerical floor, not tuned. Full reference constants included in logged loss; gradients chunked8192queries with exact chain rule. NLL is a marginal surrogate, not expected AP/IoU.',
         inference='No reference labels/coordinates/targets at inference. Categorical owner probabilities against fixed null logit0; unique maximum owner probability strictly>.5, else reject. Preserve old geometry/IDs/classification exactly; append unique real measured source coordinates, no synthetic geometry, newIDs, deletion, relabeling, reranking or cleanup.',
         budget='Six200-step fits, two existing windows x four full maps including baseline copies. No new perception, additional prompts, parameter sweep, validation checkpoint choice or robot execution before gate. All eight map artifacts sealed before scoring.',
         gate='Ours AP50:90>=baseline in both windows; mean>=baseline+.01 and strictly>both newly trained controls; >=1gained AP50 GT and zero lost; Ours decisions differ from balanced control in both windows. Failure closes version without parameter retries or independent/robot expansion.',
         scope='Both0000/0002exposed. Parent maps inherit earlier cross-window supervision; these are two-direction development comparisons, NOT leakage-free nested CV, independent tests, official leaderboard or paper-ready generalization. Prior failed branches stay frozen. New mechanism and required effective robot/independent evaluations remain unproven until actual gates pass.'))
    print('FROZEN MAP RESPONSIBILITY SUPERVISION',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    return p


def seal(name):
    d=read(OUT/(name+'.json'));assert d['complete']
    for f,h in d['files'].items():assert sha(f)==h,f
    return d


def training_assignment(projected, truth, weight, objects):
    """GT-only training assignment. Same-class, one-to-one, positive overlap."""
    objects=sorted(objects,key=lambda o:o['object_id']);oids=np.array([o['object_id'] for o in objects]);mass=np.bincount(truth,weights=weight)
    gt=np.flatnonzero(mass>=100);cost=np.zeros((len(oids),len(gt)+len(oids)));rows=[]
    for i,obj in enumerate(objects):
        own=projected==obj['object_id'];inter=np.bincount(truth[own],weights=weight[own],minlength=len(mass));pmass=weight[own].sum()
        allowed=gt//1000==obj['predictions']['mean']['train_id'];cost[i,:len(gt)]=np.where(allowed,inter[gt]/np.maximum(1e-12,mass[gt]+pmass-inter[gt]),0)
    ii,jj=linear_sum_assignment(-cost);by_owner={int(o):-1 for o in oids}
    for i,j in zip(ii,jj):
        if j<len(gt) and cost[i,j]>0:by_owner[int(oids[i])]=int(gt[j]);rows.append(dict(owner=int(oids[i]),gt=int(gt[j]),iou=float(cost[i,j])))
    return by_owner,rows


def prepare():
    p=verify();assert not (OUT/'prepared.json').exists();files={};stats={};taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    for scene in p['scenes']:
        key=scene['key'];ref=dict(np.load(scene['reference']));raw=read_static_ply(p['references'][key]['raw']);train=np.array([taxonomy[int(s)].trainId for s in raw['semantic']]);expected=train*1000+raw['instance'].astype(np.int64)%1000;expected[train==255]=0
        np.testing.assert_array_equal(expected,ref['train_instance']);weight=np.where(train!=255,ref['confidence'],0).astype(float);assert np.all(weight>=0)
        projected=np.load(AUDIT/key/'baseline_projection.npz')['labels'];objects=read(PARENT/'maps'/key/'baseline/classification.json')['objects'];assigned,rows=training_assignment(projected,ref['train_instance'],weight,objects)
        target=np.zeros(len(weight),np.int64)
        for oid,gid in assigned.items():
            if gid>=0:target[ref['train_instance']==gid]=oid
        d=dict(np.load(PARENT/'features'/key/'candidates.npz'));source=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'][d['source_ids']]
        nearest=transfer(ref['xyz'],np.arange(1,len(weight)+1),source,.2);ix=np.maximum(nearest-1,0);source_weight=np.where(nearest>0,weight[ix],0);source_target=target[ix].copy();available=(d['owners']==source_target[:,None])&d['valid'];source_target[~available.any(1)]=0
        a=dict(np.load(AUDIT/key/'adjacency.npz'));active=(np.diff(a['offsets'])>0)&(weight>0);indices=np.flatnonzero(active);reference_weight=normalized_weights(target,weight,True)
        constant=float((-np.log(1e-8))*reference_weight[(~active)&(projected!=target)].sum())
        folder=OUT/'training_only'/key;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'labels.npz',source_target=source_target,source_weight=source_weight,
                            source_plain=normalized_weights(source_target,source_weight,False),source_balanced=normalized_weights(source_target,source_weight,True),
                            reference_target=target,reference_weight=reference_weight,active_indices=indices,constant_loss=np.asarray(constant))
        dump(folder/'assignment.json',dict(owner_to_gt=assigned,matches=rows,training_only=True))
        stat=dict(candidates=len(source),known_source_labels=int((source_weight>0).sum()),positive_source_labels=int(((source_weight>0)&(source_target>0)).sum()),
                  matched_objects=len(rows),unmatched_objects=len(objects)-len(rows),known_reference_locations=int((weight>0).sum()),active_reference_locations=len(indices),constant_loss=constant)
        dump(folder/'run.json',stat);stats[key]=stat
        for f in folder.iterdir():files[str(f)]=sha(f)
        print('PREPARED',key,stat,flush=True)
    dump(OUT/'prepared.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json')))


def train():
    import torch
    p=verify();seal('prepared');assert not (OUT/'trained.json').exists();assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);files={};stats={};device='cuda:0'
    for target in KEYS:
        key=next(k for k in KEYS if k!=target);d=dict(np.load(PARENT/'features'/key/'candidates.npz'));y=dict(np.load(OUT/'training_only'/key/'labels.npz'));a=dict(np.load(AUDIT/key/'adjacency.npz'))
        xx=d['object_features'];valid=torch.tensor(d['valid'],device=device);owners=torch.tensor(d['owners'],device=device);label_count=max(int(d['owners'].max()),int(a['fallback'].max()))+1
        mean=xx[d['valid']].mean(0);std=np.maximum(xx[d['valid']].std(0),1e-6);x=torch.tensor((xx-mean)/std,device=device);st=torch.tensor(y['source_target'],device=device)
        # Compile once; identical fixed spatial support reused for all iterations.
        chunks=[];indices=y['active_indices']
        for lo in range(0,len(indices),8192):
            ii=indices[lo:lo+8192];chunks.append((CompiledReadout(select_queries(a,ii),d['owners'],label_count,device),
                torch.tensor(y['reference_target'][ii],device=device),torch.tensor(y['reference_weight'][ii],device=device,dtype=torch.float32)))
        for method in METHODS[1:]:
            torch.manual_seed(0);torch.cuda.manual_seed_all(0);model=OwnershipScorer(50).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001);curve=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            sw=torch.tensor(y['source_plain' if method=='source_nll' else 'source_balanced'],device=device,dtype=torch.float32)
            for step in range(200):
                optimizer.zero_grad();prob=categorical(model(x),valid)
                if method=='ours_spatial_nll':
                    leaf=prob.detach().requires_grad_(True);total=float(y['constant_loss'])
                    for operator,tt,ww in chunks:
                        loss=marginal_nll(operator(leaf),tt,ww);loss.backward();total+=float(loss.detach())
                    prob.backward(leaf.grad)
                else:
                    loss=marginal_nll(source_labels(prob,owners,label_count),st,sw);loss.backward();total=float(loss.detach())
                assert np.isfinite(total) and all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
                optimizer.step();curve.append(total)
                if step in [0,49,99,149,199]:print('FIT',target,method,step+1,total,flush=True)
            folder=OUT/'models'/target/method;folder.mkdir(parents=True,exist_ok=False)
            torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},mean=mean,std=std,dimensions=50,training=[key],method=method),folder/'model.pt')
            stat=dict(training=[key],steps=200,loss=curve,seconds=time.monotonic()-start,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30);dump(folder/'training.json',stat)
            for f in folder.iterdir():files[str(f)]=sha(f)
            stats[target+'/'+method]=stat
        del chunks,x,owners,valid;torch.cuda.empty_cache()
    dump(OUT/'trained.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json'),torch_version=torch.__version__,gpu=torch.cuda.get_device_name()))


def predict():
    import torch
    from run_object_context_acceptance import logits
    p=verify();seal('trained');assert not (OUT/'predictions_complete.json').exists();files={};stats={};identifiability={};torch.set_num_threads(1)
    for scene in p['scenes']:
        key=scene['key'];d=dict(np.load(PARENT/'features'/key/'candidates.npz'));source=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];base=dict(np.load(PARENT/'maps'/key/'baseline/geometry.npz'));classification=read(PARENT/'maps'/key/'baseline/classification.json');owners={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False)
            if method=='baseline':
                for name in ['geometry.npz','classification.json']:shutil.copyfile(PARENT/'maps'/key/'baseline'/name,folder/name)
                stat=dict(added_points=0,exact_copy=True)
            else:
                m=torch.load(OUT/'models'/key/method/'model.pt',map_location='cpu',weights_only=False);assert m['training']==[k for k in KEYS if k!=key]
                z=logits(m,d['object_features']);owner,prob=allocate(z,d['valid'],d['owners']);g=append_owned(base['xyz'],base['instance'],source,d['source_ids'],owner)
                np.savez_compressed(folder/'geometry.npz',**g);np.savez_compressed(folder/'decisions.npz',source_ids=d['source_ids'],owners=owner,confidence=prob);dump(folder/'classification.json',classification)
                stat=dict(added_points=int((owner>0).sum()),old_point_ids_changed=0,new_objects=0);owners[method]=owner
            dump(folder/'run.json',stat);stats[key+'/'+method]=stat
            for f in folder.iterdir():files[str(f)]=sha(f)
            print('MAP',key,method,stat,flush=True)
        identifiability[key]=dict(ours_vs_balanced=int((owners['ours_spatial_nll']!=owners['balanced_source_nll']).sum()))
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,identifiability=identifiability,protocol_sha256=sha(OUT/'protocol.json'),sealed_before_scoring=True))


def score():
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p=verify();s=seal('predictions_complete');assert not (OUT/'summary.json').exists();results={};means={m:[] for m in METHODS};gate=True;gained=lost=0
    for scene in p['scenes']:
        key=scene['key'];ref=dict(np.load(scene['reference']));rows={}
        for method in METHODS:
            f=OUT/'maps'/key/method;g=dict(np.load(f/'geometry.npz'));objects=read(f/'classification.json')['objects'];projection=transfer(g['xyz'],g['instance'],ref['xyz'],.2)
            r=evaluate(ref['train_instance'],ref['confidence'],projection,{o['object_id']:o['predictions']['mean'] for o in objects},EVALUATOR)
            dump(f/'matches.json',clean_json(r['matches']));rows[method]=dict(metrics=clean_json(r['averages']),matched_gt_ids=sorted(matched_ids(r['matches'])),run=read(f/'run.json'));means[method].append(float(r['averages']['allAp']))
            print('SCORE',key,method,rows[method]['metrics'],flush=True)
        expected=read(PARENT/'summary.json')['results'][key]['methods']['baseline']['metrics'];assert abs(rows['baseline']['metrics']['allAp']-expected['allAp'])<1e-12
        base=set(rows['baseline']['matched_gt_ids'])
        for r in rows.values():r['changes_vs_baseline']=dict(gained=sorted(set(r['matched_gt_ids'])-base),lost=sorted(base-set(r['matched_gt_ids'])))
        change=rows['ours_spatial_nll']['changes_vs_baseline'];gained+=len(change['gained']);lost+=len(change['lost']);gate&=rows['ours_spatial_nll']['metrics']['allAp']>=rows['baseline']['metrics']['allAp'];gate&=s['identifiability'][key]['ours_vs_balanced']>0
        results[key]=dict(methods=rows)
    mean={m:float(np.mean(v)) for m,v in means.items()};gate&=mean['ours_spatial_nll']>=mean['baseline']+.01 and mean['ours_spatial_nll']>max(v for m,v in mean.items() if m!='ours_spatial_nll') and gained>0 and lost==0
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,advance_gate=bool(gate),gained=gained,lost=lost,development_only=True,official_leaderboard=False,prediction_seal_sha256=sha(OUT/'predictions_complete.json')))
    print('GATE',bool(gate),mean,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','train','predict','score']);globals()[parser.parse_args().action]()
