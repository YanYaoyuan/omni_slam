"""Object-anchored, image-context-disjoint evidence for map membership revision.

All identities and proposals are frozen predictions. This is an offline
window-end revision experiment, not causal online instance discovery.
"""
import argparse,copy,shutil,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha
from audit_mask_observation_code import source_anchors
from omni_semantic_map.persistent_source_memory import sample_patch_features
from omni_semantic_map.provenance_object_memory import (
    ProvenanceObjectMemory,ViewEvidenceAllocator,disjoint_inputs,revision_map)
from omni_semantic_map.intrinsic_surface_geometry import geometry_features
from omni_semantic_map.surface_ownership import allocate,grouped_loss

OUT=ROOT/'build/benchmarks/provenance_object_memory_v1'
PREV=ROOT/'build/benchmarks/persistent_source_memory_v1'
INPUT=ROOT/'build/benchmarks/discovery_measured_inputs_v1'
SEM=ROOT/'build/benchmarks/closed_vocabulary_semantics_v1'
POOL=ROOT/'build/benchmarks/object_context_acceptance_v1'
BASE=ROOT/'build/benchmarks/signed_query_association_v1/maps'
KEYS=['kitti360_0000','kitti360_0002']
TRAINED=['mean_disjoint','retrieval_disjoint','attention_all','ours_revision']
METHODS=['baseline']+TRAINED[:3]+['ours_append_only','ours_revision']
EVALUATOR=ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'


def freeze():
    assert not OUT.exists();parent=read(SEM/'protocol.json');inputs={}
    for root,name in [(PREV,'capture_complete.json'),(POOL,'features_complete.json')]:
        f=root/name;d=read(f);assert d['complete'];inputs[str(f)]=sha(f)
        for path,h in d['files'].items():assert sha(path)==h;inputs[path]=h
    for scene in parent['scenes']:
        key=scene['key'];m=read(INPUT/key/'manifest.json')
        paths=[INPUT/key/'manifest.json',m['geometry'],INPUT/'query_cells'/key/'source_to_cell.npy',
            ROOT/'build/benchmarks/surface_ownership_v1/features'/key/'core_sources.json',
            ROOT/'build/benchmarks/intrinsic_surface_ownership_v1/maps'/key/'intrinsic_geometry/geometry.npz',
            scene['reference'],parent['references'][key]['raw']]
        paths += [BASE/key/'geometry_graph'/n for n in ['geometry.npz','classification.json','groups.json']]
        paths += [SEM/'frames'/key/f'{r["frame_id"]:06d}.npz' for r in m['records']]
        paths += [r['file'] for r in m['records']]
        for f in paths:inputs[str(f)]=sha(f)
    summary=ROOT/'build/benchmarks/signed_query_association_v1/summary.json';inputs[str(summary)]=sha(summary)
    sources=[Path(__file__),ROOT/'scripts/benchmarks/audit_mask_observation_code.py',
        ROOT/'scripts/benchmarks/evaluate_kitti360_development.py',EVALUATOR,
        ROOT/'build/datasets/kitti360_prepare/labels.py']
    sources += [ROOT/'omni_semantic_map/omni_semantic_map'/n for n in
        ['provenance_object_memory.py','persistent_source_memory.py','intrinsic_surface_geometry.py','surface_ownership.py','kitti360_evaluation.py']]
    sources += [ROOT/'omni_semantic_map/test/test_provenance_object_memory.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=parent['scenes'],references=parent['references'],methods=METHODS,
        input_hashes=inputs,source_hashes={str(f):sha(f) for f in sources},
        hypothesis='Condition historical point ownership on preserved, object-identified view evidence. Test whether excluding overlapping VGGT RGB contexts reduces self-confirming revisions beyond mean prototype, nearest-descriptor retrieval and unrestricted learned view attention. Stable provenance alone is not claimed novel.',
        bank='Frozen geometry_graph predicted IDs and exact source provenance, including earlier additions. Conflicting source ownership ignored. For each observed object/frame, unit-normalized VGGT point descriptors averaged with equal20cm-cell mass, then normalized; original source IDs and19-class means retained. Only that object receives this view; absent objects unchanged. No random slots or recursive global mixing. Final predicted IDs may use future upstream observations: offline window-end refinement, NOT online discovery.',
        query='All original fused-map points with measured-visible image support plus the frozen object_context_acceptance raw proposal pool. Map coordinates/namespace separate from original measured-source coordinates/IDs. Last measured-visible RGB patch descriptor and corresponding last-frame SegFormer19-probabilities only. Original invisible map points stay unchanged. No previous learned candidate decisions or GT selected subsets.',
        candidates='Old point keeps its original object as a candidate; up to3nearest OTHER source-anchor objects within0.4m are alternatives. Added measured sources keep exactly the prior unfiltered raw-query proposed owners. No class pre-veto. Current-owner memory must contain a view with disjoint VGGT RGB context; otherwise an old point cannot be revised. Candidate with no disjoint view is unavailable in EVERY learned arm, ensuring common eligibility. Missing evidence abstains, never forces null.',
        disjoint='Each cached descriptor uses causal RGB ranks[max(0,t-3),t]. Bank view g is input-disjoint from query t iff abs(g-t)>=4. This only excludes shared image inputs in these descriptors/last-frame semantics; upstream predicted identity and shared geometry still couple evidence. NOT statistical independence, independent test data or new perception.',
        representation='Static32D=12frozen intrinsic point-to-object/other-anchor geometry +19last-frame semantic probabilities +original-owner flag. Per-view5D=raw VGGT cosine, object-to-camera viewing-direction cosine, log1p(anchor source count), anchor own-class mean probability, log1p(abs(frame-rank gap)). All available object views retained, no best-view/GT selection or view-budget sweep.',
        model='Shared architecture: viewMLP5->16->16, scalar learned view quality, pool encoded16D and quality1D, concatenate static32D, MLP49->32->32->1. mean uses uniform disjoint views; retrieval uses maximal raw descriptor cosine among disjoint views; attention_all uses learned softmax weights on all views of common-eligible candidates; ours uses same learned softmax on disjoint views. Quality is also read by final head in every arm, so all parameters participate. Unknown candidates masked; explicit null logit0.',
        supervision='Training-only one-to-one same-class Hungarian assignment of frozen baseline objects to known reference GT (confidence mass>=100, positive IoU), with dummy unmatched slots. Query targets from nearest nonignored reference within strict20cm, positive confidence; road remains known background, ignored/unmatched geometry unsupervised. A candidate is positive iff its assigned GT equals query GT. If none positive, null is target. Grouped NLL and reference-confidence weights; no target labels in inference.',
        training='8fixed fits:4pooling modes x2directions, train other window only. Seed17,Adam1e-3,40epochs,batch1024,train-only static-channel standardization,view channels unstandardized,final checkpoint only. No new backbone inference, epoch/seed/threshold search or positive resampling. Inherited parent maps used earlier cross-window supervision; NOT nestedCV or generalization.',
        export='Unique best categorical object probability>0.5 else null; exact ties abstain. Shared eligibility means an unseen old point remains original. Revision may remove/reassign named old point IDs and append unique actual source measurements. Existing object IDs/classes/ranking retained (drop only empty objects). No new IDs, SOR,NMS,coordinate synthesis or reclassification. ours_append_only uses identical ours weights/decisions but leaves all original geometry/membership untouched, isolating historical revision.',
        budget='No new foundation-model calls;8fixed fits;2windows x6full maps, including baseline copies. Models/predictions sealed before complete-map scoring. Existing failed global-slot model stays frozen; its features are reused, weights/output labels are not.',
        gate='Ours AP50:90>=baseline eachwindow;mean>=baseline+0.01 and strictly>mean_disjoint,retrieval_disjoint,attention_all,ours_append_only;>=1gained and0lostAP50GT. At least one historical membership changed, all unseen-point/provenance invariants pass. Otherwise close version without rescoring adjusted thresholds/epochs or robot/independent expansion.',
        scope='Fixed-predicted-identity membership revision, not missing-instance discovery or all-error recovery. Two previously exposed outdoor windows; not official leaderboard. Same-version robot and truly new-scene evaluation only after method gate, not claimed complete by input preparation.'))
    print('FROZEN PROVENANCE MEMORY',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    return p


def seal(name):
    p=read(OUT/f'{name}.json');assert p['complete']
    for f,h in p['files'].items():assert sha(f)==h,f
    return p


def gpu():
    import torch
    torch.set_num_threads(4);assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info();assert free>=14*1024**3
    torch.cuda.set_per_process_memory_fraction(12*1024**3/total)
    return torch


def unit_rows(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def prepare():
    p=verify();assert not (OUT/'preparation_execution.json').exists();torch=gpu()
    dump(OUT/'preparation_execution.json',dict(started_epoch=time.time(),no_gt_read=True));files={};stats={}
    for scene in p['scenes']:
        start=time.monotonic();key=scene['key'];m=read(INPUT/key/'manifest.json');records=m['records'];frames=len(records)
        xyz=np.load(m['geometry'])['xyz'];old=np.load(BASE/key/'geometry_graph/geometry.npz');objects=read(BASE/key/'geometry_graph/classification.json')['objects']
        object_ids=np.array(sorted(o['object_id'] for o in objects));classes={o['object_id']:o['predictions']['mean']['train_id'] for o in objects}
        oid_column={int(o):i for i,o in enumerate(object_ids)};anchors=source_anchors(key,len(xyz));cells=np.load(INPUT/'query_cells'/key/'source_to_cell.npy')
        core={o:np.flatnonzero(anchors==o) for o in object_ids};assert all(len(ids) for ids in core.values())
        pool=np.load(POOL/'features'/key/'candidates.npz');new_ids=pool['source_ids'];new_lookup=np.full(len(xyz),-1,int);new_lookup[new_ids]=np.arange(len(new_ids))
        bank=ProvenanceObjectMemory(object_ids,2048);grids=[];map_frame=np.full(len(old['xyz']),-1,int);map_xy=np.zeros((len(map_frame),2),int);map_prob=np.zeros((len(map_frame),19),np.float32)
        new_frame=np.full(len(new_ids),-1,int);new_xy=np.zeros((len(new_ids),2),int);new_prob=np.zeros((len(new_ids),19),np.float32)
        for i,rec in enumerate(records):
            grid=np.load(PREV/'features'/key/f'{rec["frame_id"]:06d}.npz')['grid'];grids.append(grid)
            d=np.load(SEM/'frames'/key/f'{rec["frame_id"]:06d}.npz');sid=d['source_ids'];take=anchors[sid]>0
            f=unit_rows(sample_patch_features(grid,d['source_pixel_xy'][take],(376,1408)))
            bank.observe(i,sid[take],anchors[sid[take]],cells[sid[take]],f,d['source_probabilities'][take])
            mi=d['map_ids'];map_frame[mi]=i;map_xy[mi]=d['map_pixel_xy'];map_prob[mi]=d['map_probabilities']
            ix=new_lookup[sid];take=ix>=0;ix=ix[take]
            new_frame[ix]=i;new_xy[ix]=d['source_pixel_xy'][take];new_prob[ix]=d['source_probabilities'][take]
        assert np.all(new_frame>=0)
        old_indices=np.flatnonzero(map_frame>=0);n_old=len(old_indices)
        query_xyz=np.r_[old['xyz'][old_indices],xyz[new_ids]];query_frame=np.r_[map_frame[old_indices],new_frame]
        query_xy=np.r_[map_xy[old_indices],new_xy];probability=np.r_[map_prob[old_indices],new_prob]
        n=len(query_xyz);width=max(4,pool['owners'].shape[1]);candidate=np.zeros((n,width),np.int64)
        candidate[:n_old,0]=old['instance'][old_indices]
        distances=np.column_stack([cKDTree(xyz[core[o]]).query(query_xyz[:n_old],workers=1)[0] for o in object_ids])
        distances[object_ids[None]==candidate[:n_old,:1]]=np.inf
        nearest=np.argsort(distances,axis=1,kind='stable')[:,:3];dd=np.take_along_axis(distances,nearest,axis=1)
        candidate[:n_old,1:4]=np.where(dd<=.4,object_ids[nearest],0)
        candidate[n_old:,:pool['owners'].shape[1]]=np.where(pool['valid'],pool['owners'],0)
        descriptors=np.zeros((len(object_ids),frames,2048),np.float32);view_valid=np.zeros((len(object_ids),frames),bool)
        meta=np.zeros((len(object_ids),frames,2),np.float32);provenance=[]
        for col,oid in enumerate(object_ids):
            for event in bank.views[int(oid)]:
                i=event['frame'];descriptors[col,i]=event['descriptor'];view_valid[col,i]=True
                meta[col,i]=[np.log1p(event['source_count']),event['semantic'][classes[int(oid)]]]
                provenance.append(dict(object_id=int(oid),frame_index=i,source_ids=event['source_ids'].tolist(),cell_count=event['cell_count']))
        columns=np.zeros_like(candidate)
        for oid,col in oid_column.items():columns[candidate==oid]=col
        available=view_valid[columns]&(candidate>0)[...,None]
        separated=available&disjoint_inputs(query_frame[:,None,None],np.arange(frames)[None,None])
        valid=separated.any(-1)
        # Unknown current-owner view is abstention for the whole old point.
        lock=~valid[:n_old,0];valid[:n_old][lock]=False
        available &= valid[...,None];separated &= valid[...,None]
        static=np.zeros((n,width,32),np.float32);static[:,:,12:31]=probability[:,None]
        static[:n_old,0,31]=1
        allcore=np.flatnonzero(anchors>0)
        for oid in object_ids:
            ii,jj=np.where((candidate==oid)&valid)
            if not len(ii):continue
            static[ii,jj,:12]=geometry_features(query_xyz[ii],xyz[core[oid]],xyz[allcore[anchors[allcore]!=oid]],11 if classes[int(oid)]==2 else 26)
        centers=np.array([xyz[core[o]].mean(0) for o in object_ids]);camera=np.array([r['camera_pose'] for r in records])[:,:3,3]
        view_directions=unit_rows(camera[None]-centers[:,None])
        view_features=np.zeros((n,width,frames,5),np.float16)
        bank_tensor=torch.tensor(descriptors.reshape(-1,2048).T,device='cuda')
        with torch.inference_mode():
            for lo in range(0,n,2048):
                hi=min(lo+2048,n);q=np.zeros((hi-lo,2048),np.float32)
                for fr in np.unique(query_frame[lo:hi]):
                    take=query_frame[lo:hi]==fr;q[take]=sample_patch_features(grids[fr],query_xy[lo:hi][take],(376,1408))
                q=unit_rows(q);cos=(torch.tensor(q,device='cuda')@bank_tensor).cpu().numpy().reshape(hi-lo,len(object_ids),frames)
                cols=columns[lo:hi];r=np.arange(hi-lo)[:,None]
                view_features[lo:hi,:,:,0]=cos[r,cols]
                direction=unit_rows(camera[query_frame[lo:hi],None]-centers[cols])
                view_features[lo:hi,:,:,1]=np.einsum('bkd,bkvd->bkv',direction,view_directions[cols])
                view_features[lo:hi,:,:,2:4]=meta[cols]
                view_features[lo:hi,:,:,4]=np.log1p(np.abs(query_frame[lo:hi,None,None]-np.arange(frames)[None,None]))
        view_features[~available]=0
        folder=OUT/'features'/key;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'candidates.npz',static=static,views=view_features,available=available,separated=separated,valid=valid,
            owners=candidate,old_indices=old_indices,new_source_ids=new_ids,query_frame=query_frame,query_xyz=query_xyz)
        np.savez_compressed(folder/'bank.npz',object_ids=object_ids,descriptors=descriptors,observed=view_valid,metadata=meta)
        dump(folder/'provenance.json',provenance)
        row=dict(objects=len(object_ids),view_records=int(view_valid.sum()),old_total=len(old['xyz']),old_observed=n_old,
            old_eligible=int(valid[:n_old].any(1).sum()),new_proposals=len(new_ids),new_eligible=int(valid[n_old:].any(1).sum()),
            candidate_pairs=int(valid.sum()),all_view_pairs=int(available.sum()),disjoint_view_pairs=int(separated.sum()),seconds=time.monotonic()-start)
        dump(folder/'run.json',row);stats[key]=row
        for f in folder.iterdir():files[str(f)]=sha(f)
        print('OBJECT MEMORY FEATURES',key,row,flush=True)
        del bank_tensor;torch.cuda.empty_cache()
    dump(OUT/'features_complete.json',dict(complete=True,files=files,stats=stats,no_gt_read=True,foundation_model_calls=0,protocol_sha256=sha(OUT/'protocol.json')))


def labels():
    from evaluate_kitti360_development import read_static_ply,transfer
    from run_map_responsibility_supervision import training_assignment
    import runpy
    p=verify();seal('features_complete');assert not (OUT/'labels_complete.json').exists();files={};stats={}
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    for scene in p['scenes']:
        key=scene['key'];ref=np.load(scene['reference']);raw=read_static_ply(p['references'][key]['raw'])
        train=np.array([taxonomy[int(s)].trainId for s in raw['semantic']]);truth=train*1000+raw['instance'].astype(np.int64)%1000;truth[train==255]=0
        np.testing.assert_array_equal(truth,ref['train_instance']);weight=np.where(train!=255,ref['confidence'],0.)
        base=np.load(BASE/key/'geometry_graph/geometry.npz');objects=read(BASE/key/'geometry_graph/classification.json')['objects']
        projected=transfer(base['xyz'],base['instance'],ref['xyz'],.2);binding,matches=training_assignment(projected,truth,weight,objects)
        d=np.load(OUT/'features'/key/'candidates.npz');nearest=transfer(ref['xyz'],np.arange(1,len(truth)+1),d['query_xyz'],.2)
        ix=np.maximum(nearest-1,0);w=np.where(nearest>0,weight[ix],0.);valid=d['valid'];eligible=(w>0)&valid.any(1)
        target=np.full(d['owners'].shape,-1,np.int64)
        for oid,gid in binding.items():target[d['owners']==oid]=gid
        positive=valid&(target==truth[ix,None])&(target>0)
        folder=OUT/'training_only'/key;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'labels.npz',eligible=eligible,positive=positive,weight=w)
        dump(folder/'binding.json',dict(training_only=True,binding=binding,matches=matches))
        stats[key]=dict(eligible=int(eligible.sum()),positive=int((positive.any(1)&eligible).sum()),null=int((~positive.any(1)&eligible).sum()))
        for f in folder.iterdir():files[str(f)]=sha(f)
        print('LABELS',key,stats[key],flush=True)
    dump(OUT/'labels_complete.json',dict(complete=True,files=files,stats=stats))


def mode_for(method):
    return 'mean' if method=='mean_disjoint' else 'retrieval' if method=='retrieval_disjoint' else 'attention'


def train():
    p=verify();seal('features_complete');seal('labels_complete');assert not (OUT/'training_execution.json').exists();torch=gpu()
    dump(OUT/'training_execution.json',dict(started_epoch=time.time()));files={};stats={}
    for target in KEYS:
        key=next(k for k in KEYS if k!=target);d=np.load(OUT/'features'/key/'candidates.npz');y=np.load(OUT/'training_only'/key/'labels.npz');use=y['eligible']
        xx=d['static'][use];v=d['valid'][use];mean=xx[v].mean(0);std=np.maximum(xx[v].std(0),1e-6)
        x=torch.tensor((xx-mean)/std,device='cuda');views=torch.tensor(d['views'][use].astype('float32'),device='cuda')
        valid=torch.tensor(v,device='cuda');positive=torch.tensor(y['positive'][use],device='cuda');w=y['weight'][use];weight=torch.tensor(w/w.mean(),device='cuda',dtype=torch.float32)
        all_views=torch.tensor(d['available'][use],device='cuda');separated=torch.tensor(d['separated'][use],device='cuda')
        for method in TRAINED:
            torch.manual_seed(17);torch.cuda.manual_seed_all(17);rng=np.random.default_rng(17)
            model=ViewEvidenceAllocator().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001);curve=[];start=time.monotonic()
            seen=all_views if method=='attention_all' else separated;torch.cuda.reset_peak_memory_stats()
            for epoch in range(40):
                order=rng.permutation(len(x));total=0.
                for lo in range(0,len(order),1024):
                    ix=torch.tensor(order[lo:lo+1024],device='cuda');logits=model(x[ix],views[ix],seen[ix],valid[ix],mode_for(method))
                    loss=(grouped_loss(logits,valid[ix],positive[ix])*weight[ix]).mean();assert torch.isfinite(loss)
                    optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
                    optimizer.step();total+=float(loss.detach())*len(ix)
                curve.append(total/len(x))
                if epoch==0 or (epoch+1)%10==0:print('FIT OBJECT MEMORY',target,method,epoch+1,curve[-1],flush=True)
            folder=OUT/'models'/target/method;folder.mkdir(parents=True,exist_ok=False)
            torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=mean.tolist(),std=std.tolist(),training=[key],method=method),folder/'model.pt')
            row=dict(training=[key],queries=len(x),loss=curve,epochs=40,seconds=time.monotonic()-start,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
            dump(folder/'training.json',row);stats[target+'/'+method]=row
            for f in folder.iterdir():files[str(f)]=sha(f)
            del model,optimizer
        del x,views,valid,positive,weight,all_views,separated;torch.cuda.empty_cache()
    dump(OUT/'trained.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json')))


def predict():
    p=verify();seal('trained');seal('features_complete');assert not (OUT/'prediction_execution.json').exists();torch=gpu()
    dump(OUT/'prediction_execution.json',dict(started_epoch=time.time()));files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];d=np.load(OUT/'features'/key/'candidates.npz');base=np.load(BASE/key/'geometry_graph/geometry.npz')
        cls=read(BASE/key/'geometry_graph/classification.json');known=[o['object_id'] for o in cls['objects']]
        xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];old_indices=d['old_indices'];n_old=len(old_indices)
        eligible=d['valid'].any(1);indices=np.flatnonzero(eligible);decisions={};stats[key]={}
        for method in TRAINED:
            checkpoint=torch.load(OUT/'models'/key/method/'model.pt',map_location='cpu',weights_only=True)
            assert checkpoint['training']==[k for k in KEYS if k!=key]
            model=ViewEvidenceAllocator().cuda().eval();model.load_state_dict(checkpoint['state'],strict=True);model.requires_grad_(False)
            owners=np.r_[base['instance'][old_indices],np.zeros(len(d['new_source_ids']),np.int64)];confidence=np.full(len(owners),np.nan)
            viewmask=d['available'] if method=='attention_all' else d['separated'];static=d['static'];view=d['views'];valid=d['valid'];candidate=d['owners']
            with torch.inference_mode():
                for lo in range(0,len(indices),2048):
                    ix=indices[lo:lo+2048]
                    x=torch.tensor(((static[ix]-np.array(checkpoint['mean']))/np.array(checkpoint['std'])).astype('float32'),device='cuda')
                    z=model(x,torch.tensor(view[ix].astype('float32'),device='cuda'),torch.tensor(viewmask[ix],device='cuda'),
                        torch.tensor(valid[ix],device='cuda'),mode_for(method)).cpu().numpy()
                    chosen,prob=allocate(z,valid[ix],candidate[ix]);owners[ix]=chosen;confidence[ix]=prob
            decisions[method]=(owners,confidence);del model;torch.cuda.empty_cache()
        for method in METHODS:
            folder=OUT/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False)
            if method=='baseline':
                for name in ['geometry.npz','classification.json']:shutil.copyfile(BASE/key/'geometry_graph'/name,folder/name)
                row=dict(exact_copy=True)
            else:
                owners,confidence=decisions['ours_revision' if method=='ours_append_only' else method]
                g=revision_map(base['xyz'],base['instance'],xyz,old_indices,owners[:n_old],d['new_source_ids'],owners[n_old:],known,revise=method!='ours_append_only')
                np.savez_compressed(folder/'geometry.npz',**g);np.savez_compressed(folder/'decisions.npz',owners=owners,confidence=confidence)
                classification=copy.deepcopy(cls);present=set(g['instance']);classification['objects']=[o for o in cls['objects'] if o['object_id'] in present]
                dump(folder/'classification.json',classification)
                row=dict(objects=len(present),points=len(g['xyz']),added_sources=len(g['added_source_ids']),
                    removed_old=int((g['revised_base_owners']==0).sum()),
                    reassigned_old=int(((g['revised_base_owners']!=base['instance'])&(g['revised_base_owners']>0)).sum()),
                    skipped_unavailable_old=n_old-int(eligible[:n_old].sum()),old_total=len(base['xyz']))
            dump(folder/'run.json',row);stats[key][method]=row
            for f in folder.iterdir():files[str(f)]=sha(f)
            print('MAP OBJECT MEMORY',key,method,row,flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,sealed_before_scoring=True,
        trained_seal_sha256=sha(OUT/'trained.json')))


def score():
    from evaluate_kitti360_development import transfer,clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p=verify();prediction=seal('predictions_complete');assert not (OUT/'summary.json').exists()
    results={};means={m:[] for m in METHODS};gained=lost=0;gate=True
    for scene in p['scenes']:
        key=scene['key'];ref=np.load(scene['reference']);rows={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=np.load(folder/'geometry.npz');objects=read(folder/'classification.json')['objects']
            pred=transfer(g['xyz'],g['instance'],ref['xyz'],.2)
            r=evaluate(ref['train_instance'],ref['confidence'],pred,{o['object_id']:o['predictions']['mean'] for o in objects},EVALUATOR)
            dump(folder/'matches.json',clean_json(r['matches']));np.savez_compressed(folder/'reference_prediction.npz',instance=pred)
            rows[method]=dict(metrics=clean_json(r['averages']),matched_gt_ids=sorted(matched_ids(r['matches'])))
            means[method].append(float(r['averages']['allAp']));print('OBJECT MEMORY SCORE',key,method,rows[method]['metrics'],flush=True)
        expected=read(ROOT/'build/benchmarks/signed_query_association_v1/summary.json')['results'][key]['methods']['geometry_graph']['metrics']
        assert abs(rows['baseline']['metrics']['allAp']-expected['allAp'])<1e-12
        old=set(rows['baseline']['matched_gt_ids'])
        for row in rows.values():
            ids=set(row['matched_gt_ids']);row['changes_vs_baseline']=dict(gained=sorted(ids-old),lost=sorted(old-ids))
        change=rows['ours_revision']['changes_vs_baseline'];gained+=len(change['gained']);lost+=len(change['lost'])
        gate &= rows['ours_revision']['metrics']['allAp']>=rows['baseline']['metrics']['allAp'];results[key]=dict(methods=rows)
    mean={m:float(np.mean(v)) for m,v in means.items()}
    gate &= mean['ours_revision']>=mean['baseline']+.01 and mean['ours_revision']>max(v for m,v in mean.items() if m!='ours_revision') and gained>=1 and lost==0
    changed=any(row['ours_revision']['removed_old']+row['ours_revision']['reassigned_old']>0 for row in prediction['stats'].values());gate &= changed
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,gained=gained,lost=lost,advance_gate=bool(gate),
        historical_changes=changed,official_leaderboard=False,independent_test=False,prediction_seal_sha256=sha(OUT/'predictions_complete.json')))
    print('OBJECT MEMORY GATE',bool(gate),mean,'gained',gained,'lost',lost,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','labels','train','predict','score']);globals()[ap.parse_args().action]()
