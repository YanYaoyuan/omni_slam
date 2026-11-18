"""Isolated execution of the registered, unchanged quality candidate on0006/7.

References are read only by score, after all ten final predictions are sealed.
Parent modules are loaded in isolated namespaces; no old experiment is edited.
"""
import argparse,copy,time,shutil,runpy
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from prepare_batch_transport_assessment import OUT,INPUT,FULL,verify as verify_registration
from run_intrinsic_public_holdout import isolated_adapter
from run_batch_stable_transport import graph_and_seeds
from omni_semantic_map.batch_stable_transport import native_library,nearest_groups,nearest_owner,certify

METHODS=['baseline','same_cell','full_nearest','full_two_cells','ours_frame']
BASE=OUT/'baseline'
SEM=OUT/'semantics'
BACK=OUT/'backbone'


def seal(path):
    d=read(path);assert d['complete']
    for f,h in d.get('files',{}).items():assert sha(f)==h,f
    return d


def source_owners(core_file,intrinsic_folder,graph_folder,n):
    core=read(core_file);groups=read(graph_folder/'groups.json');assignment={int(k):int(v) for k,v in groups['old_id_assignment'].items()};labels=np.zeros(n,np.int64)
    def assign(ids,oid):
        old=labels[ids];bad=(old!=0)&(old!=oid);labels[ids]=oid;labels[ids[bad]]=-1
    for oid,ids in core.items():assign(np.asarray(ids,np.int64),assignment[int(oid)])
    for path,remap in [(intrinsic_folder,True),(graph_folder,False)]:
        g=np.load(path/'geometry.npz');ids=g['added_source_ids']
        if not len(ids):continue
        owner=g['instance'][-len(ids):]
        if remap:owner=np.array([assignment[int(i)] for i in owner])
        for oid in np.unique(owner):assign(ids[owner==oid],oid)
    return labels


def freeze():
    p=verify_registration();assert not (OUT/'execution_protocol.json').exists()
    seal(OUT/'prepared_complete.json');assert read(OUT/'input_delivery_audit.json')['complete'] and read(OUT/'backbone_validation/validation.json')['complete']
    for key in ['kitti360_0000','kitti360_0002']:
        expected=np.load(ROOT/'build/benchmarks/closed_vocabulary_semantics_v1/aggregate'/key/'source.npz')['owners']
        actual=source_owners(ROOT/'build/benchmarks/surface_ownership_v1/features'/key/'core_sources.json',ROOT/'build/benchmarks/intrinsic_surface_ownership_v1/maps'/key/'intrinsic_geometry',ROOT/'build/benchmarks/signed_query_association_v1/maps'/key/'geometry_graph',len(expected))
        np.testing.assert_array_equal(actual,expected)
    sources={str(Path(__file__)):sha(__file__)}
    for name in ['run_discovery_query_benchmark.py','discovery_cg_backend.py','run_intrinsic_public_holdout.py','run_surface_ownership.py','run_intrinsic_surface_ownership.py','run_robot_geometry_reference.py','capture_prethreshold_candidates.py','capture_closed_vocabulary_semantics.py','evaluate_closed_vocabulary_semantics.py','audit_semantic_counterevidence.py','run_batch_stable_transport.py','evaluate_kitti360_development.py']:
        f=ROOT/'scripts/benchmarks'/name;sources[str(f)]=sha(f)
    inputs={str(OUT/name):sha(OUT/name) for name in ['protocol.json','prepared_complete.json','input_golden.json','input_delivery_audit.json','backbone_validation/validation.json']}
    for path in [ROOT/'build/benchmarks/intrinsic_surface_ownership_v1/protocol.json',ROOT/'build/benchmarks/discovery_query_benchmark_v1/protocol.json']:
        inputs[str(path)]=sha(path)
    model=ROOT/'build/models/segformer_b5_cityscapes';manifest=read(model/'source_manifest.json')
    for name,row in manifest['files'].items():assert sha(model/name)==row['sha256'];inputs[str(model/name)]=row['sha256']
    for path,row in read(ROOT/'build/benchmarks/ovo_full_20261110/model_artifacts.json')['files'].items():assert sha(path)==row['sha256'];inputs[path]=row['sha256']
    dump(OUT/'execution_protocol.json',dict(complete=True,created_epoch=time.time(),source_hashes=sources,input_hashes=inputs,
        methods=METHODS,source_owner_golden=['kitti360_0000','kitti360_0002'],
        adaptation='Reuse pinned kernels with isolated module globals and new output roots only. Original masks/features/supports/CG/intrinsic/rawquery/reference/SegFormer kernels unchanged. Fullgraph and certificate called directly from frozen implementation. New wrappers only stage/persist/score.',
        labels='No reference decode until alltenfinalmapseal. Olddevelopment owner replay uses cached predictions only,notGT.',
        stage_policy='Every stage gets a fresh execution record; a failed/interrupted stage cannot silently restart. Verify live handles before any recovery. Independent GPU calls may overlap only on separate devices and separate output subtrees.'))
    for sub in ['raw','ownership_axis','ownership_intrinsic','semantics']:
        folder=OUT/sub;folder.mkdir(exist_ok=True);shutil.copyfile(OUT/'protocol.json',folder/'protocol.json')
    print('ASSESSMENT EXECUTION FROZEN',flush=True)


def verify(data=False):
    e=read(OUT/'execution_protocol.json')
    for f,h in {**e['source_hashes'],**e['input_hashes']}.items():assert sha(f)==h,f
    p=read(OUT/'protocol.json')
    if data:
        verify_registration();seal(OUT/'prepared_complete.json')
    # Required native-stage metadata; scientific values are already registered.
    p=copy.deepcopy(p);p.update(prompts=['car','building'],semantic_ids=[26,11],sam3_threshold=.5,sam3_resolution=1008,
        checkpoint=p['sam3_checkpoint'],checkpoint_sha256=p['sam3_sha256'],rotations=read(ROOT/'build/benchmarks/intrinsic_surface_ownership_v1/protocol.json')['rotations'])
    return p


def stage_start(name):
    verify(True);file=OUT/'stages'/(name+'.json');assert not file.exists(),'Inspect existing stage handle/status before recovery'
    dump(file,dict(started_epoch=time.time(),stage=name,execution_protocol_sha256=sha(OUT/'execution_protocol.json')))


def frontend(action):
    stage_start(action);n=isolated_adapter('run_discovery_query_benchmark.py');n.OUT=OUT;n.INPUT=INPUT;n.verify=verify;getattr(n,action)()


def base_maps():
    stage_start('base_maps');n=isolated_adapter('run_intrinsic_public_holdout.py');n.OUT=OUT;n.INPUT=INPUT;n.verify=verify;n.sealed=lambda name:seal(OUT/(name+'.json'));n.maps()


def candidates():
    stage_start('candidates');n=isolated_adapter('run_intrinsic_public_holdout.py');n.OUT=OUT;n.INPUT=INPUT;n.verify=verify;n.sealed=lambda name:seal(OUT/(name+'.json'));n.candidates()


def intrinsic():
    import torch
    from run_intrinsic_surface_ownership import inference
    from omni_semantic_map.surface_ownership import append_owned
    stage_start('intrinsic');p=verify();torch.set_num_threads(1);seal(OUT/'ownership_intrinsic/features_complete.json');files={};stats={}
    model=torch.load(p['selected_model'],map_location='cpu',weights_only=False);assert model['training']==['kitti360_0000','kitti360_0002']
    for scene in p['scenes']:
        key=scene['key'];d=np.load(OUT/'ownership_intrinsic/features'/key/'candidates.npz');core=np.load(OUT/'maps'/key/'largest/geometry.npz');source=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];owner,prob=inference(model,d['features'],d['valid'],d['owners'])
        for method in ['baseline','intrinsic_geometry']:
            folder=BACK/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False)
            if method=='baseline':shutil.copyfile(OUT/'maps'/key/'largest/geometry.npz',folder/'geometry.npz')
            else:np.savez_compressed(folder/'geometry.npz',**append_owned(core['xyz'],core['instance'],source,d['source_ids'],owner))
            shutil.copyfile(OUT/'maps'/key/'largest/classification.json',folder/'classification.json')
            for f in folder.iterdir():files[str(f)]=sha(f)
        folder=BACK/'maps'/key/'intrinsic_geometry';np.savez_compressed(folder/'decisions.npz',source_ids=d['source_ids'],owners=owner,confidence=prob);files[str(folder/'decisions.npz')]=sha(folder/'decisions.npz')
        stats[key]=dict(candidate_sources=len(owner),added_sources=int((owner>0).sum()));print('ASSESSMENT INTRINSIC',key,stats[key],flush=True)
    dump(OUT/'intrinsic_complete.json',dict(complete=True,files=files,stats=stats))


def raw():
    stage_start('raw');seal(OUT/'masks_complete.json');n=isolated_adapter('capture_prethreshold_candidates.py');n.OUT=OUT/'raw';n.FRONT=OUT;n.INPUT=INPUT;n.verify=verify;n.predict()


def geometry():
    stage_start('geometry');p=verify();seal(OUT/'intrinsic_complete.json');seal(OUT/'raw/predictions_complete.json');files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];n=isolated_adapter('run_robot_geometry_reference.py');n.OUT=OUT/'geometry_reference'/key;n.INPUT=INPUT;n.BASE=BACK;n.OWN=OUT/'ownership_axis';n.KEY=key;n.verify=verify
        (n.OUT/'raw').mkdir(parents=True,exist_ok=False);shutil.copyfile(OUT/'raw/predictions_complete.json',n.OUT/'raw/predictions_complete.json');shutil.copyfile(OUT/'protocol.json',n.OUT/'protocol.json')
        n.index();n.maps();seal(n.OUT/'maps_complete.json');source=n.OUT/'maps/geometry_graph_reference';dest=BASE/key/'geometry_graph';dest.mkdir(parents=True,exist_ok=False)
        for name in ['geometry.npz','classification.json','groups.json','run.json']:
            shutil.copyfile(source/name,dest/name);files[str(dest/name)]=sha(dest/name)
        stats[key]=read(dest/'run.json');print('ASSESSMENT STRONG REFERENCE',key,stats[key]['objects'],flush=True)
    dump(OUT/'baseline_complete.json',dict(complete=True,files=files,stats=stats))


def anchors():
    stage_start('anchors');p=verify();seal(OUT/'baseline_complete.json');files={}
    for scene in p['scenes']:
        key=scene['key'];n=len(np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz']);owner=source_owners(OUT/'ownership_axis/features'/key/'core_sources.json',BACK/'maps'/key/'intrinsic_geometry',BASE/key/'geometry_graph',n);declared=np.full(n,-1,np.int16)
        for obj in read(BASE/key/'geometry_graph/classification.json')['objects']:declared[owner==obj['object_id']]=obj['predictions']['mean']['train_id']
        assert (declared[owner>0]>=0).all();folder=OUT/'anchors'/key;folder.mkdir(parents=True,exist_ok=False);f=folder/'predictions.npz';np.savez_compressed(f,owners=owner,declared_train_id=declared);files[str(f)]=sha(f)
    dump(OUT/'anchors_complete.json',dict(complete=True,files=files))


def semantics():
    stage_start('semantics');seal(OUT/'baseline_complete.json');n=isolated_adapter('capture_closed_vocabulary_semantics.py');n.OUT=SEM;n.BASE=BASE;n.INPUT=INPUT;n.verify=verify;n.predict()


def aggregate():
    stage_start('aggregate');seal(OUT/'anchors_complete.json');seal(SEM/'predictions_complete.json');n=isolated_adapter('evaluate_closed_vocabulary_semantics.py');n.OUT=SEM;n.BASE=BASE;n.INPUT=INPUT;n.SAM_AUDIT=OUT/'anchors';n.verify=verify;n.aggregate()


def fields():
    stage_start('fields');p=verify();seal(SEM/'aggregation_complete.json');lib=native_library(OUT/'native');files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];start=time.monotonic();full=np.load(FULL/key/'geometry.npz');mapping=np.load(FULL/key/'old_to_full.npy');sem=np.load(SEM/'aggregate'/key/'source.npz')
        graph,d,stat=graph_and_seeds(full['xyz'],mapping,sem,full['source_frame']);seeds=d['seeds'];labels,dist=nearest_groups(graph,seeds[:,0],seeds[:,1],lib)
        d.update(label_nearest=labels,label_distance=dist,nearest_owner=nearest_owner(labels,dist))
        for grouping,column in [('cell',0),('frame',2)]:
            pairs,tokens=np.unique(seeds[:,[1,column]],axis=0,return_inverse=True);tg,td=nearest_groups(graph,seeds[:,0],tokens,lib);pred=certify(labels,dist,tg,td,pairs[:,0])
            d.update({grouping+'_groups':pairs,grouping+'_nearest':tg,grouping+'_distance':td,grouping+'_owner':pred});stat[grouping+'_foreground_cells']=int((pred>0).sum())
        d['same_cell_owner']=np.where(dist[:,0]==0,d['nearest_owner'],-1)
        folder=OUT/'fields'/key;folder.mkdir(parents=True,exist_ok=False);f=folder/'field.npz';np.savez_compressed(f,**d,graph_indptr=graph.indptr,graph_indices=graph.indices,graph_weights=graph.data);files[str(f)]=sha(f);stat['seconds']=time.monotonic()-start;stats[key]=stat;print('ASSESSMENT FIELD',key,stat,flush=True)
    dump(OUT/'fields_complete.json',dict(complete=True,files=files,stats=stats))


def maps():
    stage_start('maps');p=verify();seal(OUT/'fields_complete.json');files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];base=np.load(BASE/key/'geometry_graph/geometry.npz');full=np.load(FULL/key/'geometry.npz');sem=np.load(SEM/'aggregate'/key/'source.npz');mapping=np.load(FULL/key/'old_to_full.npy');owners=np.zeros(len(full['xyz']),np.int64);owners[mapping]=sem['owners'];d=np.load(OUT/'fields'/key/'field.npz');stats[key]={}
        for method in METHODS:
            folder=OUT/'final_maps'/key/method;folder.mkdir(parents=True,exist_ok=False);shutil.copyfile(BASE/key/'geometry_graph/classification.json',folder/'classification.json')
            if method=='baseline':shutil.copyfile(BASE/key/'geometry_graph/geometry.npz',folder/'geometry.npz');added=np.empty(0,np.int64)
            else:
                name=dict(same_cell='same_cell',full_nearest='nearest',full_two_cells='cell',ours_frame='frame')[method];pred=d[name+'_owner'][d['source_to_cell']];added=np.flatnonzero((owners==0)&(pred>0))
                np.savez_compressed(folder/'geometry.npz',xyz=np.r_[base['xyz'],full['xyz'][added]],instance=np.r_[base['instance'],pred[added]],added_full_source_ids=added)
            stats[key][method]=dict(added_sources=len(added),added_new_lidar_sources=int((full['old_source_id'][added]<0).sum()),objects=len(np.unique(base['instance'])),new_ids=0,old_map_preserved=True);dump(folder/'run.json',stats[key][method])
            for f in folder.iterdir():files[str(f)]=sha(f)
            print('ASSESSMENT MAP',key,method,len(added),flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,reference_labels_decoded=False,execution_protocol_sha256=sha(OUT/'execution_protocol.json')))


def score():
    from evaluate_kitti360_development import read_static_ply,transfer,clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    stage_start('score');p=verify();seal(OUT/'predictions_complete.json');assert not (OUT/'summary.json').exists();taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label'];results={};means={m:[] for m in METHODS};noharm=True;gained=lost=0
    for scene in p['scenes']:
        key=scene['key'];job=read(scene['job']);raw=read_static_ply(scene['opaque_reference']);xyz=np.column_stack([raw[k] for k in ['x','y','z']]).astype(np.float64)-np.asarray(job['origin_world']);train=np.array([taxonomy[int(v)].trainId for v in raw['semantic']],np.int64);truth=train*1000+raw['instance'].astype(np.int64)%1000;truth[train==255]=0;weight=raw['confidence'].astype(np.float64)
        folder=OUT/'evaluation'/key;folder.mkdir(parents=True,exist_ok=False);np.savez_compressed(folder/'reference.npz',xyz=xyz,train_instance=truth,confidence=weight);methods={}
        for method in METHODS:
            f=OUT/'final_maps'/key/method;g=np.load(f/'geometry.npz');cls=read(f/'classification.json')['objects'];pred=transfer(g['xyz'],g['instance'],xyz,.2);r=evaluate(truth,weight,pred,{o['object_id']:o['predictions']['mean'] for o in cls},ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
            dump(f/'matches.json',clean_json(r['matches']));np.savez_compressed(f/'reference_prediction.npz',instance=pred);methods[method]=dict(metrics=clean_json(r['averages']),matched_gt_ids=sorted(matched_ids(r['matches'])),run=read(f/'run.json'));means[method].append(float(r['averages']['allAp']));print('INDEPENDENT SCORE',key,method,methods[method]['metrics'],flush=True)
        old=set(methods['baseline']['matched_gt_ids'])
        for method,row in methods.items():current=set(row['matched_gt_ids']);row['changes_vs_baseline']=dict(gained=sorted(current-old),lost=sorted(old-current))
        ours=methods['ours_frame'];gained+=len(ours['changes_vs_baseline']['gained']);lost+=len(ours['changes_vs_baseline']['lost'])
        noharm &= all(ours['metrics'][m] is not None and methods['baseline']['metrics'][m] is not None and ours['metrics'][m]>=methods['baseline']['metrics'][m] for m in ['allAp','allAp50%']);results[key]=dict(methods=methods,reference_points=len(xyz))
    mean={m:float(np.mean(v)) for m,v in means.items()};gate=bool(noharm and all(np.isfinite(v) for v in mean.values()) and mean['ours_frame']>=mean['baseline']+.02 and all(mean['ours_frame']>mean[m] for m in ['same_cell','full_nearest','full_two_cells']))
    dump(OUT/'summary.json',clean_json(dict(complete=True,results=results,mean_primary_ap=mean,independent_gate=gate,no_metric_regression=bool(noharm),gained=gained,lost=lost,
        protocol_sha256=sha(OUT/'protocol.json'),execution_protocol_sha256=sha(OUT/'execution_protocol.json'),predictions_sha256=sha(OUT/'predictions_complete.json'),posthoc_development_selection=True,independent_new_windows=True,official_leaderboard=False,robot_completed=False)))
    print('INDEPENDENT QUALITY GATE',gate,mean,'gained',gained,'lost',lost,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','masks','features','supports','base_maps','candidates','intrinsic','raw','geometry','anchors','semantics','aggregate','fields','maps','score']);action=ap.parse_args().action
    frontend(action) if action in ['masks','features','supports'] else globals()[action]()
