"""Independent assessment of the simpler candidate selected on development.

The prior full-evidence candidate failed its gate and remains closed. Selection
of its simpler geometry-only control is explicitly posthoc development choice,
frozen here before new-window inference and before any reference decoding.
"""
import argparse,time,shutil,runpy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_intrinsic_surface_ownership import OUT as DEV,PARENT as AXIS,verify as verify_dev,sealed as dev_sealed

OUT=ROOT/'build/benchmarks/intrinsic_public_holdout_v1'
INPUT=OUT/'measured'
DATASET=ROOT/'build/datasets/kitti360_unseen_v2'
META=ROOT/'build/datasets/kitti360_prepare'
METHODS=['baseline','nearest_core','axis_geometry','axis_evidence','ours_intrinsic_geometry','intrinsic_evidence']


def isolated_adapter(filename):
    # Original verification functions retain references to their module globals.
    # Never redirect those modules' OUT/INPUT while reusing their stage kernels.
    import importlib.util
    path=ROOT/'scripts/benchmarks'/filename
    spec=importlib.util.spec_from_file_location('_holdout_adapter_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def freeze():
    assert not OUT.exists();development=verify_dev(True);dev_sealed('trained');dev_sealed('predictions_complete');assert not read(DEV/'summary.json')['advance_gate']
    front=read(ROOT/'build/benchmarks/discovery_query_benchmark_v1/protocol.json');selection=read(DATASET/'window_selection.json');dataset=read(DATASET/'verification.json');assert dataset['complete']
    sources={str(Path(__file__)):sha(__file__)};sources.update(front['source_hashes']);sources.update(development['sources'])
    for filename in ['prepare_kitti360_reserved.py','run_surface_ownership.py','prepare_strong_jobs.py']:
        f=ROOT/'scripts/benchmarks'/filename;sources[str(f)]=sha(f)
    for name in ['kitti360_input','measured_query_cells','surface_ownership','sparse_lidar_support']:
        f=ROOT/'omni_semantic_map/omni_semantic_map'/f'{name}.py';sources[str(f)]=sha(f)
    models=dict(axis_geometry=DEV/'models/rosbag2245/axis_geometry/model.pt',axis_evidence=AXIS/'models/rosbag2245/independent/model.pt',
        ours_intrinsic_geometry=DEV/'models/rosbag2245/intrinsic_geometry/model.pt',intrinsic_evidence=DEV/'models/rosbag2245/ours_intrinsic/model.pt')
    inputs={str(DATASET/'window_selection.json'):sha(DATASET/'window_selection.json'),str(DATASET/'verification.json'):sha(DATASET/'verification.json'),str(DEV/'summary.json'):sha(DEV/'summary.json')};scenes=[]
    for w in selection['windows']:
        key='kitti360_'+w['scene'].split('_')[-2];job=OUT/'prepared'/key/'jobs'/(w['scene']+'.json');ref=DATASET/'data/opaque_references'/w['member'];inputs[str(ref)]=sha(ref)
        for kind in ['image_00','velodyne']:
            f=DATASET/'data'/f'{w["scene"]}_{kind}_download.json';m=read(f);assert m['complete'];inputs[str(f)]=sha(f)
            for r in m['files']:
                path=ROOT/r['path'];assert sha(path)==r['sha256'];inputs[str(path)]=r['sha256']
        for f in (META/'extracted'/w['scene']).rglob('*.txt'):inputs[str(f)]=sha(f)
        scenes.append(dict(key=key,window=w,job=str(job),reference=str(ref)))
    for f in (META/'extracted/calibration').glob('*.txt'):inputs[str(f)]=sha(f)
    for f in models.values():inputs[str(f)]=sha(f)
    for f in [ROOT/'build/benchmarks/hypersim_shared_v1/jobs/ai_001_001.json',ROOT/'build/benchmarks/ovo_full_20261110/model_artifacts.json',META/'labels.py']:
        inputs[str(f)]=sha(f)
    for f in (ROOT/'build/benchmarks/kitti360_development_v1/taxonomy').iterdir():
        if f.is_file():inputs[str(f)]=sha(f)
    p={k:front[k] for k in ['sam3_checkpoint','sam3_sha256','prompts','semantic_ids','sam3_threshold','sam3_resolution']}
    p.update(created_epoch=time.time(),scenes=scenes,methods=METHODS,models={k:str(v) for k,v in models.items()},source_hashes=sources,input_hashes=inputs,
        rotations=development['rotations'],development_protocol_sha256=sha(DEV/'protocol.json'),
        selection_history='Posthoc development selection of intrinsic_geometry: meanAP50:90=22.639 versus baseline19.889/nearest20.540/axis_geometry21.768; both windows primary improved,3gained/1lost AP50 GT. Prior full-evidence candidate failed its frozen gate; that failure is retained. This is a new independent assessment of the simpler control, not a claim that the previous gate passed.',
        selection_change='Explicit follow-up decision after development ablations: assess the simpler positive geometry-only candidate on untouched windows once, despite the prior full-candidate stop. No adjustment of previous results/gate and no assertion of prespecified development winner. No further fitting, threshold/architecture search or selection on these heldout results.',
        candidate='Ours=geometry-only invariant surface allocation. MLP12->32->32->1, independent BCE, sigmoid>0.5, one candidate owner per source. Append exact previously unowned measured sources; core/identity/semantic scores fixed. Training on0000+0002 only. The already-frozen weights located under prior robot target directory were trained solely on public data, with no robot labels.',
        controls='Same full-frame SAM3/SigLIP/native CG-history2 frontend/backend. Original map; nearest core<=0.4m; axis geometry; axis+temporal evidence; selected intrinsic geometry; intrinsic+temporal evidence. Same existing weights trained on0000+0002, no new training. Exact same candidate source/owner pairs.',
        reference_policy='Do not decode either reference until every method prediction in BOTH windows is sealed. RGB inference begins only after this protocol. Hashing opaque references allowed. New sequences0004/0005 selected by split/pose metadata; input95frames already downloaded/verified.',
        evaluation='Full released static windows, local raw-camera/LiDAR mapping adaptation, nearest predicted point<=20cm, original confidence-weighted KITTI AP50:90/AP50 kernel. Different input task from official accumulated XYZRGB leaderboard. Report every window/class/method and all gained/lost GT; no substitutions.',
        independent_gate='Selected ours meanAP50:90>=baseline+0.02 and>=nearest_core+0.02, no per-window primary or AP50 decrease vsbaseline; meanprimary>axis_geometry; total unique gainedAP50 GT>lost. Report losses explicitly even if this gate passes. Both fixed rigid transforms yield zero changed ownership decisions. This gate applies only to untouched independent windows, not retroactively to development.',
        budget='95 shared image inputs,2native maps,10derived maps;0new fits,0parameter sweeps,0new prompts. All12predictions sealed together before scores. No heldout-driven method switch, retraining, replacement windows or additional variants.',
        scope='Two new independent validation windows, not all KITTI-360 or official server results. Foundation pretraining overlap unknown. Static object map, not moving-object tracking; supplied public poses, not LIVO run on KITTI. Robot execution remains the separate existing real-LIVO comparison.')
    dump(OUT/'protocol.json',p);print('FROZEN independent assessment',[(s['key'],len(s['window']['input_frames'])) for s in scenes],flush=True)


def verify(data=False):
    p=read(OUT/'protocol.json');verify_dev(False);assert sha(DEV/'protocol.json')==p['development_protocol_sha256']
    for f,h in p['source_hashes'].items():assert sha(f)==h,f
    if data:
        for f,h in p['input_hashes'].items():assert sha(f)==h,f
    return p


def sealed(name):
    d=read(OUT/(name+'.json'))
    for f,h in d['files'].items():assert sha(f)==h,f
    return d


def prepare():
    from PIL import Image
    import prepare_kitti360_reserved as native
    from omni_semantic_map.sparse_lidar_support import MeasuredVoxelMap,project_measured_visibility
    from omni_semantic_map.measured_query_cells import query_cell_ids
    p=verify(True);assert not (OUT/'prepared_complete.json').exists();files={};stats={};meta=OUT/'metadata';meta.mkdir();shutil.copyfile(DATASET/'window_selection.json',meta/'window_selection.json');(meta/'extracted').symlink_to(META/'extracted',target_is_directory=True);native.META=meta;native.DATA=DATASET/'data'
    for s in p['scenes']:
        key=s['key'];native.prepare(s['window'],OUT/'prepared'/key);job=read(s['job']);folder=INPUT/key;(folder/'frames').mkdir(parents=True);records=[];hashes={s['job']:sha(s['job'])};voxel=MeasuredVoxelMap(.05)
        for rank,r in enumerate(job['frames']):
            z=np.load(r['measured']);rgb=np.asarray(Image.open(r['rgb']).convert('RGB'));px=z['pixel_xy'];voxel.append(z['xyz'],rgb[px[:,1],px[:,0]],r['frame_id'],z['raw_indices'],px);xyz=voxel.arrays()['xyz'];n=len(xyz)
            x,y,_,visible=project_measured_visibility(xyz,np.asarray(r['pose']),job['config']['cam'],z['depth'],SimpleNamespace(depth_absolute_m=.05,depth_relative=0.));ids=np.flatnonzero(visible);dest=folder/'frames'/f'{r["frame_id"]:06d}.npz';np.savez_compressed(dest,visible_source_ids=ids,pixel_xy=np.column_stack([x[ids],y[ids]]));files[str(dest)]=sha(dest)
            records.append(dict(frame_id=r['frame_id'],rgb=r['rgb'],file=str(dest),sha256=sha(dest),map_points=n,visible_points=len(ids),camera_pose=r['pose']))
            for f in [r['rgb'],r['measured']]:hashes[f]=sha(f)
            if rank%25==0:print('INPUT',key,rank,'points',n,flush=True)
        geometry=folder/'geometry.npz';np.savez_compressed(geometry,**voxel.arrays());files[str(geometry)]=sha(geometry)
        dump(folder/'manifest.json',dict(complete=True,key=key,kind='outdoor',records=records,geometry=str(geometry),geometry_sha256=sha(geometry),input_hashes=hashes,source_points=n,frames=len(records),no_masks_or_gt=True));files[str(folder/'manifest.json')]=sha(folder/'manifest.json');files[s['job']]=sha(s['job'])
        cells=query_cell_ids(xyz,.2);dest=INPUT/'query_cells'/key;dest.mkdir(parents=True);np.save(dest/'source_to_cell.npy',cells);files[str(dest/'source_to_cell.npy')]=sha(dest/'source_to_cell.npy')
        for r in records[::max(1,len(records)//4)]:np.testing.assert_array_equal(query_cell_ids(xyz[:r['map_points']],.2),cells[:r['map_points']])
        stats[key]=dict(frames=len(records),source_points=n,prefix_cells_verified=True)
    dump(OUT/'prepared_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json')))


def frontend(action):
    native=isolated_adapter('run_discovery_query_benchmark.py')
    verify(True);sealed('prepared_complete');native.OUT=OUT;native.INPUT=INPUT;native.verify=verify
    getattr(native,action)()


def maps():
    from discovery_cg_backend import build_map
    p=verify(True);sealed('prepared_complete');masks=sealed('masks_complete');sealed('features_complete');sealed('supports_complete');assert not (OUT/'base_maps_complete.json').exists();files={};stats={}
    tax=ROOT/'build/benchmarks/kitti360_development_v1/taxonomy';text=np.load(tax/'text_features.npy');vocab=read(tax/'vocabulary.json')
    for s in p['scenes']:
        key=s['key'];manifest=read(INPUT/key/'manifest.json');geometry=np.load(manifest['geometry']);records=masks['scenes'][key]['records'];byframe={r['frame_id']:r for r in manifest['records']};features=[];frames=[];areas=[];classes=[];scores=[]
        for r in records:
            z=np.load(r['file']);features.append(np.load(OUT/'frontend'/key/'features'/f'{r["frame_id"]:06d}.npy'));frames.extend([r['frame_id']]*r['masks']);areas.extend(z['areas']);classes.extend(z['semantic_ids']);scores.extend(z['scores'])
            base=OUT/'frontend'/key/'supports'/f'{r["frame_id"]:06d}.npz';dest=OUT/'supports'/key/'largest'/base.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base,dest);files[str(dest)]=sha(dest)
            measured=np.load(byframe[r['frame_id']]['file']);ids=measured['visible_source_ids'];px=measured['pixel_xy'];labels=z['segmentation'][px[:,1],px[:,0]];pieces=[ids[labels==j] for j in range(r['masks'])];points=np.concatenate(pieces) if pieces else np.empty(0,np.int64);offsets=np.r_[0,np.cumsum([len(v) for v in pieces])];dest=OUT/'supports'/key/'all'/base.name;dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,point_ids=points,offsets=offsets);files[str(dest)]=sha(dest)
        def loader(r):return np.load(OUT/'supports'/key/'largest'/f'{r["frame_id"]:06d}.npz')
        folder=OUT/'maps'/key/'largest';stat=build_map(folder,geometry,records,loader,np.concatenate(features),np.asarray(frames),np.asarray(areas),np.asarray(classes),np.asarray(scores),text,vocab)
        for f in folder.iterdir():files[str(f)]=sha(f)
        stats[key]=stat;print('BASE MAP',key,stat['objects'],flush=True)
    dump(OUT/'base_maps_complete.json',dict(complete=True,files=files,stats=stats))


def candidates():
    axis=isolated_adapter('run_surface_ownership.py')
    intrinsic=isolated_adapter('run_intrinsic_surface_ownership.py')
    verify(True);sealed('base_maps_complete');axis.OUT=OUT/'ownership_axis';axis.PRIOR=OUT;axis.INPUT=INPUT;axis.FRONT=OUT;axis.SUPPORT=OUT;axis.verify=verify;axis.features()
    intrinsic.OUT=OUT/'ownership_intrinsic';intrinsic.PARENT=OUT/'ownership_axis';intrinsic.PRIOR=OUT;intrinsic.INPUT=INPUT;intrinsic.verify=verify;intrinsic.features()


def predict():
    import torch
    from run_intrinsic_surface_ownership import inference
    from omni_semantic_map.surface_ownership import append_owned
    p=verify(True);sealed('base_maps_complete');assert not (OUT/'predictions_complete.json').exists();torch.set_num_threads(1);files={};stats={};invariance={}
    for root in [OUT/'ownership_axis',OUT/'ownership_intrinsic']:
        for f,h in read(root/'features_complete.json')['files'].items():assert sha(f)==h
    for s in p['scenes']:
        key=s['key'];axis=np.load(OUT/'ownership_axis/features'/key/'candidates.npz');intrinsic=np.load(OUT/'ownership_intrinsic/features'/key/'candidates.npz');rigid=np.load(OUT/'ownership_intrinsic/features'/key/'rigid_features.npz')['features'];core=np.load(OUT/'maps'/key/'largest/geometry.npz');classification=read(OUT/'maps'/key/'largest/classification.json');source=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];ids=axis['source_ids'];valid=axis['valid'];owners=axis['owners']
        for field in ['source_ids','valid','owners']:np.testing.assert_array_equal(axis[field],intrinsic[field])
        for method in METHODS:
            start=time.monotonic();owner=np.zeros(len(ids),np.int64);prob=np.zeros(len(ids))
            if method=='nearest_core':
                d=np.where(valid,np.expm1(axis['features'][:,:,0]),np.inf);j=d.argmin(1);best=d[np.arange(len(j)),j];owner=np.where((best<=.4)&((d==best[:,None]).sum(1)==1),owners[np.arange(len(j)),j],0)
            elif method!='baseline':
                model=torch.load(p['models'][method],map_location='cpu',weights_only=False);assert model['training']==['kitti360_0000','kitti360_0002'];x=axis['features'] if method.startswith('axis') else intrinsic['features'];owner,prob=inference(model,x,valid,owners)
                if not method.startswith('axis'):
                    check={}
                    for k,t in enumerate(p['rotations']):
                        alternate,conf=inference(model,rigid[k],valid,owners);check[t['name']]=dict(changed_decisions=int((alternate!=owner).sum()),max_probability_error=float(np.max(np.abs(conf-prob))) if len(prob) else 0.)
                    invariance[key+'/'+method]=check
            result=append_owned(core['xyz'],core['instance'],source,ids,owner);n=len(core['xyz']);np.testing.assert_array_equal(result['xyz'][:n],core['xyz']);np.testing.assert_array_equal(result['instance'][:n],core['instance']);folder=OUT/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False);np.savez_compressed(folder/'geometry.npz',**result);np.savez_compressed(folder/'decisions.npz',source_ids=ids,owners=owner,confidence=prob);dump(folder/'classification.json',classification)
            stat=dict(complete=True,objects=len(classification['objects']),old_points=n,added_points=int((owner>0).sum()),map_points=len(result['xyz']),seconds=time.monotonic()-start,core_bitwise_equal=True,unique_actual_sources=True,model=p['models'].get(method));dump(folder/'run.json',stat)
            for f in folder.iterdir():files[str(f)]=sha(f)
            stats[key+'/'+method]=stat;print('PREDICTED',key,method,stat['added_points'],flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,rigid_invariance=invariance,protocol_sha256=sha(OUT/'protocol.json'),reference_labels_not_decoded=True))


def score():
    from evaluate_kitti360_development import read_static_ply,transfer,clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p=verify(True);seal=sealed('predictions_complete');assert not (OUT/'summary.json').exists();taxonomy=runpy.run_path(str(META/'labels.py'))['id2label'];results={};means={m:[] for m in METHODS};no_harm=True;gained=0;lost=0
    for s in p['scenes']:
        key=s['key'];job=read(s['job']);raw=read_static_ply(s['reference']);xyz=np.column_stack([raw[k] for k in ['x','y','z']]).astype(np.float64)-np.asarray(job['origin_world']);train=np.array([taxonomy[int(v)].trainId for v in raw['semantic']],np.int64);truth=train*1000+raw['instance'].astype(np.int64)%1000;truth[train==255]=0;confidence=raw['confidence'].astype(np.float64);dest=OUT/'evaluation'/key;dest.mkdir(parents=True);np.savez_compressed(dest/'reference.npz',xyz=xyz,train_instance=truth,confidence=confidence);methods={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=np.load(folder/'geometry.npz');projected=transfer(g['xyz'],g['instance'],xyz,.2);cls=read(folder/'classification.json')['objects'];r=evaluate(truth,confidence,projected,{o['object_id']:o['predictions']['mean'] for o in cls},META/'evalInstanceLevelSemanticLabeling.py');row=dict(metrics=clean_json(r['averages']),matched_gt_ids=sorted(matched_ids(r['matches'])),run=read(folder/'run.json'));dump(folder/'matches.json',clean_json(r['matches']));means[method].append(r['averages']['allAp']);methods[method]=row;print('SCORED',key,method,row['metrics'],flush=True)
        base=set(methods['baseline']['matched_gt_ids'])
        for method,row in methods.items():current=set(row['matched_gt_ids']);row['changes_vs_baseline']=dict(gained=sorted(current-base),lost=sorted(base-current))
        a=methods['ours_intrinsic_geometry'];b=methods['baseline'];gained+=len(a['changes_vs_baseline']['gained']);lost+=len(a['changes_vs_baseline']['lost']);no_harm&=all(a['metrics'][m]>=b['metrics'][m]-1e-12 for m in ['allAp','allAp50%']);results[key]=dict(methods=methods,reference_points=len(xyz))
    mean={m:float(np.mean(v)) for m,v in means.items()};invariant=all(r['changed_decisions']==0 for name,checks in seal['rigid_invariance'].items() if name.endswith('/ours_intrinsic_geometry') for r in checks.values());gate=bool(no_harm and invariant and gained>lost and all(mean['ours_intrinsic_geometry']>=mean[c]+.02 for c in ['baseline','nearest_core']) and mean['ours_intrinsic_geometry']>mean['axis_geometry'])
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,independent_gate=gate,no_metric_regression=bool(no_harm),gained_gt=gained,lost_gt=lost,rigid_invariance_pass=invariant,protocol_sha256=sha(OUT/'protocol.json'),prediction_seal_sha256=sha(OUT/'predictions_complete.json'),selection_posthoc_on_development=True,independent_new_windows=True,official_leaderboard=False));print('INDEPENDENT GATE',gate,mean,'gained',gained,'lost',lost,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','masks','features','supports','maps','candidates','predict','score']);action=parser.parse_args().action
    frontend(action) if action in ['masks','features','supports'] else globals()[action]()
