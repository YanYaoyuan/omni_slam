"""One-shot, preselected Hypersim comparison; predictions precede GT decoding.

This is an observed-surface class-agnostic instance evaluation, not an official
Hypersim challenge score. No test-driven parameter or scene selection.
"""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np
from paper_final_common import ROOT,read,dump,sha,FROZEN,CGPY
from omni_semantic_map.visibility_cleanup import CleanupConfig,frame_evidence,project_visibility,rejection_mask,statistical_outliers
from omni_semantic_map.mask_recall_revision import RecallConfig,revise

OUT=ROOT/'build/benchmarks/hypersim_independent_20261113'
SHARED=ROOT/'build/benchmarks/hypersim_shared_v1'
DATA=ROOT/'build/datasets/hypersim_holdout_v1'
META=ROOT/'build/datasets/hypersim_prepare'
METHODS=['cg_shared','history2','sor','visibility1','simple_ratio','mc_point_ratio',
         'recall_pooled','recall_per_view','recall_leave_point_out','maskclustering_shared']


def freeze():
    assert not (OUT/'protocol.json').exists()
    selection=read(META/'reserved_test_scenes.json');scenes=list(selection['scenes'])
    paths=[Path(__file__),ROOT/'scripts/benchmarks/paper_final_mapping.py',ROOT/'scripts/benchmarks/run_livo_history_graph.py',
        ROOT/'scripts/benchmarks/maskclustering_shared.py',ROOT/'scripts/benchmarks/run_split_queue.py',
        ROOT/'scripts/benchmarks/prepare_hypersim_job.py',FROZEN/'config/cg_rgbd.json']
    paths += [ROOT/'omni_semantic_map/omni_semantic_map'/n for n in ['visibility_cleanup.py','mask_recall_revision.py',
        'history_graph.py','replay_fusion.py','fusion_diagnostics.py','shared_observations.py','evidence_readout.py',
        'full_ovo_evaluation.py','mesh_instance_evaluation.py','instance_error_audit.py']]
    from maskclustering_shared import UP,CPU,FILES
    paths += [UP/n for n in FILES]+[CPU/n for n in FILES]
    jobs={};inputs={str(META/'reserved_test_scenes.json'):sha(META/'reserved_test_scenes.json')}
    for s in scenes:
        path=SHARED/'jobs'/(s+'.json');job=read(path)
        assert job['kind']=='hypersim_reserved_test' and len(job['frames'])==100
        assert [r['frame_id'] for r in job['frames']]==selection['scenes'][s]
        assert not Path(job['output']).exists(),'Freeze before test inference'
        jobs[s]=str(path);inputs[str(path)]=sha(path)
        inputs[str(DATA/(s+'_download.json'))]=sha(DATA/(s+'_download.json'))
        for p,h in job['input_sha256'].items():assert sha(p)==h
    paths=list(dict.fromkeys(paths));source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=scenes,jobs=jobs,input_sha256=inputs,
        source_sha256=source_hashes,methods=METHODS,cleanup_config=asdict(CleanupConfig()),recall_config=asdict(RecallConfig()),
        selection='Previously reserved first two lexicographic official test scenes, cam_00, all 100 public frames each; no scene replacement after scoring.',
        status='One-shot evaluation of frozen candidates, not a claim that a development gate has passed. User requested prioritizing public benchmark after robot annotation was unavailable.',
        frontend='Identical frozen SAM2/SigLIP learned-merger cache for every method, official HDR exposure normalization, metric axial depth, given poses. No frontend or association tuning on these scenes.',
        comparison='CG and MaskClustering are shared-input core adaptations, not their original full detector pipelines. SOR, visibility1, ratio filters and recall variants act on the exact same history2 map. Classes and point identities are unchanged by filters.',
        final_processing='CG native denoise/filter/guarded merge inherited. History2 adds two-frame historical bridges with coframe veto before the same final processing. Each cleanup is one alternative final point-rejection stage; do not stack SOR and another filter.',
        primary='Class-agnostic observed-surface instance PQ/SQ/RQ using strict IoU>0.5 and the existing assigned-point nearest-5cm projector. All nonempty reference instances included. Stable source export order resolves exact ties.',
        reference=dict(pixel_stride=4,voxel_size_m=.05,excluded_nyu40=[1,2,22],
            valid='Finite positive released radial depth with axial depth <=65.535 m (the converted input range), semantic class 1..40 and nonnegative instance ID. Unannotated pixels excluded. Wall/floor/ceiling map to background 0; other raw instance IDs shifted by +1.',
            aggregation='Union of every selected input frame sampled on a fixed 4-pixel grid. Voxel centroid of sampled points; majority instance label including background, lowest ID on ties. No predicted map defines reference support.',
            small_object_sensitivity='Also score after excluding reference-instance voxels belonging to instances smaller than 20 reference voxels; keep background voxels. Report separately, never select the more favorable metric.',
            limitation='Observed surfaces only; not complete purchased meshes or official Hypersim benchmark. Original foreground vocabulary retains its existing structural exclusions, so ontology mismatches are not silently removed from GT.'),
        sealing='All geometry/classification outputs from both scenes hashed before semantic or instance GT is decoded. Common reference and evaluator for every method.',
        stop='One frozen configuration. Report all methods/scenes, gained/lost matches and resource costs. No test-score feedback into this version. Any implementation correction requires an explicit provenance record before re-scoring.'))
    for p in paths:
        dest=OUT/'sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    print('FROZEN',scenes,len(METHODS),'methods',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for path,h in p['source_sha256'].items():assert sha(ROOT/path)==h,path
    for path,h in p['input_sha256'].items():assert sha(path)==h,path
    return p


def paths_for(scene):
    maps=OUT/'maps'/scene
    paths={'cg_shared':maps/'cg/cg','history2':maps/'cg/history2','maskclustering_shared':maps/'maskclustering'}
    paths.update({m:maps/'cleanup'/m for m in METHODS if m not in paths})
    return paths


def clean(scene):
    protocol=verify();job=read(protocol['jobs'][scene]);cache=Path(job['output'])/'cache'
    source=OUT/'maps'/scene/'cg/history2';target=OUT/'maps'/scene/'cleanup';target.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);cfg=CleanupConfig(**protocol['cleanup_config']);start=time.monotonic()
    m=read(cache/'manifest.json');assert m['complete'];g=np.load(source/'geometry.npz');xyz,labels=g['xyz'],g['instance']
    for name,h in m['files_sha256'].items():assert sha(cache/name)==h
    rows=read(source/'classification.json')['objects'];kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
    fg=np.flatnonzero(np.isin(labels,kept));points,ids=xyz[fg],labels[fg]
    owner={int(o):r['object_id'] for r in rows for o in r['observation_ids']}
    poses=np.load(cache/'poses.npy');frames={r['frame_id']:r for r in job['frames']};cam=m['config']['cam']
    observations=np.zeros((len(poses),len(fg)),np.uint8);known_counts=np.zeros(len(fg),np.uint16)
    votes=np.zeros((2,len(fg)),np.uint16);events=[]
    for i,rec in enumerate(m['records']):
        f=cache/'frames'/rec['file'];assert sha(f)==rec['sha256']
        d=frames[rec['frame_id']]['depth'];assert sha(d)==job['input_sha256'][d]
        depth=cv2.imread(d,cv2.IMREAD_UNCHANGED).astype(np.float32)/cam['depth_scale']
        with np.load(f) as z:masks=np.unpackbits(z['masks_packed'],axis=2)[:,:,:cam['W']].astype(bool)
        owners=np.array([owner.get(rec['observation_offset']+n,0) for n in range(rec['masks'])])
        p,n,_,_,visible=frame_evidence(points,ids,poses[i],cam,depth,masks,owners,cfg);votes+=np.array([p,n],np.uint16)
        x,y,_,valid=project_visibility(points,poses[i],cam,depth,cfg);known=masks.any(0)
        for oid in np.intersect1d(owners,kept):
            local=np.flatnonzero(valid&(ids==oid));union=masks[owners==oid].any(0)
            observations[i,local]=1+union[y[local],x[local]].astype(np.uint8)
            known_counts[local]+=known[y[local],x[local]]
        events.append(dict(frame_id=int(rec['frame_id']),visible=int(visible.sum()),positive=int(p.sum()),negative=int(n.sum())))
    visible=(observations>0).sum(0);positive=(observations==2).sum(0)
    reject={'sor':statistical_outliers(points,ids,cfg),'visibility1':rejection_mask(votes[0],votes[1],cfg,single_frame=True),
            'simple_ratio':(visible>0)&(positive<=.5*visible),'mc_point_ratio':positive<=.5*(known_counts+1e-6)}
    iterative={}
    for mode in ['pooled','per_view','leave_point_out']:
        decision=np.zeros(len(fg),bool);details=[];tic=time.monotonic()
        for oid in kept:
            local=np.flatnonzero(ids==oid);result=revise(observations[:,local],mode,RecallConfig(**protocol['recall_config']))
            decision[local]=result['reject'];details.append(dict(object_id=int(oid),iterations=result['iterations'],maximum_change=result['maximum_change']))
        reject['recall_'+mode]=decision;iterative[mode]=dict(kernel_seconds=time.monotonic()-tic,objects=details)
    records={}
    for name,removed in reject.items():
        dest=target/name;dest.mkdir();keep=np.ones(len(xyz),bool);keep[fg]=~removed
        np.savez_compressed(dest/'geometry.npz',xyz=xyz[keep],instance=labels[keep]);np.save(dest/'retained_source_indices.npy',np.flatnonzero(keep))
        (dest/'classification.json').write_bytes((source/'classification.json').read_bytes())
        records[name]=dict(removed=int(removed.sum()),original_foreground_points=len(fg),remaining_foreground_points=int((~removed).sum()))
    np.savez_compressed(target/'evidence.npz',source_indices=fg,observations=observations,known_counts=known_counts,positive=votes[0],negative=votes[1])
    dump(target/'events.json',events);dump(target/'run.json',dict(complete=True,no_gt=True,methods=records,iterative=iterative,
        total_seconds=time.monotonic()-start,source_geometry_sha256=sha(source/'geometry.npz'),
        source_classification_sha256=sha(source/'classification.json'),
        files_sha256={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}))


def map_scene(scene,kind):
    p=verify();assert scene in p['scenes'];cache=Path(read(p['jobs'][scene])['output'])/'cache'
    if kind=='cg':
        from paper_final_mapping import run_cg
        run_cg(cache,OUT/'maps'/scene/'cg')
    elif kind=='mc':
        from maskclustering_shared import run
        run(cache,OUT/'maps'/scene/'maskclustering')
    else:clean(scene)


def seal():
    p=verify();assert not (OUT/'predictions_complete.json').exists();records={}
    for scene in p['scenes']:
        maps=OUT/'maps'/scene
        for folder in ['cg','maskclustering','cleanup']:
            assert read(maps/folder/'run.json')['complete'],folder
        methods={}
        for method,path in paths_for(scene).items():
            methods[method]={name:dict(path=str(path/name),sha256=sha(path/name)) for name in ['geometry.npz','classification.json']}
        records[scene]=methods
    dump(OUT/'predictions_complete.json',dict(complete=True,methods=records,protocol_sha256=sha(OUT/'protocol.json'),created_epoch=time.time()))


def gate():
    p=verify();g=read(OUT/'predictions_complete.json');assert g['complete'] and g['protocol_sha256']==sha(OUT/'protocol.json')
    for scene in g['methods'].values():
        for method in scene.values():
            for r in method.values():assert sha(r['path'])==r['sha256']
    return p,g


def reference(scene):
    p,_=gate();assert scene in p['scenes'];sys.path.insert(0,str(META/'python_deps'))
    import h5py
    import csv
    def array(path):
        with h5py.File(path,'r') as h:return h['dataset'][:]
    target=OUT/'reference'/scene;target.mkdir(parents=True,exist_ok=False)
    d=read(DATA/(scene+'_download.json'));hashes={str((ROOT/r['path']).resolve()):r['sha256'] for r in d['files']}
    job=read(p['jobs'][scene]);cam=job['config']['cam'];stride=p['reference']['pixel_stride']
    h,w=cam['H'],cam['W'];y,x=np.meshgrid(np.arange(0,h,stride),np.arange(0,w,stride),indexing='ij')
    rays=np.stack([(x-cam['cx'])/cam['fx'],(y-cam['cy'])/cam['fy'],np.ones_like(x)],-1).astype(np.float64)
    rays/=np.linalg.norm(rays,axis=2,keepdims=True)
    clouds=[];labels=[];input_hashes={};frame_stats=[]
    for r in job['frames']:
        prefix=f"frame.{r['frame_id']:04d}";folder=DATA/'data'/scene/'images/scene_cam_00_geometry_hdf5'
        values=[]
        for name in ['depth_meters','semantic','semantic_instance']:
            path=folder/(prefix+'.'+name+'.hdf5');expected=hashes[str(path.resolve())];assert sha(path)==expected
            input_hashes[str(path)]=expected;values.append(array(path)[::stride,::stride])
        distance,semantic,instance=values;distance=distance.astype(float)
        valid=np.isfinite(distance)&(distance>0)&(distance*rays[:,:,2]<=65.535)&(semantic>=1)&(semantic<=40)&(instance>=0)
        pose=np.asarray(r['pose']);world=(rays*distance[:,:,None])@pose[:3,:3].T+pose[:3,3]
        lab=np.where(np.isin(semantic,p['reference']['excluded_nyu40']),0,instance.astype(np.int64)+1)
        clouds.append(world[valid]);labels.append(lab[valid]);frame_stats.append(dict(frame_id=r['frame_id'],valid_samples=int(valid.sum())))
    xyz=np.concatenate(clouds);lab=np.concatenate(labels);assert len(xyz)>0
    vox=np.floor(xyz/p['reference']['voxel_size_m']).astype(np.int64);keys,inverse=np.unique(vox,axis=0,return_inverse=True)
    counts=np.bincount(inverse);centers=np.column_stack([np.bincount(inverse,weights=xyz[:,k])/counts for k in range(3)])
    pairs,n=np.unique(np.column_stack([inverse,lab]),axis=0,return_counts=True)
    order=np.lexsort((pairs[:,1],-n,pairs[:,0]));ordered=pairs[order]
    first=np.r_[True,ordered[1:,0]!=ordered[:-1,0]];truth=ordered[first,1]
    np.testing.assert_array_equal(ordered[first,0],np.arange(len(keys)))
    np.savez_compressed(target/'reference.npz',xyz=centers.astype(np.float32),instance=truth,voxel=keys)
    dump(target/'manifest.json',dict(complete=True,scene=scene,protocol_sha256=sha(OUT/'protocol.json'),
        prediction_gate_sha256=sha(OUT/'predictions_complete.json'),input_sha256=input_hashes,
        reference_sha256=sha(target/'reference.npz'),samples=len(xyz),voxels=len(keys),
        foreground_instances=len(np.unique(truth[truth>0])),frames=frame_stats))
    print('REFERENCE',scene,len(keys),'voxels',flush=True)


def evaluate():
    from omni_semantic_map.mesh_instance_evaluation import transfer_labels
    from omni_semantic_map.full_ovo_evaluation import voxel_metrics
    from omni_semantic_map.instance_error_audit import audit_instances,filter_transition
    p,g=gate();assert not (OUT/'summary.json').exists();results={}
    for scene in p['scenes']:
        path=OUT/'reference'/scene/'reference.npz';manifest=read(path.parent/'manifest.json');assert sha(path)==manifest['reference_sha256']
        ref=np.load(path);truth=ref['instance'];gt,count=np.unique(truth,return_counts=True)
        small=gt[(gt>0)&(count<20)];large_domain=~np.isin(truth,small);methods={};audits={}
        for method,folder in paths_for(scene).items():
            geom=np.load(folder/'geometry.npz');rows=read(folder/'classification.json')['objects']
            kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
            lab=np.where(np.isin(geom['instance'],kept),geom['instance'],0)
            projected,_=transfer_labels(geom['xyz'],lab,ref['xyz'])
            audit=audit_instances(truth,projected);audits[method]=audit
            methods[method]=dict(primary=voxel_metrics(truth,projected),minimum20_sensitivity=voxel_metrics(truth[large_domain],projected[large_domain]))
            target=OUT/'evaluation'/scene;dump(target/(method+'_audit.json'),audit);np.savez_compressed(target/(method+'.npz'),instance=projected)
        for method in methods:
            methods[method]['vs_history2']=filter_transition(audits['history2'],audits[method])
            methods[method]['vs_simple_ratio']=filter_transition(audits['simple_ratio'],audits[method])
        results[scene]=dict(methods=methods,reference_voxels=len(truth),small_gt_instances=len(small))
        print(scene,{m:round(v['primary']['observed_voxel_instance_pq']*100,3) for m,v in methods.items()},flush=True)
    means={m:float(np.mean([r['methods'][m]['primary']['observed_voxel_instance_pq']*100 for r in results.values()])) for m in METHODS}
    dump(OUT/'summary.json',dict(complete=True,results=results,scene_mean_pq100=means,protocol_sha256=sha(OUT/'protocol.json'),
        scope=p['primary'],official_benchmark_score=False,never_used_for_tuning_this_version=True))


def queue():
    p=verify();state_path=OUT/'queue_status.json';assert not state_path.exists()
    state=dict(pid=os.getpid(),complete=False,stage='wait_training_adapter',child_pid=None,completed=[])
    def publish():state['updated_epoch']=time.time();dump(state_path,state)
    publish();smoke_path=SHARED/'smoke_queue_status.json'
    while True:
        smoke=read(smoke_path)
        if smoke['complete']:break
        if smoke.get('state')=='failed':raise RuntimeError('Training adapter failed; inspect before test inference')
        os.kill(smoke['pid'],0)  # A status file alone is not a live process.
        state['waiting_on_pid']=smoke['pid'];publish();time.sleep(10)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'omni_semantic_map'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    def child(cmd,label):
        verify();log=logs/(label+'.log')
        with log.open('x') as f:
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            state.update(stage=label,child_pid=process.pid,log=str(log));publish()
            while process.poll() is None:publish();time.sleep(2)
            if process.returncode:
                state.update(stage='failed',failed_step=label,returncode=process.returncode,child_pid=None);publish()
                raise RuntimeError(label+' failed')
        state['completed'].append(label);state['child_pid']=None;publish()
    child([str(CGPY),'scripts/benchmarks/run_split_queue.py','--jobs',*[p['jobs'][s] for s in p['scenes']],
           '--gpu','0','--status',str(OUT/'frontend_queue_status.json')],'shared_frontend')
    script=str(Path(__file__).resolve())
    for scene in p['scenes']:
        for action in ['cg','mc','cleanup']:child([str(CGPY),script,action,'--scene',scene],scene+'_'+action)
    child([str(CGPY),script,'seal'],'seal_all_predictions')
    for scene in p['scenes']:child([str(CGPY),script,'reference','--scene',scene],scene+'_reference')
    child([str(CGPY),script,'evaluate'],'evaluate_once')
    state.update(complete=True,stage='complete',child_pid=None);publish()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','queue','cg','mc','cleanup','seal','reference','evaluate']);a.add_argument('--scene');args=a.parse_args()
    if args.action in ['cg','mc','cleanup']:map_scene(args.scene,args.action)
    elif args.action=='reference':reference(args.scene)
    else:globals()[args.action]()
