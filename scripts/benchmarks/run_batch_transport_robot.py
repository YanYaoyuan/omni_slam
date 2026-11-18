"""Same frozen mechanism and four controls on all 330 robot frames.

This fulfills the requested robot comparison even if the public quality gate
fails. It is a bounded diagnostic, never a promotion or semantic-accuracy claim.
"""
import argparse
import inspect
import shutil
import time
from pathlib import Path
import numpy as np
import run_batch_transport_assessment as native
from run_batch_transport_assessment_corrected import verify_amendment
from audit_transport_unknown_seeds import corrected_graph_and_seeds
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.discovery_measured_input import robot_measured_visibility

PUBLIC=ROOT/'build/benchmarks/batch_transport_assessment_v1'
OUT=ROOT/'build/benchmarks/batch_transport_robot_v1'
INPUT=ROOT/'build/benchmarks/discovery_measured_inputs_v1'
KEY='rosbag2245'
BASE=OUT/'baseline'
FULL=OUT/'full_measured'
SEM=OUT/'semantics'
native_adapter=native.isolated_adapter


def robot_adapter(filename):
    module=native_adapter(filename)
    if filename=='capture_closed_vocabulary_semantics.py':
        source=inspect.getsource(module.predict)
        old="raw=frames[rec['frame_id']];np.testing.assert_allclose(rec['camera_pose'],raw['pose'],rtol=0,atol=1e-10);depth=np.load(raw['measured'])['depth'];x,y,_,visible=project_measured_visibility(base['xyz'],np.asarray(raw['pose']),job['config']['cam'],depth,SimpleNamespace(depth_absolute_m=.05,depth_relative=0.));ids=np.flatnonzero(visible);"
        new="raw=frames[rec['frame_id']];np.testing.assert_allclose(rec['camera_pose'],raw['camera_pose'],rtol=0,atol=1e-10);ids,px,_=robot_measured_visibility(base['xyz'],np.load(raw['cloud'])['points'],raw);x=np.zeros(len(base['xyz']),np.int32);y=x.copy();x[ids]=px[:,0];y[ids]=px[:,1];"
        assert source.count(old)==1;source=source.replace(old,new);namespace=dict(module.__dict__);namespace['robot_measured_visibility']=robot_measured_visibility
        exec(compile(source,str(__file__)+':robot_projection','exec'),namespace);fn=namespace['predict']
        def predict():
            namespace.update(module.__dict__)
            return fn()
        module.predict=predict
    return module


def freeze():
    verify_amendment();native.verify(True);assert not OUT.exists()
    summary=read(PUBLIC/'summary.json');assert summary['complete'] and read(PUBLIC/'prediction_audit.json')['complete']
    old=ROOT/'build/benchmarks/robot_geometry_reference_v1';native.seal(old/'maps_complete.json')
    manifest=read(INPUT/KEY/'manifest.json');assert len(manifest['records'])==330
    scene=next(s for s in read(ROOT/'build/benchmarks/discovery_query_benchmark_v1/protocol.json')['scenes'] if s['key']==KEY)
    inputs={}
    paths=[PUBLIC/'protocol.json',PUBLIC/'summary.json',PUBLIC/'prediction_audit.json',PUBLIC/'unknown_semantic_amendment.json',PUBLIC/'empty_input_amendment.json',old/'maps_complete.json',INPUT/KEY/'manifest.json',Path(manifest['geometry']),Path(scene['job'])]
    for name in ['geometry.npz','classification.json','groups.json','run.json']:paths.append(old/'maps/geometry_graph_reference'/name)
    paths += [ROOT/'build/benchmarks/surface_ownership_v1/features'/KEY/'core_sources.json',ROOT/'build/benchmarks/intrinsic_surface_ownership_v1/maps'/KEY/'intrinsic_geometry/geometry.npz']
    for r in manifest['records']:paths.extend([Path(r['rgb']),Path(r['file'])])
    job=read(scene['job'])
    for r in job['frames']:paths.append(Path(r['cloud']))
    # Robot packets carry tracking pose AND camera pose; use packet intrinsics,
    # floor-pixel projection and current measured cloud, never the RGB-D job's
    # unrelated default camera configuration.
    robot_adapter('capture_closed_vocabulary_semantics.py')
    sources_xyz=np.load(manifest['geometry'])['xyz'];byframe={r['frame_id']:r for r in job['frames']}
    for rec in manifest['records']:
        frame=byframe[rec['frame_id']];ids,px,_=robot_measured_visibility(sources_xyz[:rec['map_points']],np.load(frame['cloud'])['points'],frame);expected=np.load(rec['file'])
        np.testing.assert_array_equal(ids,expected['visible_source_ids']);np.testing.assert_array_equal(px,expected['pixel_xy']);np.testing.assert_array_equal(rec['camera_pose'],frame['camera_pose'])
    model=ROOT/'build/models/segformer_b5_cityscapes'
    paths += [model/name for name in read(model/'source_manifest.json')['files']]
    for f in paths:inputs[str(f)]=sha(f)
    sources={str(Path(__file__)):sha(__file__)}
    for name in ['run_batch_transport_assessment.py','audit_transport_unknown_seeds.py','capture_closed_vocabulary_semantics.py','evaluate_closed_vocabulary_semantics.py','run_batch_stable_transport.py']:
        f=ROOT/'scripts/benchmarks'/name;sources[str(f)]=sha(f)
    for name in ['batch_stable_transport.py','batch_stable_transport.cpp','discovery_measured_input.py']:
        f=ROOT/'omni_semantic_map/omni_semantic_map'/name;sources[str(f)]=sha(f)
    OUT.mkdir();p=dict(complete=True,created_epoch=time.time(),scenes=[scene],methods=native.METHODS,input_hashes=inputs,source_hashes=sources,
        scope='Same source-frame certificate and four controls on existing330robotframes, no parameter/model/ID changes. Fixed measured archived LIVO tracking geometry, not original beams.',
        public_gate_passed=summary['independent_gate'],explicit_scope_amendment='Public gate remains failed. Supersede only self-imposed robot execution stop to deliver the user-requested same-version comparison as a diagnostic. No candidate promotion, replacement public scene, extra fit or sweep.',
        budget='330sharedSegFormerFP32forwards,onegraph,fivefinalmaps; reuse cached strong reference. No new training, prompt/threshold search or robot-specific parameter.',
        metrics='No independent robotGT: report source/identity invariants, per-object geometry changes, actual RGB/3D, and module runtime only. No semanticAP/accuracy or S100 realtime claim.',
        mechanism='Exact public corrected_graph_and_seeds and native nearest_groups/certify; same20cmcells,16NNstrict0.4m,source-acquisitionframebatch and1e-10comparison. RGB semantics unknowns abstain. OriginalcoreandIDs/classes/scores fixed.',
        robot_projection='Existing robot_measured_visibility floor-pixel current-cloud zbuffer, packet640x360intrinsics/camera_pose; all330cachedsourcevisibilityoutputs replayed exactly before freeze. SharedSegFormer unchanged; only map diagnostic projection uses robot sensor adapter.')
    dump(OUT/'protocol.json',p);SEM.mkdir();shutil.copyfile(OUT/'protocol.json',SEM/'protocol.json')
    folder=BASE/KEY/'geometry_graph';folder.mkdir(parents=True);files={}
    for name in ['geometry.npz','classification.json','groups.json','run.json']:
        f=folder/name;shutil.copyfile(old/'maps/geometry_graph_reference'/name,f);files[str(f)]=sha(f)
    dump(OUT/'baseline_complete.json',dict(complete=True,files=files,reused_existing_strong_reference=True))
    geometry=dict(np.load(manifest['geometry']));n=len(geometry['xyz']);geometry['old_source_id']=np.arange(n,dtype=np.int64)
    f=FULL/KEY;f.mkdir(parents=True);np.savez_compressed(f/'geometry.npz',**geometry);np.save(f/'old_to_full.npy',np.arange(n,dtype=np.int64))
    owners=native.source_owners(ROOT/'build/benchmarks/surface_ownership_v1/features'/KEY/'core_sources.json',ROOT/'build/benchmarks/intrinsic_surface_ownership_v1/maps'/KEY/'intrinsic_geometry',folder,n);declared=np.full(n,-1,np.int16)
    for obj in read(folder/'classification.json')['objects']:declared[owners==obj['object_id']]=obj['predictions']['mean']['train_id']
    assert (declared[owners>0]>=0).all();a=OUT/'anchors'/KEY;a.mkdir(parents=True);np.savez_compressed(a/'predictions.npz',owners=owners,declared_train_id=declared)
    dump(OUT/'anchors_complete.json',dict(complete=True,files={str(a/'predictions.npz'):sha(a/'predictions.npz')}))
    dump(OUT/'prepared_complete.json',dict(complete=True,files={str(v):sha(v) for v in f.iterdir()},processed_measured_sources=n,frames=330,new_raw_lidar_sources=0))
    print('ROBOT COMPARISON FROZEN',n,'sources / 330frames / 5maps',flush=True)


def verify(data=False):
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    native.seal(OUT/'prepared_complete.json');return p


def stage_start(name):
    verify(True);f=OUT/'stages'/(name+'.json');assert not f.exists()
    dump(f,dict(started_epoch=time.time(),stage=name,protocol_sha256=sha(OUT/'protocol.json')))


def audit():
    verify();native.seal(OUT/'predictions_complete.json');base=np.load(BASE/KEY/'geometry_graph/geometry.npz');source=np.load(FULL/KEY/'geometry.npz')['xyz'];owners=np.load(OUT/'anchors'/KEY/'predictions.npz')['owners'];field=np.load(OUT/'fields'/KEY/'field.npz');n=len(base['xyz']);results={};assignments={}
    for method in native.METHODS:
        f=OUT/'final_maps'/KEY/method;g=np.load(f/'geometry.npz');np.testing.assert_array_equal(g['xyz'][:n],base['xyz']);np.testing.assert_array_equal(g['instance'][:n],base['instance']);assert sha(f/'classification.json')==sha(BASE/KEY/'geometry_graph/classification.json')
        labels=np.zeros(len(source),np.int64);added=np.empty(0,np.int64)
        if method!='baseline':
            name=dict(same_cell='same_cell',full_nearest='nearest',full_two_cells='cell',ours_frame='frame')[method];decisions=field[name+'_owner'][field['source_to_cell']];added=np.flatnonzero((owners==0)&(decisions>0));np.testing.assert_array_equal(added,g['added_full_source_ids']);np.testing.assert_array_equal(g['xyz'][n:],source[added]);np.testing.assert_array_equal(g['instance'][n:],decisions[added]);labels[added]=decisions[added]
        assert set(np.unique(g['instance']))==set(np.unique(base['instance']));assignments[method]=labels
        objects=[]
        for oid in np.unique(base['instance']):
            before=base['xyz'][base['instance']==oid];after=g['xyz'][g['instance']==oid];objects.append(dict(object_id=int(oid),old_points=len(before),new_points=len(after),added_points=len(after)-len(before),old_extent=np.ptp(before,axis=0).tolist(),new_extent=np.ptp(after,axis=0).tolist()))
        results[method]=dict(points=len(g['xyz']),objects=len(objects),added_sources=len(added),per_object=objects,original_geometry_identity_classification_exact=True,all_added_measured_sources_exact=True)
    differences={a+' vs '+b:int(np.count_nonzero(assignments[a]!=assignments[b])) for i,a in enumerate(native.METHODS) for b in native.METHODS[i+1:]}
    dump(OUT/'summary.json',dict(complete=True,results=results,different_source_assignments=differences,frames=330,geometry_field=read(OUT/'fields_complete.json')['stats'][KEY],semantics=read(SEM/'predictions_complete.json')['seconds'],semantic_accuracy_available=False,public_gate_passed=read(PUBLIC/'summary.json')['independent_gate'],scope='Robot diagnostic comparison only; changed geometry is not verified semantic improvement.',protocol_sha256=sha(OUT/'protocol.json'),prediction_seal_sha256=sha(OUT/'predictions_complete.json')))
    print('ROBOT ALL FIVE MAPS AUDITED', {k:{a:v[a] for a in ['points','objects','added_sources']} for k,v in results.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','semantics','aggregate','fields','maps','audit']);action=parser.parse_args().action
    if action=='freeze':freeze()
    else:
        native.OUT=OUT;native.INPUT=INPUT;native.FULL=FULL;native.BASE=BASE;native.SEM=SEM;native.verify=verify;native.stage_start=stage_start;native.graph_and_seeds=corrected_graph_and_seeds;native.isolated_adapter=robot_adapter
        audit() if action=='audit' else getattr(native,action)()
