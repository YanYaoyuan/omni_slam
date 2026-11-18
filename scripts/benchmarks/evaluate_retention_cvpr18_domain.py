"""Original instance IDs with official CVPR18 vertex classes for FG/BG/void.

Evaluation-only correction after v1 scores: blank 2016 XML names cannot define
foreground. The 2018 reannotation supplies NYU classes at exact original mesh
coordinates. No predictions, memberships, text prompts or thresholds change.
This remains our observed-voxel diagnostic, not an official SceneNN leaderboard.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from audit_evidence_retention import ROOT, OUT, REPORT, dump
sys.path.insert(0,str(OUT/'tools/hdf5_reader'))
import h5py
from prepare_retention_holdout import DATA, SCENES
from evaluate_retention_holdout import mesh
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.flat_surface_evaluation import majority_labels
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.instance_error_audit import audit_instances

ANNOTATION=OUT/'official_annotation_audit/cvpr18'
PROTOCOL=OUT/'stage3_cvpr18_domain_protocol.json'


def freeze():
    assert not PROTOCOL.exists();download=json.loads((ANNOTATION/'download.json').read_text());assert download['complete']
    files={r['path']:r['sha256'] for r in download['files']}
    sources=[Path(__file__),ROOT/'scripts/benchmarks/evaluate_retention_holdout.py']
    sources += [ROOT/'omni_semantic_map/omni_semantic_map'/n for n in ['flat_benchmark.py','flat_surface_evaluation.py','full_ovo_evaluation.py','mesh_instance_evaluation.py','instance_error_audit.py']]
    for s in SCENES:
        sources += [DATA/'main'/s/(s+'.ply'),OUT/'metric_inputs'/s/'poses.npz',OUT/'heldout_runs'/s/'run.json',OUT/'heldout_runs'/s/'geometry.npz',OUT/'heldout_runs'/s/'classification.json']
    files.update({str(p):sha256(p) for p in sources})
    dump(PROTOCOL,dict(scope=__doc__,scenes=SCENES,methods=['mean','area_density'],no_prediction_changes=True,
        correction='Post-hoc GT-domain correction, not a new method or a preregistered improvement. Preserve v1 and named-only diagnostics.',
        reference='Original instance mesh in frozen canonical coordinates. Shared native mapper points <=5cm to mesh, same 5cm voxels.',
        annotation='Deduplicate HDF5 coords, require every coordinate to exactly equal an original mesh vertex and duplicate class labels to agree. No GT fitting.',
        semantic_transfer='Nearest official annotation coordinate <=5cm in canonical metric coordinates; include label0 samples in nearest-neighbour search. Otherwise void.',
        foreground='NYU classes 1 wall, 2 floor, 22 ceiling map to background 0; class0 or unavailable annotation becomes void -1; other classes retain original instance ID. Original instance0 always void.',
        vote='Vote foreground IDs, background0 and void-1 on all observed supports together; majority then smallest integer tie. Ignore voxels whose majority is void.',
        comparison='Original predictions; nearest assigned predicted point <=5cm on whole reference, then restrict to shared annotated domain; strict IoU>.5 PQ and scene macro.',
        limits='Annotation samples omit parts of the full-height scene; report coverage. Cross-release region-instance diagnostic, not official 3D instance AP or whole-map recall.',
        source_urls=['https://github.com/hkust-vgd/pointwise','https://hkust-vgd.github.io/scenenn/',
            'https://hkust-vgd.ust.hk/scenenn/home/cvpr18/data/scenenn_seg_76.zip'],files_sha256=files))
    dump(REPORT/'stage3/cvpr18_domain_protocol.json',json.loads(PROTOCOL.read_text()))


def evaluate():
    protocol=json.loads(PROTOCOL.read_text())
    for name,h in protocol['files_sha256'].items():assert sha256(Path(name))==h,name
    results=[]
    for scene in SCENES:
        output=OUT/'heldout_cvpr18_evaluation'/scene;output.mkdir(parents=True,exist_ok=False)
        run=OUT/'heldout_runs'/scene;status=json.loads((run/'run.json').read_text());assert status['complete'] and status['no_gt']
        for name,h in status['files_sha256'].items():assert sha256(run/name)==h
        cache=DATA/'strong_frontend/v1'/scene/'cache';assert sha256(cache/'manifest.json')==status['cache_manifest_sha256']
        m=json.loads((cache/'manifest.json').read_text());assert sha256(cache/'geometry.npz')==m['files_sha256']['geometry.npz']
        prediction=np.load(run/'geometry.npz');rows=json.loads((run/'classification.json').read_text())['objects']
        # Predictions verified before opening the evaluation annotation.
        xyz,instances=mesh(DATA/'main'/scene/(scene+'.ply'))
        with h5py.File(ANNOTATION/f'scenenn_coords_{scene}.hdf5') as f:coords=f['coords'][:].reshape(-1,3)
        with h5py.File(ANNOTATION/f'scenenn_seg_{scene}.hdf5') as f:classes=f['label'][:].reshape(-1)
        assert len(coords)==len(classes) and np.isfinite(coords).all() and classes.min()>=0 and classes.max()<=40
        coords,first,inv=np.unique(coords,axis=0,return_index=True,return_inverse=True)
        unique_classes=classes[first];assert np.array_equal(unique_classes[inv],classes),'conflicting duplicate semantic GT'
        d,idx=cKDTree(xyz).query(coords,workers=2);assert np.all(d==0),'annotation coordinates differ from original mesh; do not fit'
        basis=np.load(OUT/'metric_inputs'/scene/'poses.npz')['canonical_from_raw'];xyz=xyz@basis[:3,:3].T+basis[:3,3];coords=coords@basis[:3,:3].T+basis[:3,3]
        d,idx=cKDTree(coords).query(xyz,workers=2);semantic=np.where(d<=.05,unique_classes[idx],0)
        mesh_truth=np.where(np.isin(semantic,[1,2,22]),0,instances)
        mesh_truth[(semantic==0)|(instances==0)]=-1
        points=np.load(cache/'geometry.npz')['xyz'];d,idx=cKDTree(xyz).query(points,workers=2);near=d<=.05
        vox,inv=np.unique(np.floor(points[near]/.05).astype(np.int64),axis=0,return_inverse=True)
        truth=majority_labels(inv,mesh_truth[idx[near]],len(vox));valid=truth>=0;vertices=(vox+.5)*.05
        result=dict(scene=scene,complete=False,protocol_sha256=sha256(PROTOCOL),prediction_run_sha256=sha256(run/'run.json'),
            domain=dict(unique_official_annotation_vertices=len(coords),annotation_vertices_exact=True,
                observed_voxels=len(vox),valid_voxels=int(valid.sum()),valid_fraction=float(valid.mean()),
                void_voxels=int((~valid).sum()),background_voxels=int((truth==0).sum()),gt_instances=int(len(np.unique(truth[truth>0])))),methods={})
        np.savez_compressed(output/'reference.npz',xyz=vertices,instance=truth,valid=valid)
        for method in protocol['methods']:
            kept=[r['object_id'] for r in rows if r['predictions'][method]['retained']]
            labels=np.where(np.isin(prediction['instance'],kept),prediction['instance'],0)
            projected,_=transfer_labels(prediction['xyz'],labels,vertices,mode='nearest_5cm')
            result['methods'][method]=voxel_metrics(truth[valid],projected[valid])
            dump(output/(method+'_audit.json'),audit_instances(truth[valid],projected[valid]))
            np.savez_compressed(output/(method+'.npz'),instance=projected)
        result['complete']=True;dump(output/'metrics.json',result);results.append(result)
        print('CVPR18 DOMAIN',scene,{k:100*v['observed_voxel_instance_pq'] for k,v in result['methods'].items()},result['domain'],flush=True)
    means={m:float(np.mean([r['methods'][m]['observed_voxel_instance_pq'] for r in results])) for m in protocol['methods']}
    summary=dict(complete=True,scope=__doc__,protocol_sha256=sha256(PROTOCOL),scenes=results,macro_pq=means,
        paired_delta_pq100=100*(means['area_density']-means['mean']),official_leaderboard=False,
        no_model_retuning=True,all_three_frozen_scenes_evaluated=True)
    dump(OUT/'heldout_cvpr18_evaluation/summary.json',summary);dump(REPORT/'stage3/cvpr18_domain_summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    if a.run:evaluate()
