"""Supplementary named-annotation diagnostic, never a full-scene benchmark.

Unspecified XML text does not identify whether a segment is foreground or
background. Evaluate only named main-release labels, keep named background,
report ignored coverage, and preserve all original predictions and v1 scores.
This protocol was introduced after inspecting v1 scores, not preregistered.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree
from audit_evidence_retention import ROOT, OUT, REPORT, dump
from prepare_retention_holdout import DATA, SCENES
from evaluate_retention_holdout import mesh
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.flat_surface_evaluation import majority_labels
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.instance_error_audit import audit_instances

PROTOCOL=OUT/'stage3_named_domain_protocol.json'
PREFIXES=('wall','floor','ceiling','unlabeled')


def freeze():
    assert not PROTOCOL.exists()
    sources=[Path(__file__),ROOT/'scripts/benchmarks/evaluate_retention_holdout.py']
    sources += [ROOT/'omni_semantic_map/omni_semantic_map'/n for n in ['flat_benchmark.py','flat_surface_evaluation.py','full_ovo_evaluation.py','mesh_instance_evaluation.py','instance_error_audit.py']]
    files={str(p):sha256(p) for p in sources}
    for s in SCENES:
        for p in [DATA/'main'/s/(s+'.xml'),DATA/'main'/s/(s+'.ply'),OUT/'metric_inputs'/s/'poses.npz',OUT/'heldout_runs'/s/'run.json',OUT/'heldout_runs'/s/'geometry.npz',OUT/'heldout_runs'/s/'classification.json']:
            files[str(p)]=sha256(p)
    dump(PROTOCOL,dict(scope=__doc__,scenes=SCENES,methods=['mean','area_density'],no_prediction_changes=True,
        status='post-hoc evaluation-scope diagnostic; not a replacement full-scene benchmark',
        selection_sha256=sha256(OUT/'stage3_selection.json'),original_evaluation_sha256=sha256(OUT/'stage3_evaluation_protocol.json'),
        geometry='Same label-independent mapper, canonical coordinates, <=5cm GT vertex match, 5cm floor voxels.',
        vote='Vote original instance IDs before masking. Ties use smaller ID. Do not merge all unknown IDs into background.',
        valid='Majority label has nonempty stripped XML text, ID>0, and text does not start unlabeled. All other voxels ignored.',
        background='Within valid domain, wall/floor/ceiling prefixes mapped to 0; all other named segments retain instance ID.',
        predictions='Original retained prediction geometry and classes, identical transfer kernel, no clipping before nearest-neighbour transfer.',
        limitations='Conditional precision/recall on named domain. Unknown surfaces are excluded, so full-scene false positives and object recall remain unmeasured.',
        sources={'official_release':'https://github.com/hkust-vgd/scenenn','extra_annotation_readme':'https://hkust-vgd.ust.hk/scenenn/contrib/nyu_class/README.txt'},
        files_sha256=files))
    dump(REPORT/'stage3/named_domain_protocol.json',json.loads(PROTOCOL.read_text()))


def evaluate():
    protocol=json.loads(PROTOCOL.read_text())
    for name,h in protocol['files_sha256'].items():assert sha256(Path(name))==h,name
    summaries=[]
    for scene in SCENES:
        output=OUT/'heldout_named_evaluation'/scene;output.mkdir(parents=True,exist_ok=False)
        run=OUT/'heldout_runs'/scene;status=json.loads((run/'run.json').read_text());assert status['complete'] and status['no_gt']
        for name,h in status['files_sha256'].items():assert sha256(run/name)==h
        cache=DATA/'strong_frontend/v1'/scene/'cache'
        assert sha256(cache/'manifest.json')==status['cache_manifest_sha256']
        m=json.loads((cache/'manifest.json').read_text())
        assert sha256(cache/'geometry.npz')==m['files_sha256']['geometry.npz']
        prediction=np.load(run/'geometry.npz');rows=json.loads((run/'classification.json').read_text())['objects']
        # All prediction files exist and verify before GT enters the evaluator.
        xyz,raw=mesh(DATA/'main'/scene/(scene+'.ply'));basis=np.load(OUT/'metric_inputs'/scene/'poses.npz')['canonical_from_raw']
        xyz=xyz@basis[:3,:3].T+basis[:3,3]
        xml={int(x.get('id')):x.get('text','').strip().lower() for x in ET.parse(DATA/'main'/scene/(scene+'.xml')).getroot().iter('label')}
        points=np.load(cache/'geometry.npz')['xyz'];dist,idx=cKDTree(xyz).query(points,workers=1);near=dist<=.05
        vox,inv=np.unique(np.floor(points[near]/.05).astype(np.int64),axis=0,return_inverse=True)
        majority=majority_labels(inv,raw[idx[near]],len(vox));vertices=(vox+.5)*.05
        named=[k for k,name in xml.items() if k>0 and name and not name.startswith('unlabeled')]
        bg=[k for k in named if xml[k].startswith(PREFIXES[:3])]
        valid=np.isin(majority,named);truth=np.where(np.isin(majority,bg),0,majority)
        result=dict(scene=scene,complete=False,protocol_sha256=sha256(PROTOCOL),prediction_run_sha256=sha256(run/'run.json'),
            domain=dict(observed_voxels=len(vox),named_voxels=int(valid.sum()),named_fraction=float(valid.mean()),
                        ignored_voxels=int((~valid).sum()),gt_instances=int(len(np.unique(truth[valid&(truth>0)]))),
                        named_background_voxels=int((valid&(truth==0)).sum()),xml_segments=len(xml),unnamed_xml_segments=sum(not v for v in xml.values())),methods={})
        np.savez_compressed(output/'reference.npz',xyz=vertices,raw_instance=majority,instance=truth,valid=valid)
        for method in protocol['methods']:
            retained=[r['object_id'] for r in rows if r['predictions'][method]['retained']]
            labels=np.where(np.isin(prediction['instance'],retained),prediction['instance'],0)
            projected,_=transfer_labels(prediction['xyz'],labels,vertices,mode='nearest_5cm')
            result['methods'][method]=voxel_metrics(truth[valid],projected[valid])
            dump(output/(method+'_audit.json'),audit_instances(truth[valid],projected[valid]))
            np.savez_compressed(output/(method+'.npz'),instance=projected)
        result['complete']=True;dump(output/'metrics.json',result);summaries.append(result)
        print('NAMED DOMAIN',scene,{k:100*v['observed_voxel_instance_pq'] for k,v in result['methods'].items()},result['domain'],flush=True)
    means={m:float(np.mean([r['methods'][m]['observed_voxel_instance_pq'] for r in summaries])) for m in protocol['methods']}
    summary=dict(complete=True,scope=__doc__,protocol_sha256=sha256(PROTOCOL),scenes=summaries,macro_pq=means,
        paired_delta_pq100=100*(means['area_density']-means['mean']),full_scene_validation_complete=False)
    dump(OUT/'heldout_named_evaluation/summary.json',summary);dump(REPORT/'stage3/named_domain_summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    if a.run:evaluate()
