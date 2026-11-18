"""Evaluate frozen predictions on shared GT-matched observed 5 cm voxels.

No geometry/threshold fitting: the metric coordinate basis is already frozen
from poses alone. GT mesh labels are opened only here, after prediction files
exist. This is a paired diagnostic, not the official SceneNN benchmark.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree
from audit_evidence_retention import ROOT, OUT, REPORT, dump
from prepare_retention_holdout import DATA, SCENES
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.flat_surface_evaluation import majority_labels
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.instance_error_audit import audit_instances


def mesh(path):
    with path.open('rb') as f:
        lines=[]
        while True:
            raw=f.readline()
            if not raw:raise ValueError('unterminated PLY header')
            line=raw.decode('ascii').strip();lines.append(line)
            if line=='end_header':break
        assert 'format binary_little_endian 1.0' in lines
        k=next(i for i,s in enumerate(lines) if s.startswith('element vertex '));n=int(lines[k].split()[-1])
        types={'float':'<f4','float32':'<f4','double':'<f8','uint':'<u4','uint32':'<u4','int':'<i4','uchar':'u1','uint8':'u1'};fields=[]
        for line in lines[k+1:]:
            if not line.startswith('property '):break
            _,kind,name=line.split();fields.append((name,types[kind]))
        values=np.fromfile(f,dtype=np.dtype(fields),count=n)
        assert len(values)==n and 'label' in values.dtype.names
    return np.column_stack([values[k] for k in ['x','y','z']]),values['label'].astype(np.int64)


def freeze():
    path=OUT/'stage3_evaluation_protocol.json';assert not path.exists()
    files=[Path(__file__),ROOT/'scripts/benchmarks/retention_holdout_backend.py']
    files += [ROOT/'omni_semantic_map/omni_semantic_map'/name for name in ['evidence_readout.py','replay_fusion.py','fusion_diagnostics.py','full_ovo_evaluation.py','flat_benchmark.py','flat_surface_evaluation.py','mesh_instance_evaluation.py','instance_error_audit.py']]
    dump(path,dict(scenes=SCENES,methods=['mean','area_density'],no_tuning=True,
        selection_sha256=sha256(OUT/'stage3_selection.json'),coordinate_protocol_sha256=sha256(OUT/'stage3_coordinate_protocol.json'),
        reference='All final label-independent native mapper points within <=5cm of canonical GT mesh vertices; floor(xyz/.05) voxels, majority GT IDs, query at voxel center.',
        invalid_geometry='Report unmatched input geometry fraction; exclude it from labeled domain. Domain is identical for both methods.',
        background='XML label text prefixes: wall, floor, ceiling, unlabeled; label0 is void/background',
        transfer='assigned predicted points only, 1NN <=5cm, unchanged frozen kernel',
        metric='class-agnostic instance PQ strict IoU>.5; scene macro mean; include all GT foreground instance sizes',
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in files}))
    dump(REPORT/'stage3/evaluation_protocol.json',json.loads(path.read_text()))


def evaluate(scene,cache,run,output):
    protocol_path=OUT/'stage3_evaluation_protocol.json';protocol=json.loads(protocol_path.read_text())
    for file,h in protocol['source_sha256'].items():assert sha256(ROOT/file)==h,file
    assert scene in protocol['scenes']
    finished=json.loads((run/'run.json').read_text());assert finished['complete'] and finished['no_gt']
    assert finished['cache_manifest_sha256']==sha256(cache/'manifest.json')
    for name,h in finished['files_sha256'].items():assert sha256(run/name)==h,name
    manifest=json.loads((cache/'manifest.json').read_text());assert manifest['complete'] and manifest['frames_cached']==200 and manifest['frames']==400
    for name,h in manifest['files_sha256'].items():assert sha256(cache/name)==h
    prediction=np.load(run/'geometry.npz');rows=json.loads((run/'classification.json').read_text())['objects']
    labels={method:np.where(np.isin(prediction['instance'],[r['object_id'] for r in rows if r['predictions'][method]['retained']]),prediction['instance'],0) for method in protocol['methods']}
    output.mkdir(parents=True,exist_ok=False)
    # Ground truth enters only after the above predictions are complete.
    paths=[DATA/'main'/scene/(scene+'.ply'),DATA/'main'/scene/(scene+'.xml')]
    downloaded=json.loads((OUT/'holdout_download.json').read_text());hashes={x['path']:x['sha256'] for x in downloaded['files']}
    for path in paths:assert sha256(path)==hashes[str(path)]
    mesh_xyz,mesh_labels=mesh(paths[0]);basis_path=OUT/'metric_inputs'/scene/'poses.npz'
    bm=json.loads(basis_path.with_name('manifest.json').read_text());assert sha256(basis_path)==bm['output_sha256']
    basis=np.load(basis_path)['canonical_from_raw'];mesh_xyz=mesh_xyz@basis[:3,:3].T+basis[:3,3]
    xml=ET.parse(paths[1]).getroot();background=[0]+[int(x.attrib['id']) for x in xml.iter('label') if x.attrib.get('text','').lower().startswith(('wall','floor','ceiling','unlabeled'))]
    mesh_labels[np.isin(mesh_labels,background)]=0
    points=np.load(cache/'geometry.npz')['xyz'];dist,index=cKDTree(mesh_xyz).query(points,workers=1);valid=dist<=.05
    if not valid.any():raise ValueError('no input geometry near GT mesh')
    vox,inverse=np.unique(np.floor(points[valid]/.05).astype(np.int64),axis=0,return_inverse=True)
    truth=majority_labels(inverse,mesh_labels[index[valid]],len(vox));vertices=(vox+.5)*.05
    np.savez_compressed(output/'reference.npz',xyz=vertices,instance=truth)
    result=dict(scene=scene,complete=False,protocol_sha256=sha256(protocol_path),run_sha256=sha256(run/'run.json'),
        domain=dict(map_points=len(points),matched_points=int(valid.sum()),matched_fraction=float(valid.mean()),
                    voxels=len(vox),gt_instances=len(np.unique(truth[truth>0])),background_ids=background),methods={})
    for name,lab in labels.items():
        projected,_=transfer_labels(prediction['xyz'],lab,vertices,mode='nearest_5cm')
        score=voxel_metrics(truth,projected);audit=audit_instances(truth,projected)
        score.update(predicted_error_counts=audit['predicted_counts'],gt_error_counts=audit['gt_counts'])
        result['methods'][name]=score
        np.savez_compressed(output/(name+'.npz'),instance=projected);dump(output/(name+'_audit.json'),audit)
    result.update(complete=True,files_sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file()})
    dump(output/'metrics.json',result);print('EVALUATED',scene,{k:v['observed_voxel_instance_pq']*100 for k,v in result['methods'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--scene',choices=SCENES);p.add_argument('--cache',type=Path);p.add_argument('--run',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    if args.freeze:freeze()
    if args.scene:evaluate(args.scene,args.cache,args.run,args.output)
