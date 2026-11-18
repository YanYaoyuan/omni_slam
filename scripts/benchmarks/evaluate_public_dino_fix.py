"""Audited evaluation-only repair: h5py path and missing return_inverse=True.

No mapping, domain policy, labels, thresholds or metric definitions are changed.
The original frozen source remains intact. Record this amendment before scoring.
"""
import argparse
from pathlib import Path
from collections import Counter,defaultdict
import sys
import numpy as np
from scipy.spatial import cKDTree
from audit_evidence_retention import OUT,dump
from public_dino_protocol import read,verify
from omni_semantic_map.flat_benchmark import sha256
import evaluate_public_dino as original

sys.path.insert(0,str(OUT/'tools/hdf5_reader'))
old_classes=original.classes_for_scene


def classes_for_scene(entry,gt_ids):
    if entry['dataset']=='Replica':return old_classes(entry,gt_ids)
    import h5py
    from evaluate_retention_holdout import mesh
    assert sha256(entry['semantic_metadata'])==entry['semantic_metadata_sha256']
    old=read(OUT/'stage3_cvpr18_domain_protocol.json')['files_sha256'];scene=entry['scene']
    coord=OUT/'official_annotation_audit/cvpr18'/f'scenenn_coords_{scene}.hdf5'
    meshpath=Path(read(OUT/'stage3_selection.json')['data_root'])/'main'/scene/(scene+'.ply')
    for p in [coord,meshpath,Path(entry['semantic_metadata'])]:assert sha256(p)==old[str(p)]
    with h5py.File(coord) as f:coords=f['coords'][:].reshape(-1,3)
    with h5py.File(entry['semantic_metadata']) as f:labels=f['label'][:].reshape(-1)
    coords,first,inv=np.unique(coords,axis=0,return_index=True,return_inverse=True);unique_labels=labels[first]
    assert np.array_equal(unique_labels[inv],labels)
    xyz,instances=mesh(meshpath);dist,idx=cKDTree(xyz).query(coords,workers=1);assert np.all(dist==0)
    counts=defaultdict(Counter)
    pairs,num=np.unique(np.column_stack([instances[idx],unique_labels]),axis=0,return_counts=True)
    for (iid,cls),n in zip(pairs,num):counts[int(iid)][int(cls)]+=int(n)
    mapping={};purities={}
    for iid in gt_ids:
        if not counts[int(iid)]:continue
        cls,n=max(counts[int(iid)].items(),key=lambda x:(x[1],-x[0]));purity=n/sum(counts[int(iid)].values())
        purities[int(iid)]=dict(dominant_class=cls,purity=purity,annotation_vertices=sum(counts[int(iid)].values()))
        if cls>0 and purity>=.7:mapping[int(iid)]=cls
    provenance=dict(sources={str(p):sha256(p) for p in [coord,meshpath,Path(entry['semantic_metadata'])]},instance_annotation_purities=purities)
    return {i:mapping[i] for i in map(int,gt_ids) if i in mapping},provenance


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',choices=['public_dino_v1','public_dino_scope_v2'],required=True)
    p.add_argument('--freeze',action='store_true');p.add_argument('--scene');p.add_argument('--summarize',action='store_true');a=p.parse_args()
    original.DEST=OUT/a.version;record=original.DEST/'evaluation_repair.json';source=Path(__file__).resolve()
    if a.freeze:
        assert not record.exists()
        dump(record,dict(source=str(source),sha256=sha256(source),protocol_sha256=sha256(original.DEST/'protocol.json'),
            reason='Missing h5py import search path and missing return_inverse=True prevented SceneNN evaluation before producing scores.',
            preserves='All frozen mapping, category-domain rules, ontology mapping and metrics; original source left intact.'))
    audit=read(record);assert sha256(source)==audit['sha256'];assert sha256(original.DEST/'protocol.json')==audit['protocol_sha256']
    verify(read(original.DEST/'protocol.json'));original.classes_for_scene=classes_for_scene
    if a.scene:original.evaluate(a.scene)
    if a.summarize:original.summarize()
