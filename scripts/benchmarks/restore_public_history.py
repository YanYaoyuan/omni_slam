"""Reconstruct an offline CG snapshot, with mandatory exact output checks.

Stored world points and ordered observation members are sufficient because the
frozen native runs disable periodic filtering/merging and ingest singleton
detections. Recompute the exact recursive torch feature updates. RGB colors
are placeholders, unused by this geometry/feature-only evaluation export.
"""
import sys
from pathlib import Path
import numpy as np
from audit_evidence_retention import ROOT
from public_dino_protocol import PARENT,read
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.shared_observations import observation_table
from run_livo_history_graph import finalize


def restore(entry,verification_output):
    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    parent=PARENT/'runs'/entry['key'];status=read(parent/'run.json');assert status['complete']
    for f,h in status['artifacts'].items():assert sha256(parent/f)==h
    cfg=OmegaConf.create(read(entry['config']));assert cfg.filter_interval<0 and cfg.merge_interval<0
    cache=Path(entry['cache']);m=read(cache/'manifest.json')
    for r in m['records']:assert sha256(cache/'frames'/r['file'])==r['sha256']
    features,frames,areas=observation_table(cache);text=np.load(cache/'text_features.npy');vocab=read(cache/'vocabulary.json')
    poses=np.load(cache/'poses.npy');byframe={r['frame_id']:poses[i] for i,r in enumerate(m['records'])}
    engine=ConceptGraphsBackend(cfg,seed=0);nodes=read(parent/'nodes.json');snap=np.load(parent/'snapshot.npz')
    for node in nodes:
        ids=node['members'];clip=torch.tensor(features[ids[0]]);txt=torch.tensor(text[int((text@features[ids[0]]).argmax())])
        for n,oid in enumerate(ids[1:],start=1):
            clip=F.normalize((clip*n+torch.tensor(features[oid]))/(n+1),dim=0)
            txt=F.normalize((txt*n+torch.tensor(text[int((text@features[oid]).argmax())]))/(n+1),dim=0)
        points=snap['xyz'][snap['instance']==node['alias']]
        pcd=engine.o3d.geometry.PointCloud();pcd.points=engine.o3d.utility.Vector3dVector(points)
        pcd.colors=engine.o3d.utility.Vector3dVector(np.zeros_like(points))
        engine._objects.append(dict(pcd=pcd,bbox=engine.utils.get_bounding_box(cfg,pcd),num_detections=len(ids),
            observation_ids=ids.copy(),class_id=[0]*len(ids),clip_ft=clip,text_ft=txt))
    table=features,frames,areas,text/np.linalg.norm(text,axis=1,keepdims=True),vocab,byframe
    verification_output.mkdir(parents=True,exist_ok=False)
    for name in ['baseline','native_bridge_2']:
        groups=read(parent/name/'graph.json')['groups'];out=verification_output/name
        finalize(engine,groups,out,table)
        assert read(out/'classification.json')==read(parent/name/'classification.json'),(entry['key'],name,'classification')
        actual,expected=np.load(out/'geometry.npz'),np.load(parent/name/'geometry.npz')
        for k in actual.files:np.testing.assert_array_equal(actual[k],expected[k])
    return engine,table
