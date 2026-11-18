"""Versioned 2x2 sparse-scale diagnostic; no frozen baseline mutations.

The original CG implementation is called with explicit process-local overrides.
Only overlap queries see the sensor scale; voxel downsampling remains 2.5cm.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import numpy as np
import yaml
from audit_evidence_retention import ROOT, OUT, REPORT, dump
from omni_semantic_map.flat_benchmark import sha256

ROOT_OUT = OUT/'scale_ablation_v1'
VARIANTS = ['native','no_dbscan','sensor_overlap','both']


def freeze():
    path=ROOT_OUT/'protocol.json'; assert not path.exists()
    config=ROOT/'build/experiments/semantic_20261106/managed_config/livo.yaml'
    params=yaml.safe_load(config.read_text())['/**']['ros__parameters']
    # Use the native point-cloud sampling setting, not a fit to object labels.
    leaf=params['common']['filter_size_surf'] if 'filter_size_surf' in params.get('common',{}) else None
    if leaf is None:
        matches=[v['filter_size_surf'] for v in params.values() if isinstance(v,dict) and 'filter_size_surf' in v]
        assert len(matches)==1, 'locate unique lidar filter_size_surf'
        leaf=matches[0]
    paths=[Path(__file__),config,ROOT/'scripts/benchmarks/retention_holdout_backend.py',
        ROOT/'omni_semantic_map/omni_semantic_map/replay_fusion.py',
        ROOT/'omni_semantic_map/omni_semantic_map/fusion_diagnostics.py']
    paths += [ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs/conceptgraph/slam'/f for f in ['mapping.py','utils.py']]
    dump(path,dict(scope=__doc__,version=1,variants=VARIANTS,inputs=['cache','cache_supportcc'],
        sensor_overlap_m=float(leaf),native_overlap_m=.025,dbscan_control='Disable DBSCAN at ingestion, periodic denoising, merging and finalization; keep voxel downsampling.',
        overlap_control='Change nearest-neighbor overlap radius only, for causal association and final merging. Keep AABB/OBB overlap prefilter and all similarity/merge thresholds.',
        no_threshold_sweep=True,no_gt=True,no_feature_or_mask_changes=True,
        development_scope='Previously viewed rosbag2245; engineering diagnosis, not independent validation or paper improvement claim.',
        fixed_visual_targets=[6,2647,2652,3075],
        acceptance='Do not select by object count alone. Inspect alias survival/merges, extent collapse, observation ownership, same-frame conflicts and original-image evidence. Baseline must reproduce exactly.',
        files_sha256={str(p):sha256(p) for p in paths},
        cache_sha256={c:sha256(OUT/'livo_full_strong_v2'/c/'manifest.json') for c in ['cache','cache_supportcc']}))


def execute(cache_name,variant):
    protocol=ROOT_OUT/'protocol.json';spec=json.loads(protocol.read_text())
    for p,h in spec['files_sha256'].items(): assert sha256(Path(p))==h,p
    cache=OUT/'livo_full_strong_v2'/cache_name
    assert sha256(cache/'manifest.json')==spec['cache_sha256'][cache_name]
    output=ROOT_OUT/cache_name/variant
    sys.path.insert(0,str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    from conceptgraph.slam import mapping,utils
    from omegaconf import OmegaConf
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend
    from retention_holdout_backend import run
    init=ConceptGraphsBackend.__init__;integrate=ConceptGraphsBackend.integrate
    overlap=mapping.compute_overlap_matrix_2set; final_overlap=utils.compute_overlap_matrix
    holder={};history=[]
    def initialize(self,cfg,seed=0):
        if variant in ['no_dbscan','both']:cfg.dbscan_remove_noise=False
        holder['config']=OmegaConf.to_container(cfg,resolve=True)
        init(self,cfg,seed=seed);holder['engine']=self
    def online(self,observations,pose):
        result=integrate(self,observations,pose)
        history.append(dict(frame_id=observations[0].frame_id if observations else None,
                            objects=len(self._objects),rejected=len(self.rejected)))
        return result
    def sensor_cfg(cfg):
        cfg=deepcopy(cfg);cfg.downsample_voxel_size=spec['sensor_overlap_m'];return cfg
    ConceptGraphsBackend.__init__=initialize;ConceptGraphsBackend.integrate=online
    if variant in ['sensor_overlap','both']:
        mapping.compute_overlap_matrix_2set=lambda cfg,*args:overlap(sensor_cfg(cfg),*args)
        utils.compute_overlap_matrix=lambda cfg,*args:final_overlap(sensor_cfg(cfg),*args)
    try:run(cache,output)
    finally:
        ConceptGraphsBackend.__init__=init;ConceptGraphsBackend.integrate=integrate
        mapping.compute_overlap_matrix_2set=overlap;utils.compute_overlap_matrix=final_overlap
    manifest=json.loads((cache/'manifest.json').read_text())
    frames={i:r['frame_id'] for r in manifest['records'] for i in range(r['observation_offset'],r['observation_offset']+r['masks'])}
    engine=holder['engine'];final=json.loads((output/'classification.json').read_text())['objects'];geom=np.load(output/'geometry.npz')
    def members(row):
        ids=list(map(int,row['observation_ids']));counts=Counter(frames[i] for i in ids)
        return dict(alias=min(ids)+1,distinct_frames=len(counts),same_frame_extra_members=sum(n-1 for n in counts.values()),observations=len(ids))
    online_objects=[dict(**members(o),points=len(o['pcd'].points)) for o in engine._objects]
    results=[]
    for obj in final:
        points=geom['xyz'][geom['instance']==obj['object_id']]
        results.append(dict(**members(obj),evaluation_id=obj['object_id'],name=obj['predictions']['mean']['name'],
            retained=obj['predictions']['mean']['retained'],points=len(points),extent_m=(points.max(0)-points.min(0)).tolist(),
            center_m=points.mean(0).tolist(),fixed_target_observations=[a for a in spec['fixed_visual_targets'] if a-1 in obj['observation_ids']]))
    if variant=='native':
        ref=OUT/'livo_full_strong_v2'/('cg' if cache_name=='cache' else 'cg_supportcc')
        g=np.load(ref/'geometry.npz')
        for k in g.files:np.testing.assert_array_equal(geom[k],g[k])
        assert final==json.loads((ref/'classification.json').read_text())['objects']
    dump(output/'diagnostics.json',dict(complete=True,protocol_sha256=sha256(protocol),cache=cache_name,variant=variant,
        effective_config=holder['config'],overlap_radius_m=spec['sensor_overlap_m'] if variant in ['sensor_overlap','both'] else spec['native_overlap_m'],
        native_exact_reproduction=variant=='native',online_objects=online_objects,final_objects=results,history=history,
        rejected_observations=len(engine.rejected),scope='Same-frame extras are conflict candidates (overlapping masks can also explain them), not GT identity errors.'))
    state=json.loads((output/'run.json').read_text());state['intervention_protocol_sha256']=sha256(protocol)
    state['effective_config']=holder['config'];state['overlap_radius_m']=spec['sensor_overlap_m'] if variant in ['sensor_overlap','both'] else spec['native_overlap_m']
    state['files_sha256']['diagnostics.json']=sha256(output/'diagnostics.json');dump(output/'run.json',state)
    print('FINISHED',cache_name,variant,'pre-final',len(online_objects),'retained',sum(o['retained'] for o in results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--cache',choices=['cache','cache_supportcc']);p.add_argument('--variant',choices=VARIANTS);a=p.parse_args()
    if a.freeze:freeze()
    if a.cache:execute(a.cache,a.variant)
