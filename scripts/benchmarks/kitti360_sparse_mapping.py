"""Transfer the already fixed garage sparse-LiDAR scale adapter, without tuning."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time
from paper_final_common import ROOT, read, dump, sha
import kitti360_development_mapping as base

OUT = ROOT/'build/benchmarks/kitti360_sparse_development_v1'
GARAGE = ROOT/'build/benchmarks/evidence_retention_20261111/scale_ablation_v1/protocol.json'


def freeze():
    p, job = base.verify()
    assert not (OUT/'mapping_protocol.json').exists()
    prior = base.OUT/'maps/cg/run.json'
    assert read(prior)['complete']
    assert not read(base.OUT/'maps/cg/history2/classification.json')['objects']
    garage = read(GARAGE)
    assert garage['sensor_overlap_m'] == .2 and garage['native_overlap_m'] == .025
    sources = dict(p['source_sha256'])
    for path in [Path(__file__), GARAGE, ROOT/'scripts/benchmarks/run_livo_scale_ablation.py',
        ROOT/'omni_semantic_map/omni_semantic_map/replay_fusion.py',
        ROOT/'omni_semantic_map/omni_semantic_map/fusion_diagnostics.py',
        ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs/conceptgraph/slam/mapping.py',
        ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs/conceptgraph/slam/utils.py']:
        sources[str(path.relative_to(ROOT))] = sha(path)
    p.update(created_epoch=time.time(), source_sha256=sources,
        parent_protocol_sha256=sha(base.OUT/'mapping_protocol.json'), prior_native_run_sha256=sha(prior),
        amendment='Reuse garage both adapter: disable DBSCAN; 20cm nearest-neighbor overlap radius for association and final merge, keep 2.5cm voxel processing and all similarity/retention thresholds. No frontend changes.',
        reason='Native indoor transfer yielded 262 pre-final fragments: 259 seen once, 3 twice; no fragment survived the unchanged 3-observation final gate. Zero historical proposals. Development geometry only; no semantic reference decoded.',
        sensor_overlap_m=garage['sensor_overlap_m'], no_parameter_sweep=True,
        scope='Second version of development adapter. Previously tested robot configuration transferred once. Reserved validation not run; no accuracy claim from object counts.')
    dump(OUT/'mapping_protocol.json', p)


def main(action):
    if action == 'freeze': return freeze()
    base.OUT = OUT
    p, job = base.verify()
    if action == 'cleanup': return base.cleanup()
    sys.path.insert(0, str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    from conceptgraph.slam import mapping, utils
    from omegaconf import OmegaConf
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend
    original_init = ConceptGraphsBackend.__init__
    original_overlap = mapping.compute_overlap_matrix_2set
    original_final_overlap = utils.compute_overlap_matrix
    holder = {}
    def initialize(self, cfg, seed=0):
        cfg = deepcopy(cfg)
        cfg.dbscan_remove_noise = False
        holder['effective_config'] = OmegaConf.to_container(cfg, resolve=True)
        original_init(self, cfg, seed=seed)
    def overlap_config(cfg):
        cfg = deepcopy(cfg)
        cfg.downsample_voxel_size = p['sensor_overlap_m']
        return cfg
    ConceptGraphsBackend.__init__ = initialize
    mapping.compute_overlap_matrix_2set = lambda cfg, *args:original_overlap(overlap_config(cfg), *args)
    utils.compute_overlap_matrix = lambda cfg, *args:original_final_overlap(overlap_config(cfg), *args)
    try: base.cg()
    finally:
        ConceptGraphsBackend.__init__ = original_init
        mapping.compute_overlap_matrix_2set = original_overlap
        utils.compute_overlap_matrix = original_final_overlap
    path = OUT/'maps/cg/run.json'
    status = read(path)
    status.update(effective_config=holder['effective_config'], overlap_radius_m=p['sensor_overlap_m'],
        amendment_protocol_sha256=sha(OUT/'mapping_protocol.json'))
    dump(path, status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'cg', 'cleanup'])
    args = parser.parse_args()
    main(args.action)
