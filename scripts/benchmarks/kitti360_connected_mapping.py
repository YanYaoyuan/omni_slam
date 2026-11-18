"""Restore the full pre-existing garage baseline, including support components."""
import argparse
from pathlib import Path
import time
from paper_final_common import ROOT, read, dump, sha
import kitti360_development_mapping as base
import kitti360_sparse_mapping as sparse

OUT = ROOT/'build/benchmarks/kitti360_connected_development_v1'


def freeze():
    assert not (OUT/'mapping_protocol.json').exists()
    support = read(OUT/'cache/support_protocol.json')
    assert support['radius_m'] == .4 and support['minimum_points'] == 16
    assert read(OUT/'cache/support_changes.json')['complete']
    source_job = base.JOB
    job = read(source_job)
    job.update(output=str(OUT), kind='kitti360_development_connected_support',
        source_job_sha256=sha(source_job), source_job=str(source_job),
        frontend_scope='No new GPU inference: exact source masks/features with already established garage largest-component supports.')
    job_path = OUT/'job.json'
    dump(job_path, job)
    p = read(sparse.OUT/'mapping_protocol.json')
    p.update(created_epoch=time.time(), job_sha256=sha(job_path),
        parent_protocol_sha256=sha(sparse.OUT/'mapping_protocol.json'),
        amendment='Full existing garage baseline: largest measured component per exclusive mask at radius0.4m/min16 points, then no DBSCAN and 20cm association overlap. Masks, features, mean readout, all history and final retention thresholds unchanged.',
        scope='Exposed development window. Native and scale-only outputs were already evaluated; this is completion of an omitted established input stage, not unseen validation or new methodological contribution.',
        cache_manifest_sha256=sha(OUT/'cache/manifest.json'),
        prior_scale_scores_sha256=sha(sparse.OUT/'evaluation/summary.json'))
    for path in [Path(__file__), source_job, ROOT/'scripts/benchmarks/filter_livo_sparse_support.py',
        ROOT/'omni_semantic_map/omni_semantic_map/sparse_support_components.py',
        OUT/'cache/support_protocol.json', OUT/'cache/support_changes.json']:
        p['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    dump(OUT/'mapping_protocol.json', p)


def main(action):
    if action == 'freeze': return freeze()
    base.JOB = OUT/'job.json'
    sparse.OUT = OUT
    sparse.main(action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze','cg','cleanup'])
    args = parser.parse_args()
    main(args.action)
