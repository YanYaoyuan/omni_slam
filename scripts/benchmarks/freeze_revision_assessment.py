"""Prospectively fix an outdoor replication, without inspecting new images/GT.

This scope amendment permits a bounded test of the frozen, unsuccessful
development candidate. It neither repairs that failed gate nor selects a new
winner. Data acquisition, execution validation and scoring have separate seals.
"""
from pathlib import Path
import time
from paper_final_common import ROOT, read, dump, sha

OUT = ROOT / 'build/benchmarks/revision_assessment_v1'
DATA = ROOT / 'build/datasets/kitti360_reserved_v4'
B = ROOT / 'build/benchmarks'


def main():
    assert not OUT.exists()
    selection = read(DATA / 'window_selection.json')
    assert [w['scene'] for w in selection['windows']] == [
        '2013_05_28_drive_0009_sync', '2013_05_28_drive_0010_sync']
    for path, digest in selection['source_sha256'].items():
        assert sha(ROOT / path) == digest, path
    original = read(B / 'revision_authorization_v1/summary.json')
    factorial = read(B / 'revision_factorial_v1/summary.json')
    assert not factorial['original_development_gate']
    assert not factorial['candidate_promoted']
    assert read(B / 'revision_robot_v1/completion_record.json')['complete']
    assert read(B / 'revision_factorial_v1/completion_record.json')['stage_complete']
    inputs = [DATA / 'window_selection.json',
              B / 'revision_authorization_v1/protocol.json',
              B / 'revision_authorization_v1/summary.json',
              B / 'revision_factorial_v1/protocol.json',
              B / 'revision_factorial_v1/summary.json',
              B / 'revision_factorial_v1/completion_record.json',
              B / 'revision_robot_v1/protocol.json',
              B / 'revision_robot_v1/completion_record.json',
              B / 'batch_transport_assessment_v1/protocol.json']
    sources = [Path(__file__)]
    for name in ['prepare_kitti360_unseen_v4.py',
                 'verify_kitti360_unseen_v4.py',
                 'audit_reserved_geometry_separation_v4.py',
                 'prepare_batch_transport_assessment.py',
                 'run_batch_transport_assessment.py',
                 'prepare_censored_object_partition.py',
                 'run_object_mask_cover.py', 'run_revision_authorization.py',
                 'run_revision_factorial.py', 'evaluate_kitti360_development.py']:
        sources.append(ROOT / 'scripts/benchmarks' / name)
    for name in ['mask_cover_revision.py', 'revision_authorization.py',
                 'batch_stable_transport.py', 'batch_stable_transport.cpp',
                 'kitti360_evaluation.py', 'kitti360_reference_identity.py']:
        sources.append(ROOT / 'omni_semantic_map/omni_semantic_map' / name)
    old = read(B / 'batch_transport_assessment_v1/protocol.json')
    model = Path(old['selected_model'])
    inputs.append(model)
    front = read(B / 'discovery_query_benchmark_v1/protocol.json')
    inputs += [Path(front['sam3_checkpoint']),
               B / 'ovo_full_20261110/model_artifacts.json',
               ROOT / 'build/models/segformer_b5_cityscapes/source_manifest.json']
    scenes = [dict(key='kitti360_' + w['scene'].split('_')[-2], kind='outdoor',
                   window=w,
                   job=str(OUT / 'prepared' / ('kitti360_' + w['scene'].split('_')[-2])
                           / 'jobs' / (w['scene'] + '.json')),
                   opaque_reference=str(DATA / 'data/opaque_references' / w['member']))
              for w in selection['windows']]
    dump(OUT / 'protocol.json', dict(
        created_epoch=time.time(), scenes=scenes,
        methods=['strong_baseline', 'unconditional', 'history_only', 'full_authorization'],
        primary_candidate='full_authorization',
        original_development_gate=False, candidate_promoted=False, goal_complete=False,
        selected_model=str(model), sam3_checkpoint=front['sam3_checkpoint'],
        sam3_sha256=front['sam3_sha256'],
        input_hashes={str(p): sha(p) for p in inputs},
        source_hashes={str(p): sha(p) for p in sources},
        authorization='The persistent user objective explicitly includes a new mechanism, '
            'robot comparison and additional assessment. This bounded prospective replication '
            'supersedes only our self-imposed prohibition on new-scene assessment after a '
            'failed development gate. It does not promote any candidate, rewrite past results, '
            'claim successful development, or authorize parameter tuning on new scenes.',
        selection='Two first lexicographic official validation windows in previously unused '
            'sequences with >=100 posed frames and >200m trajectory XY AABB separation from '
            'all seven exposed sequences and each other. Fixed before RGB inspection and '
            'reference decode. 50+34 sampled paired frames, not full sequences or official server.',
        primary='Exact unchanged revision_authorization_v1 Ours: selected cover mask witnesses, '
            'complete-link legacy identity authorization with one-view-deletion margin, local '
            'identity competition and certified writes. It failed development; this is a '
            'prospective replication of its mechanism, not another revised method.',
        controls={
            'strong_baseline': 'Unchanged SAM3/SigLIP CG history2 plus frozen intrinsic geometry, '
                'raw mask geometric expansion and source-frame certified full-LiDAR support. '
                'Same as the current strong baseline (ours_frame in the prior assessment).',
            'unconditional': 'Unchanged cover_history complete-map output.',
            'history_only': 'Exact cover_history XYZ, additions, new labels and ranking scores; '
                'replace only old-point labels by full_authorization authorized history. '
                'This is the factorial control, not a newly named Ours.'},
        backbone='Reuse the separately validated input/front-end/geometry/semantic pipeline from '
            'batch_transport_assessment_v1. Intrinsic model was trained on 0000/0002 only. '
            'SAM3 car/building threshold0.5 frontend, all positive raw masks for cover, '
            'frozen SigLIP+CGhistory2, frozen 19-class SegFormer. No new fitting or model switch.',
        budgets='Exactly 2 new windows, 8 final maps, 24 full scores (primary, raw-reference '
            'diagnostic and uniform-score diagnostic). Shared inference only once per original '
            'pipeline stage/frame; no seed, parameter, class, window replacement or model sweep. '
            'Intermediate backbone outputs are not additional candidate selection trials.',
        efficacy='Evaluate full_authorization without candidate selection: primary AP and AP50 '
            'must be >= strong_baseline in EACH window; mean primary AP must exceed strong by '
            'at least 0.01 and strictly exceed both controls; at least one new raw-target AP50 '
            'match and zero lost raw matches; nonzero historical edits. Empty-class/window '
            'scores remain NA with actual reference counts, not zeros or replacement scenes. '
            'Fewer than two eligible windows cannot establish this criterion. Secondary control '
            'success does not promote it; report all differences and losses.',
        scoring='Same full static reference, confidence-weighted building/car AP50:90, AP50, '
            'AP25, strict20cm nearest-map readout; all reference targets retained, including '
            'uncovered targets. Raw-target and uniform-score views remain diagnostic. '
            'Report all per-object best IoU/IDs and new/lost matches, no oracle region splicing.',
        execution_gate='Before new RGB/model inference: finish data SHA/CRC/timing/calibration '
            'checks, prove measurement-space separation, seal isolated execution adapters, '
            'and validate changed adapters against exposed golden outputs. All eight final '
            'maps must be frozen before decoding either reference. Separate execution seals '
            'may add paths/wrappers only, not alter this scientific protocol.',
        robot='The same four versions already have full330frame rosbag2245 outputs in '
            'revision_robot_v1 and revision_factorial_v1. Independently verify correspondence '
            'and link those sealed outputs, rather than rerunning identical maps and counting '
            'them as new validation. No robot GT: no semantic accuracy, S100 runtime or '
            'deployment claim; those remain explicit limits.',
        scope='A single-dataset outdoor local adaptation, not official leaderboard ranking, '
            'domain-general novelty proof or a new contribution by itself. Existing failure '
            'mechanism remains: no newborn identity inside a component with old footprints. '
            'That limitation is preserved here rather than silently changed between domains.',
        status='registered_not_executed'))
    print('FROZEN OUTDOOR REVISION REPLICATION', [(s['key'], len(s['window']['input_frames']))
                                                 for s in scenes], flush=True)


if __name__ == '__main__':
    main()
