"""Requirement-level completion evidence for the three-stage experiment.

Completion here means execution, evaluation, failure analysis and reporting.
It does not certify a positive method gain, production quality or online latency.
"""
import json
import re
from pathlib import Path
import numpy as np
from audit_evidence_retention import ROOT, OUT, REPORT, dump
from omni_semantic_map.flat_benchmark import sha256


def read(path):
    return json.loads(path.read_text())


def main():
    checked = {}
    def verify(path, expected):
        path = Path(path)
        if not path.is_absolute(): path = ROOT/path
        actual = sha256(path)
        assert actual == expected, path
        checked[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = actual
    def retain(path):
        verify(path, sha256(path))
    p2 = read(OUT/'stage2_protocol.json')
    assert len(p2['scenes']) == 8 and len(p2['methods']) == 6
    for f, h in p2['source_sha256'].items(): verify(f, h)
    total_objects = 0
    for scene in p2['scenes']:
        path = REPORT/'stage1'/f'{scene}.json'; d = read(path); retain(path)
        assert d['complete'] and d['classification_reproduced']
        total_objects += len(d['objects'])
        for f, h in d['input_sha256'].items(): verify(f, h)
        for obj in d['objects']:
            assert all(k in obj for k in ['before','after','unique_frames','median_area_px',
                'median_supports','median_exclusive_fraction','truncated_view_fraction',
                'max_view_angle_deg','feature_agreement'])
    assert total_objects == 667
    cases = read(REPORT/'stage1/case_manifest.json')['cases']; assert len(cases) == 22
    for case in cases:
        assert Path(case['rgb']).is_file() and Path(case['mask']).is_file()
    for row in read(REPORT/'completion_audit_stage12.json')['runs']:
        for f, h in row['files_sha256'].items(): verify(OUT/'readout_runs'/row['scene']/f, h)
    for name in ['stage3_cvpr18_domain_protocol.json','stage3_named_domain_protocol.json']:
        path = OUT/name; retain(path)
        for f,h in read(path)['files_sha256'].items(): verify(f,h)
    holdout = read(REPORT/'stage3/cvpr18_domain_summary.json')
    assert [r['scene'] for r in holdout['scenes']] == ['014','015','021']
    for r in holdout['scenes']:
        verify(OUT/'heldout_runs'/r['scene']/'run.json', r['prediction_run_sha256'])
        verify(OUT/'stage3_cvpr18_domain_protocol.json', r['protocol_sha256'])
        for m in r['methods'].values():
            assert all(x['iou'] > .5 for x in m['matches'])
            pq = sum(x['iou'] for x in m['matches'])/(m['tp']+.5*m['fp']+.5*m['fn'])
            assert abs(pq-m['observed_voxel_instance_pq']) < 1e-12
    state = read(OUT/'livo_full_validation_v2_status.json')
    assert state['complete'] and state['phase'] == 'complete' and len(state['steps']) == 9
    verify(ROOT/'scripts/benchmarks/run_livo_full_validation.py', state['source_sha256'])
    for step in state['steps']:
        assert step['returncode'] == 0 and step['wall_s'] > 0
        verify(OUT/'logs'/f'livo_full_v2_{step["phase"]}.log',step['log_sha256'])
    job_path=OUT/'jobs/livo_full_v2.json'; job=read(job_path)
    assert job['no_gt_in_inference'] and len(job['frames']) == 330
    for key in ['input_sha256','source_sha256']:
        for f,h in job[key].items(): verify(f,h)
    cap=REPORT/'stage3/livo_full_v2_capture_audit.json'; capture=read(cap)
    verify(Path(job['capture']),capture['capture_sha256'])
    verify(ROOT/'scripts/benchmarks/audit_livo_capture.py',capture['source_sha256'])
    assert all(capture['gate'].values()) and capture['missing_visual_packet_ids']==[0,2]
    assert capture['calibration_matches_config'] and capture['atomic_files_and_native_positions_verified']
    review_path=REPORT/'stage3/livo_full_visual_review.json';review=read(review_path)
    assert review['complete'] and not review['independent_ground_truth']
    for name,cache_name,run_name in [('livo_full_v2','cache','cg'),('livo_full_v2_supportcc','cache_supportcc','cg_supportcc')]:
        root=Path(job['output']); cache=root/cache_name; run=root/run_name
        manifest=read(cache/'manifest.json'); result=read(run/'run.json')
        assert manifest['complete'] and result['complete'] and len(manifest['records'])==330
        verify(cache/'manifest.json',result['cache_manifest_sha256'])
        for f,h in manifest['files_sha256'].items(): verify(cache/f,h)
        for row in manifest['records']: verify(cache/'frames'/row['file'],row['sha256'])
        for f,h in result['files_sha256'].items(): verify(run/f,h)
        r=review['branches'][name]; assert r['all_retained_predictions_reviewed']
        verify(REPORT/'stage3'/name/'summary.json',r['summary_sha256'])
        verify(REPORT/'stage3'/(name+'_audit')/'audit.json',r['audit_sha256'])
        verify(run/'causal_diagnostics.json',r['causal_sha256'])
        summary=read(REPORT/'stage3'/name/'summary.json')
        assert {o['alias'] for o in r['objects']}=={o['persistent_alias'] for o in summary['retained_objects']}
        for o in r['objects']:
            assert o['semantic_verdict'] in ['supported','contradicted','ambiguous','insufficient']
            assert len(o['panels_reviewed'])==3
            verify(Path(o['image']),o['image_sha256'])
        assert summary['processing']['semantic_workers_peak_rss_bytes']>0
    post=root/'cg_supportcc_postprocessing_audit'; p=read(post/'postprocessing_audit.json')
    verify(root/'cg_supportcc/run.json',p['reference_sha256'])
    verify(ROOT/'scripts/benchmarks/audit_livo_postprocessing.py',p['source_sha256'])
    for key in ['xyz','instance']:
        np.testing.assert_array_equal(np.load(post/'geometry.npz')[key],np.load(root/'cg_supportcc/geometry.npz')[key])
    assert read(post/'classification.json')==read(root/'cg_supportcc/classification.json')
    tests=read(REPORT/'stage3/unit_tests.json');assert tests['returncode']==0 and '19 passed' in tests['output']
    for doc in [ROOT/'docs/benchmarks/evidence_retention_2026-11-11.md',ROOT/'docs/experiments/evidence_retention_2026-11-11.md',ROOT/'docs/paper/dino_livo_architecture_2026-11-10.md']:
        retain(doc)
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            if '://' not in target and not target.startswith('#'):
                assert (doc.parent/target.split('#')[0]).exists(),target
    for path in [cap,review_path,REPORT/'stage3/unit_tests.json',REPORT/'stage3/cvpr18_domain_summary.json',
        REPORT/'stage1/image_cases.png',REPORT/'stage1/evidence_statistics.png',REPORT/'stage3/livo_full_comparison.png']:
        retain(path)
    requirements=[
        dict(stage=1,requirement='Eight-scene per-object audit, foreground losses, evidence statistics and fixed image cases',evidence='667 objects, 22 cases; input hashes and per-object fields verified',status='complete'),
        dict(stage=2,requirement='Fixed-input mechanism plus all six controls on eight scenes, costs and paired failures',evidence='48 runs verified; original mean reproduced; candidate does not recover GT matches',status='complete'),
        dict(stage=3,requirement='Frozen unused scenes and unchanged model predictions; documented evaluation scope',evidence='SceneNN 014/015/021, frozen input/prediction hashes and PQ arithmetic verified; post-hoc annotation correction disclosed',status='complete'),
        dict(stage=3,requirement='Actual LIVO recording to strong semantic objects, queries, identity and costs',evidence='589s recording, 330 packets, two completed branches, all 9 retained outputs visually reviewed, full resources recorded',status='complete'),
        dict(stage=3,requirement='Failure analysis and final recommendation',evidence='Candidate rejected; support adaptation loses visible vehicle; return duplicate and 377-to-11 point denoising collapse documented by exact replay',status='complete'),
        dict(stage='all',requirement='Reproducible reports and tests',evidence='Reports and local links checked, 19 relevant tests previously passed, snapshots and hashed artifacts retained',status='complete')]
    result=dict(complete=True,scope=__doc__,requirements=requirements,verified_files=len(checked),files_sha256=checked,
        quality_acceptance='Neither candidate readout nor sparse support adaptation is approved as a general quality improvement.',
        future_work_outside_this_experiment=['DINO association ablation','new learned model or SAM3','production online latency','S100 deployment','independent GT for VBot object position and identity'])
    dump(REPORT/'completion_audit.json',result)
    print('COMPLETION AUDIT PASSED',len(requirements),'requirements,',len(checked),'hashed files')


if __name__=='__main__':main()
