"""Independent fixed-domain scores, paired object changes, and pilot decision."""
from pathlib import Path
import numpy as np
from run_visibility_cleanup import OUT, HISTORY, KEYS, VARIANTS, verify, read, dump, sha
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.instance_error_audit import audit_instances, filter_transition


def evaluate():
    protocol = verify(); gate = read(OUT/'predictions_complete.json')
    assert gate['complete'] and gate['protocol_sha256'] == sha(OUT/'protocol.json')
    for p, h in gate['runs'].items(): assert sha(OUT/p) == h, p
    results = []
    for scene in protocol['scenes']:
        key = scene['key']; target = OUT/'runs'/key; run = read(target/'run.json')
        for p, h in scene['immutable_hashes'].items(): assert sha(p) == h, p
        for p, h in run.get('artifacts', {}).items(): assert sha(target/p) == h, p
        folder = OUT/'evaluation'/key; folder.mkdir(parents=True, exist_ok=False)
        reference = np.load(scene['reference']); vertices = reference['xyz']; truth = reference['instance']
        valid = reference['valid'] if 'valid' in reference.files else np.ones(len(truth), bool)
        baseline_scores = read(HISTORY/'evaluation'/key/'metrics.json'); methods = {}; audits = {}
        rows = read(scene['classification'])['objects']; kept = [r['object_id'] for r in rows if r['predictions']['mean']['retained']]
        original = np.load(scene['geometry'])
        check, _ = transfer_labels(original['xyz'], np.where(np.isin(original['instance'], kept), original['instance'], 0), vertices, mode='nearest_5cm')
        assert np.array_equal(check, np.load(HISTORY/'evaluation'/key/'native_bridge_2.npz')['instance']), key
        for name in ['baseline', 'native_bridge_2'] + VARIANTS:
            if name in VARIANTS and not run['complete']:
                methods[name] = dict(complete=False, error=run['error']); continue
            if name in ['baseline', 'native_bridge_2']:
                projected = np.load(HISTORY/'evaluation'/key/(name+'.npz'))['instance']
            else:
                path = target/name; geometry = np.load(path/'geometry.npz'); selection = np.load(path/'retained_source_indices.npy')
                np.testing.assert_array_equal(geometry['xyz'], original['xyz'][selection])
                np.testing.assert_array_equal(geometry['instance'], original['instance'][selection])
                assert sha(path/'classification.json') == sha(scene['classification'])
                labels = np.where(np.isin(geometry['instance'], kept), geometry['instance'], 0)
                projected, _ = transfer_labels(geometry['xyz'], labels, vertices, mode='nearest_5cm')
            score = voxel_metrics(truth[valid], projected[valid]); audit = audit_instances(truth[valid], projected[valid]); audits[name] = audit
            if name in ['baseline', 'native_bridge_2']:
                for k in ['observed_voxel_instance_pq','tp','fp','fn','sq','rq']:
                    assert score[k] == baseline_scores['methods'][name][k], (key, name, k)
            transition = filter_transition(audits['native_bridge_2'], audit) if 'native_bridge_2' in audits else None
            methods[name] = dict(complete=True, score=score, transition=transition,
                assigned_valid_fraction=float((projected[valid]>0).mean()),
                predicted_error_counts=audit['predicted_counts'], gt_error_counts=audit['gt_counts'],
                removal=run.get('methods', {}).get(name),
                matched_iou_sum=float(sum(o['best_iou'] for o in audit['predicted'] if o['matched_gt'] is not None)))
            np.savez_compressed(folder/(name+'.npz'), instance=projected); dump(folder/(name+'_audit.json'), audit)
        gt_rows = []
        before = {g['gt_id']: g for g in audits['native_bridge_2']['ground_truth']}
        for gid, b in before.items():
            row = dict(gt_id=gid,baseline_best_iou=b['best_iou'],baseline_coverage=b['prediction_coverage'],methods={})
            for name in VARIANTS:
                if name not in audits: continue
                g = next(g for g in audits[name]['ground_truth'] if g['gt_id']==gid)
                row['methods'][name] = dict(best_iou=g['best_iou'],delta_iou=g['best_iou']-b['best_iou'],coverage=g['prediction_coverage'],matched=g['category']=='matched')
            gt_rows.append(row)
        dump(folder/'per_object.json', gt_rows)
        result = dict(key=key,dataset=scene['dataset'],methods=methods,valid_voxels=int(valid.sum()),
            baseline_projection_exact=True,inputs_unchanged=True,geometry_is_exact_subset=True,
            timing={k: run.get(k) for k in ['total_wall_s','evidence_s','evidence_frame_p95_s','geometric_control_s','cpu_peak_rss_bytes']})
        dump(folder/'metrics.json',result); results.append(result)
        print('SCORED',key,{k:round(v['score']['observed_voxel_instance_pq']*100,4) for k,v in methods.items() if v['complete']},flush=True)
    groups = {}
    for dataset in ['Replica','SceneNN']:
        selected = [r for r in results if r['dataset']==dataset]; groups[dataset] = {}
        for name in ['baseline','native_bridge_2']+VARIANTS:
            if not all(r['methods'][name]['complete'] for r in selected):
                groups[dataset][name] = dict(complete=False); continue
            scores = [r['methods'][name]['score'] for r in selected]
            groups[dataset][name] = dict(complete=True,pq100=float(np.mean([s['observed_voxel_instance_pq'] for s in scores])*100),
                **{k:sum(s[k] for s in scores) for k in ['tp','fp','fn']},
                gained=sum(len(r['methods'][name]['transition']['gained_matched_gt']) for r in selected) if name in VARIANTS else 0,
                lost=sum(len(r['methods'][name]['transition']['lost_matched_gt']) for r in selected) if name in VARIANTS else 0)
    all_complete = all(r['methods'][v]['complete'] for r in results for v in VARIANTS)
    checks = {}
    if all_complete:
        g = protocol['gate']
        checks = dict(dataset_mean_gain=all(d['multiview']['pq100']-d['native_bridge_2']['pq100']>=g['dataset_mean_delta_pq100_min'] for d in groups.values()),
            nondecreasing_scenes=sum(r['methods']['multiview']['score']['observed_voxel_instance_pq']>=r['methods']['native_bridge_2']['score']['observed_voxel_instance_pq']-1e-12 for r in results)>=g['nondecreasing_scenes_min'],
            tp_not_lower_each_dataset=all(d['multiview']['tp']>=d['native_bridge_2']['tp'] for d in groups.values()),
            new_matches_each_dataset=all(d['multiview']['gained']>0 for d in groups.values()),
            positive_vs_sor_each_dataset=all(d['multiview']['pq100']>d['geometric_sor']['pq100'] for d in groups.values()),
            visibility_ablation_positive=sum(d['multiview']['pq100']-d['no_visibility']['pq100'] for d in groups.values())>0,
            persistence_ablation_positive=sum(d['multiview']['pq100']-d['single_refutation']['pq100'] for d in groups.values())>0)
    passed = all_complete and all(checks.values())
    summary = dict(complete=True,all_predictions_complete=all_complete,datasets=groups,gate_checks=checks,gate_passed=passed,
        decision='Eligible for a separately frozen independent evaluation; not a publication claim.' if passed else 'STOP this fixed cleanup configuration; no automatic retuning or expansion.',
        new_geometry_outputs=16,development_only=True,no_post_score_tuning=True,
        protocol_sha256=sha(OUT/'protocol.json'),prediction_gate_sha256=sha(OUT/'predictions_complete.json'),
        scene_results={r['key']:sha(OUT/'evaluation'/r['key']/'metrics.json') for r in results})
    dump(OUT/'summary.json',summary); print('DECISION',summary['decision'],flush=True)


if __name__=='__main__': evaluate()
