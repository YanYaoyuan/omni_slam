"""Independent vectorized verification of every exported support and target.

Equivalent full-population checks avoid repeated per-token membership sorting
and NumPy testing dispatch. No subsampling or relaxed acceptance conditions.
"""
from collections import Counter
from pathlib import Path
import time
import numpy as np
import audit_full_object_maps as task


def archive(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def main():
    start=time.monotonic();p=task.protocol()
    for key in ['input_sha256','reference_sha256','map_artifact_sha256']:task.verify_hashes(p[key])
    maps=task.read(task.OUT/'map_summary.json');dataset=task.read(task.OUT/'supervision_summary.json')
    for r in maps['scenes']:
        entry=next(s for s in p['scenes'] if s['key']==r['key']);ref=archive(entry['reference'])
        valid=ref.get('valid',np.ones(len(ref['xyz']),bool));truth=ref['instance'][valid]
        coverage=archive(task.OUT/'maps'/r['key']/'coverage.npz');pred=archive(task.OUT/'maps'/r['key']/'predictions.npz')
        objects=task.read(task.OUT/'maps'/r['key']/'objects.json')
        for name in task.STAGES:
            eligible=0
            for row in objects:
                hit=int(coverage[name][valid][truth==row['gt']].sum());n=int((truth==row['gt']).sum())
                assert row['coverage'][name]==hit/n;eligible+=2*hit>n
            assert eligible==r['stages'][name]['reachable_gt_over_half']
        for name,outname in [('v1','actual_v1'),('all_regions_control','actual_all_regions_control')]:
            assert task.voxel_metrics(truth,pred[name][valid])==r[outname]
    for row in dataset['scenes']:
        assert task.sha(row['manifest'])==row['sha256'];manifest=task.read(row['manifest']);folder=Path(row['manifest']).parent
        entry=next(s for s in p['scenes'] if s['key']==row['key']);cache,m,geo,supports,frames=task.data(entry)
        assert task.sha(folder/'point_targets.npz')==manifest['point_targets_sha256']
        labels=archive(folder/'point_targets.npz')['instance'];_,_,independent=task.point_targets(entry,geo['xyz'])
        np.testing.assert_array_equal(labels,independent)
        counts=Counter();base=len(labels)
        for rec in manifest['records']:
            assert task.sha(folder/rec['file'])==rec['sha256'];assert task.sha(folder/rec['meta'])==rec['meta_sha256']
            z=archive(folder/rec['file']);meta=task.read(folder/rec['meta']);orig=archive(meta['feature_source'])
            np.testing.assert_array_equal(z['features'],orig['local_features']);np.testing.assert_array_equal(z['offsets'],orig['offsets'])
            n=len(z['features']);token_observation=np.repeat(z['observation_ids'],np.diff(z['offsets']))
            lengths=np.diff(z['point_offsets']);occurrence_token=np.repeat(np.arange(n),lengths)
            assert len(occurrence_token)==len(z['point_ids'])
            # Source membership for EVERY support occurrence, including mask ID.
            source_arrays=[int(oid)*base+supports[int(oid)] for oid in z['observation_ids']]
            source_pairs=np.unique(np.concatenate(source_arrays)) if source_arrays else np.empty(0,np.int64)
            pairs=token_observation[occurrence_token]*base+z['point_ids']
            indices=np.searchsorted(source_pairs,pairs)
            assert (indices<len(source_pairs)).all()
            np.testing.assert_array_equal(source_pairs[indices],pairs)
            observed_labels=labels[z['point_ids']]
            pairs,count=np.unique(np.column_stack([occurrence_token,observed_labels]),axis=0,return_counts=True)
            saved_tokens=np.repeat(np.arange(n),np.diff(z['target_offsets']))
            np.testing.assert_array_equal(pairs[:,0],saved_tokens)
            np.testing.assert_array_equal(pairs[:,1],z['target_instance_ids'])
            np.testing.assert_array_equal(count,z['target_counts'])
            valid_counts=np.bincount(occurrence_token,weights=observed_labels>=0,minlength=n).astype(np.int64)
            foreground=np.bincount(occurrence_token,weights=observed_labels>0,minlength=n).astype(np.int64)
            instances=np.bincount(pairs[pairs[:,1]>0,0],minlength=n)
            np.testing.assert_array_equal(np.column_stack([lengths,valid_counts,foreground,instances]),z['target_stats'])
            counts.update(tokens=n,tokens_with_valid_support=int((valid_counts>0).sum()),
                tokens_with_foreground=int((foreground>0).sum()),tokens_multiple_foreground_instances=int((instances>=2).sum()),
                token_support_occurrences=len(occurrence_token),observations=len(z['observation_ids']))
            for ob in meta['observations']:
                d=task.label_distribution(labels[supports[ob['observation_id']]])
                assert d==ob['distribution'];counts['observations_multiple_foreground_instances']+=int(d['foreground_instances']>=2)
        assert dict(counts)==manifest['counts']==row['counts']
        print('AUDITED',row['key'],counts['tokens'],'tokens',flush=True)
    result=dict(complete=True,all_inputs_and_original_maps_unchanged=True,
        all_full_map_metrics_recomputed=True,coverage_counts_recomputed=True,
        token_features_bytewise_array_equal=True,all_token_target_histograms_recomputed=True,
        source_support_membership_checked=True,all_scene_point_targets_recomputed=True,
        no_new_inference_training_or_mapping=True,
        map_summary_sha256=task.sha(task.OUT/'map_summary.json'),supervision_summary_sha256=task.sha(task.OUT/'supervision_summary.json'),
        verification_source_sha256=task.sha(Path(__file__)),all_tokens_checked_no_sampling=True)
    task.dump(task.OUT/'audit.json',result)
    task.dump(task.OUT/'verification_runtime.json',dict(complete=True,wall_s=time.monotonic()-start,
        adaptation='Independent vectorized full-population checks; no change to predictions, labels or scores.',
        prior_attempts='Original/eager per-token checks were interrupted for slow repeated work. First vectorized attempt stopped at an empty-mask frame; empty-array handling was corrected and all checks rerun. No model, label, score, sampling or acceptance condition changed.',
        source_sha256=task.sha(Path(__file__)),audit_sha256=task.sha(task.OUT/'audit.json')))
    print('FULL AUDIT PASS',flush=True)


if __name__=='__main__':main()
