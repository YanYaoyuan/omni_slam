"""Read-only developmental diagnosis; oracle labels never enter the mapper."""
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.instance_error_audit import audit_instances

OUT = ROOT / 'build/benchmarks/reversible_revision_20261113'
KEYS = ['replica_office_0', 'scenenn_014']


def main():
    target = OUT / 'diagnosis.json'
    assert not target.exists(), 'Preserve prior diagnosis'
    protocol = read(ROOT / 'build/benchmarks/visibility_cleanup_20261112/protocol.json')
    records = []
    for scene in protocol['scenes']:
        if scene['key'] not in KEYS:
            continue
        reference = np.load(scene['reference'])
        valid = reference['valid'] if 'valid' in reference.files else np.ones(len(reference['xyz']), bool)
        truth = reference['instance'][valid]
        projection = ROOT / 'build/benchmarks/visibility_cleanup_20261112/evaluation' / scene['key'] / 'native_bridge_2.npz'
        predicted = np.load(projection)['instance'][valid]
        audit = audit_instances(truth, predicted)
        matched = {o['matched_gt'] for o in audit['predicted'] if o['matched_gt'] is not None}
        all_objects, candidates = [], []
        sizes = {int(g): int((truth == g).sum()) for g in np.unique(truth) if g > 0}
        for oid in np.unique(predicted):
            if oid <= 0:
                continue
            ids, counts = np.unique(truth[predicted == oid], return_counts=True)
            parts = [dict(gt_id=int(g), intersection=int(n), gt_size=sizes[int(g)],
                          coverage=float(n / sizes[int(g)]), already_matched=int(g) in matched)
                     for g, n in zip(ids, counts) if g > 0]
            row = dict(object_id=int(oid), voxels=int((predicted == oid).sum()), parts=parts)
            all_objects.append(row)
            recoverable = [v for v in parts if v['coverage'] > .5 and not v['already_matched']]
            if recoverable:
                candidates.append(dict(**row, ideal_recovery_gt=[v['gt_id'] for v in recoverable],
                                       nontrivial_recovery_gt=[v['gt_id'] for v in recoverable if v['gt_size'] >= 5]))
        records.append(dict(key=scene['key'], predicted_counts=audit['predicted_counts'],
                            gt_counts=audit['gt_counts'], all_objects=all_objects,
                            ideal_partition_candidates=candidates,
                            reference_sha256=sha(scene['reference']), projection_sha256=sha(projection)))
    dump(target, dict(development_only=True, mapping_outputs_created=0,
                      interpretation='Oracle coverage upper bound on the fixed evaluation domain; not realizable predictions or proof that observations separate the instances. Tiny GT (<5 voxels) explicitly separated.',
                      scenes=records))
    for scene in records:
        print(scene['key'], 'ideal candidates', [(o['object_id'], o['nontrivial_recovery_gt'])
                                                 for o in scene['ideal_partition_candidates']])


if __name__ == '__main__':
    main()
