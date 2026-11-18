"""Attribute frozen prediction changes on the exact foreground evaluation domain."""
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha


def main():
    parent=ROOT/'build/benchmarks/reversible_revision_20261113'
    out=ROOT/'build/benchmarks/ratio_evidence_audit_20261113'
    for scene in read(parent/'protocol.json')['scenes']:
        key=scene['key'];folder=out/key
        g=np.load(scene['geometry']);xyz,labels=g['xyz'],g['instance']
        rows=read(scene['classification'])['objects']
        retained=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
        fg=np.flatnonzero(np.isin(labels,retained))
        ref=np.load(scene['reference']);valid=ref['valid'] if 'valid' in ref.files else np.ones(len(ref['xyz']),bool)
        truth=ref['instance'][valid]
        distance,index=cKDTree(xyz[fg]).query(ref['xyz'][valid],workers=1)
        nearest=fg[index]
        before=np.load(parent/'evaluation'/key/'baseline.npz')['instance'][valid]
        reconstructed=labels[nearest].copy();reconstructed[distance>.05]=0
        np.testing.assert_array_equal(before,reconstructed)
        after=np.load(parent/'evaluation'/key/'simple_ratio.npz')['instance'][valid]
        audit=read(parent/'evaluation'/key/'baseline_audit.json')
        dominant={r['predicted_id']:r['dominant_gt'] or -1 for r in audit['predicted']}
        old_correct=(truth>0)&(truth==np.array([dominant.get(int(o),-1) for o in before]))
        new_correct=(truth>0)&(truth==np.array([dominant.get(int(o),-1) for o in after]))
        c=np.load(folder/'evidence.npz');counts=c['counts'];sectors=c['sectors']
        removed=(counts[0]<=counts[1]*.5)&(counts[1]>0)
        negative=counts[1]-counts[0]
        groups={'all':np.ones(len(labels),bool),
                'outside_all_masks_majority':(counts[1]-counts[2])>.5*negative,
                'negative_one_sector':(sectors[1]>0).sum(0)==1,
                'interior_supported':counts[3]>0,
                'interior_and_negative_one_sector':(counts[3]>0)&((sectors[1]>0).sum(0)==1),
                'visible_at_most_two':counts[1]<=2}
        changed=before!=after
        assert np.all(removed[nearest[changed&(before>0)]])
        def stats(select):
            selected=select[nearest]&changed&(before>0)
            return dict(changed_reference_voxels=int(selected.sum()),
                        lost_dominant_support=int((selected&old_correct&~new_correct).sum()),
                        gained_dominant_support=int((selected&~old_correct&new_correct).sum()),
                        removed_background_assignments=int((selected&(truth==0)&(after==0)).sum()),
                        became_unassigned=int((selected&(after==0)).sum()))
        result=dict(scene=key,exact_baseline_projection_verified=True,
            interpretation='Fixed baseline dominant-GT correspondence. Actual before/after labels use the primary evaluator; these diagnostic correctness counts are not rematched instance PQ. A removed source point may be replaced by another point of the same identity and cause no reference-label change.',
            groups={name:stats(select) for name,select in groups.items()},
            objects=[dict(object_id=int(oid),groups={name:stats(select&(labels==oid)) for name,select in groups.items()}) for oid in retained],
            input_evidence_sha256=sha(folder/'evidence.npz'),script_sha256=sha(__file__))
        dump(folder/'transition_attribution.json',result)
        print(key,result['groups'],flush=True)


if __name__=='__main__':main()
