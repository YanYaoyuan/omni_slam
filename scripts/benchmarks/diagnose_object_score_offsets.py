"""Unsupervised within-object ordinal score, for diagnosis only, no decisions."""
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from paper_final_common import ROOT, dump, sha
from audit_anchor_retraction_failures import OUT, verify


def main():
    p = verify(); target = OUT / 'object_score_offsets.json'; assert not target.exists()
    results, files = {}, {}
    for c in p['contexts']:
        a = np.load(OUT / c['key'] / 'anchor_readout.npz')
        decision = np.load(ROOT / 'build/benchmarks/anchor_counterevidence_v1/decisions' / c['key'] / 'anchors.npz')
        units = decision['units']; margin = decision['mean_margin']
        rank = np.zeros(len(units))
        # All units of the predicted object, including sources without usable
        # reference labels. The transform never sees GT or semantic probabilities.
        for oid in np.unique(units[:, 1]):
            ix = np.flatnonzero(units[:, 1] == oid)
            rank[ix] = rankdata(margin[ix], method='average') / (len(ix) + 1)
        point_rank = rank[a['unit_index']]
        known, y, w = a['known'], a['wrong_semantic_class'], a['confidence']
        groups = {}
        for name, t in [('all', known), ('building', known & (a['declared_class'] == 2)), ('car', known & (a['declared_class'] == 13))]:
            groups[name] = dict(known_anchors=int(t.sum()),
                original_margin_auc=float(roc_auc_score(y[t], a['mean_margin'][t], sample_weight=w[t])),
                within_object_rank_auc=float(roc_auc_score(y[t], point_rank[t], sample_weight=w[t])))
        path = OUT / c['key'] / 'object_relative_rank.npz'
        np.savez_compressed(path, units=units, margin=margin, within_object_rank=rank)
        files[str(path)] = sha(path)
        results[c['key']] = groups
        print('OBJECT SCORE OFFSETS', c['key'], groups, flush=True)
    dump(target, dict(complete=True, results=results, files=files, source_sha256=sha(__file__),
        transformation='Average-tie empirical rank/(object unit count+1), using ALL cached units per predicted ID. No reference labels select the ranks.',
        scope='Posthoc diagnostic of cross-object score comparability. No calibrated error probabilities, thresholds, revised decisions, maps or AP scores. Purely wrong objects cannot be identified from their internal ranks alone. Not an accepted mapping method.'))


if __name__ == '__main__': main()
