"""Existing identity-free semantic evidence on exactly the same anchor sources.

Posthoc diagnostic controls, not a new map candidate or calibrated decision.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from paper_final_common import read, dump, sha
from audit_anchor_retraction_failures import OUT, verify
from pathlib import Path


def main():
    p = verify(); assert not (OUT / 'semantic_independence.json').exists()
    results = {}
    for c in p['contexts']:
        a = np.load(OUT / c['key'] / 'anchor_readout.npz')
        source = a['source_id']; mapping = np.load(Path(c['full']) / 'old_to_full.npy')
        inverse = np.full(int(mapping.max()) + 1, -1, np.int64); inverse[mapping] = np.arange(len(mapping))
        old = inverse[source]; assert (old >= 0).all()
        sem = np.load(c['sem']); owners = sem['owners'][old]
        probabilities = sem['mean_probabilities'][old]
        observed = (sem['seen'][old] > 0) & (sem['mean_response'][old] >= 0) & (sem['mean_response'][old] < 19)
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1.000001).all()
        np.testing.assert_allclose(probabilities[observed].sum(1), 1, rtol=0, atol=1e-5)
        semantic_error_score = 1 - probabilities[np.arange(len(source)), a['declared_class']]
        known = a['known']; wrong = a['wrong_semantic_class']; weight = a['confidence']; margin = a['mean_margin']
        def auc(take, score):
            return float(roc_auc_score(wrong[take], score[take], sample_weight=weight[take])) if len(np.unique(wrong[take])) == 2 else None
        grouped = {}
        for name, take in [('all', known), ('building', known & (a['declared_class'] == 2)), ('car', known & (a['declared_class'] == 13))]:
            common = take & observed
            grouped[name] = dict(known_anchors=int(take.sum()), semantic_available=int(common.sum()),
                original_margin_auc_same_subset=auc(common, margin),
                identity_free_semantic_auc_same_subset=auc(common, semantic_error_score))
        objects = []
        for oid in np.unique(owners):
            take = known & (owners == oid); common = take & observed
            total = weight[take].sum()
            objects.append(dict(object_id=int(oid), declared_class=int(a['declared_class'][owners == oid][0]),
                known=int(take.sum()), wrong=int((take & wrong).sum()), correct=int((take & ~wrong).sum()),
                wrong_confidence_fraction=float(weight[take & wrong].sum() / total) if total else None,
                mean_margin=float(np.average(margin[take], weights=weight[take])) if total else None,
                within_object_margin_auc=auc(take, margin), semantic_available=int(common.sum()),
                within_object_semantic_auc=auc(common, semantic_error_score),
                within_object_margin_auc_on_semantic_subset=auc(common, margin)))
        per_class = {}
        for name, label in [('building', 2), ('car', 13)]:
            rows = [r for r in objects if r['declared_class'] == label and r['wrong_confidence_fraction'] is not None]
            eligible = [r for r in rows if r['wrong'] >= 10 and r['correct'] >= 10]
            mixed = [r for r in rows if r['within_object_margin_auc'] is not None]
            rho = spearmanr([r['wrong_confidence_fraction'] for r in rows], [r['mean_margin'] for r in rows])[0] if len(rows) > 2 else np.nan
            per_class[name] = dict(objects=len(rows), mixed_label_objects=len(mixed), objects_with_at_least10_each=len(eligible),
                mean_within_object_margin_auc=float(np.mean([r['within_object_margin_auc'] for r in eligible])) if eligible else None,
                wrong_purity_vs_mean_margin_spearman=float(rho) if np.isfinite(rho) else None,
                all_known_sources_wrong_objects=[r['object_id'] for r in rows if r['correct'] == 0],
                all_known_sources_correct_objects=[r['object_id'] for r in rows if r['wrong'] == 0])
        results[c['key']] = dict(common_source_controls=grouped, object_conditioned=per_class, all_objects=objects)
        print('SEMANTIC INDEPENDENCE', c['key'], grouped, per_class, flush=True)
    dump(OUT / 'semantic_independence.json', dict(complete=True, results=results, source_sha256=sha(__file__),
        scope='Posthoc descriptive controls. Existing SegFormer probabilities do not consume predicted object IDs; the RGB observations may still be shared, so statistical independence is not claimed. Same-source coverage enforced. No decision threshold, map, AP score or method promotion.',
        thresholds='At least10positive/negative source counts only defines a descriptive within-object macro; all object rows and small/empty subsets remain visible.',
        parent_summary_sha256=sha(OUT / 'summary.json')))


if __name__ == '__main__': main()
