"""Paired object changes for every readout, including regressions."""
import json
import numpy as np
from audit_evidence_retention import OUT, REPORT, dump


def main():
    summary=json.loads((REPORT/'stage2/summary.json').read_text());results=[]
    for scene in summary['scenes']:
        folder=OUT/'readout_runs'/scene
        predictions=json.loads((folder/'predictions.json').read_text())['methods']
        baseline={r['object_id']:r for r in predictions['mean']}
        audit=json.loads((folder/'mean_audit.json').read_text())
        before={r['matched_gt'] for r in audit['predicted'] if r['matched_gt'] is not None}
        originally_lost={r['gt_id'] for r in json.loads((REPORT/'stage1'/(scene+'.json')).read_text())['filter_losses']}
        methods={}
        for name,rows in predictions.items():
            current=json.loads((folder/(name+'_audit.json')).read_text())
            after={r['matched_gt'] for r in current['predicted'] if r['matched_gt'] is not None}
            changes=[dict(object_id=r['object_id'],before_class=baseline[r['object_id']]['name'],after_class=r['name'],
                         retained_before=baseline[r['object_id']]['retained'],retained_after=r['retained'])
                     for r in rows if r['replica_class_id']!=baseline[r['object_id']]['replica_class_id'] or r['retained']!=baseline[r['object_id']]['retained']]
            methods[name]=dict(changes=changes,gained_matched_gt=sorted(after-before),lost_matched_gt=sorted(before-after),
                recovered_original_filter_losses=sorted(after & originally_lost))
        results.append(dict(scene=scene,methods=methods))
    rng=np.random.default_rng(20261111);indices=rng.integers(0,len(results),size=(10000,len(results)))
    comparisons={}
    for m,values in summary['methods'].items():
        delta=(np.array(values['pq_by_scene'])-summary['methods']['mean']['pq_by_scene'])*100
        comparisons[m]=dict(mean_delta_pq100=float(delta.mean()),exploratory_scene_bootstrap_interval=np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),
            gt_gained=sum(len(s['methods'][m]['gained_matched_gt']) for s in results),
            gt_lost=sum(len(s['methods'][m]['lost_matched_gt']) for s in results),
            original_filter_losses_recovered=sum(len(s['methods'][m]['recovered_original_filter_losses']) for s in results))
    dump(REPORT/'stage2/paired_changes.json',dict(scenes=results,comparisons=comparisons,
        bootstrap_scope='10000 paired scene resamples, seed20261111, eight previously viewed development scenes; descriptive uncertainty only, not held-out significance'))
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
