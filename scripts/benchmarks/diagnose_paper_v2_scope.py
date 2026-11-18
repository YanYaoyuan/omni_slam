"""Read-only, post-hoc bottleneck diagnosis. Never a new benchmark result.

Uses already viewed v1 outputs. GT proxies only explain failure patterns; they
must never be consumed by a mapper. Preserves all frozen predictions and scores.
"""
from collections import Counter,defaultdict
from pathlib import Path
import json
import numpy as np
from paper_final_common import ROOT,RUN,read,sha,dump

DEST=ROOT/'build/benchmarks/paper_v2_diagnosis_20261112'


def gt_patterns(truth,prediction,matched):
    gt_ids,gt_counts=np.unique(truth[truth>0],return_counts=True)
    pids,counts=np.unique(prediction[prediction>0],return_counts=True);sizes=dict(zip(pids,counts))
    pairs,n=np.unique(np.column_stack([truth,prediction]),axis=0,return_counts=True)
    overlap=defaultdict(dict)
    for (g,p),k in zip(pairs,n):
        if g>0 and p>0:overlap[int(g)][int(p)]=int(k)
    result=[]
    for g,total in zip(gt_ids,gt_counts):
        row=overlap[int(g)];covered=sum(row.values());best_cover=max(row.values(),default=0)
        best_iou=max((v/(int(total)+int(sizes[p])-v) for p,v in row.items()),default=0.)
        if int(g) in matched:kind='matched'
        elif not covered:kind='no_assigned_prediction_support'
        elif covered/total<=.5:kind='insufficient_assigned_coverage_for_any_split_or_merge_only_fix'
        elif best_cover/total>.5:kind='single_prediction_has_coverage_but_contamination_prevents_match'
        else:kind='support_distributed_across_predictions_needs_combination_or_reassignment'
        result.append(dict(gt=int(g),gt_voxels=int(total),covered_voxels=covered,assigned_coverage=covered/int(total),
            largest_intersection_fraction=best_cover/int(total),best_current_iou=best_iou,pattern=kind,
            overlap_voxels={str(p):v for p,v in row.items()}))
    assert sum(x['pattern']=='matched' for x in result)==len(matched)
    return dict(counts=dict(Counter(x['pattern'] for x in result)),objects=result)


def conflict_audit(scene):
    graph=read(RUN/'evaluation'/scene/'graph_diagnostics.json');nodes=read(RUN/'maps'/scene/'cg/nodes.json')
    manifest=read(RUN/'frontend'/scene/'cache/manifest.json');obsframes={}
    for r in manifest['records']:
        for i in range(r['observation_offset'],r['observation_offset']+r['masks']):obsframes[i]=r['frame_id']
    labels=graph['observation_GT_labels'];parents={n['alias']:n['alias'] for n in nodes}
    members={n['alias']:set(n['members']) for n in nodes}
    def root(k):
        while k!=parents[k]:k=parents[k]
        return k
    rows=[]
    for edge in graph['variants']['history2']['edges']:
        a,b=root(edge['a']),root(edge['b'])
        if edge['reason']=='coframe_conflict':
            af=defaultdict(list);bf=defaultdict(list)
            for i in members[a]:af[obsframes[i]].append(i)
            for i in members[b]:bf[obsframes[i]].append(i)
            frames=sorted(set(af)&set(bf));assert frames==edge['conflict_frames']
            witnesses=[]
            for frame in frames:
                for i in af[frame]:
                    for j in bf[frame]:
                        x,y=labels[i]['gt'],labels[j]['gt']
                        witnesses.append(dict(frame=frame,a=i,b=j,gt_a=x,gt_b=y,
                            relation='unknown' if x is None or y is None else 'same_dominant_gt' if x==y else 'different_dominant_gt'))
            c=Counter(x['relation'] for x in witnesses)
            kind=('has_different_gt_witness' if c['different_dominant_gt'] else
                  'all_witnesses_judgable_same_gt' if c['same_dominant_gt']==len(witnesses) else 'inconclusive')
            rows.append(dict(a=edge['a'],b=edge['b'],diagnosis=kind,witness_counts=dict(c),witnesses=witnesses,
                endpoint_same_gt=edge['correct']))
        if edge['reason']=='joined':
            lo,hi=sorted([a,b]);parents[hi]=lo;members[lo]|=members[hi]
    return dict(rejections=len(rows),counts=dict(Counter(x['diagnosis'] for x in rows)),rows=rows,
        limitation='Dominant GT observation labels are >=.7-purity proxies; same-frame evidence is audited across whole components, not just edge endpoints.')


def main():
    summary=read(RUN/'summary.json');before={p:sha(RUN/p) for p in ['summary.json','predictions_complete.json','execution_manifest.json']}
    totals=Counter();veto=Counter();results=[]
    for scene in summary['scenes']:
        m=read(RUN/'evaluation'/scene/'metrics.json');ref=np.load(RUN/'evaluation'/scene/'reference.npz')
        pred=np.load(RUN/'evaluation'/scene/'history2_projection.npz')['instance'];valid=ref['valid']
        matched={x['gt_id'] for x in m['methods']['M']['instance']['matches']}
        patterns=gt_patterns(ref['instance'][valid],pred[valid],matched)
        assert sum(patterns['counts'].values())==m['methods']['M']['instance']['gt_visible_instances']
        conflicts=conflict_audit(scene);totals.update(patterns['counts']);veto.update(conflicts['counts'])
        result=dict(scene=scene,patterns=patterns,conflicts=conflicts,
            metrics_sha256=sha(RUN/'evaluation'/scene/'metrics.json'),
            native_final_merge_conflicts=read(RUN/'evaluation'/scene/'graph_diagnostics.json')['variants']['history2']['post_native_merge_conflict_pairs'])
        dump(DEST/(scene+'.json'),result);results.append(result)
        print(scene,patterns['counts'],conflicts['counts'],flush=True)
    assert before=={p:sha(RUN/p) for p in before}
    report=dict(complete=True,post_hoc=True,already_viewed_scenes=summary['scenes'],not_independent_v2_test=True,
        no_model_inference=True,no_parameters_changed=True,no_new_benchmark_scores=True,
        interpretation='Coverage is assigned predicted support on the original valid voxel domain. Not evidence of detector misses by itself. Split/merge-only ceiling excludes adding previously unassigned points.',
        false_negative_patterns=dict(totals),coframe_rejection_witness_patterns=dict(veto),
        native_final_merge_conflicts=sum(x['native_final_merge_conflicts'] for x in results),
        source_sha256=sha(Path(__file__)),unchanged_original_artifacts=before,
        scene_results={x['scene']:sha(DEST/(x['scene']+'.json')) for x in results})
    dump(DEST/'summary.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
