"""Optimistic capacity of the frozen proposal/identity space; NOT a predictor.

Relax each reference query independently: any nearer candidate owner or the
unchanged-map fallback may be chosen. Real source decisions couple queries,
so this relaxed best IoU is only an upper bound, never achievable AP.
"""
import itertools
import numpy as np
from paper_final_common import read,dump,sha
from run_map_credit_training import OUT,verify


def bounds(truth,weight,fallback,rows,owners,object_ids):
    n=len(truth);labels=int(max(object_ids))+1
    pairs=np.unique(rows*labels+owners)
    q=pairs//labels;o=pairs%labels;counts=np.bincount(q,minlength=n)
    gt_mass=np.bincount(truth,weights=weight);result={}
    for oid in object_ids:
        can=np.zeros(n,bool);can[q[o==oid]]=True
        old=fallback==oid
        forced=old&((counts==0)|((counts==1)&can))
        can|=old
        inter=np.bincount(truth[can],weights=weight[can],minlength=len(gt_mass))
        forced_gt=np.bincount(truth[forced],weights=weight[forced],minlength=len(gt_mass))
        unavoidable=weight[forced].sum()-forced_gt
        result[int(oid)]=inter/np.maximum(1e-12,gt_mass+unavoidable)
    return result


def enumerate_check():
    rng=np.random.default_rng(0)
    for _ in range(30):
        n=5;truth=rng.integers(0,3,n);w=rng.uniform(.1,1,n);fallback=rng.integers(0,3,n)
        sets=[np.flatnonzero(rng.random(2)>.5)+1 for i in range(n)]
        rows=np.repeat(np.arange(n),[len(v) for v in sets]);owners=np.concatenate(sets)
        got=bounds(truth,w,fallback,rows,owners,[1,2])
        attainable=[sorted(set([int(fallback[i]),*map(int,sets[i])])) for i in range(n)]
        for oid in [1,2]:
            for gt in [1,2]:
                if gt>=len(got[oid]):continue
                best=0.
                for choice in itertools.product(*attainable):
                    pred=np.asarray(choice)==oid;target=truth==gt;inter=w[pred&target].sum();union=w[pred|target].sum()
                    best=max(best,float(inter/union) if union else 0.)
                assert abs(best-got[oid][gt])<1e-12


def main():
    enumerate_check();p=verify();assert not (OUT/'capacity_audit.json').exists()
    scored=read(OUT/'summary.json');detail=read(OUT/'per_object.json');results={}
    for c in p['contexts']:
        key=c['key'];a=np.load(OUT/'inputs'/key/'adjacency.npz');d=np.load(OUT/'inputs'/key/'candidates.npz');ref=np.load(c['reference'])
        records=read(__import__('pathlib').Path(c['ours'])/'classification.json')['objects']
        label={r['object_id']:r['predictions']['mean']['train_id'] for r in records}
        q=np.repeat(np.arange(len(a['fallback'])),np.diff(a['offsets']));src=a['sources'];v=d['valid'][src]
        rows=np.broadcast_to(q[:,None],v.shape)[v];owners=d['owners'][src][v]
        upper=bounds(ref['train_instance'],ref['confidence'],a['fallback'],rows,owners,sorted(label))
        oldmatches=set(scored['results'][key]['methods']['baseline']['matched_gt_ids']);items=[]
        for row in detail['results'][key]:
            gt=row['gt_id'];eligible=[o for o,cls in label.items() if cls==gt//1000]
            best=max(eligible,key=lambda o:upper[o][gt]);limit=float(upper[best][gt])
            assert all(v['best_iou']<=limit+1e-10 for v in row['methods'].values())
            items.append(dict(gt_id=gt,class_name=row['class_name'],baseline_matched=gt in oldmatches,
                baseline_best_iou=row['methods']['baseline']['best_iou'],querywise_upper_iou=limit,
                bound_owner=int(best),potential_new_ap50=gt not in oldmatches and limit>.5))
        results[key]=dict(objects=items,existing_matches=len(oldmatches),
            potential_new_gt=[r['gt_id'] for r in items if r['potential_new_ap50']],
            missed_buildings_cannot_cross_half=sum(r['class_name']=='building' and not r['baseline_matched'] and r['querywise_upper_iou']<=.5 for r in items),
            actual_maps_below_relaxation=True)
        print('MAP CREDIT CAPACITY',key,'potential newGT',results[key]['potential_new_gt'],
              'missed buildings below bound',results[key]['missed_buildings_cannot_cross_half'],flush=True)
    dump(OUT/'capacity_audit.json',dict(complete=True,results=results,source_sha256=sha(__file__),
        exhaustive_random_checks=30,new_fits=0,new_maps=0,new_ap=0,
        scope='Posthoc GT-assisted upper bound for this EXACT fixed candidate/identity space with immutable old map. Per-query choices relaxed independently and per-GT optimal owners may conflict; not realizable map or AP. Does not bound new proposals,history revision or new identity formation.',
        formula='For an owner o and GT g: maximum possible intersection / (GTmass + unavoidable outside-GT mass). Query admits fallback or any nearer candidate owner. Outside-GT mass unavoidable only when o is its sole attainable owner.'))


if __name__=='__main__':main()
