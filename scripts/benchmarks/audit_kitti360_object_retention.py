"""Post-hoc GT coverage diagnostic. Never a prediction or tuning input."""
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT, read, dump, sha
from run_kitti360_reserved import OUT, verify
from evaluate_kitti360_development import transfer

DEST=ROOT/'build/benchmarks/kitti360_retention_audit_v1'
STAGES=['measurements','raw_mask_support','connected_support','three_frame_groups',
        'final_object_observations','target_object_observations']


def covered(source,mask,reference):
    result=np.zeros(len(reference),bool)
    if not mask.any():return result
    tree=cKDTree(source[mask])
    for start in range(0,len(reference),200000):
        result[start:start+200000]=np.isfinite(tree.query(reference[start:start+200000],
            distance_upper_bound=.2,workers=1)[0])
    return result


def main():
    assert not (DEST/'summary.json').exists(), 'Preserve completed diagnostic'
    p=verify(); seal=read(OUT/'predictions_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    dump(DEST/'protocol.json',dict(created_epoch=time.time(),posthoc=True,gt_diagnostic_only=True,
        source_sha256=sha(__file__),sealed_input_sha256=sha(OUT/'predictions_complete.json'),
        stages=STAGES,radius_m=.2,min_confidence_mass=100,coverage_gate='strictly > 0.5',
        meaning='Perfect pointwise identity/background oracle within each support domain; necessary coverage diagnostic, NOT achievable instance recall or AP. No algorithm predictions are created.',
        attribution='Nested source supports show first coverage loss. Three-frame stage uses history2 groups BEFORE final filtering, including multi-node groups; other final filtering/merging bundled in final-members stage. New centroid geometry is diagnosed separately and is not nested.',
        scope='Two exposed reserved windows, baseline only, no tuning or new inference. GT-derived output remains separate from predictor inputs.'))
    results=[]
    for w in p['windows']:
        scene=w['scene']; root=OUT/scene; cache=root/'connected/cache'
        front=root/'frontend'/scene/'cache'; manifest=read(cache/'manifest.json')
        source=np.load(cache/'geometry.npz')['xyz']; n=len(source)
        assert sha(cache/'geometry.npz')==sha(front/'geometry.npz')
        rows=read(root/'base/maps/cg/history2/classification.json')['objects']
        nodes=read(root/'base/maps/cg/nodes.json')
        by_alias={node['alias']:node for node in nodes}
        groups=read(root/'base/maps/cg/history2/graph.json')['groups']
        eligible=set()
        for group in groups:
            frames={f for alias in group for f in by_alias[alias]['frames']}
            if len(frames)>=3:eligible.update(o for alias in group for o in by_alias[alias]['members'])
        final={o for row in rows for o in row['observation_ids']}
        target={o for row in rows if row['predictions']['mean']['retained'] for o in row['observation_ids']}
        assert target<=final<=eligible
        domains={k:np.zeros(n,bool) for k in STAGES};domains['measurements'][:]=True
        counts=[]
        for rec in manifest['records']:
            raw=np.load(front/'frames'/rec['file']); con=np.load(cache/'frames'/rec['file'])
            domains['raw_mask_support'][raw['point_ids']]=True
            domains['connected_support'][con['point_ids']]=True
            for j,(lo,hi) in enumerate(zip(con['offsets'][:-1],con['offsets'][1:])):
                oid=rec['observation_offset']+j; ids=con['point_ids'][lo:hi]
                assert np.isin(ids,raw['point_ids'][raw['offsets'][j]:raw['offsets'][j+1]]).all()
                for name,group in [('three_frame_groups',eligible),('final_object_observations',final),('target_object_observations',target)]:
                    if oid in group:domains[name][ids]=True
                counts.append(dict(observation_id=oid,frame_id=rec['frame_id'],raw_points=int(raw['offsets'][j+1]-raw['offsets'][j]),
                    connected_points=len(ids),three_frame_group=oid in eligible,final_object=oid in final,target_object=oid in target))
        for a,b in zip(STAGES[:-1],STAGES[1:]):assert not np.any(domains[b]&~domains[a])
        reference=np.load(root/'evaluation/reference.npz');truth=reference['train_instance']; weights=reference['confidence']
        # Only target reference points are needed for this diagnostic.
        select=np.isin(truth//1000,[2,13]); xyz=reference['xyz'][select];truth=truth[select];weights=weights[select]
        gids=np.unique(truth); objects=[]
        for gid in gids:
            total=float(weights[truth==gid].sum())
            if total>=100:objects.append(dict(gt_id=int(gid),category='building' if gid//1000==2 else 'car',mass=total,coverage={}))
        for stage in STAGES:
            domain=covered(source,domains[stage],xyz)
            for row in objects:
                hit=truth==row['gt_id']; row['coverage'][stage]=float(weights[hit&domain].sum()/row['mass'])
            print(scene,stage,'source',int(domains[stage].sum()),'recoverable',sum(r['coverage'][stage]>.5 for r in objects),flush=True)
        geometry=np.load(root/'base/maps/cg/history2/geometry.npz')
        all_labels=transfer(geometry['xyz'],geometry['instance'],xyz,.2)
        id_to_row={r['object_id']:r for r in rows}
        for obj in objects:
            hit=truth==obj['gt_id']; labels=all_labels[hit]; mass=weights[hit]
            contributions=[]
            for oid in np.unique(labels[labels>0]):
                row=id_to_row[int(oid)]; contributions.append(dict(object_id=int(oid),
                    predicted_class=row['predictions']['mean']['name'],retained=row['predictions']['mean']['retained'],
                    gt_fraction=float(mass[labels==oid].sum()/obj['mass'])))
            obj['final_map_covered_fraction']=float(mass[labels>0].sum()/obj['mass'])
            obj['final_map_contributions']=sorted(contributions,key=lambda r:-r['gt_fraction'])
            obj['first_stage_below_half']=next((s for s in STAGES if obj['coverage'][s]<=.5),None)
        totals={}
        for klass in ['building','car']:
            local=[r for r in objects if r['category']==klass]
            totals[klass]=dict(eligible_gt=len(local),coverage_over_half={s:sum(r['coverage'][s]>.5 for r in local) for s in STAGES},
                first_loss={s:sum(r['first_stage_below_half']==s for r in local) for s in STAGES},
                remains_covered=sum(r['first_stage_below_half'] is None for r in local))
        result=dict(scene=scene,totals=totals,objects=objects,source_points={k:int(v.sum()) for k,v in domains.items()},
            observations=dict(all=manifest['observations'],with_connected_support=sum(r['connected_points']>0 for r in counts),
                three_frame_group=len(eligible),final=len(final),target=len(target)))
        dump(DEST/scene/'observations.json',counts);dump(DEST/scene/'results.json',result);results.append(result)
    dump(DEST/'summary.json',dict(complete=True,gt_diagnostic_only=True,protocol_sha256=sha(DEST/'protocol.json'),scenes=results))


if __name__=='__main__':main()
