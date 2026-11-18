"""Posthoc coverage and per-target merging oracle; never predictor inputs."""
import itertools
import time
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_discovery_query_benchmark import OUT,INPUT,verify
from audit_kitti360_object_retention import covered
from evaluate_kitti360_development import transfer

DEST=ROOT/'build/benchmarks/discovery_full_map_audit_v1'
STAGES=['measurements','raw_masks','connected_support','three_frame_groups','final_members']


def best_union(intersections,masses,gt_mass):
    """Exact best subset IoU for disjoint predicted regions against one GT.

    At optimum, adding a region is beneficial iff its intersection/(mass-
    intersection) exceeds the current IoU. Therefore an optimum is a prefix
    sorted by intersection/mass. This oracle may allocate a region to multiple
    GTs in separate calls; it is not a joint feasible map or AP prediction.
    """
    inter=np.asarray(intersections,float);mass=np.asarray(masses,float)
    assert np.all(inter>=0) and np.all(inter<=mass+1e-8) and inter.sum()<=gt_mass+1e-5
    ids=np.flatnonzero((inter>0)&(mass>0))
    ids=ids[np.argsort(-inter[ids]/mass[ids],kind='stable')]
    if not len(ids):return 0.,[]
    hit=np.cumsum(inter[ids]);union=gt_mass+np.cumsum(mass[ids])-hit;iou=hit/union
    k=int(iou.argmax());return float(iou[k]),ids[:k+1].tolist()


def self_check():
    rng=np.random.default_rng(0)
    for _ in range(50):
        hit=rng.integers(0,100,6);mass=hit+rng.integers(0,200,6);gt=float(hit.sum()+10)
        brute=max(sum(hit[i] for i in subset)/(gt+sum(mass[i]-hit[i] for i in subset))
            for n in range(7) for subset in itertools.combinations(range(6),n))
        assert abs(best_union(hit,mass,gt)[0]-brute)<1e-12


def main():
    self_check();assert not (DEST/'summary.json').exists()
    protocol=verify(True);seal=read(OUT/'predictions_complete.json');masks=read(OUT/'masks_complete.json')
    for s in [seal,masks,read(OUT/'supports_complete.json')]:
        for f,h in s['files'].items():assert sha(f)==h,f
    dump(DEST/'protocol.json',dict(created_epoch=time.time(),posthoc=True,source_sha256=sha(__file__),
        parent_prediction_seal_sha256=sha(OUT/'predictions_complete.json'),radius_m=.2,
        purpose='Localize missing building/car support in sealed all-frame maps. No inference, tuning, new predictions or AP result.',
        oracle='Independent per-target optimal same-class union of disjoint projected predicted regions, including small fragments. Not jointly attainable, does not relabel or delete points. Necessary point coverage also grants perfect pointwise identity/background.',
        tests='50 small random cases: optimal prefix union equals exhaustive subset enumeration.'))
    results={}
    for s in protocol['scenes']:
        if not s['reference']:continue
        key=s['key'];assert sha(s['reference'])==protocol['reference_hashes'][s['reference']]
        manifest=read(INPUT/key/'manifest.json');measured={r['frame_id']:r for r in manifest['records']}
        source=np.load(manifest['geometry'])['xyz'];folder=OUT/'maps'/key/'all_frames';rows=read(folder/'classification.json')['objects']
        nodes={n['alias']:n for n in read(folder/'nodes.json')};groups=read(folder/'graph.json')['groups'];eligible=set()
        for group in groups:
            if len({f for a in group for f in nodes[a]['frames']})>=3:eligible.update(o for a in group for o in nodes[a]['members'])
        final={o for row in rows for o in row['observation_ids']};assert final<=eligible
        domains={k:np.zeros(len(source),bool) for k in STAGES};domains['measurements'][:]=True;observation_stats=[]
        for r in masks['scenes'][key]['records']:
            z=np.load(r['file']);measurement=np.load(measured[r['frame_id']]['file']);ids=measurement['visible_source_ids'];px=measurement['pixel_xy'];label=z['segmentation'][px[:,1],px[:,0]]
            domains['raw_masks'][ids[label>=0]]=True
            support=np.load(OUT/'frontend'/key/'supports'/f'{r["frame_id"]:06d}.npz');points=support['point_ids'];offsets=support['offsets']
            domains['connected_support'][points]=True
            for j in range(r['masks']):
                oid=r['observation_offset']+j;current=points[offsets[j]:offsets[j+1]]
                for name,members in [('three_frame_groups',eligible),('final_members',final)]:
                    if oid in members:domains[name][current]=True
                observation_stats.append(dict(observation_id=oid,frame=r['frame_id'],semantic_id=int(z['semantic_ids'][j]),raw_points=int(np.count_nonzero(label==j)),connected_points=len(current),eligible=oid in eligible,final=oid in final))
        for a,b in zip(STAGES,STAGES[1:]):assert not np.any(domains[b]&~domains[a])
        ref=np.load(s['reference']);truth=ref['train_instance'];weight=ref['confidence'].astype(float);selected=np.isin(truth//1000,[2,13]);objects=[]
        for gid in np.unique(truth[selected]):
            mass=float(weight[truth==gid].sum())
            if mass>=100:objects.append(dict(gt_id=int(gid),category='car' if gid//1000==13 else 'building',mass=mass,coverage={}))
        for stage in STAGES:
            hit=covered(source,domains[stage],ref['xyz'][selected]);gt=truth[selected];w=weight[selected]
            for obj in objects:obj['coverage'][stage]=float(w[(gt==obj['gt_id'])&hit].sum()/obj['mass'])
            print(key,stage,{c:sum(o['category']==c and o['coverage'][stage]>.5 for o in objects) for c in ['building','car']},flush=True)
        geom=np.load(folder/'geometry.npz');projected=transfer(geom['xyz'],geom['instance'],ref['xyz'],.2)
        maximum=int(projected.max(initial=0));mass=np.bincount(projected,weights=weight,minlength=maximum+1)
        class_ids={r['object_id']:r['predictions']['mean']['semantic_id'] for r in rows}
        for obj in objects:
            hit=truth==obj['gt_id'];inter=np.bincount(projected[hit],weights=weight[hit],minlength=maximum+1)
            ids=np.asarray([i for i in range(1,maximum+1) if class_ids.get(i)==(26 if obj['category']=='car' else 11)],int)
            valid=ids[mass[ids]>=100];single=inter[valid]/(obj['mass']+mass[valid]-inter[valid])
            best,subset=best_union(inter[ids],mass[ids],obj['mass'])
            obj.update(best_valid_single_iou=float(single.max(initial=0)),best_same_class_union_iou=best,
                oracle_union_object_ids=ids[subset].tolist(),same_class_coverage=float(inter[ids].sum()/obj['mass']),
                final_map_coverage=float(inter[1:].sum()/obj['mass']),
                first_support_loss=next((stage for stage in STAGES if obj['coverage'][stage]<=.5),None))
        totals={c:dict(gt=sum(o['category']==c for o in objects),
            coverage_over_half={stage:sum(o['category']==c and o['coverage'][stage]>.5 for o in objects) for stage in STAGES},
            single_matched=sum(o['category']==c and o['best_valid_single_iou']>.5 for o in objects),
            oracle_merge_matched=sum(o['category']==c and o['best_same_class_union_iou']>.5 for o in objects)) for c in ['car','building']}
        results[key]=dict(totals=totals,objects=objects,source_points={k:int(v.sum()) for k,v in domains.items()})
        dump(DEST/key/'observations.json',observation_stats);dump(DEST/key/'results.json',results[key]);print('ORACLES',key,totals,flush=True)
    dump(DEST/'summary.json',dict(complete=True,diagnostic_only=True,results=results,protocol_sha256=sha(DEST/'protocol.json')))


if __name__=='__main__':main()
