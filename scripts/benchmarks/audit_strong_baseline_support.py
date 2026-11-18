"""Read-only full-window coverage and error inventory on the strong baseline.

References are diagnostic only. Coverage is a necessary geometric condition,
not attainable AP and not a label-free object selector. No new map is emitted.
"""
import time
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha
from prepare_partition_revision import PARENT,INPUT

OUT=ROOT/'build/benchmarks/strong_baseline_support_audit_v1'
CAPTURE=ROOT/'build/benchmarks/prethreshold_candidates_v1'


def main():
    assert not OUT.exists()
    parent=read(PARENT/'protocol.json');capture=read(CAPTURE/'predictions_complete.json')
    details=read(PARENT/'object_diagnostics.json')['results'];files={}
    for f in [PARENT/'protocol.json',PARENT/'predictions_complete.json',PARENT/'object_diagnostics.json',CAPTURE/'predictions_complete.json']:
        files[str(f)]=sha(f)
    for scene in parent['scenes']:
        key=scene['key'];manifest=read(INPUT/key/'manifest.json')
        for f in [scene['reference'],manifest['geometry'],INPUT/key/'manifest.json',PARENT/'maps'/key/'geometry_graph/geometry.npz']:
            files[str(f)]=sha(f)
        for r in manifest['records']:files[r['file']]=sha(r['file'])
    for f,h in capture['files'].items():assert sha(f)==h;files[f]=h
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),source_sha256=sha(__file__),inputs=files,
        scope='Existing strong geometry_graph maps, all eligible car/building GTs in exposed0000/0002. No new inference, maps, fitting, candidate selection, robot data or independent test access.',
        coverage='At fixed existing20cm reference transfer radius, measure weighted GT surface coverage by all measured points; union of all same-class raw native positive masks; same-class raw score>0.5 masks; final same-class map points. Mask unions ignore identity, exclusivity and component truncation, so they are optimistic necessary coverage, never map AP.',
        support='Visible original measured source IDs and original packed masks; no new29validation frames. ALL source points retained in the raw support baseline. Score>0 and>0.5 are existing capture definitions; no threshold sweep.',
        diagnostic='Every eligible GT mass>=100. Recover best actual object IoU, precision, recall and count of fragments sharing dominant GT; these use reference labels and are never predictor inputs. Report counts at fixed IoU50/75/90. Final map resampling is not guaranteed to be a subset of raw source coordinates, so do not assert that nesting.',
        interpretation='Coverage below threshold rules out that geometry pool under this distance transfer. Coverage above threshold does not prove feasible object assignment, purity or AP; this audit is not an oracle method result.'))
    results={}
    for scene in parent['scenes']:
        started=time.monotonic();key=scene['key'];manifest=read(INPUT/key/'manifest.json');source=np.load(manifest['geometry'])['xyz'];ref=dict(np.load(scene['reference']))
        truth=ref['train_instance'];weight=ref['confidence'];mass=np.bincount(truth,weights=weight)
        target=np.isin(truth//1000,[2,13]);points=ref['xyz'][target];gt=truth[target];w=weight[target]
        supported={sem:{name:np.zeros(len(source),bool) for name in ['any_native','high_native']} for sem in [11,26]}
        views={r['frame_id']:r for r in manifest['records']}
        for row in capture['results'][key]:
            d=dict(np.load(row['file']));v=dict(np.load(views[row['frame_id']]['file']));ids=v['visible_source_ids'];xy=v['pixel_xy']
            for sem in [11,26]:
                for name,threshold in [('any_native',0.),('high_native',.5)]:
                    selected=(d['semantic_ids']==sem)&(d['scores']>threshold)
                    if not selected.any():continue
                    packed=np.bitwise_or.reduce(d['masks_packed'][selected],axis=0)
                    inside=((packed[xy[:,1],xy[:,0]//8]>>(7-xy[:,0]%8))&1).astype(bool)
                    supported[sem][name][ids[inside]]=True
        d=cKDTree(source).query(points,distance_upper_bound=.2,workers=1)[0];coverage={'all_measured':np.isfinite(d)}
        geometry=np.load(PARENT/'maps'/key/'geometry_graph/geometry.npz');objects=details[key]['geometry_graph']
        for name in ['any_native','high_native','final_same_class']:
            hit=np.zeros(len(points),bool)
            for sem,tid in [(11,2),(26,13)]:
                take=gt//1000==tid
                if name=='final_same_class':cloud=geometry['xyz'][np.isin(geometry['instance'],[o['object_id'] for o in objects if o['semantic_id']==sem])]
                else:cloud=source[supported[sem][name]]
                if len(cloud):hit[take]=np.isfinite(cKDTree(cloud).query(points[take],distance_upper_bound=.2,workers=1)[0])
            coverage[name]=hit
        assert np.all(~coverage['high_native']|coverage['any_native'])
        assert np.all(~coverage['any_native']|coverage['all_measured'])
        bygt={name:np.bincount(gt,weights=w*hit,minlength=len(mass))/np.maximum(mass,1e-12) for name,hit in coverage.items()}
        rows=[]
        for gid in np.unique(gt):
            if mass[gid]<100:continue
            sem=11 if gid//1000==2 else 26
            candidates=[(q['iou'],o,q) for o in objects if o['semantic_id']==sem for q in o['overlaps'] if q['gt_id']==gid]
            best=max(candidates,key=lambda x:x[0]) if candidates else None
            fragments=[o['object_id'] for o in objects if o['overlaps'] and o['overlaps'][0]['gt_id']==gid]
            record=dict(gt_id=int(gid),category='building' if sem==11 else 'car',mass=float(mass[gid]),coverage={name:float(v[gid]) for name,v in bygt.items()},dominant_fragments=fragments)
            if best:
                iou,o,q=best;inter=q['intersection'];record['best_actual']=dict(object_id=o['object_id'],iou=iou,precision=inter/o['reference_mass'],recall=inter/q['gt_mass'],false_mass=o['reference_mass']-inter,missing_mass=q['gt_mass']-inter)
            else:record['best_actual']=dict(object_id=None,iou=0.,precision=0.,recall=0.,false_mass=0.,missing_mass=float(mass[gid]))
            rows.append(record)
        counts={}
        for category in ['building','car']:
            eligible=[r for r in rows if r['category']==category];counts[category]=dict(gt_count=len(eligible),thresholds={})
            for threshold in [.5,.75,.9]:
                counts[category]['thresholds'][str(threshold)]={name:sum(r['coverage'][name]>threshold for r in eligible) for name in coverage}
                counts[category]['thresholds'][str(threshold)]['best_actual_iou']=sum(r['best_actual']['iou']>threshold for r in eligible)
            counts[category]['unmatched_but_raw_coverage_over_half']=[r['gt_id'] for r in eligible if r['best_actual']['iou']<=.5 and r['coverage']['all_measured']>.5]
            counts[category]['matched_precision_lower_than_recall']=sum(r['best_actual']['iou']>.5 and r['best_actual']['precision']<r['best_actual']['recall'] for r in eligible)
        result=dict(objects=rows,counts=counts,seconds=time.monotonic()-started,source_counts={str(s):{k:int(v.sum()) for k,v in d.items()} for s,d in supported.items()})
        dump(OUT/(key+'.json'),result);results[key]=result;print('AUDIT',key,counts,flush=True)
    dump(OUT/'summary.json',dict(complete=True,diagnostic_only=True,no_new_map=True,results=results,protocol_sha256=sha(OUT/'protocol.json')))


if __name__=='__main__':main()
