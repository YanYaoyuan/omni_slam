"""Separate historical identity collisions from unsupported new writes.

No map changes, GT-assisted prediction or new threshold selection. Replays
the already-sealed cover candidate and records evidence for every addition
and every executed old-ID transition, on both complete development windows.
"""
from pathlib import Path
import time
import runpy
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_object_mask_cover import OUT as PARENT,RAW,verify,PARENT as OBS
from evaluate_kitti360_development import transfer,read_static_ply
from omni_semantic_map.kitti360_reference_identity import encode_reference

OUT=ROOT/'build/benchmarks/object_revision_provenance_v1'


def run():
    p=verify();assert not OUT.exists();inputs={}
    files=[PARENT/'protocol.json',PARENT/'summary.json',PARENT/'maps_complete.json',
           PARENT/'delivery_audit.json',PARENT/'per_object.json',RAW]
    for c in p['contexts']:
        k=c['key'];files += [c['reference'],c['raw_reference'],c['field'],c['job'],
            Path(c['full'])/'geometry.npz',Path(c['ours'])/'geometry.npz',Path(c['ours'])/'classification.json',
            PARENT/'prepared'/k/'components.npz',PARENT/'prepared'/k/'explanations.json',
            PARENT/'transport'/k/'cover/field.npz',OBS/'prepared'/k/'views.json']
        for m in ['cover_current','cover_history']:
            files += [PARENT/'maps'/k/m/'geometry.npz',PARENT/'maps'/k/m/'classification.json',
                      PARENT/'evaluation'/k/m/'reference_prediction.npz']
        files += [PARENT/'evaluation'/k/'baseline/reference_prediction.npz']
        files += [r['file'] for r in read(RAW)['results'][k]]
        files += [r['file'] for r in read(OBS/'prepared'/k/'views.json')]
    for f in files:inputs[str(f)]=sha(f)
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),source_sha256=sha(__file__),inputs=inputs,
        scope='Post-hoc provenance and observational identifiability, not a new method, score or map. Every executed old-ID transition and every new measured source in both full windows. No fitting, parameter search, selected GT success subset, robot or independent input.',
        support='For each actually visible full-source pixel, replay selected native-mask unions from accepted geometric components. A point supports its assigned object only if inside a union whose component exported that objectID. Count at most once per frame. Also retain any foreign-object union support and visible frames where its assigned object has an explanation but this pixel lies outside it. Outside partial masks is NOT certified negative evidence. Unknown/hidden views are not negatives.',
        lineage='Record every actual historical oldID->newID transition, its exact component/root, and reference-nearest raw identity composition as a diagnostic only. Pure old-ID footprints in a measured cell are built from baseline oldXYZ by exact20cmcell lookup, excluding cells occupied by multiple oldIDs. All pixels, not reference labels, form the observational footprints.',
        pair_evidence='For every executed transition with two surviving oldIDs, replay ALL positive native-score same-class masks in every jointly footprint-visible frame. Coverage A(q),B(q) uses per-visible-cell mass. Shared=max(score*min(A,B)); exclusive=sqrt(max(score*A*(1-B))*max(score*B*(1-A))). Report both continuously, all frames and all pairs; no decision threshold or merge policy learned here. Native scores are not calibrated identity probabilities.',
        reference_influence='Recompute nearest final-map INDEX for every reference point with stable lowest-index ties. Verify current/history labels exactly against sealed evaluator projections. For every eligible raw physical target, hold its original best baseline objectID fixed and split reference-weighted TP/FP and losses by old/new nearest source and mask support counts. No counterfactual map or AP exported.',
        interpretation='This can identify whether geometric transport supplied unobserved or unselected-mask evidence and whether an existing ID collision has distinguishable native-mask footprints. It does not prove a source filter or new identity module would improve whole-map AP. Correlated image views and nearest-reference labels are not independent measurements of true identity.',
        budget='One complete cached observation pass, no GPU/model calls/training/maps/scores. Output supports a concrete next mechanism decision, not goal completion.'))
    rawseal=read(RAW);previous=read(PARENT/'per_object.json');taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    results={}
    for c in p['contexts']:
        started=time.monotonic();key=c['key'];dest=OUT/key;dest.mkdir()
        base=dict(np.load(Path(c['ours'])/'geometry.npz'));cur=dict(np.load(PARENT/'maps'/key/'cover_current/geometry.npz'))
        hist=np.load(PARENT/'maps'/key/'cover_history/geometry.npz')['instance'];nold=len(base['xyz'])
        full=np.load(Path(c['full'])/'geometry.npz')['xyz'];nf=len(full)
        data=dict(np.load(PARENT/'transport'/key/'cover/field.npz'));parts=dict(np.load(PARENT/'prepared'/key/'components.npz'))
        field=np.load(c['field']);sourcecell=field['source_to_cell'];nc=len(field['cells'])
        assigned=data['certified_owner'][sourcecell];added=cur['added_full_source_ids']
        np.testing.assert_array_equal(cur['xyz'][:nold],base['xyz']);np.testing.assert_array_equal(cur['xyz'][nold:],full[added])
        owner_of_component=data['root_owner'][parts['roots']]
        oldclasses={r['object_id']:r['predictions']['mean']['train_id'] for r in read(Path(c['ours'])/'classification.json')['objects']}
        changed=hist[:nold]!=base['instance'];stride=int(max(hist.max(),base['instance'].max()))+1
        transitions=np.unique(base['instance'][changed]*stride+hist[:nold][changed])
        pair_keys=[tuple(map(int,divmod(int(e),stride))) for e in transitions]
        pair_rows={str(a)+'_'+str(b):[] for a,b in pair_keys if a in oldclasses and b in oldclasses and oldclasses[a]==oldclasses[b]}
        # Independently remap all oldXYZ to exact cells, then compute pure-ID cells.
        dtype=np.dtype([('x',np.int64),('y',np.int64),('z',np.int64)])
        fkeys=np.ascontiguousarray(field['cells'],np.int64).view(dtype).ravel()
        bkeys=np.ascontiguousarray(np.floor(base['xyz']/.2),np.int64).view(dtype).ravel()
        pos=np.searchsorted(fkeys,bkeys);valid=pos<len(fkeys);valid[valid]&=fkeys[pos[valid]]==bkeys[valid]
        oldcell=np.full(nold,-1,np.int64);oldcell[valid]=pos[valid];np.testing.assert_array_equal(oldcell,data['old_cell'])
        minimum=np.full(nc,np.iinfo(np.int64).max,np.int64);maximum=np.full(nc,-1,np.int64)
        np.minimum.at(minimum,pos[valid],base['instance'][valid]);np.maximum.at(maximum,pos[valid],base['instance'][valid])
        pure=np.where((minimum==maximum)&(minimum>0),minimum,0)
        seen=np.zeros(nf,np.int16);own=np.zeros_like(seen);foreign=np.zeros_like(seen);outside=np.zeros_like(seen);ambiguous=np.zeros_like(seen)
        records=read(PARENT/'prepared'/key/'explanations.json');rawby={r['frame_id']:r for r in rawseal['results'][key]}
        views=read(OBS/'prepared'/key/'views.json');checked=0
        for rank,rec in enumerate(views):
            v=np.load(rec['file']);ids=v['source_ids'];px=v['pixel_xy'];cell=v['cells'];owner=assigned[ids]
            weights=1./np.bincount(cell,minlength=nc)[cell]
            raw=dict(np.load(rawby[rec['frame_id']]['file']));cache={};by_owner={}
            def bits(q):
                if q not in cache:cache[q]=np.unpackbits(raw['masks_packed'][q],axis=1)[px[:,1],px[:,0]].astype(bool)
                return cache[q]
            for r in records:
                component=r['component'];oid=int(owner_of_component[component])
                if r['frame_id']!=rec['frame_id'] or oid<=0 or not parts['cover_accepted'][component]:continue
                union=np.zeros(len(ids),bool)
                for q in r['cover']['native_rows']:union |= bits(q)
                if oid in by_owner:by_owner[oid] |= union
                else:by_owner[oid]=union
            own_frame=np.zeros(len(ids),bool);foreign_frame=np.zeros(len(ids),bool);outside_frame=np.zeros(len(ids),bool)
            for oid,mask in by_owner.items():
                own_frame |= (owner==oid)&mask
                foreign_frame |= (owner!=oid)&mask
                outside_frame |= (owner==oid)&~mask
            seen[ids]+=1;own[ids]+=own_frame;foreign[ids]+=foreign_frame;outside[ids]+=outside_frame;ambiguous[ids]+=own_frame&foreign_frame
            footprint=pure[cell]
            for tag,rows in pair_rows.items():
                a,b=map(int,tag.split('_'));ta=footprint==a;tb=footprint==b
                if not ta.any() or not tb.any():continue
                rid=11 if oldclasses[a]==2 else 26;qq=np.flatnonzero((raw['semantic_ids']==rid)&(raw['scores']>0))
                if not len(qq):continue
                mass_a=float(weights[ta].sum());mass_b=float(weights[tb].sum())
                ca=np.array([weights[ta&bits(int(q))].sum()/mass_a for q in qq])
                cb=np.array([weights[tb&bits(int(q))].sum()/mass_b for q in qq]);score=raw['scores'][qq]
                joint=score*np.minimum(ca,cb);xa=score*ca*(1-cb);xb=score*cb*(1-ca)
                ji,ai,bi=map(int,[joint.argmax(),xa.argmax(),xb.argmax()])
                rows.append(dict(frame_id=rec['frame_id'],mass_a=mass_a,mass_b=mass_b,
                    shared=float(joint[ji]),exclusive=float(np.sqrt(xa[ai]*xb[bi])),shared_query=int(qq[ji]),
                    exclusive_query_a=int(qq[ai]),exclusive_query_b=int(qq[bi]),
                    shared_coverage_a=float(ca[ji]),shared_coverage_b=float(cb[ji])))
                # Fixed scalar pixel spot check, independent of truth and utility.
                for q in [int(qq[ji]),int(qq[ai]),int(qq[bi])]:
                    for i in [0,len(px)//2,len(px)-1]:
                        x,y=map(int,px[i]);expected=bool(raw['masks_packed'][q,y,x//8] & (1 << (7-x%8)))
                        assert expected==bits(q)[i];checked+=1
            if (rank+1)%12==0:print('REVISION SOURCE LEDGER',key,rank+1,'/',len(views),flush=True)
        assert np.all(own<=seen) and np.all(ambiguous<=own) and np.all(own+outside<=seen)
        np.savez_compressed(dest/'source_ledger.npz',seen=seen,own=own,foreign=foreign,outside=outside,ambiguous=ambiguous)
        # All actual source additions, including ones that never own a GT query.
        def source_summary(take):
            return dict(points=int(len(take)),never_visible=int((seen[take]==0).sum()),
                zero_own_support=int((own[take]==0).sum()),one_own_support=int((own[take]==1).sum()),
                at_least_two_own_support=int((own[take]>=2).sum()),any_foreign_support=int((foreign[take]>0).sum()),
                any_ambiguous_frame=int((ambiguous[take]>0).sum()),outside_own_explanation_any_frame=int((outside[take]>0).sum()))
        additions=source_summary(added)
        by_object={int(o):source_summary(added[cur['instance'][nold:]==o]) for o in np.unique(cur['instance'][nold:])}
        ref=dict(np.load(c['reference']));rawref=read_static_ply(c['raw_reference'])
        shifted=np.column_stack([rawref[t] for t in ['x','y','z']]).astype(float)-np.asarray(read(c['job'])['origin_world'])
        np.testing.assert_array_equal(shifted,ref['xyz'])
        truth,_=encode_reference(rawref['semantic'],rawref['instance'],taxonomy,'target_only')
        weight=ref['confidence'];base_truth=transfer(ref['xyz'],truth,base['xyz'],.2)
        lineage=[]
        for a,b in pair_keys:
            take=(base['instance']==a)&(hist[:nold]==b)
            gt,cnt=np.unique(base_truth[take],return_counts=True)
            dgt,dcnt=np.unique(base_truth[base['instance']==b],return_counts=True)
            tag=str(a)+'_'+str(b);rows=pair_rows.get(tag,[])
            lineage.append(dict(old_id=a,new_id=b,changed_points=int(take.sum()),old_total_points=int((base['instance']==a).sum()),
                changed_reference_identity_counts={int(g):int(n) for g,n in zip(gt,cnt)},
                recipient_original_reference_identity_counts={int(g):int(n) for g,n in zip(dgt,dcnt)},
                new_roots=np.flatnonzero(data['root_owner']==b).tolist(),joint_footprint_views=len(rows),
                mean_shared=float(np.mean([r['shared'] for r in rows])) if rows else None,
                mean_exclusive=float(np.mean([r['exclusive'] for r in rows])) if rows else None))
        # Index transfer provides exact provenance; old/current/history share XYZ.
        index=transfer(cur['xyz'],np.arange(len(cur['xyz']),dtype=np.int64)+1,ref['xyz'],.2)-1
        curpred=np.zeros(len(index),np.int64);histpred=curpred.copy();valid=index>=0
        curpred[valid]=cur['instance'][index[valid]];histpred[valid]=hist[index[valid]]
        for method,pred in [('cover_current',curpred),('cover_history',histpred)]:
            np.testing.assert_array_equal(pred,np.load(PARENT/'evaluation'/key/method/'reference_prediction.npz')['instance'])
        basepred=np.load(PARENT/'evaluation'/key/'baseline/reference_prediction.npz')['instance']
        new=index>=nold;ref_own=np.zeros(len(index),np.int16);ref_own[new]=own[added[index[new]-nold]]
        ref_seen=np.zeros(len(index),np.int16);ref_seen[new]=seen[added[index[new]-nold]]
        np.savez_compressed(dest/'reference_provenance.npz',nearest_index=index,own_mask_frames=ref_own,visible_frames=ref_seen)
        object_rows=[]
        for oldrow in previous['results'][key]['raw_targets']:
            gt=oldrow['gt_id'];oid=oldrow['methods']['baseline']['object_id'];target=truth==gt
            row=dict(gt_id=gt,fixed_baseline_object_id=oid,class_name=oldrow['class_name'],
                     best_iou=oldrow['methods'])
            def mass(t):return float(weight[t].sum())
            for name,pred in [('baseline',basepred),('current',curpred),('history',histpred)]:
                owned=pred==oid;inter=mass(owned&target);union=mass(owned|target)
                row[name]=dict(tp=inter,fp=mass(owned&~target),fn=mass(~owned&target),fixed_id_iou=inter/union if union else 0.)
            owned=curpred==oid
            row['current_new_reference_area']=dict(tp=mass(new&owned&target),fp=mass(new&owned&~target),
                tp_zero_own=mass(new&owned&target&(ref_own==0)),fp_zero_own=mass(new&owned&~target&(ref_own==0)),
                tp_two_or_more=mass(new&owned&target&(ref_own>=2)),fp_two_or_more=mass(new&owned&~target&(ref_own>=2)),
                previously_correct_stolen_by_another_id=mass(new&target&(basepred==oid)&(curpred!=oid)))
            row['historical_reference_change']=dict(tp_lost=mass(target&(curpred==oid)&(histpred!=oid)),
                tp_gained=mass(target&(curpred!=oid)&(histpred==oid)),
                fp_added=mass(~target&(curpred!=oid)&(histpred==oid)),fp_removed=mass(~target&(curpred==oid)&(histpred!=oid)))
            object_rows.append(row)
        result=dict(all_additions=additions,additions_by_object=by_object,executed_transitions=lineage,
            pair_observations=pair_rows,all_reference_objects=object_rows,scalar_mask_checks=checked,
            exact_reference_index_replays=2,seconds=time.monotonic()-started)
        dump(dest/'result.json',result);results[key]=result
        print('REVISION PROVENANCE COMPLETE',key,additions,'transitions',len(lineage),'pixel checks',checked,flush=True)
    proto=read(OUT/'protocol.json');assert sha(__file__)==proto['source_sha256']
    for f,h in proto['inputs'].items():assert sha(f)==h,f
    dump(OUT/'summary.json',dict(complete=True,results=results,protocol_sha256=sha(OUT/'protocol.json'),
        files={str(f):sha(f) for f in OUT.glob('*/*') if f.is_file()},
        new_maps=0,new_scores=0,new_model_calls=0,new_training=0,new_robot_runs=0,new_independent_scenes=0,goal_complete=False))


if __name__=='__main__':run()
