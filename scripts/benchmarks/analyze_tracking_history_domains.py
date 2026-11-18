"""Diagnose measured-cell evidence coverage without treating cells as points.

Uses the existing frozen 20cm grid, not a resolution sweep. A cell with both
inside and outside measurements abstains in that frame. This is a post-hoc
transport diagnostic, not a new map, chosen pruning rule or method result.
"""
import runpy
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from audit_tracking_history import OUT, BASE, count, evidence_groups
from probe_world_anchor_tracking import OUT as TRACK, INPUT, METHODS, prepared
from evaluate_kitti360_development import transfer, read_static_ply


def main():
    assert not (OUT/'cell_summary.json').exists()
    prior=read(OUT/'summary.json'); assert prior['complete']
    for f,h in prior['files'].items(): assert sha(f)==h,f
    p=read(TRACK/'protocol.json'); cases=prepared()['cases']
    for f,h in read(TRACK/'predictions_complete.json')['files'].items(): assert sha(f)==h,f
    pins={str(OUT/'summary.json'):sha(OUT/'summary.json'),str(TRACK/'predictions_complete.json'):sha(TRACK/'predictions_complete.json')}
    for key in prior['results']:
        f=INPUT/'query_cells'/key/'source_to_cell.npy'; pins[str(f)]=sha(f)
    dump(OUT/'cell_protocol.json',dict(created_epoch=time.time(),source_sha256=sha(__file__),input_hashes=pins,
        rule='Existing20cm world-grid IDs. Every currently measured visible source contributes its binary mask membership. Per frame/cell: all inside=support, all outside=exclusion, mixed=unknown. Unknown/missing cells abstain. One vote per cell/frame, then read at old-map point cell; no GT in construction.',
        difference='Cell co-observation is not exact-point depth visibility. Multiple surfaces or instances can share a cell. This deliberately diagnoses coverage and contamination, not a safe revision policy. Offline inherited map/prompt limitations remain.',
        scope='Post-hoc, same8clips/common nonseed frames/3sealed masks. No new sampling, mask generation, fitted parameters, geometry/identity edits or AP. Errors classified against fixed baseline GT association; N/A retained.'))
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    results={}; files={}
    for scene in p['scenes']:
        key=scene['key']; base=dict(np.load(BASE/key/'geometry_graph/geometry.npz')); ref=dict(np.load(scene['reference'])); raw=read_static_ply(p['references'][key]['raw']); train=np.array([taxonomy[int(v)].trainId for v in raw['semantic']])
        nn=transfer(ref['xyz'],np.arange(1,len(train)+1),base['xyz'],.2); ix=np.maximum(nn-1,0); known=(nn>0)&(train[ix]!=255)&(ref['confidence'][ix]>0); truth=ref['train_instance'][ix]; weights=ref['confidence'][ix]
        geometry=np.load(read(INPUT/key/'manifest.json')['geometry']); cell=np.load(INPUT/'query_cells'/key/'source_to_cell.npy'); ids,first=np.unique(cell,return_index=True); np.testing.assert_array_equal(ids,np.arange(len(ids)))
        keys=np.floor(geometry['xyz'][first]/.2).astype(np.int64); lookup={tuple(k):int(i) for i,k in enumerate(keys)}; assert len(lookup)==len(keys)
        map_cells=np.array([lookup.get(tuple(k),-1) for k in np.floor(base['xyz']/.2).astype(np.int64)])
        rows=[]
        for case in [c for c in cases if c['key']==key]:
            direct=next(r for r in prior['results'][key]['cases'] if r['case']==case['name']); oid=case['object_id']; gid=direct['gt_target']; mids=np.flatnonzero(base['instance']==oid); mc=map_cells[mids]; eligible=mc>=0
            masks={m:np.load(TRACK/'cases'/case['name']/(m+'.npz')) for m in METHODS}; counters={m:np.zeros((2,len(mids)),np.int16) for m in METHODS}; frame_counts=[]
            for j in case['common_frames']:
                obs=np.load(case['records'][j]['file']); xy=obs['pixel_xy']; sid=obs['visible_source_ids']; cells,inverse,counts=np.unique(cell[sid],return_inverse=True,return_counts=True); stat={}
                for method in METHODS:
                    packed=masks[method]['masks_packed'][j]; inside=((packed[xy[:,1],xy[:,0]//8]>>(7-xy[:,0]%8))&1).astype(bool); yes=np.bincount(inverse,weights=inside,minlength=len(cells)); votes=np.zeros(len(ids),np.int8); votes[cells[yes==counts]]=1; votes[cells[yes==0]]=-1
                    v=np.zeros(len(mids),np.int8); v[eligible]=votes[mc[eligible]]; counters[method][0]+=v==1; counters[method][1]+=v==-1
                    stat[method]=dict(measured_cells=len(cells),mixed_cells=int(((yes>0)&(yes<counts)).sum()),map_points_voted=int((v!=0).sum()))
                frame_counts.append(dict(frame=case['records'][j]['frame_id'],methods=stat))
            valid=known[mids] if gid is not None else np.zeros(len(mids),bool); correct=truth[mids]==gid if gid is not None else np.zeros(len(mids),bool); w=weights[mids]
            row=dict(case=case['name'],gt_target=gid,missing_cell_points=int((~eligible).sum()),frames=frame_counts,cell_methods={},error_semantics={}); arrays={'map_ids':mids}
            direct_arrays=np.load(OUT/'cases'/(case['name']+'.npz'))
            for method in METHODS:
                pos,neg=counters[method]; assert np.all(pos+neg<=len(case['common_frames']))
                row['cell_methods'][method]={name:count(take,valid,correct,w) for name,take in evidence_groups(pos,neg).items()}
                arrays[method+'_positive']=pos; arrays[method+'_negative']=neg
                row['error_semantics'][method]={}
                for domain,positive,negative in [('point',direct_arrays['map_'+method+'_positive'],direct_arrays['map_'+method+'_negative']),('cell',pos,neg)]:
                    detail={}
                    for name,take in evidence_groups(positive,negative).items():
                        bad=take&valid&~correct; same=(truth[mids]//1000==gid//1000) if gid is not None else np.zeros(len(mids),bool)
                        detail[name]=dict(same_class_other_instance=int((bad&same).sum()),other_class=int((bad&~same).sum()))
                    row['error_semantics'][method][domain]=detail
            f=OUT/'cell_cases'/(case['name']+'.npz'); f.parent.mkdir(exist_ok=True); np.savez_compressed(f,**arrays); files[str(f)]=sha(f); rows.append(row)
        totals={}; semantic={}
        for method in METHODS:
            totals[method]={}; semantic[method]={}
            for group in evidence_groups(np.zeros(1),np.zeros(1)):
                entries=[r['cell_methods'][method][group] for r in rows]; totals[method][group]={k:sum(e[k] for e in entries) for k in entries[0]}
            for domain in ['point','cell']:
                semantic[method][domain]={g:{k:sum(r['error_semantics'][method][domain][g][k] for r in rows) for k in ['same_class_other_instance','other_class']} for g in totals[method]}
        results[key]=dict(cases=rows,totals=totals,error_semantics=semantic)
        print('CELL DIAGNOSTIC',key,totals['video_seed'],flush=True)
        print('ERROR CLASSES',key,semantic['video_seed'],flush=True)
    dump(OUT/'cell_summary.json',dict(complete=True,protocol_sha256=sha(OUT/'cell_protocol.json'),results=results,files=files,new_maps=0,new_ap=0))
    print('CELL AUDIT COMPLETE',flush=True)


if __name__=='__main__': main()
