"""Post-hoc audit of whether sealed video masks can revise existing geometry.

No new masks, fitting, map changes, threshold selection or benchmark scoring.
Counts distinguish original measured sources from the actual fused-map points.
Exclusion from an object mask is not evidence for another object/background.
"""
import time
import runpy
from types import SimpleNamespace
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from probe_world_anchor_tracking import OUT as TRACK, INPUT, METHODS, verify, prepared
from evaluate_kitti360_development import transfer, read_static_ply
from omni_semantic_map.sparse_lidar_support import project_measured_visibility

OUT = ROOT/'build/benchmarks/tracking_history_audit_v1'
BASE = ROOT/'build/benchmarks/signed_query_association_v1/maps'


def evidence_groups(positive, negative):
    return dict(unseen=(positive+negative == 0),
                supported_only=(positive > 0)&(negative == 0),
                excluded_once=(positive == 0)&(negative == 1),
                excluded_repeated=(positive == 0)&(negative >= 2),
                contradictory=(positive > 0)&(negative > 0))


def count(take, known, correct, weight):
    k=take&known
    return dict(points=int(take.sum()), known=int(k.sum()), unknown=int((take&~known).sum()),
                correct=int((k&correct).sum()), incorrect=int((k&~correct).sum()),
                correct_weight=float(weight[k&correct].sum()),
                incorrect_weight=float(weight[k&~correct].sum()))


def main():
    assert not (OUT/'summary.json').exists(), 'Do not overwrite a completed audit'
    resume=(OUT/'protocol.json').exists()
    if resume:
        assert not (OUT/'cases').exists(), 'Inspect partially written cases before resuming'
        assert not (OUT/'implementation_fix.json').exists(), 'Inspect repeated failure before resuming'
        previous=read(OUT/'protocol.json')
        for f,h in previous['input_hashes'].items(): assert sha(f)==h, f
        dump(OUT/'implementation_fix.json',dict(previous_protocol_sha256=sha(OUT/'protocol.json'),
             corrected_source_sha256=sha(__file__), reason='camera_pose is stored in the manifest record, not the source visibility NPZ. Initial attempt stopped before writing any case. No input, mask, target, gate or diagnostic policy changed.'))
    p=verify(); cases=prepared()['cases']; seal=read(TRACK/'predictions_complete.json')
    for f,h in seal['files'].items(): assert sha(f)==h, f
    targets={r['case']:r['gt_target'] for s in read(TRACK/'summary.json')['results'].values() for r in s['cases']}
    pins={str(TRACK/n):sha(TRACK/n) for n in ['protocol.json','prepared.json','predictions_complete.json','summary.json']}
    for scene in p['scenes']:
        key=scene['key']; job=read(scene['job'])
        needed={r['frame_id'] for c in cases if c['key']==key for j,r in enumerate(c['records']) if j in c['common_frames']}
        files=[scene['job'],scene['reference'],p['references'][key]['raw'],BASE/key/'geometry_graph/geometry.npz']
        files += [f['measured'] for f in job['frames'] if f['frame_id'] in needed]
        for f in files: pins[str(f)]=sha(f)
    protocol=dict(created_epoch=time.time(), source_sha256=sha(__file__), input_hashes=pins,
        purpose='Post-hoc mechanism diagnosis on all eight existing clips. Not a new method/gate, map evaluation or independent test.',
        observation_domain='Same common eligible nonseed frames and three sealed mask arms as prior probe. Actual fused map points projected using exact original poses, sparse measured depth and unchanged abs5cm/relative0 gate. No dilation/interpolation/neighbour substitution.',
        domains='source: only predicted provenance owned by this object, seen at exact recorded source pixels, and source_frame<current frame. map: original fused geometry of this object with current measured depth support. Offline window-end maps; map point acquisition times are not inferred.',
        classification='Unseen, supported-only, excluded once, excluded in>=2frames with no support, or contradictory. Repeated video outputs are correlated, not independent measurements. Exclusion is not a positive class or alternative-owner label.',
        reference='Post-hoc nearest reference within20cm, confidence>0 and raw trainId!=255. Fixed prior best same-class GT per object. N/A case retained; no GT-based selection, mask/ownership change or parameter tuning.',
        scope='Two exposed windows, inherited supervised baseline. Full map remains unchanged; no new AP/PQ, training or GPU inference.')
    if resume: assert pins==previous['input_hashes']
    else: dump(OUT/'protocol.json',protocol)
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    results={}; files={}
    for scene in p['scenes']:
        key=scene['key']; ref=dict(np.load(scene['reference'])); raw=read_static_ply(p['references'][key]['raw'])
        train=np.array([taxonomy[int(v)].trainId for v in raw['semantic']]); expected=train*1000+raw['instance'].astype(np.int64)%1000; expected[train==255]=0
        np.testing.assert_array_equal(expected,ref['train_instance'])
        manifest=read(INPUT/key/'manifest.json'); source=dict(np.load(manifest['geometry'])); anchors=np.load(TRACK/f'{key}_anchors.npz')['owners']
        base=dict(np.load(BASE/key/'geometry_graph/geometry.npz')); job=read(scene['job']); frames={r['frame_id']:r for r in job['frames']}
        labels={}
        for domain,xyz in [('source',source['xyz']),('map',base['xyz'])]:
            nn=transfer(ref['xyz'],np.arange(1,len(train)+1),xyz,.2); ix=np.maximum(nn-1,0)
            labels[domain]=dict(known=(nn>0)&(train[ix]!=255)&(ref['confidence'][ix]>0), truth=ref['train_instance'][ix], weight=ref['confidence'][ix])
        rows=[]
        for case in [c for c in cases if c['key']==key]:
            oid=case['object_id']; gid=targets[case['name']]; masks={m:np.load(TRACK/'cases'/case['name']/(m+'.npz')) for m in METHODS}
            selected={'source':np.flatnonzero(anchors==oid), 'map':np.flatnonzero(base['instance']==oid)}
            counters={d:{m:np.zeros((2,len(v)),np.int16) for m in METHODS} for d,v in selected.items()}
            inverse=np.full(len(source['xyz']),-1,np.int64); inverse[selected['source']]=np.arange(len(selected['source']))
            frame_rows=[]
            for j in case['common_frames']:
                rec=case['records'][j]; f=frames[rec['frame_id']]; obs=np.load(rec['file']); sid=obs['visible_source_ids']
                take=(inverse[sid]>=0)&(source['source_frame'][sid]<rec['frame_id'])
                sx,sy=obs['pixel_xy'][take].T; si=inverse[sid[take]]
                assert len(si)==len(np.unique(si))
                np.testing.assert_allclose(rec['camera_pose'],np.asarray(f['pose']),rtol=0,atol=1e-10)
                depth=np.load(f['measured'])['depth']; mx,my,_,visible=project_measured_visibility(base['xyz'][selected['map']],np.asarray(f['pose']),job['config']['cam'],depth,SimpleNamespace(depth_absolute_m=.05,depth_relative=0.))
                mi=np.flatnonzero(visible); mx=mx[mi]; my=my[mi]
                frame_rows.append(dict(frame=rec['frame_id'],historical_sources=len(si),visible_old_map_points=len(mi)))
                for method in METHODS:
                    assert masks[method]['valid'][j]
                    mask=masks[method]['masks_packed'][j]
                    for domain,ix,x,y in [('source',si,sx,sy),('map',mi,mx,my)]:
                        inside=((mask[y,x//8]>>(7-x%8))&1).astype(bool)
                        counters[domain][method][0,ix]+=inside
                        counters[domain][method][1,ix]+=~inside
            row=dict(case=case['name'],object_id=oid,gt_target=gid,frames=frame_rows,domains={})
            arrays={}
            for domain,ids in selected.items():
                lab=labels[domain]; known=lab['known'][ids] if gid is not None else np.zeros(len(ids),bool)
                correct=lab['truth'][ids]==gid if gid is not None else np.zeros(len(ids),bool); weight=lab['weight'][ids]
                row['domains'][domain]=dict(total=count(np.ones(len(ids),bool),known,correct,weight), methods={})
                arrays[domain+'_ids']=ids
                for method in METHODS:
                    positive,negative=counters[domain][method]; groups=evidence_groups(positive,negative)
                    assert np.all(np.sum(list(groups.values()),axis=0)==1)
                    row['domains'][domain]['methods'][method]={name:count(take,known,correct,weight) for name,take in groups.items()}
                    arrays[domain+'_'+method+'_positive']=positive; arrays[domain+'_'+method+'_negative']=negative
            path=OUT/'cases'/(case['name']+'.npz'); path.parent.mkdir(exist_ok=True); np.savez_compressed(path,**arrays); files[str(path)]=sha(path)
            rows.append(row)
            print('HISTORY',case['name'],'target',gid,'map',row['domains']['map']['total'],flush=True)
        totals={}
        for domain in ['source','map']:
            totals[domain]={}
            for method in METHODS:
                totals[domain][method]={}
                for group in evidence_groups(np.zeros(1),np.zeros(1)):
                    entries=[r['domains'][domain]['methods'][method][group] for r in rows]
                    totals[domain][method][group]={k:sum(v[k] for v in entries) for k in entries[0]}
        results[key]=dict(cases=rows,totals=totals)
        print('TOTAL',key,'video/map',totals['map']['video_seed'],flush=True)
    dump(OUT/'summary.json',dict(complete=True,protocol_sha256=sha(OUT/'protocol.json'),source_sha256=sha(__file__),files=files,results=results,new_maps=0,new_inference=0,new_ap=0))
    for f,h in pins.items(): assert sha(f)==h, f
    print('AUDIT COMPLETE',flush=True)


if __name__=='__main__': main()
