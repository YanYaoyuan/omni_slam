"""Read-only audit of sealed outdoor predictions; no inference or tuning."""
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from run_kitti360_reserved import OUT, META, METHODS, verify
from evaluate_kitti360_development import transfer, clean_json
from omni_semantic_map.kitti360_evaluation import evaluate

DEST = ROOT/'output/kitti360_equal_score_audit_v1'


def evidence_audit(root, arm):
    """Independently reconstruct the point/frame funnel from masks and depths."""
    folder = root/arm
    job = read(folder/'job.json'); cache = Path(job['output'])/'cache'
    m = read(cache/'manifest.json'); cam = m['config']['cam']
    cfg = read(folder/'mapping_protocol.json')['cleanup']
    g = np.load(folder/'maps/cg/history2/geometry.npz')
    rows = read(folder/'maps/cg/history2/classification.json')['objects']
    owners = {int(o):r['object_id'] for r in rows for o in r['observation_ids']}
    ev = np.load(folder/'maps/cleanup/evidence.npz')
    xyz = g['xyz'][ev['source_indices']]; labels = g['instance'][ev['source_indices']]
    poses = np.load(cache/'poses.npy'); totals = Counter(); events = []
    positives = np.zeros(len(xyz), np.int64); negatives = positives.copy()
    k = np.ones((2*cfg['boundary_pixels']+1,)*2, np.uint8)
    for rec, frame, pose in zip(m['records'], job['frames'], poses):
        assert rec['frame_id'] == frame['frame_id']
        depth = np.load(frame['measured'])['depth']
        packed = np.load(cache/'frames'/rec['file'])['masks_packed']
        masks = np.unpackbits(packed, axis=2)[:, :, :cam['W']].astype(bool)
        assigned = np.array([owners.get(rec['observation_offset']+i, 0) for i in range(rec['masks'])])
        inv = np.linalg.inv(pose); pc = xyz@inv[:3,:3].T + inv[:3,3]
        z = pc[:,2]; u = pc[:,0]*cam['fx']/np.maximum(z,1e-9)+cam['cx']
        v = pc[:,1]*cam['fy']/np.maximum(z,1e-9)+cam['cy']
        fov = (z>.2)&(u>=-.5)&(u<cam['W']-.5)&(v>=-.5)&(v<cam['H']-.5)
        x = np.floor(np.clip(u+.5,0,cam['W']-1)).astype(int)
        y = np.floor(np.clip(v+.5,0,cam['H']-1)).astype(int)
        d = depth[y,x]; measured = fov & np.isfinite(d) & (d>0)
        visible = measured & (abs(z-d)<=cfg['depth_absolute_m']+cfg['depth_relative']*d)
        owner_present = np.isin(labels,assigned[assigned>0])
        known = masks.any(0)[y,x]
        pos = np.zeros(len(xyz),bool); neg = pos.copy(); outside = pos.copy()
        for oid in np.unique(labels):
            if oid not in assigned: continue
            union = masks[assigned==oid].any(0).astype(np.uint8)
            inner = cv2.erode(union,k,borderType=cv2.BORDER_CONSTANT,borderValue=0)[y,x]>0
            outer = cv2.dilate(union,k,borderType=cv2.BORDER_CONSTANT,borderValue=0)[y,x]>0
            local = (labels==oid)&fov
            pos |= local & inner; outside |= local & ~outer
            neg |= local & ~outer & known
        positives += pos & visible; negatives += neg & visible
        row = dict(point_frame_pairs=len(xyz),in_fov=int(fov.sum()),
            measured_pixel=int(measured.sum()),depth_agreement=int(visible.sum()),
            owner_present_in_fov=int((fov&owner_present).sum()),
            visible_owner_present=int((visible&owner_present).sum()),
            visible_owner_absent=int((visible&~owner_present).sum()),
            refutation_before_depth_gate=int(neg.sum()),
            refutation_has_measured_pixel=int((neg&measured).sum()),
            refutation_rejected_by_missing_depth=int((neg&~measured).sum()),
            refutation_rejected_by_depth_disagreement=int((neg&measured&~visible).sum()),
            outside_expanded_own_mask_visible=int((outside&visible).sum()),
            positive=int((pos&visible).sum()),negative=int((neg&visible).sum()))
        totals.update(row); events.append(dict(frame_id=rec['frame_id'],**row))
    np.testing.assert_array_equal(positives,ev['positive'])
    np.testing.assert_array_equal(negatives,ev['negative'])
    archived = read(folder/'maps/cleanup/events.json')
    for a,b in zip(archived,events):
        assert (a['positive'],a['negative'],a['visible']) == (b['positive'],b['negative'],b['depth_agreement'])
    return dict(totals=dict(totals),pointwise_votes_reproduced=True,events=events)


def main():
    protocol = verify(); seal = read(OUT/'predictions_complete.json')
    for f,h in seal['files'].items(): assert sha(f)==h,f
    DEST.mkdir(parents=True,exist_ok=True)
    report = dict(sealed_files_verified=len(seal['files']),scenes={},source_sha256=sha(__file__),
        scope='Post-hoc audit of existing two-window local adapted evaluation; no new predictions or tuning.')
    for w in protocol['windows']:
        scene=w['scene']; root=OUT/scene; ep=read(root/'evaluation/protocol.json')
        ref=np.load(root/'evaluation/reference.npz'); truth=ref['train_instance']; weights=ref['confidence']
        details={}; canonical={}; controls={}; table=[]
        base_rows=read(root/'base/maps/cg/history2/classification.json')['objects']
        base_keys={tuple(sorted(r['observation_ids'])):r['object_id'] for r in base_rows}
        for method,(arm,subpath) in METHODS.items():
            folder=root/arm/subpath
            assert Path(ep['methods'][method])==folder
            g=np.load(folder/'geometry.npz'); rows=read(folder/'classification.json')['objects']
            preds={r['object_id']:r['predictions']['mean'] for r in rows if r['predictions']['mean']['retained']}
            labels=np.where(np.isin(g['instance'],list(preds)),g['instance'],0)
            projected=transfer(g['xyz'],labels,ref['xyz'],ep['export_radius_m'])
            saved=np.load(root/'evaluation'/f'{method}_prediction.npz')['instance']
            np.testing.assert_array_equal(projected,saved)
            canon=np.zeros_like(projected)
            for r in rows:
                if r['object_id'] in preds:
                    key=tuple(sorted(r['observation_ids'])); assert key in base_keys
                    canon[projected==r['object_id']]=base_keys[key]
            canonical[method]=canon
            domains={}
            for name,domain in [('full_window',np.ones(len(truth),bool)),('observed_support',ref['observed_support'])]:
                result=evaluate(truth[domain],weights[domain],projected[domain],preds,META/'evalInstanceLevelSemanticLabeling.py')
                assert clean_json(result['matches'])==read(root/'evaluation'/f'{method}_{name}_matches.json')
                domains[name]=clean_json(result['averages'])
                if name=='full_window':
                    for p in result['matches']['window']['prediction']['car']:
                        for gt in p['matchedGt']:
                            iou=gt['intersection']/(p['vert_count']+gt['vert_count']-gt['intersection'])
                            table.append(dict(method=method,gt_id=gt['instance_id'],iou=float(iou),
                                confidence=p['confidence'],threshold_passes=(iou>result['thresholds']).tolist()))
            details[method]=dict(path=str(folder),geometry_sha256=sha(folder/'geometry.npz'),
                point_count=len(g['xyz']),foreground_points=int((labels>0).sum()),
                target_objects=len(preds),reference_target_points=int((projected>0).sum()),
                changed_reference_points_vs_history2=int(np.count_nonzero(canon!=canonical['history2'])),
                projection_pointwise_reproduced=True,matching_records_reproduced=True,scores=domains)
            if method=='history2':
                controls['empty_predictions_ap50']=clean_json(evaluate(truth,weights,np.zeros_like(projected),{},META/'evalInstanceLevelSemanticLabeling.py')['averages']['allAp50%'])
                dropped=next(iter(preds)); p2={k:v for k,v in preds.items() if k!=dropped}
                controls['drop_one_car_object_id']=dropped
                controls['drop_one_car_ap50']=clean_json(evaluate(truth,weights,np.where(projected==dropped,0,projected),p2,META/'evalInstanceLevelSemanticLabeling.py')['averages']['allAp50%'])
                assert controls['empty_predictions_ap50']==0
                assert controls['drop_one_car_ap50']<domains['full_window']['allAp50%']
            print(scene,method,'projection and scores verified',flush=True)
        comparisons={}
        bg=np.load(root/'base/maps/cg/history2/geometry.npz')
        vr=read(root/'vggt_single/maps/cg/history2/classification.json')['objects']
        vg=np.load(root/'vggt_single/maps/cg/history2/geometry.npz')
        for r in vr:
            if not r['predictions']['mean']['retained']: continue
            bid=base_keys[tuple(sorted(r['observation_ids']))]
            a=bg['xyz'][bg['instance']==bid]; b=vg['xyz'][vg['instance']==r['object_id']]
            a=a[np.lexsort(a.T[::-1])]; b=b[np.lexsort(b.T[::-1])]
            np.testing.assert_array_equal(a,b)
            comparisons[str(bid)]=dict(vggt_id=r['object_id'],points=len(a),xyz_exactly_equal=True,observations=r['observation_ids'])
        nodes={a:read(root/a/'maps/cg/nodes.json') for a in ['base','vggt_single']}
        partition={a:{tuple(sorted(n['members'])) for n in ns} for a,ns in nodes.items()}
        ev={a:evidence_audit(root,a) for a in ['base','vggt_single']}
        report['scenes'][scene]=dict(methods=details,scorer_controls=controls,car_ious=table,
            target_geometry_comparison=comparisons,evidence=ev,
            native_nodes={a:len(ns) for a,ns in nodes.items()},
            native_membership_groups_symmetric_difference=len(partition['base']^partition['vggt_single']),
            final_objects={'base':len(base_rows),'vggt_single':len(vr)})
        dump(DEST/'audit.json',clean_json(report))
        print(scene,'visibility funnel',ev['base']['totals'],flush=True)
    report['complete']=True; dump(DEST/'audit.json',clean_json(report))


if __name__=='__main__':main()
