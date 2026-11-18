"""Development-only attribution of frozen ratio filtering; produces no maps.

Reference labels are used only after evidence extraction, on a fixed original
nearest-source assignment. These weighted counts are diagnostics, not PQ or a
claim that every removed point is an error. No ground truth enters prediction.
"""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.visibility_cleanup import CleanupConfig, project_visibility

PARENT = ROOT/'build/benchmarks/reversible_revision_20261113'
OUT = ROOT/'build/benchmarks/ratio_evidence_audit_20261113'
NAMES = ['support', 'visible', 'known', 'interior_support', 'far_absence',
         'far_known_absence', 'first_half_support', 'last_half_support']


def main(key):
    target=OUT/key
    target.mkdir(parents=True, exist_ok=False)
    protocol=read(PARENT/'protocol.json')
    scene=next(s for s in protocol['scenes'] if s['key']==key)
    cfg=CleanupConfig(**protocol['visibility_config']);cv2.setNumThreads(1)
    for k in ['geometry','classification','reference']:
        assert sha(scene[k])==scene['immutable_hashes'][scene[k]]
    source=np.load(scene['geometry']);xyz,labels=source['xyz'],source['instance']
    rows=read(scene['classification'])['objects']
    kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
    foreground=np.flatnonzero(np.isin(labels,kept))
    owner={o:r['object_id'] for r in rows for o in r['observation_ids']}
    cache=Path(scene['cache']);manifest=read(cache/'manifest.json')
    for name,h in manifest['files_sha256'].items():assert sha(cache/name)==h
    cam=manifest['config']['cam'];poses=np.load(cache/'poses.npy')
    counts=np.zeros((len(NAMES),len(xyz)),np.uint16)
    sectors=np.zeros((2,8,len(xyz)),np.uint16)
    centers={oid:xyz[labels==oid].mean(0) for oid in kept}
    kernel=np.ones((2*cfg.boundary_pixels+1,)*2,np.uint8)
    start=time.monotonic()
    for fi,(rec,dep) in enumerate(zip(manifest['records'],scene['depths'])):
        assert rec['frame_id']==dep['frame_id']
        path=cache/'frames'/rec['file']
        assert sha(path)==rec['sha256'] and sha(dep['path'])==dep['sha256']
        depth=cv2.imread(dep['path'],cv2.IMREAD_UNCHANGED).astype(np.float32)/cam['depth_scale']
        with np.load(path) as z:masks=np.unpackbits(z['masks_packed'],axis=2)[:,:,:cam['W']].astype(bool)
        obs=np.arange(rec['observation_offset'],rec['observation_offset']+len(masks))
        owners=np.array([owner.get(int(o),0) for o in obs])
        x,y,_,valid=project_visibility(xyz[foreground],poses[fi],cam,depth,cfg)
        ids=foreground[valid];px=x[valid];py=y[valid]
        known=masks[:,py,px].any(0)
        for oid in np.intersect1d(owners,kept):
            local=np.flatnonzero(labels[ids]==oid);idx=ids[local]
            if not len(idx):continue
            ux,uy=px[local],py[local]
            union=masks[owners==oid].any(0).astype(np.uint8)
            support=union[uy,ux]>0
            interior=cv2.erode(union,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)[uy,ux]>0
            outside=cv2.dilate(union,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)[uy,ux]==0
            values=[support,np.ones(len(idx),bool),known[local],interior,outside,
                    outside&known[local],support&(fi<len(poses)//2),support&(fi>=len(poses)//2)]
            for j,v in enumerate(values):counts[j,idx]+=v
            direction=poses[fi,:3,3]-centers[oid]
            sector=int(np.floor((np.arctan2(direction[1],direction[0])+np.pi)/(2*np.pi)*8))%8
            sectors[0,sector,idx]+=support;sectors[1,sector,idx]+=~support
        if fi%50==0:print(key,fi,'frames',flush=True)
    reject=(counts[0]<=.5*counts[1])&(counts[1]>0)
    expected=np.ones(len(xyz),bool)
    expected[foreground]=~reject[foreground]
    original_keep=np.load(PARENT/'runs'/key/'simple_ratio/retained_source_indices.npy')
    np.testing.assert_array_equal(np.flatnonzero(expected),original_keep)
    np.savez_compressed(target/'evidence.npz',counts=counts,sectors=sectors,names=np.array(NAMES))

    # Diagnose only after evidence extraction and exact old-result reproduction.
    ref=np.load(scene['reference']);valid=ref['valid'] if 'valid' in ref.files else np.ones(len(ref['xyz']),bool)
    distance,nearest=cKDTree(xyz).query(ref['xyz'][valid],workers=1)
    truth=ref['instance'][valid];inside=distance<=.05
    gt=truth[inside];ni=nearest[inside];pl=labels[ni]
    before=read(PARENT/'evaluation'/key/'baseline_audit.json')
    dominant={o['predicted_id']:o['dominant_gt'] for o in before['predicted']}
    mapped=np.array([dominant.get(int(o)) or -1 for o in pl])
    correct=(gt>0)&(gt==mapped)
    weights=np.vstack([np.bincount(ni[correct],minlength=len(xyz)),
                       np.bincount(ni[~correct],minlength=len(xyz))])
    # Groups are descriptive, overlapping, and are not candidate map outputs.
    negative=counts[1]-counts[0]
    groups={
        'all_removed':reject,
        'no_own_support':reject&(counts[0]==0),
        'one_visible_frame':reject&(counts[1]==1),
        'at_most_two_visible_frames':reject&(counts[1]<=2),
        'absence_mostly_outside_all_masks':reject&((counts[1]-counts[2])>.5*negative),
        'absence_mostly_near_own_boundary':reject&((negative-counts[4])>.5*negative),
        'negative_views_one_azimuth_sector':reject&((sectors[1]>0).sum(0)==1),
        'has_interior_support':reject&(counts[3]>0),
        'support_both_temporal_halves':reject&(counts[6]>0)&(counts[7]>0)}
    fg=np.isin(labels,kept)
    def stats(select):
        select=select&fg
        a,b=weights[:,select].sum(1)
        return dict(source_points=int(select.sum()),dominant_gt_reference_support=int(a),
                    other_reference_support=int(b),dominant_fraction=float(a/max(a+b,1)))
    objects=[]
    for oid in kept:
        select=labels==oid
        objects.append(dict(object_id=oid,dominant_gt=dominant.get(oid),before=stats(select),
                            groups={k:stats(select&v) for k,v in groups.items()}))
    dump(target/'audit.json',dict(scene=key,development_only=True,new_maps=0,
        interpretation='Fixed original nearest-source assignment within 5 cm. Dominant GT is a diagnostic correspondence, not independent point truth; mixed objects and rematching require separate inspection. Groups overlap. Azimuth is a coarse diagnostic, not independent-view proof.',
        evidence_names=NAMES,groups={k:stats(v) for k,v in groups.items()},objects=objects,
        reference_unassigned=int((~inside).sum()),source_sha256=sha(__file__),
        parent_protocol_sha256=sha(PARENT/'protocol.json'),evidence_sha256=sha(target/'evidence.npz'),
        exact_ratio_reproduction=True,elapsed_seconds=time.monotonic()-start))
    print(key,{k:stats(v) for k,v in groups.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('scene',choices=['replica_office_0','scenenn_014'])
    main(parser.parse_args().scene)
