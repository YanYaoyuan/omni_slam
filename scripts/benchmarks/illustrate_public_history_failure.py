"""Post-hoc office_1 ablation case: preserve the one-frame failure explicitly."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni-public-history-mpl')
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from audit_evidence_retention import REPORT,dump
from run_public_history_graph import DEST, read
from omni_semantic_map.flat_benchmark import sha256
from pathlib import Path


def main():
    spec=read(DEST/'protocol.json');entry=next(s for s in spec['scenes'] if s['key']=='replica_office_1')
    dest=REPORT/'public_history_v1';assert dest.exists()
    cache=Path(entry['cache']);manifest=read(cache/'manifest.json')
    lookup={r['observation_offset']+k:(r,k) for r in manifest['records'] for k in range(r['masks'])}
    rgb_root=cache.parents[2]/'OVO_Replica_full/office_1/results'
    rgb_paths=sorted(rgb_root.glob('frame*.jpg'));assert len(rgb_paths)==2000
    panels=[];images=[]
    for oid,title in [(846,'Query: bridge at frame 480'),(18,'Existing history 19'),(35,'Short history 36')]:
        rec,k=lookup[oid];f=np.load(cache/'frames'/rec['file']);path=rgb_paths[rec['frame_id']]
        im=cv2.imread(str(path));im=cv2.resize(im,(1200,680));mask=f['segmentation']==k
        im[mask]=(im[mask]*.55+np.array([0,200,255])*.45).astype(np.uint8)
        im=cv2.resize(im,(600,340))
        cv2.rectangle(im,(0,0),(600,30),(18,25,31),-1)
        cv2.putText(im,title+f' | obs {oid+1}',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
        panels.append(im);images.append(dict(observation_id=oid,frame=rec['frame_id'],rgb=str(path),rgb_sha256=sha256(path)))
    assert cv2.imwrite(str(dest/'office1_failure_rgb.jpg'),np.concatenate(panels,axis=1))
    ref=np.load(entry['reference']);xyz,gt=ref['xyz'],ref['instance']
    run=DEST/'runs'/entry['key'];evaluation=DEST/'evaluation'/entry['key']
    variants=['baseline','native_bridge_1','native_bridge_2'];selected=[];details=[]
    for name in variants:
        rows=read(run/name/'classification.json')['objects']
        obj=next(o for o in rows if 18 in o['observation_ids'])
        pred=np.load(evaluation/(name+'.npz'))['instance'];sel=pred==obj['object_id'];selected.append(sel)
        intersection=int((sel&(gt==2)).sum());union=int((sel|(gt==2)).sum())
        details.append(dict(variant=name,prediction_id=obj['object_id'],member_aliases=[i+1 for i in obj['observation_ids']],
                            gt2_iou=intersection/union,gt_counts={str(int(k)):int((sel&(gt==k)).sum()) for k in np.unique(gt[sel])}))
    domain=np.logical_or.reduce(selected)|(gt==2)
    points=xyz[domain];center=points.mean(0)
    _,_,basis=np.linalg.svd(points-center,full_matrices=False)
    uv=(points-center)@basis[:2].T
    fig,axes=plt.subplots(1,4,figsize=(13,3.8),layout='constrained')
    green,orange,gray='#208376','#ca7842','#c6cdd2'
    for ax,mask,title in zip(axes,[np.ones(len(points),bool)]+[s[domain] for s in selected],
            ['GT on fixed local domain']+[f'{d["variant"]}\nGT 2 IoU = {d["gt2_iou"]:.3f}' for d in details]):
        colors=np.full(len(points),gray,dtype='U7');colors[mask&(gt[domain]==2)]=green;colors[mask&(gt[domain]==6)]=orange
        ax.scatter(uv[:,0],uv[:,1],s=23,c=colors,edgecolors='white',linewidths=.35)
        ax.set_title(title,fontsize=10);ax.set_aspect('equal');ax.set_xlabel('Display coordinate 1 (m)');ax.set_ylabel('Display coordinate 2 (m)')
        ax.grid(alpha=.12)
    fig.legend(handles=[Patch(color=green,label='GT instance 2'),Patch(color=orange,label='GT instance 6'),Patch(color=gray,label='Other / not assigned to this prediction')],loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('A single bridge adds contamination and loses an instance match',fontsize=14,fontweight='bold')
    for ext in ['png','svg']:fig.savefig(dest/f'office1_failure_geometry.{ext}',dpi=180)
    plt.close(fig)
    dump(dest/'office1_failure.json',dict(selection='Post-hoc explanation of office_1: one-frame rule loses GT2; two-frame rule preserves it. Not an independently selected validation case.',
        graph_edge=[19,36],query_frame=480,source_images=images,variants=details,
        display='Common PCA plane for display of the same fixed reference positions; not registration or geometry correction.',
        images_sha256={p.name:sha256(p) for p in dest.glob('office1_failure*') if p.is_file()}))
    print(details)


if __name__=='__main__':main()
