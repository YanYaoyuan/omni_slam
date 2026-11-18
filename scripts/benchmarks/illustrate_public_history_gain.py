"""Post-hoc sole newly matched GT case for the primary public graph rule."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni-public-history-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from pathlib import Path

from audit_evidence_retention import REPORT,dump
from run_public_history_graph import DEST,read
from omni_semantic_map.flat_benchmark import sha256


def main():
    key='scenenn_021'; gt_id=286
    entry=next(s for s in read(DEST/'protocol.json')['scenes'] if s['key']==key)
    dest=REPORT/'public_history_v1';assert dest.exists()
    ref=np.load(entry['reference']);xyz,truth,valid=ref['xyz'],ref['instance'],ref['valid']
    run=DEST/'runs'/key;ev=DEST/'evaluation'/key
    variants=['baseline','native_bridge_2'];predictions=[];details=[]
    domain=(truth==gt_id)&valid
    for name in variants:
        audit=read(ev/(name+'_audit.json'))
        relevant=[p for p in audit['predicted'] if p['dominant_gt']==gt_id]
        ids=[p['predicted_id'] for p in relevant]
        rows=read(run/name/'classification.json')['objects']
        selected=[r for r in rows if r['object_id'] in ids]
        pred=np.load(ev/(name+'.npz'))['instance'];predictions.append(pred)
        domain|=np.isin(pred,ids)&valid
        details.append(dict(variant=name,objects=[dict(alias=min(r['observation_ids'])+1,
            evaluation_id=r['object_id'],name=r['predictions']['mean']['name'],members=len(r['observation_ids'])) for r in selected],
            best_iou=max(p['best_iou'] for p in relevant)))
    points=xyz[domain];center=points.mean(0);_,_,basis=np.linalg.svd(points-center,full_matrices=False)
    uv=(points-center)@basis[:2].T
    palette={21:'#2877a0',320:'#d18a3b',345:'#9863a6',159:'#9863a6',188:'#4f9b82'}
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    gray='#d5dade'
    colors=np.where(truth[domain]==gt_id,'#263e4a',gray)
    axes[0].scatter(uv[:,0],uv[:,1],c=colors,s=5,alpha=.7,rasterized=True)
    axes[0].set_title('Reference instance 286\nFixed valid observed domain',fontsize=11)
    for ax,pred,detail in zip(axes[1:],predictions,details):
        colors=np.full(len(points),gray,dtype='U7')
        for obj in detail['objects']:
            colors[pred[domain]==obj['evaluation_id']]=palette[obj['alias']]
        ax.scatter(uv[:,0],uv[:,1],c=colors,s=5,alpha=.7,rasterized=True)
        ax.set_title(f'{detail["variant"]}\nBest instance IoU = {detail["best_iou"]:.3f}',fontsize=11)
    for ax in axes:
        ax.set_aspect('equal');ax.set_xlabel('Display coordinate 1 (m)');ax.set_ylabel('Display coordinate 2 (m)');ax.grid(alpha=.1)
    fig.legend(handles=[Patch(color=palette[x],label=label) for x,label in [(21,'History 21'),(320,'History 320'),(345,'Other bed fragment'),(188,'History 188')]],loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('A fragmented instance gains a match after history union',fontsize=14,fontweight='bold')
    for ext in ['png','svg']:fig.savefig(dest/f'scenenn021_gain.{ext}',dpi=180)
    plt.close(fig)
    dump(dest/'scenenn021_gain.json',dict(selection='Sole GT instance newly matched by primary two-frame rule across all 11 scenes; post-hoc explanation, no parameter selection.',
        scene=key,gt_id=gt_id,variants=details,
        caveat='Class-agnostic PQ only: merged alias21 predicted class changes from bed to pillow. Other fragments remain. This does not prove semantic classification improvement.',
        images_sha256={p.name:sha256(p) for p in dest.glob('scenenn021_gain.*')}))
    print(details)


if __name__=='__main__':main()
