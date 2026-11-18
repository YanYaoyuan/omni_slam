"""Uncropped car support geometry before/after the live sparse adaptation."""
import json
from itertools import product,combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from audit_evidence_retention import OUT,REPORT


def main():
    root=OUT/'livo_strong_v1';panels=[]
    for name,run,title in [('livo','cg','Unadapted sparse supports'),('livo_supportcc','cg_supportcc','Connected measured supports')]:
        summary=json.loads((REPORT/'stage3'/name/'summary.json').read_text());car=next(r for r in summary['retained_objects'] if r['name']=='car')
        g=np.load(root/run/'geometry.npz');points=g['xyz'][g['instance']==car['evaluation_id']]
        panels.append((title,car,points))
    all_points=np.concatenate([p[2] for p in panels]);low=all_points.min(0)-.3;high=all_points.max(0)+.3
    fig=plt.figure(figsize=(12,9),layout='constrained');grid=fig.add_gridspec(2,2,height_ratios=[1,1.5])
    for col,(title,car,points) in enumerate(panels):
        ax=fig.add_subplot(grid[0,col]);ax.imshow(Image.open(car['evidence_image']));ax.axis('off');ax.set_title(title,fontweight='bold')
        ax=fig.add_subplot(grid[1,col],projection='3d');ax.scatter(*points.T,s=8,c='#dc5a32' if col==0 else '#177cb0',alpha=.85)
        bounds=np.array(list(product(*zip(points.min(0),points.max(0)))))
        for a,b in combinations(bounds,2):
            if np.count_nonzero(a!=b)==1:ax.plot(*np.stack([a,b]).T,c='#dc5a32' if col==0 else '#177cb0',lw=1.3)
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
        ax.set_box_aspect(high-low);ax.view_init(25,-45)
        extent=points.max(0)-points.min(0)
        ax.set_title(f'{len(points)} points | observed AABB: '+ ' x '.join(f'{x:.2f}' for x in extent)+' m',fontsize=10)
    fig.suptitle('Same LIVO packets, masks and descriptors; change only measured support validation\nObserved surfaces, not full vehicle dimensions or ground-truth accuracy',fontsize=12)
    path=REPORT/'stage3/livo_sparse_comparison.png';fig.savefig(path,dpi=160);plt.close(fig);print(path)


if __name__=='__main__':main()
