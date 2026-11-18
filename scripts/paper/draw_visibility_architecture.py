"""Conference-style, data-backed method diagram. No generated experiment assets."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Ellipse, Rectangle
from PIL import Image
from scipy.spatial import cKDTree

INK='#23344a'; BLUE='#4d80bb'; TEAL='#008f80'; CORAL='#dc5670'; GOLD='#dfaa56'; MUTED='#758495'
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/benchmarks/visibility_independent_20261112'
OUT=ROOT/'docs/paper/visibility_revision/figures'

def build(d,use):
    scene=d['scene'];pid=d['pid'];cache=d['cache']
    g=np.load(use(BASE/'maps'/scene/'cg/history2/geometry.npz'));xyz=g['xyz'];labels=g['instance']
    retained=np.load(use(BASE/'maps'/scene/'cleanup/visibility1/retained_source_indices.npy'))
    keep=np.zeros(len(xyz),bool);keep[retained]=True
    e=np.load(use(BASE/'maps'/scene/'cleanup/evidence.npz'))
    fg=[o['object_id'] for o in d['objs'] if o['predictions']['mean']['retained']];isfg=np.isin(labels,fg)
    obj=np.flatnonzero(labels==pid);removed=obj[~keep[obj]]
    votes=np.flatnonzero(labels[e['source_indices']]==pid);P=e['positive'][votes].astype(int);N=e['negative'][votes].astype(int)
    assert len(removed)==int(((N>=1)&(N>=2*P)).sum())
    pose=np.array(d['frame']['pose']);basis=pose[:3,:3].copy()
    # Tilt the real viewing basis upward for an orthographic 3D inspection.
    theta=np.deg2rad(22);rot=np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    display=basis@rot
    fgxyz=xyz[isfg];center=fgxyz.mean(0)
    def project(q):
        a=(q-center)@display;return np.c_[a[:,0],-a[:,1],a[:,2]]
    uv=project(xyz);limits=np.array([uv[isfg,:2].min(0),uv[isfg,:2].max(0)])
    # Explicitly sampled context, foreground complete; no geometrical completion.
    context=np.flatnonzero(~isfg)[::8];fgid=np.flatnonzero(isfg)
    colors=np.asarray(plt.get_cmap('turbo')((labels*.61803398875)%1))[:,:3]
    fig=plt.figure(figsize=(12.8,7.2));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,16),ylim=(0,9));ax.axis('off')
    def text(x,y,s,size=12,c=INK,bold=False,**kw):ax.text(x,y,s,fontsize=size,color=c,weight='bold' if bold else 'normal',**kw)
    def line(a,b,c=INK,lw=1.4,style='-'):ax.plot([a[0],b[0]],[a[1],b[1]],color=c,lw=lw,ls=style,zorder=3)
    def arrow(a,b,c=INK,rad=0):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=1.5,color=c,connectionstyle=f'arc3,rad={rad}',zorder=8))
    def card(x,y,w,h,fc,ec='none',lw=1):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=.12',facecolor=fc,edgecolor=ec,lw=lw,zorder=0))
    def image(x,y,w,h,im,edge=None):
        a=fig.add_axes([x/16,y/9,w/16,h/9]);a.imshow(im,aspect='auto');a.set_xticks([]);a.set_yticks([])
        for spine in a.spines.values():spine.set_visible(edge is not None);spine.set_edgecolor(edge or 'white');spine.set_linewidth(.8)
        return a
    def cloud(x,y,w,h,after=False,object_only=False,ghost=False):
        a=fig.add_axes([x/16,y/9,w/16,h/9]);a.axis('off')
        if object_only:
            ids=obj[keep[obj]] if after else obj
            lim=np.array([uv[obj,:2].min(0),uv[obj,:2].max(0)]);col=np.tile(np.array([.05,.57,.49]),(len(ids),1))
            # Gentle depth cue preserves actual point positions.
            z=uv[ids,2];shade=.62+.38*(z.max()-z)/max(np.ptp(z),1e-6);col=col*shade[:,None]+(1-shade[:,None])*.16
            draw=np.argsort(-uv[ids,2]);a.scatter(uv[ids[draw],0],uv[ids[draw],1],s=5,c=col[draw],lw=0,rasterized=True)
            if ghost:a.scatter(uv[removed,0],uv[removed,1],s=9,c=CORAL,lw=0,rasterized=True,alpha=.85)
        else:
            ids=np.r_[context,fgid[keep[fgid]] if after else fgid];ids=ids[np.argsort(-uv[ids,2])]
            col=colors[ids].copy();col[~isfg[ids]]=[.78,.81,.84]
            a.scatter(uv[ids,0],uv[ids,1],s=.55,c=col,lw=0,rasterized=True)
            lim=limits
        span=lim[1]-lim[0];pad=np.maximum(span*.06,.025);a.set_xlim(lim[0,0]-pad[0],lim[1,0]+pad[0]);a.set_ylim(lim[0,1]-pad[1],lim[1,1]+pad[1]);a.set_aspect('equal')
        return a
    # A: four-stage pipeline with real multiview input and spatial map renders.
    text(.22,8.66,'(a)  Object mapping with fixed identities',15,bold=True)
    card(.20,4.82,4.12,3.48,'#eef3fb');card(4.64,4.82,3.22,3.48,'#f4f2ed')
    card(8.20,4.82,3.48,3.48,'#e5f4ef',TEAL,1.7);card(12.03,4.82,3.75,3.48,'#f1f5f9')
    text(.39,8.02,'Multi-view observations',13,bold=True)
    # Three actual owned-observation frames; no duplicated artificial views.
    manifest=json.loads(use(cache/'manifest.json').read_text());job=json.loads(use(BASE/'jobs'/(scene+'.json')).read_text());owner=next(o for o in d['objs'] if o['object_id']==pid)
    obs=set(owner['observation_ids']);recs=[r for r in manifest['records'] if any(r['observation_offset']+i in obs for i in range(r['masks']))]
    chosen=[recs[0],recs[len(recs)//2],next(r for r in recs if r['frame_id']==d['frame']['frame_id'])];byframe={r['frame_id']:r for r in job['frames']}
    for j,rec in enumerate(chosen):
        im=np.array(Image.open(use(byframe[rec['frame_id']]['rgb'])).convert('RGB'));image(.45+j*.14,6.00-j*.12,1.78,1.34,im,'white')
    text(.47,5.42,'RGB-D + poses',11,c=BLUE)
    image(2.70,6.02,1.35,1.02,d['overlay'],'white');text(2.61,5.72,'SAM2.1',11,bold=True)
    # Feature token stack: actual cached feature excerpt, projected as slabs.
    rec=chosen[-1];f=np.load(use(cache/'frames'/rec['file']));feature=f['features'][:6,:24]
    for j in range(3):
        ax.add_patch(Polygon([[2.72+j*.065,5.00+j*.07],[3.65+j*.065,5.00+j*.07],[3.85+j*.065,5.21+j*.07],[2.92+j*.065,5.21+j*.07]],closed=True,facecolor=['#b7c8e8','#d5e1f2','#edf2fa'][j],edgecolor=BLUE,lw=.65,zorder=3))
    image(2.85,5.15,.96,.24,feature)
    text(3.10,4.87,'SigLIP',10,c=BLUE)
    arrow((2.38,6.48),(2.64,6.48),BLUE)
    text(4.82,8.02,'Associated object map',13,bold=True)
    cloud(4.74,5.57,3.01,2.15)
    text(4.91,5.25,'CG + history2',12,bold=True);text(4.91,4.96,'IDs  /  masks  /  geometry',10,c=MUTED)
    arrow((4.34,6.65),(4.61,6.65));arrow((7.87,6.65),(8.18,6.65),TEAL)
    text(8.43,8.02,'Visibility revision',13,c=TEAL,bold=True)
    # A visual point bank: source geometry and evidence are the content of Ours.
    cloud(8.63,5.75,2.55,1.8,object_only=True,ghost=True)
    card(9.07,5.32,1.72,.35,TEAL);text(9.93,5.49,'OURS',11,c='white',bold=True,ha='center',va='center')
    text(9.94,4.99,'Support  +  refutation',11,c=TEAL,ha='center')
    arrow((11.70,6.65),(12.01,6.65),TEAL)
    text(12.20,8.02,'Revised object map',13,bold=True)
    cloud(12.12,5.57,3.52,2.15,after=True)
    text(12.31,5.25,'Preserve identity',12,c=TEAL,bold=True);text(12.31,4.96,'Keep a subset of source points',10,c=MUTED)
    # Expanded method band: diagram and equations rather than a text table.
    ax.add_patch(Polygon([[8.63,4.78],[11.21,4.78],[15.78,4.35],[.2,4.35]],facecolor='#e5f4ef',alpha=.55,edgecolor='none',zorder=-1))
    line((.2,4.35),(15.78,4.35),TEAL,1.5)
    for x in [5.35,10.55]:line((x,.53),(x,4.13),'#cddbdc',.8,'--')
    text(.33,4.00,'(b)  Visibility-gated evidence',13,bold=True)
    text(5.58,4.00,'(c)  Accumulate & decide',13,bold=True)
    text(10.78,4.00,'(d)  Geometry revision',13,bold=True)
    # Oblique image plane + camera frustum (explicit mechanism schematic).
    plane=np.array([[1.55,1.67],[3.20,1.90],[3.20,3.38],[1.55,3.15]])
    ax.add_patch(Polygon(plane,closed=True,facecolor='#dce8f5',edgecolor=BLUE,lw=1,alpha=.8))
    ax.add_patch(Polygon([[1.63,2.10],[2.30,2.19],[2.30,3.03],[1.63,2.94]],closed=True,facecolor='#70c4b1',edgecolor=TEAL,lw=.8))
    ax.add_patch(Polygon([[2.47,2.00],[3.11,2.09],[3.11,2.75],[2.47,2.66]],closed=True,facecolor='#ec9daf',edgecolor=CORAL,lw=.8))
    camera=(.65,2.18)
    for p in plane:line(camera,p,'#9aafc2',.8)
    ax.add_patch(Polygon([[.42,2.07],[.70,2.07],[.70,2.30],[.42,2.30]],facecolor=INK,edgecolor='none'))
    targets=[((2.02,2.72),TEAL),((2.78,2.40),CORAL),((4.47,2.81),MUTED)]
    for p,c in targets:line(camera,p,c,1.3,'--' if c==MUTED else '-');ax.scatter(*p,s=40,c=c,zorder=8,edgecolor='white',linewidth=.8)
    text(1.57,3.55,'Measured surface',10,c=BLUE)
    text(3.57,3.13,'Occluded',10,c=MUTED);text(.45,1.57,'Camera',10,c=MUTED)
    text(3.43,2.37,'Abstain',11,c=MUTED)
    text(.45,1.12,r'$|z-D|\leq 0.05+0.01D$',13)
    text(.45,.72,'Own mask',10,c=TEAL);text(1.94,.72,'Other mask',10,c=CORAL);text(3.56,.72,'No vote',10,c=MUTED)
    # All actual point-wise vote pairs. The shaded rejection half-plane is exact.
    a=fig.add_axes([5.87/16,1.35/9,3.85/16,2.20/9]);mx=max(P.max(),1);my=max(N.max(),1)
    xmax=mx*1.08;ymax=my*1.08
    xx=np.linspace(0,min(xmax,ymax/2),200);a.fill_between(xx,np.maximum(1,2*xx),ymax,color='#f9e5e9',zorder=0)
    a.plot(xx,np.maximum(1,2*xx),color=CORAL,lw=1.2);reject=(N>=1)&(N>=2*P)
    a.scatter(P[~reject],N[~reject],s=13,c=TEAL,alpha=.55,lw=0);a.scatter(P[reject],N[reject],s=17,c=CORAL,alpha=.8,lw=0)
    a.set_xlim(-.4,xmax);a.set_ylim(-.4,ymax);a.spines[['top','right']].set_visible(False);a.spines[['left','bottom']].set_color('#a4b5bd');a.tick_params(labelsize=8,length=2,color='#a4b5bd')
    a.set_xlabel('Support P',fontsize=10,color=TEAL,labelpad=1);a.set_ylabel('Refutation N',fontsize=10,color=CORAL,labelpad=2)
    a.text(.04,.89,'REMOVE',transform=a.transAxes,color=CORAL,fontsize=9,weight='bold');a.text(.69,.15,'KEEP',transform=a.transAxes,color=TEAL,fontsize=9,weight='bold')
    text(5.85,.73,r'$N\geq 1\quad\wedge\quad N\geq 2P$',13,c=TEAL,bold=True)
    text(10.84,3.49,'Source object',11,c=INK);text(13.64,3.49,'Revised',11,c=TEAL,bold=True)
    cloud(10.77,1.48,2.13,1.83,object_only=True,ghost=True);cloud(13.39,1.48,2.13,1.83,after=True,object_only=True)
    arrow((12.93,2.43),(13.34,2.43),TEAL)
    text(11.83,1.18,f'{len(obj):,} points',11,c=INK,ha='center');text(14.48,1.18,f'{keep[obj].sum():,} points',11,c=TEAL,bold=True,ha='center')
    text(10.91,.73,f'{len(removed)} removed',11,c=CORAL);text(13.50,.73,'Same object ID',11,c=TEAL)
    text(.25,.18,'SceneNN '+scene+' / predicted object '+str(pid)+'   •   Real cached inputs and geometry; frustum / feature-stack layouts are schematic.   Observation ownership and class stay fixed.',9,c=MUTED)
    for ext in ['png','svg']:fig.savefig(OUT/('method_architecture.'+ext),dpi=280,facecolor='white')
    plt.close(fig)
    return dict(architecture='Conference-style revision; actual SceneNN imagery and geometry; frustum schematic; exact point-vote decision scatter',scene=scene,object_id=pid,input_frames=[r['frame_id'] for r in chosen],source_points=len(obj),retained_points=int(keep[obj].sum()),removed_points=len(removed),votes_plotted=len(votes),display_transform=display.tolist(),display_context_stride=8,display_foreground_complete=True,no_geometry_completion=True,style_references=['https://3d-aigc.github.io/OpenGaussian/','https://buaavrcg.github.io/LEGaussians/'])
