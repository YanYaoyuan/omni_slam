"""Paper visuals from frozen RGB, masks and point clouds; no model execution.

All rendered experiment panels use measured/cached data. Schematic elements in
method_architecture are explicitly labeled. Display sampling never changes maps.
"""
from pathlib import Path
from itertools import product
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
from scipy.spatial import cKDTree
import cv2

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/benchmarks/visibility_independent_20261112'
OUT=ROOT/'docs/paper/visibility_revision/figures'
GAR=ROOT/'docs/reports/garage_current_2026-11-12'
INK='#183247'; GREEN='#008c78'; RED='#d64061'; GRAY='#7c8c9c'
PROVENANCE=[]

def palette(ids):
    return (np.asarray(plt.get_cmap('turbo')((np.asarray(ids)*.61803398875)%1))[...,:3]*255).astype(np.uint8)

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=220,facecolor='white')
    fig.savefig(OUT/(name+'.svg'),facecolor='white')
    plt.close(fig)

def imagepanel(ax,im,title):
    ax.imshow(im);ax.set_axis_off();ax.set_title(title,loc='left',fontsize=12,color=INK,pad=9,fontweight='bold')

def raster(xyz,colors,pose,cam,w=1280,h=960):
    """Nearest-depth point splats in a real camera; no mesh/hole completion."""
    T=np.linalg.inv(np.asarray(pose));p=xyz@T[:3,:3].T+T[:3,3]
    valid=(p[:,2]>.2)&np.isfinite(p).all(1);p=p[valid];c=colors[valid]
    u=np.rint((cam['fx']*p[:,0]/p[:,2]+cam['cx'])*w/cam['W']).astype(int)
    v=np.rint((cam['fy']*p[:,1]/p[:,2]+cam['cy'])*h/cam['H']).astype(int)
    valid=(u>=0)&(u<w)&(v>=0)&(v<h);u=u[valid];v=v[valid];c=c[valid];z=p[valid,2]
    order=np.argsort(z,kind='stable');u=u[order];v=v[order];c=c[order]
    # Each splat uses its true depth ordering. Empty pixels remain background.
    pix=[];col=[];ranks=[]
    for dy,dx in product([-1,0,1],repeat=2):
        xx=u+dx;yy=v+dy;ok=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        pix.append(yy[ok]*w+xx[ok]);col.append(c[ok]);ranks.append(np.flatnonzero(ok))
    pixels=np.concatenate(pix);cc=np.concatenate(col);rank=np.concatenate(ranks)
    sort=np.argsort(rank,kind='stable');pixels=pixels[sort];cc=cc[sort]
    unique,first=np.unique(pixels,return_index=True)
    out=np.full((h*w,3),244,np.uint8);out[unique]=cc[first]
    return out.reshape(h,w,3)

def selected(scene,pid,use):
    cache=BASE/'frontend'/scene/'cache'
    manifest=json.loads(use(cache/'manifest.json').read_text());job=json.loads(use(BASE/'jobs'/(scene+'.json')).read_text())
    objs=json.loads(use(BASE/'maps'/scene/'cg/history2/classification.json').read_text())['objects']
    obj=next(o for o in objs if o['object_id']==pid);obs=set(obj['observation_ids']);best=None
    for rec in manifest['records']:
        local=[i for i in range(rec['masks']) if rec['observation_offset']+i in obs]
        if not local:continue
        with np.load(use(cache/'frames'/rec['file'])) as f:
            support=sum(int(f['offsets'][i+1]-f['offsets'][i]) for i in local)
        if best is None or support>best[0]:best=(support,rec,local)
    support,rec,local=best
    frame=next(f for f in job['frames'] if f['frame_id']==rec['frame_id'])
    rgb=np.array(Image.open(use(frame['rgb'])).convert('RGB'))
    f=np.load(use(cache/'frames'/rec['file']));masks=np.unpackbits(f['masks_packed'],axis=2)[:,:,:rgb.shape[1]].astype(bool)
    overlay=rgb.copy()
    # Object ownership colors, not semantic ground truth. Unassociated masks gray.
    owner={i:o['object_id'] for o in objs for i in o['observation_ids']}
    for i in np.argsort(-masks.sum((1,2)).astype(float)):
        oid=owner.get(rec['observation_offset']+int(i),0);c=palette(oid) if oid else np.array([170,170,170])
        overlay[masks[i]]=(.55*overlay[masks[i]]+.45*c).astype(np.uint8)
        contours,_=cv2.findContours(masks[i].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,tuple(map(int,c)),1)
    PROVENANCE.append(dict(scene=scene,object_id=pid,frame_id=rec['frame_id'],rgb=frame['rgb'],cache=str(cache/'frames'/rec['file']),selection='Maximum cached support for post-hoc explanatory object',support=support))
    return dict(scene=scene,pid=pid,frame=frame,rgb=rgb,overlay=overlay,cam=manifest['config']['cam'],cache=cache,objs=objs,masks=masks,local=local)

def scene_figure(scene,pid,use):
    d=selected(scene,pid,use);raw=np.load(use(d['cache']/'geometry.npz'))
    g=np.load(use(BASE/'maps'/scene/'cleanup/visibility1/geometry.npz'))
    fg=[o['object_id'] for o in d['objs'] if o['predictions']['mean']['retained']]
    good=np.isin(g['instance'],fg);tree=cKDTree(g['xyz'][good]);dist,idx=tree.query(raw['xyz'],distance_upper_bound=.03,workers=1)
    rgbmap=raster(raw['xyz'],raw['colors'],d['frame']['pose'],d['cam'])
    # Dense context carries display labels by nearest source only, NOT evaluator.
    sem=np.tile([190,198,205],(len(raw['xyz']),1)).astype(np.uint8);ok=np.isfinite(dist)
    sem[ok]=palette(g['instance'][good][idx[ok]])
    semantic=raster(raw['xyz'],sem,d['frame']['pose'],d['cam'])
    d['rgbmap']=rgbmap;d['semantic']=semantic
    fig,axes=plt.subplots(2,2,figsize=(12,9.8));fig.subplots_adjust(left=.02,right=.98,top=.91,bottom=.07,wspace=.035,hspace=.13)
    fig.text(.02,.975,f'SceneNN {scene}  |  from observation to object map',fontsize=19,weight='bold',color=INK,va='top')
    fid=d['frame']['frame_id']
    for ax,im,title in zip(axes.ravel(),[d['rgb'],d['overlay'],rgbmap,semantic],[f'(a) Recorded RGB / frame {fid}','(b) Cached SAM2.1 masks','(c) Accumulated RGB point cloud','(d) Ours / object IDs on dense context']):imagepanel(ax,im,title)
    fig.text(.02,.03,'Same calibrated camera in all panels. Instance colors, not category labels. Gray = unlabeled context.',fontsize=10,color=GRAY)
    save(fig,'dataset_scene_'+scene)
    PROVENANCE.append(dict(scene=scene,render='Accumulated cloud projected to the selected real camera; nearest-depth 3x3 splats; no hole completion',semantic_display='Nearest retained foreground point within 3 cm colors dense context for display only; not evaluation',full_point_count=len(raw['xyz'])))
    return d

def garage_figure(use):
    run=json.loads(use(ROOT/'build/experiments/garage_current_20261112/run.json').read_text())
    import hashlib
    for p,h in run['output_sha256'].items():assert hashlib.sha256(use(ROOT/'build/experiments/garage_current_20261112'/p).read_bytes()).hexdigest()==h,p
    objs=json.loads(use(GAR/'objects.json').read_text());job=json.loads(use(ROOT/'build/benchmarks/evidence_retention_20261111/jobs/livo_full_v2.json').read_text())
    raw=np.load(use(ROOT/'build/benchmarks/evidence_retention_20261111/livo_full_strong_v2/cache_supportcc/geometry.npz'))
    xyz=raw['xyz'][::8];traj=np.array([np.asarray(f['pose'])[:3,3] for f in job['frames']])
    # Show two spatially separated vehicles with the most supported track among
    # the requested two positions; no GT assertion of unique physical identity.
    examples=[next(o for o in objs if o['id']==i) for i in ['6','2584']]
    fig=plt.figure(figsize=(12,11.8));fig.text(.025,.975,'rosbag2245  |  garage object map & visual evidence',fontsize=19,weight='bold',color=INK,va='top')
    fig.text(.025,.94,'Archived LIVO + SAM2.1 / SigLIP + history2 branch; visibility1 is not applied here',fontsize=11,color=GRAY)
    ax=fig.add_axes([.035,.475,.93,.43]);ax.set_facecolor('#132635')
    ax.scatter(xyz[:,0],xyz[:,1],s=.25,c='#607484',alpha=.65,rasterized=True)
    ax.plot(traj[:,0],traj[:,1],color='#ffce70',lw=1.1,label='LIVO trajectory')
    for o in objs:
        p=np.array(o['points']);ax.scatter(p[:,0],p[:,1],s=2.2,color=o['color'],rasterized=True)
        lo,hi=np.array(o['box']['min']),np.array(o['box']['max'])
        ax.plot([lo[0],hi[0],hi[0],lo[0],lo[0]],[lo[1],lo[1],hi[1],hi[1],lo[1]],color=o['color'],lw=.6)
    for letter,o,offset in [('A',examples[0],(-25,20)),('B',examples[1],(15,-25))]:
        ax.annotate(letter+' / ID '+o['id'],o['box']['center'][:2],xytext=offset,textcoords='offset points',fontsize=11,color='white',weight='bold',bbox=dict(facecolor='#183247',edgecolor=o['color'],pad=4),arrowprops=dict(arrowstyle='->',color='white'))
    ax.set_xlim(-90,95);ax.set_ylim(-27,43);ax.set_aspect('equal');ax.tick_params(labelsize=9);ax.set_xlabel('X (m)',fontsize=10);ax.set_ylabel('Y (m)',fontsize=10)
    ax.legend(loc='upper left',fontsize=9,facecolor='#132635',labelcolor='white');ax.set_title('Garage overview: instance surfaces + observed-surface bounds',loc='left',fontsize=12,weight='bold',color=INK)
    # For each example show original image, archived mask+measured points, and
    # the actual accumulated 3D surface plus AABB; same instance palette.
    for row,(letter,o) in enumerate(zip(['A','B'],examples)):
        y=.267-row*.205;src=o['evidence_sources'][1]
        rgb=np.array(Image.open(use(src['rgb'])).convert('RGB'))
        assert hashlib.sha256(Path(src['rgb']).read_bytes()).hexdigest()==src['rgb_sha256']
        strip=np.array(Image.open(use(GAR/o['evidence'])).convert('RGB'));w=strip.shape[1]//3
        overlay=strip[27:,w:2*w] # Remove archived text header, declared crop.
        for col,(im,title) in enumerate([(rgb,f'{letter}  RGB / KF {src["frame_id"]}'),(overlay,'SAM mask + measured LiDAR')]):
            a=fig.add_axes([.035+col*.32,y,.30,.175]);imagepanel(a,im,title)
        a=fig.add_axes([.68,y-.015,.285,.21],projection='3d');p=np.array(o['points']);a.scatter(*p.T,s=3,color=o['color'],depthshade=False)
        lo,hi=np.array(o['box']['min']),np.array(o['box']['max']);corners=np.array(list(product(*zip(lo,hi))))
        for i,c in enumerate(corners):
            for v in corners[i+1:]:
                if np.count_nonzero(c!=v)==1:a.plot(*np.array([c,v]).T,color=GRAY,lw=.6)
        a.set_box_aspect(np.maximum(hi-lo,.2));a.view_init(22,-55);a.set_axis_off();a.set_title(f'ID {o["id"]} / {o["point_count"]} measured points',fontsize=10,color=INK,y=.97)
        PROVENANCE.append(dict(garage_object=o['id'],sources=o['evidence_sources'],archived_mask_overlay=o['evidence'],display_crop='middle panel; remove 27-pixel label strip',coordinates='omni_odom',not_visibility1=True,overview_bounds_m=[[-90,95],[-27,43]],context_display_stride=8))
    fig.text(.035,.025,'330 semantic frames | 40 retained predictions (34 car, 5 door, 1 pipe). Counts are not unique physical objects.\nBoxes enclose observed surfaces; no vehicle shape completion or garage GT accuracy is claimed.',fontsize=10,color=GRAY,linespacing=1.6)
    save(fig,'garage_real_evidence')
    return examples

def architecture(d,use):
    from draw_visibility_architecture import build as draw_architecture
    use(ROOT/'scripts/paper/draw_visibility_architecture.py')
    example=selected('057',29,use)
    PROVENANCE.append(draw_architecture(example,use))


def object_comparison(use):
    records=json.loads(use(ROOT/'docs/reports/visibility_mechanism_review_2026-11-12/per_object.json').read_text())
    fig=plt.figure(figsize=(14,9.6));fig.text(.025,.972,'What changed?  |  object-level comparison',fontsize=20,weight='bold',color=INK,va='top')
    fig.text(.025,.925,'Same reference voxels, object IDs and display coordinates within each row',fontsize=11,color=GRAY)
    for row,(scene,gid,pid,label) in enumerate([('057',59005,29,'Improvement'),('057',136313,7,'Remaining failure')]):
        record=next(x for x in records if x['scene']==scene and x['gt_id']==gid)
        d=selected(scene,pid,use);reference=np.load(use(BASE/'evaluation'/scene/'reference.npz'))
        xyz=reference['xyz'];valid=reference['valid'];gt=(reference['instance']==gid)&valid
        predicted=[np.load(use(BASE/'evaluation'/scene/(m+'.npz')))['instance'] for m in ['history2','visibility1','sor']]
        union=gt.copy()
        for labels in predicted:union|=(labels==pid)&valid
        # Common orthographic orientation taken from the explanatory RGB camera.
        R=np.array(d['frame']['pose'])[:3,:3];uv=(xyz-xyz[union].mean(0))@R[:,:2];uv[:,1]*=-1
        lo=uv[union].min(0);hi=uv[union].max(0);margin=np.maximum((hi-lo)*.08,.02)
        top=.85-row*.40;y=top-.29
        fig.text(.025,top+.015,f'{label}  /  SceneNN {scene} · GT {gid} · predicted ID {pid}',fontsize=12,weight='bold',color=GREEN if row==0 else RED)
        ax=fig.add_axes([.025,y,.235,.255]);own=d['masks'][d['local']].any(0);im=d['rgb'].copy();im[own]=(.65*im[own]+.35*np.array([0,165,132])).astype('uint8')
        imagepanel(ax,im,'RGB + owned mask')
        for col,(name,method,labels) in enumerate(zip(['Strong baseline','Ours','SOR'],['history2','visibility1','sor'],predicted)):
            pred=(labels==pid)&valid;correct=pred&gt;excess=pred&~gt;missing=gt&~pred
            ax=fig.add_axes([.285+col*.235,y,.22,.255])
            for mask,color in [(missing,'#b9c2cb'),(correct,GREEN),(excess,RED)]:ax.scatter(uv[mask,0],uv[mask,1],s=8,c=color,edgecolors='none',rasterized=True)
            ax.set_xlim(lo[0]-margin[0],hi[0]+margin[0]);ax.set_ylim(lo[1]-margin[1],hi[1]+margin[1]);ax.set_aspect('equal');ax.axis('off')
            score=record['methods'][method]['iou'];assert abs(correct.sum()/((pred|gt).sum())-score)<1e-12
            ax.set_title(f'{name}  /  IoU {score:.4f}',fontsize=11,weight='bold',color=GREEN if method=='visibility1' else INK,pad=8)
            fig.text(.285+col*.235,y-.026,f'{correct.sum()} correct · {excess.sum()} excess',fontsize=10,color=GRAY)
        delta=record['delta_ours_baseline']*100
        fig.text(.025,y-.065,f'Ours − baseline: {delta:+.2f} IoU points. '+('Refined support improves this existing match.' if row==0 else 'Ours remains below 0.5; SOR gains the threshold match.'),fontsize=11,color=INK)
        PROVENANCE.append(dict(object_comparison=dict(scene=scene,gt_id=gid,predicted_id=pid),display='Same orthographic camera orientation and axis bounds for all methods; all object reference voxels shown; source RGB separately perspective',post_hoc=True))
    fig.text(.025,.035,'Green = correct support   /   Red = excess support   /   Gray = missed GT. Selected cases explain gains and limits;\nthey are not additional test scenes. IoU values come directly from the frozen per-object evaluation.',fontsize=10,color=GRAY,linespacing=1.6)
    save(fig,'object_comparison_real')


def gallery():
    items=[
      ('method_architecture','方法架构：从观测到对象几何修订','参考顶会方法图的分区、图像堆叠与空间表达重画。真实多帧、特征、三维地图、支持/反证分布和删点对照；绿色模块是 Ours，投影体与特征堆叠布局为示意。'),
      ('dataset_scene_057','公开数据集 · SceneNN 057','原始图像、缓存分割、累积彩色点云与 Ours 对象地图，使用同一标定相机视角。'),
      ('dataset_scene_061','公开数据集 · SceneNN 061','椅子、沙发和桌子的真实观测与建图结果。对象颜色表示实例编号，不代表真值类别。'),
      ('object_comparison_real','基线 / Ours / SOR · 同对象对照','同时展示明显改善和仍然落后于 SOR 的案例。绿色正确、红色多余、灰色遗漏；IoU 取自冻结评测。'),
      ('garage_real_evidence','我们的 rosbag2245 · 地下停车场','地图轨迹与车辆位置、原始 RGB、SAM 掩码、实测雷达支撑和三维框。此图属于已有 LIVO + history2 分支，不冒充 visibility1 在车库的新实验。'),
      ('rosbag_routes_1','六包实测 · 车库路线','三个车库记录的真实相机画面、完整点云和轨迹。点云按高度着色；标注时长、估计路程与点数。'),
      ('rosbag_routes_2','六包实测 · 室外路线','另外三个室外记录，包含约 835 米和 838 米路线。六个记录可能经过同一地点，不算六个独立测试场景。'),
      ('rosbag_outdoor_semantics','室外对象地图 · 盆栽与入口','约 835 米路线上的目标记录、图像检测证据与局部三维框。历史 Grounding DINO + 深度关联分支，未运行 visibility1。'),
      ('rosbag_ground_details','地面分割 · 行人与建筑入口','三个实际关键帧的原始图像、二维分割及实测雷达验证。绿色表示地面候选或验证点，行人不作为静态地标。')]
    intro='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>omni_slam · 实验与方法图册</title><style>body{margin:0;background:#f2f5f7;color:#183247;font:17px/1.8 system-ui}main{max-width:1320px;margin:auto;padding:42px 28px}h1{font-size:36px;margin:8px 0}header p{max-width:900px;color:#52677a}nav{display:flex;flex-wrap:wrap;gap:12px;margin:26px 0}a{color:#008b78}nav a,.downloads a{display:inline-block;padding:5px 12px;background:#e3f0ec;border-radius:5px;text-decoration:none}article{background:white;padding:24px;margin:30px 0;border-radius:12px;box-shadow:0 4px 22px #18324708}h2{font-size:24px;margin:0}article p{margin:7px 0 16px;color:#52677a}img{width:100%;height:auto}.downloads{font-size:14px;display:flex;gap:12px}footer{padding:20px 0;color:#52677a;font-size:14px}</style><main><header><div>omni_slam / 2026-11-13</div><h1>把真实实验放回论文</h1><p>本页直接展示已有数据生成的大图。点击图片查看原分辨率，SVG 保留可编辑文字和矢量标注。算法、预测与评测配置保持冻结；此次只改图和论文表达。</p><nav><a href="#method_architecture">方法架构</a><a href="#dataset_scene_057">公开数据集</a><a href="#object_comparison_real">逐对象对照</a><a href="#garage_real_evidence">车库实测</a><a href="#rosbag_routes_1">六包地图</a><a href="#rosbag_outdoor_semantics">室外语义</a><a href="#rosbag_ground_details">地面分割</a><a href="index.html">完整论文</a><a href="../../../output/pdf/omni_slam_visibility_revision_draft.pdf">论文 PDF</a></nav></header>'
    parts=[intro]
    for name,title,caption in items:
        parts.append(f'<article id="{name}"><h2>{title}</h2><p>{caption}</p><a href="figures/{name}.png"><img loading="lazy" src="figures/{name}.png" alt="{title}"></a><div class="downloads"><a href="figures/{name}.png">PNG 原图</a><a href="figures/{name}.svg">SVG 矢量版</a></div></article>')
    parts.append('<footer>图像与点云来源、所选帧、显示投影与版本说明：<a href="figures/visual_provenance.json">来源记录</a>。车库地图可以在<a href="../../reports/garage_current_2026-11-12/interactive_map.html">交互三维页面</a>继续查看。多包图版的逐帧来源见<a href="figures/rosbag_visual_provenance.json">ROSBAG 来源记录</a>；室外地图可在<a href="../../reports/boxes_ground_2026-11-06/interactive_map.html">室外交互三维页面</a>查看。三维框仅包围已观测表面，预测对象数不是车辆真值数。</footer></main></html>')
    (OUT.parent/'visual_gallery.html').write_text(''.join(parts))


def build(use):
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    a=scene_figure('057',7,use);scene_figure('061',10,use)
    garage_figure(use);architecture(a,use);object_comparison(use)
    from build_rosbag_visuals import build as build_rosbag_visuals
    use(ROOT/'scripts/paper/build_rosbag_visuals.py')
    build_rosbag_visuals(use)
    gallery()
    (OUT/'visual_provenance.json').write_text(json.dumps(PROVENANCE,indent=2)+'\n')

if __name__=='__main__':build(lambda p:Path(p))
