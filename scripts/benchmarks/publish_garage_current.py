"""Publish actual replay output: instance colors, measured boxes and evidence.

All display products retain original odometry coordinates. No ICP, shape
completion, fake detections, or detector reruns are performed here.
"""
from pathlib import Path
from itertools import product
from collections import Counter
import base64
import json
import html
import numpy as np
from run_garage_current import ROOT,OLD,CACHE,OUT
from memory_reseg_common import read,dump,sha

REPORT=ROOT/'docs/reports/garage_current_2026-11-12'


def write_ply(path, xyz, rgb, ids=None):
    dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')]
    if ids is not None:dtype.append(('object_id','<i4'))
    a=np.empty(len(xyz),dtype=dtype)
    for j,k in enumerate(['x','y','z']):a[k]=xyz[:,j]
    for j,k in enumerate(['red','green','blue']):a[k]=rgb[:,j]
    if ids is not None:a['object_id']=ids
    header='ply\nformat binary_little_endian 1.0\ncomment Instance display colors; observed surfaces only\nelement vertex '+str(len(a))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n'
    if ids is not None:header+='property int object_id\n'
    with path.open('wb') as f:
        f.write((header+'end_header\n').encode());a.tofile(f)


def main():
    import cv2
    import open3d as o3d
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plotly.offline import get_plotlyjs
    from omni_semantic_map.shared_observations import observation_table,select_views

    state=read(OUT/'run.json');assert state['complete']
    for name,h in state['output_sha256'].items():assert sha(OUT/name)==h,name
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'evidence').mkdir(exist_ok=True)
    m=read(CACHE/'manifest.json');job=read(OLD/'jobs/livo_full_v2.json')
    capture=read(Path(job['capture']))
    _,frames,areas=observation_table(CACHE)
    byframe={r['frame_id']:r for r in job['frames']}
    byobs={i:r for r in m['records'] for i in range(r['observation_offset'],r['observation_offset']+r['masks'])}
    geometry=np.load(OUT/'current/geometry.npz');raw=np.load(CACHE/'geometry.npz')
    rows=read(OUT/'current/objects.json')['objects'];rows=[r for r in rows if r['retained']]
    ids=[r['object_id'] for r in rows]
    fgmask=np.isin(geometry['instance'],ids);fgxyz=geometry['xyz'][fgmask];fgids=geometry['instance'][fgmask]
    # A deterministic display palette; never interpreted as camera RGB or GT.
    palette={r['object_id']:(np.array(plt.get_cmap('turbo')((i*.61803398875)%1)[:3])*255).astype(np.uint8)
             for i,r in enumerate(rows)}
    fgrgb=np.array([palette[int(i)] for i in fgids],np.uint8)
    cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(raw['xyz'])
    context=np.asarray(cloud.voxel_down_sample(.20).points)
    displayed=context[::max(1,int(np.ceil(len(context)/100000)))]
    trajectory=np.array([np.asarray(f['pose'])[:3,3] for f in job['frames']])
    write_ply(REPORT/'semantic_objects.ply',fgxyz,fgrgb,fgids)
    write_ply(REPORT/'semantic_map_with_context.ply',np.vstack([context,fgxyz]),
              np.vstack([np.tile([135,145,158],(len(context),1)),fgrgb]),np.r_[np.zeros(len(context),np.int32),fgids])
    objects=[];images={}
    for row in rows:
        oid=row['object_id'];alias=row['alias'];p=geometry['xyz'][geometry['instance']==oid]
        selected=np.array(select_views(row['observation_ids'],frames,areas,top_k=0))
        views=[int(selected[np.argmin(frames[selected])]),int(selected[np.argmax(areas[selected])]),int(selected[np.argmax(frames[selected])])]
        tiles=[];provenance=[]
        for label,obs in zip(['Earliest','Largest mask','Latest'],views):
            rec=byobs[obs];frame=byframe[rec['frame_id']];local=obs-rec['observation_offset']
            im=cv2.imread(frame['rgb']);assert im is not None
            with np.load(CACHE/'frames'/rec['file']) as z:
                mask=np.unpackbits(z['masks_packed'][local],axis=1)[:,:im.shape[1]].astype(bool)
                sid=z['point_ids'][z['offsets'][local]:z['offsets'][local+1]]
            overlay=im.copy();color=palette[oid][::-1]
            overlay[mask]=(overlay[mask]*.55+color*.45).astype(np.uint8)
            # Project actual source-frame measured supports, not completed car geometry.
            xyz=raw['xyz'][sid];T=np.linalg.inv(np.asarray(frame['camera_pose']))
            cam=xyz@T[:3,:3].T+T[:3,3];pix=cam@np.asarray(frame['intrinsic']).T
            uv=pix[:,:2]/np.maximum(pix[:,2:],1e-9)
            valid=(cam[:,2]>.2)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<im.shape[1])&(uv[:,1]>=0)&(uv[:,1]<im.shape[0])
            for x,y in np.floor(uv[valid]).astype(int):cv2.circle(overlay,(x,y),1,(80,255,80),-1)
            cv2.rectangle(overlay,(0,0),(im.shape[1],27),(15,22,30),-1)
            cv2.putText(overlay,f'{label} | KF {rec["frame_id"]} | ID {alias} | {row["prediction"]["name"]}',(7,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
            tiles.append(overlay)
            provenance.append(dict(observation_id=obs,frame_id=rec['frame_id'],rgb=frame['rgb'],rgb_sha256=sha(Path(frame['rgb'])),source_support_points=len(sid)))
        path=REPORT/'evidence'/f'object_{alias:06d}.jpg'
        assert cv2.imwrite(str(path),np.concatenate(tiles,axis=1),[cv2.IMWRITE_JPEG_QUALITY,88])
        ok,jpeg=cv2.imencode('.jpg',tiles[1],[cv2.IMWRITE_JPEG_QUALITY,88]);assert ok
        images[str(alias)]='data:image/jpeg;base64,'+base64.b64encode(jpeg.tobytes()).decode()
        box=row['box'];corners=np.array(list(product(*zip(box['min'],box['max'])))).tolist()
        objects.append(dict(id=str(alias),evaluation_id=oid,label=row['prediction']['name'],mean_label=row['mean_prediction']['name'],
            color='#'+''.join(f'{x:02x}' for x in palette[oid]),points=p.round(4).tolist(),point_count=len(p),
            observations=len(row['observation_ids']),unique_frames=row['unique_frames'],
            box=dict(box,corners=corners),evidence=str(path.relative_to(REPORT)),evidence_sources=provenance))
    # Same coordinate domain for before/after. Neither cloud is aligned to GT.
    fig,axes=plt.subplots(1,2,figsize=(17,7),layout='constrained')
    for ax,arm,title in zip(axes,['native_reference','current'],['Sparse CG reference','Current v1: history + category evidence']):
        ax.set_facecolor('#101e2b');ax.scatter(displayed[:,0],displayed[:,1],s=.25,c='#566878',alpha=.5,rasterized=True)
        gg=np.load(OUT/arm/'geometry.npz');rr=read(OUT/arm/'objects.json')['objects']
        for r in rr:
            if not r['retained']:continue
            p=gg['xyz'][gg['instance']==r['object_id']]
            ax.scatter(p[:,0],p[:,1],s=4,c='#4cc8ef',rasterized=True)
        ax.plot(trajectory[:,0],trajectory[:,1],c='#ffcf75',lw=1)
        ax.set_title(title);ax.set_aspect('equal');ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)')
    fig.savefig(REPORT/'comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(15,8),layout='constrained');ax.set_facecolor('#101e2b')
    ax.scatter(displayed[:,0],displayed[:,1],s=.25,c='#566878',alpha=.55,rasterized=True)
    ax.plot(trajectory[:,0],trajectory[:,1],c='#ffcf75',lw=1,label='LIVO trajectory')
    for obj in objects:
        p=np.array(obj['points']);box=obj['box'];lo,hi=box['min'],box['max']
        ax.scatter(p[:,0],p[:,1],s=5,color=obj['color'],rasterized=True)
        ax.plot([lo[0],hi[0],hi[0],lo[0],lo[0]],[lo[1],lo[1],hi[1],hi[1],lo[1]],color=obj['color'],lw=.8)
        ax.annotate(obj['id'],box['center'][:2],xytext=(3,4),textcoords='offset points',color='white',fontsize=6)
    ax.set_aspect('equal');ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)')
    ax.set_title(f'omni_slam current v1 | rosbag2245 | {len(objects)} predicted objects\nInstance colors + observed-surface boxes; raw odometry coordinates')
    fig.savefig(REPORT/'overview.png',dpi=170);plt.close(fig)
    old=state['methods']['native_reference'];new=state['methods']['current']
    summary=dict(complete=True,scene='rosbag2245',source_duration_s=capture['source_duration_s'],semantic_frames=state['frames'],
        method='omni_slam-paper-v1 + frozen LiDAR adaptation',classes=dict(Counter(o['label'] for o in objects)),
        native_reference=old,current=new,retained_objects=len(objects),semantic_points=len(fgxyz),
        backend_causal_s=state['backend_causal_s'],run_sha256=sha(OUT/'run.json'),
        semantic_frame_scope='All existing 330 semantic packets across full-duration recording; not every RGB frame',
        coordinates='omni_odom; no use of the older loop-corrected map as background',
        color_scope='Synthetic instance palette, NOT camera RGB',geometry_scope='Observed LiDAR surfaces, no complete-vehicle shape inference',
        gt_available=False,frontend_recomputed=False,attention_resegmentation_used=False)
    dump(REPORT/'summary.json',summary);dump(REPORT/'objects.json',objects)
    data=dict(context=displayed.round(3).tolist(),trajectory=trajectory.round(4).tolist(),objects=objects,images=images,summary=summary)
    template=(ROOT/'scripts/benchmarks/garage_viewer_template.html').read_text()
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    (REPORT/'interactive_map.html').write_text(template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',encoded))
    report=f'''# 当前 omni_slam 车库语义地图

已重新回放 rosbag2245 全时段的 {state['frames']} 个缓存语义帧。位姿、掩码和特征复用已有 LIVO + SAM2.1 + SigLIP 缓存，本次重新执行对象关联、两查询帧历史合并、最终几何处理和一致性/平方根面积类别读出；没有重新训练或调用注意力重分割。

[打开交互三维地图](interactive_map.html) · [完整对象图册](gallery.html) · [语义对象点云](semantic_objects.ply) · [含灰色环境的语义地图](semantic_map_with_context.ply)

![当前车库语义地图](overview.png)

| 项目 | 原生稀疏底座 | 当前历史图版本 |
|---|---:|---:|
| 保留的前景预测记录 | {old['retained']} | {new['retained']} |
| 前景记录包含的观测数 | {old['foreground_members']} | {new['foreground_members']} |
| 历史图执行的合并边 | {old['joined_edges']} | {new['joined_edges']} |

当前类别：{json.dumps(summary['classes'],ensure_ascii=False)}。这是预测记录数，不是独立车辆真值数量。对象数减少可能来自合并，不能直接解释为漏检或精度变化。

类别读出相对均值的变化：{json.dumps(new['category_changed'],ensure_ascii=False)}。

本次原生底座和两帧历史图的几何、成员及原均值分类均与归档版本逐项一致。因此这是冻结算法的新回放和更完整的可视化，不是本轮又取得算法涨点。

![相同坐标域下的地图对照](comparison.png)

交互图支持旋转、缩放、俯视、按类别过滤、选择对象查看坐标和三个观测证据。彩色表示实例编号，灰色是环境；三维框覆盖已观测表面，不代表完整车辆尺寸。图像覆盖来自原始 SAM 掩码，绿色点为该帧雷达支撑，未做生成式补全。

所有几何均为本次缓存的 omni_odom 坐标，未混入早期回环优化地图。历史合并为离线最终处理，不能把页面交互或缓存回放速度当作机器人在线实时性能。

[运行清单](../../../build/experiments/garage_current_20261112/run.json) · [固定配置](../../../build/experiments/garage_current_20261112/protocol.json) · [对象坐标与证据](objects.json)
'''
    (REPORT/'report.md').write_text(report)
    gallery=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>车库全部对象证据</title><style>body{max-width:1500px;margin:30px auto;padding:20px;font:16px/1.7 system-ui}img{width:100%}article{border-top:1px solid #ccc;padding-top:20px}</style><h1>车库全部预测对象</h1><p>各行依次为最早、最大掩码、最晚观测。颜色是对象显示色，绿色点是该帧的雷达支撑。</p>']
    for o in objects:
        gallery.append(f'<article><h2>{html.escape(o["label"])} · ID {o["id"]} · {o["unique_frames"]} 帧 · {o["point_count"]} 点</h2><a href="interactive_map.html#{o["id"]}">在地图中定位</a><img loading="lazy" src="{o["evidence"]}"></article>')
    (REPORT/'gallery.html').write_text('\n'.join(gallery)+'</html>')
    print('PUBLISHED',REPORT,'objects',len(objects),'classes',summary['classes'],flush=True)


if __name__=='__main__':
    main()
