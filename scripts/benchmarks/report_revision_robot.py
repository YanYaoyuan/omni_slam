"""Real images, complete robot map controls and every predicted object."""
from pathlib import Path
import html
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import read, dump, sha
from run_revision_robot import OUT, KEY, verify, check_seal
from audit_revision_robot import map_paths
from omni_semantic_map.discovery_measured_input import robot_measured_visibility

DEST = OUT.parents[2] / 'output/revision_robot_v1'
NAMES = dict(baseline='Frozen strong baseline', geometry_history='Geometry proposals',
             single_history='Single-mask explanation', cover_current='Mask-cover / old IDs locked',
             cover_history='Mask-cover / unconditional history', identity_only='Identity evidence only',
             write_only='Write witness only', current_only='Full new points / old IDs locked',
             ours='Ours candidate / authorized revision')


def ply(path, xyz, rgb, ids):
    arr = np.empty(len(xyz), dtype=[('x','<f8'), ('y','<f8'), ('z','<f8'),
                                  ('red','u1'), ('green','u1'), ('blue','u1'), ('object_id','<i4')])
    for i, k in enumerate(['x','y','z']):
        arr[k] = xyz[:, i]
    for i, k in enumerate(['red','green','blue']):
        arr[k] = rgb[:, i]
    arr['object_id'] = ids
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(xyz))+
              '\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red'+
              '\nproperty uchar green\nproperty uchar blue\nproperty int object_id\nend_header\n')
    with path.open('wb') as f:
        f.write(header.encode('ascii')); arr.tofile(f)
    # Independent file-size and full column readback, including RGB.
    with path.open('rb') as f:
        assert f.read(len(header)) == header.encode('ascii')
        copy = np.fromfile(f, dtype=arr.dtype)
    np.testing.assert_array_equal(copy, arr)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p = verify(); audit = check_seal(OUT / 'delivery_audit.json')
    if DEST.exists():
        check_seal(OUT / 'report_audit.json')
    DEST.mkdir(parents=True, exist_ok=True)
    c = p['contexts'][0]; full = np.load(Path(c['full'])/'geometry.npz')
    maps = {m:dict(np.load(path/'geometry.npz')) for m,path in map_paths(p).items()}
    job = read(c['job']); frames = job['frames']; source = full['xyz']
    lo, hi = source[:,:2].min(0), source[:,:2].max(0)
    indices = np.linspace(0,len(source)-1,min(100000,len(source)),dtype=np.int64)
    fig, axes = plt.subplots(3,3,figsize=(16,9),constrained_layout=True)
    for ax,(m,g) in zip(axes.ravel(),maps.items()):
        ax.scatter(source[indices,0],source[indices,1],c='#cbd2d8',s=.1,rasterized=True)
        ii=np.linspace(0,len(g['xyz'])-1,min(80000,len(g['xyz'])),dtype=np.int64)
        ax.scatter(g['xyz'][ii,0],g['xyz'][ii,1],c=g['instance'][ii]%20,cmap='tab20',vmin=0,vmax=19,s=.65,rasterized=True)
        r=audit['results'][m]
        ax.set_title(NAMES[m]+f'\n{r["points"]:,} points / {r["nonempty_ids"]} IDs / {r["old_labels_changed"]:,} old labels changed',fontsize=9)
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_xlabel('x / m');ax.set_ylabel('y / m')
    fig.suptitle('rosbag2245 / all 330 frames / actual measured maps / shared world bounds\nRepeated identity palette; new IDs are not aligned between proposal variants. No robot GT.')
    fig.savefig(DEST/'all_maps.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,6),constrained_layout=True)
    for ax,m in zip(axes.ravel(),['baseline','cover_history','identity_only','ours']):
        g=maps[m];carids=[o['object_id'] for o in audit['results'][m]['objects'] if o['train_class']==13]
        ii=np.flatnonzero(np.isin(g['instance'],carids))
        ax.scatter(source[indices,0],source[indices,1],c='#d8dce0',s=.1,rasterized=True)
        ax.scatter(g['xyz'][ii,0],g['xyz'][ii,1],c=g['instance'][ii]%20,cmap='tab20',vmin=0,vmax=19,s=.8,rasterized=True)
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal')
        ax.set_title(NAMES[m]+f' / {len(ii):,} predicted car points',fontsize=10)
        ax.set_xlabel('x / m');ax.set_ylabel('y / m')
    fig.suptitle('Vehicle-class readout of the SAME sealed maps / environment in grey / no robot ground truth')
    fig.savefig(DEST/'vehicles.png',dpi=150);plt.close(fig)
    image_files=['all_maps.png','vehicles.png']; projections=[]
    selected=[frames[min(len(frames)-1,int(len(frames)*q))] for q in [.125,.375,.625,.875]]
    for f in selected:
        rgb=np.asarray(Image.open(f['rgb']).convert('RGB'));cloud=np.load(f['cloud'])['points']
        fig,axes=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
        rows={}
        for ax,m in zip(axes.ravel(),['baseline','cover_history','identity_only','ours']):
            g=maps[m];ids,xy,_=robot_measured_visibility(g['xyz'],cloud,f)
            ax.imshow(rgb);ax.scatter(xy[:,0],xy[:,1],c=g['instance'][ids]%20,cmap='tab20',vmin=0,vmax=19,s=2,alpha=.75)
            ax.set_title(NAMES[m]+f' / {len(ids):,} measured-visible map points',fontsize=10);ax.axis('off')
            rows[m]=len(ids)
        fig.suptitle(f'Actual RGB frame {f["frame_id"]} / measured-depth-gated projection, not segmentation masks')
        name=f'frame_{f["frame_id"]:06d}.png';fig.savefig(DEST/name,dpi=140);plt.close(fig)
        image_files.append(name);projections.append(dict(frame_id=f['frame_id'],selection='fixed temporal eighths:1/8,3/8,5/8,7/8',visible_map_points=rows))
    g=maps['ours'];nold=len(maps['baseline']['xyz']);dist,near=cKDTree(source).query(g['xyz'][:nold],workers=1)
    rgb=np.full((len(g['xyz']),3),160,np.uint8);valid=dist<=.05
    rgb[:nold][valid]=full['colors'][near[valid]]
    rgb[nold:]=full['colors'][g['added_full_source_ids']]
    palette=(plt.get_cmap('tab20')(np.arange(20))[:,:3]*255).astype(np.uint8)
    ply(DEST/'ours_rgb.ply',g['xyz'],rgb,g['instance'])
    ply(DEST/'ours_instances.ply',g['xyz'],palette[g['instance']%20],g['instance'])
    np.savez_compressed(DEST/'color_provenance.npz',old_nearest_source=near,old_distance=dist,
                        old_has_color=valid,added_full_source_ids=g['added_full_source_ids'])
    rows=''
    for m,r in audit['results'].items():
        rows+=f'<tr><td>{NAMES[m]}</td><td>{r["points"]:,}</td><td>{r["class_points"]["13"]:,}</td><td>{r["class_points"]["2"]:,}</td><td>{r["added_measured_sources"]:,}</td><td>{r["nonempty_ids"]}</td><td>{r["old_labels_changed"]:,}</td></tr>'
    diff=''.join(f'<tr><td>{NAMES[r["a"]]} / {NAMES[r["b"]]}</td><td>{r["added_source_label_differences"]:,}</td><td>{r["old_label_differences"]:,}</td></tr>' for r in audit['differences'])
    tables=''
    for m,r in audit['results'].items():
        body=''.join(f'<tr><td>{o["object_id"]}</td><td>{o["old_points"]}</td><td>{o["points"]}</td><td>{o["net_points"]:+d}</td><td>{html.escape(str(o["extent_m"]))}</td></tr>' for o in r['objects'])
        tables+=f'<details><summary>{NAMES[m]}：全部对象与旧ID修改</summary><pre>'+html.escape(str(r['transitions']))+'</pre><table><tr><th>ID</th><th>原点数</th><th>现点数</th><th>净变化</th><th>范围 x/y/z（米）</th></tr>'+body+'</table></details>'
    images=''.join(f'<img src="{name}">' for name in image_files)
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam 同版本车库修订对照</title>
<style>body{font:16px/1.7 sans-serif;color:#203748;max-width:1500px;margin:36px auto;padding:0 24px}img{width:100%}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #d3dce2;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}.note{padding:18px;background:#fff1dc}details{margin:18px 0;overflow:auto}pre{white-space:pre-wrap}</style>
<h1>omni_slam：同版本车库对象地图修订</h1>
<p class="note">330帧、1,278,160个存档实测来源点，8份新地图和1份复用强基线。公开开发门槛仍为false；本次按用户目标执行机器人诊断，不升级候选。没有独立机器人GT，不报告AP或精度提升；并非S100在线测试。</p>
<p>沿用公开实验的原始掩码、几何组件、身份竞争与写入核验规则，没有车库特定阈值。复用400个原生查询及19类语义输出；全地图实测源点向330帧回投，使用采集包内外参及当前点云深度门。旧累计前缀可见性逐帧精确重放。此流程使用最终地图和未来观测，属于离线回放。</p>
<p>必须区分车辆与建筑类别：强基线65,189点均为车辆；Ours的717,816点中，车辆69,776点、建筑648,040点。车辆类别净增4,587点，不能把总点数的增长解释为车辆恢复。新增的巨大建筑ID62跨越约160米，完整性和对象划分均未经独立标注验证。父版本还把8,616个原车辆点改归建筑；Ours保留原类别与ID，但不能由此推断所有新点正确。</p>
<table><tr><th>方法</th><th>地图点数</th><th>车辆点</th><th>建筑点</th><th>新增实测点</th><th>非空对象ID</th><th>旧点改ID</th></tr>'''+rows+'''</table>
<p><a href="ours_rgb.ply">Ours候选真实RGB点云</a> · <a href="ours_instances.ply">Ours候选实例着色点云</a>。两份均是对象子地图；RGB来自存档相机颜色，新点严格按来源ID赋色，旧点在5厘米内寻找来源颜色。实例色循环使用，颜色本身不是跨方法关联证据。</p>'''+images+'''
<h2>全部对照的实际差异</h2><p>比较相同原始新增源点的归属和旧点标签；不同提案中新生ID未对齐，数字标签差异不能直接解释为物理身份差异。</p><table><tr><th>对照</th><th>新增来源标签不同</th><th>旧点标签不同</th></tr>'''+diff+'''</table><h2>每个对象的几何变化</h2>'''+tables+'''
<p>独立核验包括全部地图来源及导出、原始位掩码见证、每个候选身份对逐帧撤除、局部图最近身份距离、实际整帧种子撤除。核验通过只说明实现与数据来源一致，不证明语义正确。</p>
<p>主线仍是持续对象级语义建图。有效的方法优势和真正未参与开发的新增评测仍待完成；这份车库对照不代替二者，也不将点数增加当作论文涨点。</p></html>'''
    (DEST/'index.html').write_text(text)
    dump(OUT/'report_audit.json',dict(complete=True,source_sha256=sha(__file__),
         delivery_audit_sha256=sha(OUT/'delivery_audit.json'),projections=projections,
         rgb_old_points_without_source=int((~valid).sum()),ply_full_column_replays=2,
         files={str(f):sha(f) for f in DEST.iterdir()},goal_complete=False))
    print('ROBOT REVISION REPORT',DEST/'index.html',flush=True)


if __name__ == '__main__':
    main()
