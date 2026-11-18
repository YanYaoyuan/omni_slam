"""Actual robot maps and observation projections; no unannotated accuracy claim."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni_robot_transport_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_batch_transport_robot import OUT,INPUT,KEY,FULL,verify
from run_batch_transport_assessment import METHODS,seal
from omni_semantic_map.discovery_measured_input import robot_measured_visibility

DEST=ROOT/'output/batch_transport_robot_v1'
NAMES=dict(baseline='Strong baseline',same_cell='Same-cell lift',full_nearest='Nearest anchor',full_two_cells='Two-cell control',ours_frame='Ours / frame certificate')


def main():
    p=verify();seal(OUT/'predictions_complete.json');s=read(OUT/'summary.json');assert s['complete'];DEST.mkdir(parents=True,exist_ok=True)
    source=np.load(FULL/KEY/'geometry.npz')['xyz'];manifest=read(INPUT/KEY/'manifest.json');job=read(p['scenes'][0]['job']);byframe={f['frame_id']:f for f in job['frames']};images=[]
    maps={m:dict(np.load(OUT/'final_maps'/KEY/m/'geometry.npz')) for m in METHODS}
    # Fixed temporal quartiles, chosen independently of improvements or GT.
    records=[manifest['records'][int(q*(len(manifest['records'])-1))] for q in [.25,.5,.75]]
    lo=np.quantile(source[:,:2],.005,axis=0);hi=np.quantile(source[:,:2],.995,axis=0)
    fig,axes=plt.subplots(2,3,figsize=(16,6.5),layout='constrained');axes=axes.ravel();axes[0].imshow(Image.open(records[1]['rgb']));axes[0].axis('off');axes[0].set_title(f'rosbag2245 | actual RGB {records[1]["frame_id"]}')
    for ax,method in zip(axes[1:],METHODS):
        g=maps[method];xyz=g['xyz'];step=max(1,(len(xyz)+79999)//80000);backstep=max(1,(len(source)+99999)//100000)
        ax.scatter(source[::backstep,0],source[::backstep,1],s=.15,color='#cbd2d8',alpha=.45,rasterized=True)
        ax.scatter(xyz[::step,0],xyz[::step,1],c=g['instance'][::step]%20,cmap='tab20',vmin=0,vmax=19,s=.8,rasterized=True)
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.grid(alpha=.12);ax.set_xlabel('x / m');ax.set_ylabel('y / m');ax.set_title(f'{NAMES[method]}\n{len(xyz):,} points / {len(np.unique(g["instance"]))} IDs',color='#008a7d' if method=='ours_frame' else '#20364a')
    f=DEST/'maps.png';fig.savefig(f,dpi=160);plt.close(fig);images.append(f.name)
    projection_records=[]
    for record in records:
        raw=byframe[record['frame_id']];cloud=np.load(raw['cloud'])['points'];rgb=Image.open(record['rgb']);fig,axes=plt.subplots(1,3,figsize=(16,4),layout='constrained')
        row=dict(frame_id=record['frame_id'],selection='fixed temporal quartile',methods={})
        for ax,method in zip(axes,['baseline','full_two_cells','ours_frame']):
            g=maps[method];ids,px,_=robot_measured_visibility(g['xyz'],cloud,raw);ax.imshow(rgb);ax.scatter(px[:,0],px[:,1],s=3,c=g['instance'][ids]%20,cmap='tab20',vmin=0,vmax=19,alpha=.8);ax.set_xlim(0,rgb.width);ax.set_ylim(rgb.height,0);ax.axis('off');ax.set_title(f'{NAMES[method]} | {len(ids):,} visible map points');row['methods'][method]=dict(visible_map_points=len(ids))
        fig.suptitle(f'Actual RGB {record["frame_id"]} + measured-depth-gated map projection (not a segmentation mask)',fontsize=11)
        f=DEST/f'projection_{record["frame_id"]:06d}.png';fig.savefig(f,dpi=160);plt.close(fig);images.append(f.name);projection_records.append(row)
    rows=[]
    for method in METHODS:
        r=s['results'][method];rows.append(f'<tr><td>{NAMES[method]}</td><td>{r["points"]:,}</td><td>{r["added_sources"]:,}</td><td>{r["objects"]}</td></tr>')
    per_object=[];lookup={m:{r['object_id']:r for r in s['results'][m]['per_object']} for m in METHODS}
    for oid in lookup['baseline']:
        per_object.append(f'<tr><td>{oid}</td><td>{lookup["baseline"][oid]["old_points"]}</td>'+''.join(f'<td>{lookup[m][oid]["added_points"]}</td>' for m in METHODS[1:])+'</tr>')
    page=f'''<!doctype html><meta charset="utf-8"><title>omni_slam 车库同版本对照</title>
<style>body{{max-width:1350px;margin:36px auto;padding:0 24px;background:#f5f7f9;color:#20364a;font:16px/1.65 system-ui}}section{{background:white;padding:24px;margin:18px 0;border-radius:12px}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dce4e9;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%;background:white}}a{{color:#008a7d}}</style>
<h1>omni_slam：330帧车库同版本对照</h1><p>复用LIVO位姿与实测点来源、原强基线对象身份。公开评测的同一来源帧认证模块与四组控制，不训练、不改变参数。</p>
<section><h2>完成内容及边界</h2><p>全部330帧共享语义推理、五份最终地图及逐点来源核验已完成。原坐标／ID／分类分数保持，新增点均来自实测来源。公开验收仍未通过；本对照用于实际机器人数据的行为核查。没有独立机器人GT，新增点数、覆盖变化或更完整的视觉外观不代表语义准确率提升。</p><table><tr><th>方法</th><th>总点数</th><th>新增实测点</th><th>对象ID</th></tr>{''.join(rows)}</table><p>共享语义推理阶段 {s['semantics']:.1f} 秒，Ours及控制共同归属场计算 {s['geometry_field']['seconds']:.1f} 秒。工作站离线测量，不是S100实时性能。</p></section>
<section><h2>真实车库图像与完整对象地图</h2><p>灰色为实测环境；彩色为对象ID。所有方法同一视域，坐标显示范围取来源0.5%至99.5%分位，核验覆盖完整输出。</p><img src="maps.png"></section>
<section><h2>投影回真实图像</h2><p>固定时间四分位的三帧；只显示通过当前实测深度检查的地图点。使用采集包内参、相机位姿与原floor像素规则。这些彩点是三维地图投影，不是新分割掩码，也不作为正确性标注。</p>{''.join('<img src="'+f+'">' for f in images[1:])}</section>
<section><h2>全部61个对象的新增点</h2><table><tr><th>ID</th><th>基线点数</th>{''.join('<th>'+NAMES[m]+'</th>' for m in METHODS[1:])}</tr>{''.join(per_object)}</table><p>每个对象的前后包围范围、全部方法差异及核验见<a href="../../build/benchmarks/batch_transport_robot_v1/summary.json">完整结果</a>。</p></section>
<section><h2>关联证据</h2><p><a href="../batch_transport_assessment_v1/index.html">新增公开评测</a> · <a href="../../build/benchmarks/batch_transport_robot_v1/protocol.json">机器人诊断协议</a> · <a href="projection_records.json">投影计数</a></p><p>该实验完成同版本链路验证；独立机制优势和真实语义准确率仍需证据，不能以本页代替论文贡献验收。</p></section>'''
    (DEST/'index.html').write_text(page);dump(DEST/'projection_records.json',projection_records)
    dump(DEST/'manifest.json',dict(complete=True,files={str(f):sha(f) for f in DEST.iterdir() if f.name!='manifest.json'},source_sha256=sha(__file__),semantic_accuracy_available=False))
    print('ROBOT REPORT',DEST/'index.html',flush=True)


if __name__=='__main__':main()
