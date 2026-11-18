"""Render archived robot comparison and a predetermined RGB review packet."""
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha

SOURCE=ROOT/'build/experiments/garage_visibility_20261113'
OUT=ROOT/'output/garage_visibility_review'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    run=read(SOURCE/'run.json');protocol=read(SOURCE/'protocol.json')
    assert run['complete']
    for name,h in run['artifacts'].items():assert sha(SOURCE/name)==h,name
    rows=read(SOURCE/'baseline/classification.json')['objects']
    kept=[r['object_id']for r in rows if r['predictions']['mean']['retained']]
    original=np.load(SOURCE/'baseline/geometry.npz')
    fore=np.isin(original['instance'],kept);xyz=original['xyz'][fore]
    lower,upper=xyz[:,:2].min(0),xyz[:,:2].max(0)
    center=(lower+upper)/2;span=max(upper-lower)*.55
    fig,axes=plt.subplots(1,3,figsize=(15,5.3),layout='constrained')
    for ax,name,title in zip(axes,['baseline','sor','ours_visibility1'],['Strong baseline','Baseline + SOR','Ours: visibility1']):
        g=np.load(SOURCE/name/'geometry.npz');good=np.isin(g['instance'],kept)
        pts=g['xyz'][good];ids=g['instance'][good]
        colors=plt.get_cmap('hsv')((ids*.61803398875)%1)
        ax.scatter(pts[:,0],pts[:,1],c=colors,s=1.2,linewidths=0,rasterized=True)
        ax.set(xlim=(center[0]-span,center[0]+span),ylim=(center[1]-span,center[1]+span),
               xlabel='x (m)',ylabel='y (m)',aspect='equal',title=f"{title}\n{len(pts):,} foreground points")
        ax.grid(alpha=.15)
    fig.suptitle('rosbag2245 | identical coordinates and instance colors | 330 cached frames',fontsize=14)
    fig.savefig(OUT/'maps.png',dpi=180);plt.close(fig)
    stats=read(SOURCE/'per_object.json');ids=[r['object_id']for r in stats]
    fig,ax=plt.subplots(figsize=(14,4),layout='constrained')
    x=np.arange(len(stats));ax.bar(x-.18,[r['sor_removed']/max(r['points'],1)*100 for r in stats],width=.36,label='SOR')
    ax.bar(x+.18,[r['ours_removed']/max(r['points'],1)*100 for r in stats],width=.36,label='Ours: visibility1',color='#009b83')
    ax.set_xticks(x,ids,rotation=90);ax.set(xlabel='Object ID (all 40 foreground objects)',ylabel='Source points removed (%)',title='Removal amount is not removal correctness');ax.legend()
    fig.savefig(OUT/'all_objects.png',dpi=160);plt.close(fig)
    calibration=read(SOURCE/'calibration_audit.json')['frames']
    fig,axes=plt.subplots(2,3,figsize=(15,6.6),layout='constrained')
    review=[]
    for ax,index in zip(axes.flat,protocol['review_frame_indices']):
        r=calibration[index];name=f"frame_{r['frame_id']:06}.png";shutil.copyfile(r['source_rgb'],OUT/name)
        ax.imshow(Image.open(OUT/name));ax.set_title(f"Frame {r['frame_id']} | preselected index {index}");ax.axis('off')
        review.append(dict(frame_id=r['frame_id'],image=name,source_rgb=r['source_rgb'],source_cloud=r['source_cloud'],
                           image_sha256=sha(OUT/name),camera=r['camera'],annotations=[],status='unannotated'))
    fig.suptitle('Fixed review frames | raw recorded RGB | no selection by method score',fontsize=14)
    fig.savefig(OUT/'rgb_review.png',dpi=170);plt.close(fig)
    dump(OUT/'annotation_packet.json',dict(scene='rosbag2245',annotation_status='not annotated or human verified',
        purpose='Independent visible-instance polygons can support a bounded 2D check. Cross-view IDs and 3D ground truth require additional verification.',frames=review))
    dump(OUT/'provenance.json',dict(source_run_sha256=sha(SOURCE/'run.json'),source_protocol_sha256=sha(SOURCE/'protocol.json'),
                                 generated_from_recorded_data=True,gt_accuracy_available=False))
    trs=''.join(f"<tr><td>{r['object_id']}</td><td>{r['category']}</td><td>{r['points']}</td><td>{r['observed_points']}</td><td>{r['decisive_points']}</td><td>{r['sor_removed']}</td><td>{r['ours_removed']}</td></tr>"for r in stats)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>车库对象地图修订对照</title><style>
body{max-width:1200px;margin:40px auto;padding:0 24px;color:#203544;font:16px/1.65 system-ui;background:#fafcfd}h1{font-size:30px}h2{margin-top:34px}img{width:100%;background:white;border:1px solid #dbe5e9;border-radius:10px}table{width:100%;border-collapse:collapse;background:white}th,td{padding:7px 12px;border-bottom:1px solid #dde5e8;text-align:right}th:nth-child(2),td:nth-child(2){text-align:left}.note{padding:18px;background:#eaf5f3;border-left:4px solid #008e79}a{color:#007e70}</style>
<h1>rosbag2245：真实对象地图修订对照</h1><p>330 帧记录 · 40 个前景对象 · 相同坐标系与对象配色</p>
<div class="note">本页展示实际运行结果。旧 Ours visibility1 未删除点，SOR 删除 450 点。删除量不代表正确率；当前没有独立参考标注，不能由本页推断准确率提升。</div>
<h2>完整前景对象地图</h2><img src="maps.png"><p>逐帧采用实际 640×360 相机参数。未知深度像素保持未知，所有分支沿用同一原始地图版本。</p>
<h2>预先固定的真实图像</h2><img src="rgb_review.png"><p>六帧在运行前按索引固定。<a href="annotation_packet.json">参考标注数据包</a>尚未标注，不是 ground truth。</p>
<h2>全对象删除统计</h2><img src="all_objects.png"><table><tr><th>ID</th><th>类别</th><th>原点数</th><th>有深度证据</th><th>有决定性票</th><th>SOR 删除</th><th>Ours 删除</th></tr>'''+trs+'''</table>
<h2>运行代价</h2><p>输入加载后完整处理 330 帧：Ours 当前实现中位数 0.1040 秒，SOR 0.0153 秒；1 次预热、5 次测量，Ryzen 9 9950X。该计时不包含前端模型、定位与文件读取，不等同于端到端实时速度。</p></html>'''
    (OUT/'index.html').write_text(html)
    print(OUT/'index.html')


if __name__=='__main__':main()
