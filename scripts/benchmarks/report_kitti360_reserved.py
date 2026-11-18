"""Audit and report the frozen outdoor run, including failures and coverage."""
from pathlib import Path
import html
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_kitti360_reserved import OUT,META,METHODS,verify,job_path

REPORT=ROOT/'output/kitti360_reserved_v1'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from omni_semantic_map.kitti360_evaluation import kernel
    from evaluate_kitti360_development import clean_json
    p=verify();summary=read(OUT/'summary.json');seal=read(OUT/'predictions_complete.json')
    assert summary['complete'] and summary['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    assert seal['protocol_sha256']==sha(OUT/'protocol.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    args,ns=kernel(META/'evalInstanceLevelSemanticLabeling.py');checks=[]
    for w in p['windows']:
        scene=w['scene'];root=OUT/scene;scored=read(root/'evaluation/summary.json')
        for name,h in scored['files_sha256'].items():assert sha(root/'evaluation'/name)==h
        for method in METHODS:
            for domain in ['full_window','observed_support']:
                matches=read(root/'evaluation'/f'{method}_{domain}_matches.json')
                averages=clean_json(ns['computeAverages'](ns['evaluateMatches'](matches,args),args))
                assert averages==scored['methods'][method][domain]['averages']
        for arm in ['base','vggt_single']:
            src=root/arm/'maps/cg/history2';g=np.load(src/'geometry.npz')
            for filt in ['sor','visibility1','simple_ratio','mc_point_ratio']:
                dest=root/arm/'maps/cleanup'/filt;ids=np.load(dest/'retained_source_indices.npy');out=np.load(dest/'geometry.npz')
                assert np.all(np.diff(ids)>0)
                for key in ['xyz','instance']:np.testing.assert_array_equal(out[key],g[key][ids])
                assert sha(dest/'classification.json')==sha(src/'classification.json')
        checks.append(dict(scene=scene,score_recomputations=10,exact_cleanup_subsets=8))
    for method in METHODS:
        for domain in ['full_window','observed_support']:
            matches=read(OUT/'pooled_matches'/f'{method}_{domain}.json');names=[]
            for scene,m in matches.items():
                for rows in m['prediction'].values():
                    for row in rows:
                        assert row['filename'].startswith(scene+'/');names.append(row['filename'])
            assert len(names)==len(set(names))
            calculated=clean_json(ns['computeAverages'](ns['evaluateMatches'](matches,args),args))
            assert calculated==summary['pooled'][method][domain]
    REPORT.mkdir(parents=True,exist_ok=True)
    for file in ['summary.json','protocol.json','runtime_recovery.json','runtime_smoke_verified.json']:
        (REPORT/file).write_bytes((OUT/file).read_bytes())
    if (OUT/'runtime_recovery_v2.json').exists():
        (REPORT/'runtime_recovery_v2.json').write_bytes((OUT/'runtime_recovery_v2.json').read_bytes())
    labels=['History2','SOR','Ours: visibility','VGGT single','VGGT + visibility'];colors=['#8998aa','#cc9a41','#00a28e','#3577b7','#7763ac']
    fmt=lambda v:'N/A' if v is None else f'{v*100:.2f}'
    table=''
    for method,label in zip(METHODS,labels):
        a=summary['pooled'][method]['full_window'];b=summary['pooled'][method]['observed_support']
        table+=f"<tr><td>{label}</td><td>{fmt(a['allAp50%'])}</td><td>{a['allAp']*100:.3f}</td><td>{fmt(a['classes']['building']['ap50%'])}</td><td>{fmt(a['classes']['car']['ap50%'])}</td><td>{fmt(b['allAp50%'])}</td></tr>"
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,klass in zip(axes,['all','building','car']):
        vals=[]
        for method in METHODS:
            a=summary['pooled'][method]['full_window'];v=a['allAp50%'] if klass=='all' else a['classes'][klass]['ap50%']
            vals.append(v*100 if v is not None else np.nan)
        bars=ax.bar(np.arange(5),vals,color=colors);ax.bar_label(bars,fmt='%.2f',padding=3)
        ax.set_xticks(np.arange(5),['History2','SOR','Ours','VGGT','VGGT+Ours'],rotation=25,ha='right')
        ax.set_title(klass);ax.set_ylabel('Full-window AP50 x 100');ax.set_ylim(0,max(5,np.nanmax(vals)*1.3))
    fig.suptitle('KITTI-360: pooled matching records from two reserved street windows');fig.tight_layout()
    fig.savefig(REPORT/'scores.png',dpi=170);plt.close(fig)
    scene_html='';diagnostics={}
    for w in p['windows']:
        scene=w['scene'];root=OUT/scene;job=read(job_path(scene));evaluation=read(root/'evaluation/summary.json')
        fig,axs=plt.subplots(1,3,figsize=(15,3.1))
        for ax,index in zip(axs,[0,len(job['frames'])//2,len(job['frames'])-1]):
            r=job['frames'][index];ax.imshow(Image.open(r['rgb']));ax.set_title(f"Frame {r['frame_id']}");ax.axis('off')
        fig.tight_layout();fig.savefig(REPORT/(scene+'_rgb.png'),dpi=160);plt.close(fig)
        raw=np.load(root/'connected/cache/geometry.npz');points=raw['xyz'];ids=np.linspace(0,len(points)-1,min(50000,len(points)),dtype=int)
        traj=np.array([r['pose'] for r in job['frames']])[:,:3,3]
        fig,axs=plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=True)
        for ax,method in zip(axs,['history2','Ours_visibility','VGGT_single_plus_visibility']):
            arm,path=METHODS[method];folder=root/arm/path;g=np.load(folder/'geometry.npz');rows=read(folder/'classification.json')['objects']
            kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']];sel=np.flatnonzero(np.isin(g['instance'],kept))
            if len(sel)>50000:sel=sel[np.linspace(0,len(sel)-1,50000,dtype=int)]
            ax.scatter(points[ids,0],points[ids,1],s=.15,c='#d1d7dc',rasterized=True)
            ax.scatter(g['xyz'][sel,0],g['xyz'][sel,1],s=1.2,c=g['instance'][sel]%20,cmap='tab20',rasterized=True)
            ax.plot(traj[:,0],traj[:,1],color='#273d52',lw=.8);ax.set_aspect('equal');ax.set_title(method);ax.set_xlabel('x / m')
        fig.suptitle(scene+' — measured context + predicted building/car points + camera path');fig.tight_layout()
        fig.savefig(REPORT/(scene+'_map.png'),dpi=170);plt.close(fig)
        rows=''.join(f"<tr><td>{label}</td><td>{fmt(evaluation['methods'][method]['full_window']['averages']['allAp50%'])}</td><td>{fmt(evaluation['methods'][method]['observed_support']['averages']['allAp50%'])}</td></tr>" for method,label in zip(METHODS,labels))
        change=read(root/'base/maps/cleanup/run.json')['results'];candidate=read(root/'vggt_single/maps/cleanup/run.json')['results']
        eligible=evaluation['methods']['history2']['full_window'];evidence=np.load(root/'base/maps/cleanup/evidence.npz')
        diagnostics[scene]=dict(gt_eligible=eligible['gt_eligible'],predictions_eligible=eligible['predictions_eligible'],
            measured_points=len(points),reference_points=evaluation['reference_points'],
            positive_points=int(np.count_nonzero(evidence['positive'])),negative_points=int(np.count_nonzero(evidence['negative'])),
            negative_votes=int(evidence['negative'].sum()),ours_removed=change['visibility1']['removed'])
        scene_html+=f"<h2>{scene}</h2><p>{len(job['frames'])} 个采样帧，参考点 {evaluation['reference_points']:,}，测量支持范围内 {evaluation['observed_reference_points']:,}。Ours 删除 {change['visibility1']['removed']} 个前景点，SOR 删除 {change['sor']['removed']} 个；VGGT+Ours 删除 {candidate['visibility1']['removed']} 个。</p><img src='{scene}_rgb.png'><img src='{scene}_map.png'><table><tr><th>方法</th><th>完整窗口 AP50</th><th>观测范围 AP50</th></tr>{rows}</table>"
        scene_html+=f"<p>可评估真值：建筑 {eligible['gt_eligible']['building']}，车辆 {eligible['gt_eligible']['car']}；可评估预测：建筑 {eligible['predictions_eligible']['building']}，车辆 {eligible['predictions_eligible']['car']}。可见性修订累计反证票数为 {diagnostics[scene]['negative_votes']}。地图颜色表示各面板内部的预测实例，颜色差异不代表方法改善。</p>"
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>KITTI-360 室外预留窗口评测</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui,sans-serif;color:#243648;background:#fafcfd}h1{font-size:32px}h2{margin-top:35px}table{width:100%;border-collapse:collapse;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #dbe3e8}img{width:100%;background:white;border-radius:10px}.note{padding:18px;background:#edf3f7}a{color:#167cb8}</style>
    <h1>KITTI-360：室外预留街区正式跑分</h1><p>两段街区，48+53 个固定采样帧；全部预测封存后才读取标注。</p>
    <p class="note">这是官方验证划分中两个预留窗口的本地适配评测，不是整条驾驶序列、完整验证集或官方榜单。使用给定位姿；不评测 LIVO 定位精度或动态目标跟踪。官方评分内核使用已记录的兼容适配，主要报告 AP50，避免将代码 AP50:90 与网页 AP50:95 混用。</p>
    <h2>两个窗口合并匹配记录的成绩</h2><table><tr><th>方法</th><th>完整窗口 AP50</th><th>AP50:90（辅助）</th><th>建筑 AP50</th><th>车辆 AP50</th><th>观测范围 AP50（辅助）</th></tr>'''+table+'''</table><img src="scores.png"><p>上述合并成绩由跨窗口匹配记录共同计算，不是各窗口 AP 的简单平均；预测标识已加场景前缀。缺少可评估真值时显示 N/A，不填零。</p>
    <p class="note"><b>同分核查补充：</b>只有 AP50 相同，SOR 的 AP50:90 从 3.944% 降至 3.772%；三个车辆 IoU 均下降。逐组从各自地图重投影后，与存档预测逐点一致，匹配记录也一致。Ours 的 574 次二维候选反证中，509 次缺少对应深度，65 次深度不一致，最终零反证；VGGT 改变了非目标关联，但三个参与评分车辆的观测集合及点坐标完全相同。<a href="../kitti360_equal_score_audit_v1/report.md">完整核查报告</a> · <a href="../kitti360_equal_score_audit_v1/audit.json">逐帧核查数据</a></p>
    <p class="note"><b>本轮结论：五组方法整体 AP50 均为 5.56%，没有涨点。</b>参考中共有 22 个可评估建筑、27 辆车，最终只输出 0 个建筑和 3 个可评估车辆实例。第二窗口的 50% 来自仅 1 辆可评估车辆被正确匹配、建筑为零，不能将其当作室外全面表现良好。当前主要表现为目标召回严重不足；丢失究竟发生于掩码、跨帧关联还是语义/对象保留，需要进一步按阶段定位。两段可见性修订都未积累反证，因此 Ours 未改变几何。已有结果不支持继续原样扩大此配置来寻找涨点。</p>
    <p><b>Ours</b>是当前可见性证据几何修订：身份和类别保持不变，只修订既有几何。VGGT 单帧仅替换关联描述子，语义仍由 SigLIP 读出；VGGT+visibility 是明确列出的组合候选。未训练对象记忆和 GT 诊断 oracle 均未参与。</p>'''+scene_html+'''<h2>控制与限制</h2><p>相同 SAM 掩码、测量点、给定位姿、户外类别表、CG/history2 和最终处理。使用先前开发窗口确定的稀疏 LiDAR 配置，没有根据本轮得分调整阈值或替换场景。完整静态参考域为主指标，实际观测域仅帮助解释覆盖不足。模型预训练是否包含这些环境未被排除；相隔较远的轨迹也不自动保证全部观察几何互不重叠。</p><p>运行环境原 Python 路径失效；在本次实验目录恢复原 PyTorch/torchvision 版本，并在预留推理前检查一个旧开发帧的掩码与特征一致性。恢复记录和核验结果随报告提供。</p><p><a href="summary.json">完整指标</a> · <a href="protocol.json">冻结协议</a> · <a href="verification.json">核验</a> · <a href="runtime_recovery.json">环境恢复记录</a> · <a href="runtime_smoke_verified.json">旧帧一致性检查</a></p></html>'''
    (REPORT/'index.html').write_text(page)
    dump(REPORT/'diagnostics.json',diagnostics)
    dump(REPORT/'verification.json',dict(complete=True,source_sha256=sha(__file__),checks=checks,
        pooled_score_recomputations=10,no_pooled_prediction_id_collision=True,
        sealed_prediction_files_verified=len(seal['files']),summary_sha256=sha(OUT/'summary.json'),report_sha256=sha(REPORT/'index.html')))
    print(REPORT/'index.html')


if __name__=='__main__':main()
