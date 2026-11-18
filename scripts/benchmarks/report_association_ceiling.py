"""Publish oracle diagnostics without presenting them as deployable results."""
import html
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from diagnose_association_ceiling import OUT

REPORT=ROOT/'output/association_ceiling_v1'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=read(OUT/'summary.json');p=read(OUT/'protocol.json')
    assert s['complete'] and s['protocol_sha256']==sha(OUT/'protocol.json')
    for f,h in p['source_hashes'].items():assert sha(f)==h
    REPORT.mkdir(parents=True,exist_ok=True)
    for name in ['protocol.json','summary.json']:(REPORT/name).write_bytes((OUT/name).read_bytes())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,scene in zip(axes,s['scenes']):
        m=scene['methods'];xx=np.arange(3)
        baseline=[m[k]['observed_voxel_instance_pq']*100 for k in ['native','earliest_native','latest_native']]
        oracle=[m[k]['observed_voxel_instance_pq']*100 for k in ['final_merge_oracle','earliest_oracle','latest_oracle']]
        for off,vals,color,label in [(-.18,baseline,'#8193a8','Matched control'),(.18,oracle,'#d88c31','GT-assisted diagnostic')]:
            bars=ax.bar(xx+off,vals,.34,color=color,label=label);ax.bar_label(bars,fmt='%.2f',padding=3)
        ax.set_xticks(xx,['Final map\nmerge only','Raw supports\nearliest owner','Raw supports\nlatest owner'])
        ax.set_title(scene['scene']);ax.set_ylabel('Instance PQ x 100');ax.set_ylim(0,58)
    axes[0].legend(frameon=False,fontsize=9);fig.suptitle('Oracle diagnostics — not algorithm scores; compare bars within each pair')
    fig.tight_layout();fig.savefig(REPORT/'paired_diagnostics.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,scene in zip(axes,s['scenes']):
        keys=['native','ceiling_native_final','ceiling_retained_observations','ceiling_all_masks','ceiling_all_measurements']
        vals=[scene['methods'][k]['tp'] for k in keys]
        bars=ax.bar(np.arange(5),vals,color=['#167cb8']+['#d1a977']*4);ax.bar_label(bars,padding=3)
        ax.set_xticks(np.arange(5),['Actual\nmatched','Final map\nceiling','Retained obs\nceiling','All masks\nceiling','Measured\nceiling'])
        ax.set_title(scene['scene']);ax.set_ylim(0,max(vals)*1.22);ax.set_ylabel('Matched / potentially matchable GT instances')
    fig.suptitle('Optimistic fixed-domain ceilings: perfect identity + background rejection, IoU > 0.5')
    fig.tight_layout();fig.savefig(REPORT/'coverage.png',dpi=180);plt.close(fig)
    tables=[];objects=[];categories={}
    for scene in s['scenes']:
        key=scene['scene'];m=scene['methods'];rows=[]
        for order in ['earliest','latest']:
            a,b=m[order+'_native'],m[order+'_oracle'];t=scene['transitions'][order]
            rows.append(f"<tr><td>{key}</td><td>{order}</td><td>{a['observed_voxel_instance_pq']*100:.2f} → {b['observed_voxel_instance_pq']*100:.2f}</td><td>+{(b['observed_voxel_instance_pq']-a['observed_voxel_instance_pq'])*100:.2f}</td><td>{a['tp']} → {b['tp']}</td><td>{len(t['gained_matched_gt'])} / {len(t['lost_matched_gt'])}</td></tr>")
        tables+=rows
        no_mask=[o['gt_id'] for o in scene['objects'] if not o['coverage']['all_masks']['recoverable_at_iou50']]
        dropped=[o['gt_id'] for o in scene['objects'] if o['coverage']['all_masks']['recoverable_at_iou50'] and not o['coverage']['retained_observations']['recoverable_at_iou50']]
        recover=[o['gt_id'] for o in scene['objects'] if o['raw_association_recovered_both']]
        categories[key]=dict(insufficient_mask_coverage=no_mask,lost_between_all_masks_and_retained=dropped,paired_association_recovered_both=recover)
        for o in scene['objects']:
            cover=''.join(f"<td>{o['coverage'][k]['coverage']*100:.1f}%</td>" for k in ['native_final','retained_observations','all_masks'])
            checks=''.join('<td>'+('✓' if o['matched'][k] else '—')+'</td>' for k in ['native','earliest_native','earliest_oracle','latest_native','latest_oracle'])
            objects.append(f"<tr><td>{key}</td><td>{o['gt_id']}</td>{cover}{checks}</tr>")
    dump(REPORT/'bottlenecks.json',categories)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>对象关联修订：投入前诊断</title><style>
    body{max-width:1150px;margin:40px auto;padding:0 24px;background:#fafcfd;color:#243648;font:16px/1.75 system-ui,sans-serif}h1{font-size:34px}h2{margin-top:35px}img{width:100%;border-radius:12px;background:white}table{width:100%;border-collapse:collapse;background:white}td,th{padding:9px;text-align:left;border-bottom:1px solid #dde4eb}small,.note{color:#617082}.lead{border-left:5px solid #d88c31;padding:20px;background:#fff6e9}a{color:#167cb8}details{margin-top:30px}</style>
    <h1>先判断对象关联修订值不值得做</h1><p>两场景 · 已有掩码与测量点 · 不新增建图、训练或 GPU 推理</p>
    <div class="lead"><b>结论：值得投入一个有限预算的对象归属修订原型。</b><br>在固定支持点和固定重叠处理方式的配对对照中，真值辅助的整条观测归属修订在两个场景都提升超过 5 个 PQ 点、增加至少 3 个正确匹配；两种重叠处理都成立。它说明存在可修复空间，不代表无真值算法已经实现这些收益。</div>
    <h2>先区分三种数值</h2><p><b>算法成绩</b>是已有 VGGT 单帧原生地图。<b>关联 oracle</b>使用 GT 选择整条观测的归属，不生成新点、不删除点、不在观测内部拆分；它是一个具体可构造的理想诊断，而非所有关联方案的最优上限。<b>覆盖上限</b>进一步允许逐点评价位置获得正确身份和背景标签，仅用来测量现有支持范围的限制，不能算关联收益。</p>
    <h2>只改关联，有多少空间？</h2><img src="paired_diagnostics.png"><table><tr><th>场景</th><th>重叠点归属</th><th>PQ：对照 → oracle</th><th>提升</th><th>TP</th><th>新增 / 丢失匹配</th></tr>'''+''.join(tables)+'''</table>
    <p>Replica 两种归属方式均新增 GT 17、23、36、58；SceneNN 两种方式均新增 GT 189623、958685，第三个新增对象随重叠归属方式变化。四组配对对照都没有丢失原有匹配。</p>
    <p><b>无需重建观测的更保守诊断也有空间：</b>在原生最终地图上，只允许整块预测对象按 GT 合并，Replica PQ 33.38 → 45.75，TP 11 → 12；SceneNN PQ 18.00 → 24.49，TP 13 → 14。这些数值与“观测重分配”使用不同几何表达，不能直接相减来证明拆分模块的贡献。</p>
    <p class="note">原始观测支持回放省略 CG 去噪；因此对照本身不等于原生最终地图。有效的干预比较只能是同一 earliest / latest 条件内的两根柱子。重叠点采用固定最早 / 最晚观测归属，不根据 GT 选择，两个条件全部报告。</p>
    <h2>关联解决不了哪些问题？</h2><img src="coverage.png"><p>Replica 的 18 个参考对象都有足够现存覆盖，输入范围并非该场景的主要硬限制。SceneNN 的 50 个参考对象中：全部掩码支持最多使 39 个达到 IoU &gt; 0.5 的覆盖门槛；当前前景保留观测只支持 25 个。即 11 个受限于掩码支持覆盖，另 14 个在全部掩码到保留观测之间失去足够支持。</p>
    <p>这 14 个不能直接归因为某一个筛选模块：其中可能涉及语义前景保留、观测接纳和对象过滤，当前诊断没有把它们逐级拆开。即使有足够覆盖，错误点、错误归属和掩码混杂仍会阻止匹配。</p>
    <p class="note">“测量点上限接近 100%”仅针对既有参考域；参考本来就来自可观测区域，不能推出整间房间几何完整或传感器没有误差。覆盖上限允许完美拒绝背景并不输出覆盖 ≤50% 的对象，属于乐观评价空间上限，未证明存在相应可部署的三维预测。</p>
    <h2>投入决定</h2><p>运行前已固定门槛：两场景、两种归属方式，每组都至少 +5 个 PQ 点且净增至少 2 个正确匹配。结果全部通过。下一原型应聚焦“保留不可变观测，在新证据到来后修订历史观测的对象归属”，优先消除观测足够但身份分裂或混合的错误；VGGT 单帧作为固定视觉基线。</p>
    <p>本轮未证明这种修订可以无 GT 学会，也未证明 Transformer 必需或跨窗口漂移就是原因。实现路线和方法创新还须单独验证。不能把诊断标签送回这两个场景的算法推理，更不能以 oracle 分数写论文主表。</p>
    <details><summary>全部 68 个参考对象：覆盖率和匹配状态</summary><table><tr><th>场景</th><th>GT</th><th>最终点覆盖</th><th>保留观测覆盖</th><th>所有掩码覆盖</th><th>原生成绩</th><th>最早对照</th><th>最早 oracle</th><th>最晚对照</th><th>最晚 oracle</th></tr>'''+''.join(objects)+'''</table></details>
    <p><a href="summary.json">完整结果</a> · <a href="protocol.json">事先冻结的诊断协议</a> · <a href="bottlenecks.json">逐对象问题清单</a> · <a href="../vggt_fullmap_v1/index.html">上一轮真实算法成绩与图像</a></p></html>'''
    (REPORT/'index.html').write_text(page)
    dump(REPORT/'report_manifest.json',dict(complete=True,gt_diagnostic_only=True,
        summary_sha256=sha(OUT/'summary.json'),protocol_sha256=sha(OUT/'protocol.json'),
        files={f.name:sha(f) for f in REPORT.iterdir() if f.is_file() and f.name!='report_manifest.json'},
        report_source_sha256=sha(__file__)))
    print(REPORT/'index.html')


if __name__=='__main__':main()
