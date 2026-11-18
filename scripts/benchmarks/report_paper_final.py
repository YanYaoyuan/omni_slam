"""Render final fixed benchmark evidence after all predictions and scores exist."""
from pathlib import Path
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_final_common import ROOT,RUN,FROZEN,SCENES,read,sha,dump,verify_execution

DEST=ROOT/'docs/reports/paper_final_v1_2026-11-11'
NAMES={'B0':'CG + mean','B1':'OVO (3 views) + mean','B2':'History-2 + mean','B3':'History-2 + area',
       'M':'omni_slam v1','A1':'History-1 + consensus','A2':'History-2 / no veto','A3':'History-2 + sqrt area'}


def report():
    verify_execution();s=read(RUN/'summary.json');assert s['complete']
    q=read(RUN/'queue_status.json');assert q['complete']
    audit=read(RUN/'completion_audit.json');assert audit['complete'] and audit['summary_sha256']==sha(RUN/'summary.json')
    scenes=s['scenes']; n=len(scenes)
    rows=[read(RUN/'evaluation'/scene/'metrics.json') for scene in scenes]
    for scene,want in s['scene_results'].items():assert sha(RUN/'evaluation'/scene/'metrics.json')==want
    DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','axes.titleweight':'bold','figure.facecolor':'white'})
    methods=list(NAMES);colors=['#8797a8' if k not in ['M','B3'] else '#007f77' if k=='M' else '#c78736' for k in methods]
    fig,axes=plt.subplots(1,2,figsize=(13,5.1),layout='constrained')
    for ax,key,title in zip(axes,['instance_pq100','category_pq100'],['Instance PQ × 100','Class-aware pooled PQ × 100']):
        vals=[s['methods'][k][key] for k in methods];bars=ax.barh(np.arange(8),vals,color=colors,height=.65)
        ax.set_yticks(range(8),[NAMES[k] for k in methods]);ax.invert_yaxis();ax.set_title(title,loc='left',pad=15)
        ax.set_xlim(0,max(vals)*1.2 if max(vals)>0 else 1);ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=9)
        ax.set_xlabel('Equal-weight mean of available fixed scenes');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle(f'omni_slam | Fixed SceneNN benchmark\n{n} scored / 6 selected · given poses · observed 5 cm domain',fontsize=14,fontweight='bold')
    fig.savefig(DEST/'comparison.png',dpi=180);fig.savefig(DEST/'comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
    for ax,b,title in zip(axes,['B0','B3','A3'],['Association: M − CG','Category: M − area','Category: M − sqrt area']):
        d=s['primary_contrasts'][b];values=[d['per_scene_delta'][x] for x in scenes]
        ax.bar(scenes,values,color=['#007f77' if v>=0 else '#b45340' for v in values]);ax.axhline(0,c='#586b7d',lw=.8)
        ax.set_title(title,loc='left');ax.set_ylabel('PQ × 100 difference');ax.set_xlabel('Frozen scene ID')
        ax.text(.02,.96,f"Mean {d['mean_delta']:+.3f}\n95% CI [{d['paired_scene_bootstrap_ci95'][0]:+.3f}, {d['paired_scene_bootstrap_ci95'][1]:+.3f}]",
            transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    fig.suptitle(f'Prespecified comparisons | {n} scored scenes · insufficient final-test coverage',fontsize=13)
    fig.savefig(DEST/'paired_deltas.png',dpi=180);fig.savefig(DEST/'paired_deltas.svg');plt.close(fig)
    # Every selected scene/method is accounted for, including unavailable inputs.
    table=['| 方法 | 实例 PQ×100 | 类别 pooled PQ×100 | 类别宏 PQ×100 | 前景 mIoU×100 | TP / FP / FN | 类别正确 / 匹配 |',
           '|---|---:|---:|---:|---:|---|---|']
    for k in methods:
        v=s['methods'][k];table.append(f"| {k}: {NAMES[k]} | {v['instance_pq100']:.4f} | {v['category_pq100']:.4f} | {v['macro_category_pq100']:.4f} | {v['foreground_miou100']:.4f} | {v['tp']}/{v['fp']}/{v['fn']} | {v['correct_classes']}/{v['category_geometry_matches']} |")
    scene_table=['| 场景 | 方法 | 实例 PQ×100 | 类别 PQ×100 | 召回 | 错并形态 / 预测 | 碎片 GT |',
                 '|---|---|---:|---:|---:|---|---:|']
    for r in rows:
        for k in methods:
            v=r['methods'][k];ins=v['instance'];sig=v['signatures']
            scene_table.append(f"| {r['scene']} | {k} | {100*ins['observed_voxel_instance_pq']:.4f} | {100*v['semantic']['class_aware_pq']:.4f} | {ins['recall']:.4f} | {len(sig['mixed_predicted_ids'])}/{ins['predicted_instances']} | {sig['fragmented_gt']} |")
    for missing in sorted(s['unavailable_scenes']):
        reason='官方无 RGB-D' if missing=='049' else 'GT 坐标未可靠对应'
        for k in methods:scene_table.append(f'| {missing} | {k} | N/A：{reason} | N/A | N/A | N/A | N/A |')
    domains=['| 场景 | 观测体素 | 标注有效比例 | 类别有效 / 实例有效 | 实例 GT / 类别 GT |',
             '|---|---:|---:|---:|---|']
    cost=['| 方法 | 含共享前端总耗时均值(s) | 因果建图均值(s) | 离线整理均值(s) | 类别更新均值(s) |',
          '|---|---:|---:|---:|---:|']
    for r in rows:
        d=r['domain'];domains.append(f"| {r['scene']} | {d['observed_voxels']} | {d['valid_voxel_fraction']:.2%} | {d['category_valid_voxel_fraction']:.2%} | {d['gt_instances']}/{d['category_gt_instances']} |")
    for k in methods:
        avg=lambda name:np.mean([r['cost'][k][name] for r in rows])
        cost.append(f"| {k} | {avg('total_including_frontend_s'):.2f} | {avg('causal_mapping_s'):.2f} | {avg('offline_finalization_s'):.2f} | {avg('category_s'):.3f} |")
    timing_rows=['| 场景 | 前端阶段 | 逐帧数 | 中位数(ms) | P95(ms) | 原运行阶段总耗时(s) |','|---|---|---:|---:|---:|---:|']
    timing_detail={}
    for scene in s['predicted_scenes']:
        timing_detail[scene]={}
        for phase in ['masks','triplets','features','geometry']:
            path=RUN/'frontend'/scene/(phase+'_run.json');phase_run=read(path)
            replay=None
            if phase=='geometry':
                replay=read(RUN/'timing_replay'/(scene+'.json'))
                assert replay['complete'] and replay['exact_cached_arrays'] and replay['exact_cached_manifest']
                assert 'timing_replay_checks' in audit
            records=replay['records'] if replay else phase_run['records'];times=[x['wall_s'] for x in records]
            value=dict(source_sha256=sha(path),frame_records=records,phase_wall_s=phase_run['wall_s'],
                frame_median_s=float(np.median(times)) if times else None,
                frame_p95_s=float(np.quantile(times,.95)) if times else None,
                initialization_and_phase_overhead_s=replay['initialization_and_final_export_s'] if replay else phase_run['wall_s']-sum(times),
                timing_source='exact-output timing replay' if replay else 'original frozen run',
                timing_replay=replay)
            timing_detail[scene][phase]=value
            med=f"{value['frame_median_s']*1000:.3f}" if times else 'N/A'
            p95=f"{value['frame_p95_s']*1000:.3f}" if times else 'N/A'
            timing_rows.append(f"| {scene} | {phase} | {len(times)} | {med} | {p95} | {value['phase_wall_s']:.3f} |")
    dump(DEST/'frontend_timing.json',timing_detail)
    timing_table='\n'.join(timing_rows)
    assoc='达到冻结效果门槛' if s['association_claim_supported'] else '未达到完整冻结门槛'
    cat='达到冻结效果门槛' if s['independent_category_claim_supported'] else '未达到完整冻结门槛，不能宣称独立类别精度贡献'
    coverage=f'仅 {n} 个可评分场景，低于预定至少四场景要求；本轮不能支撑充分的最终泛化主张。' if not s['sufficient_final_scene_coverage'] else f'{n} 个场景可评分，达到预定覆盖下限。'
    contrasts=['| 预定对比 | 均值变化（PQ百分点） | 配对 bootstrap 95% 区间 | 涨 / 平 / 降 | 数值效果门槛 |',
               '|---|---:|---|---|---|']
    for key,label in [('B0','实例 M−CG'),('B3','类别 M−面积'),('A3','类别 M−无筛选平方根面积')]:
        c=s['primary_contrasts'][key];lo,hi=c['paired_scene_bootstrap_ci95']
        contrasts.append(f"| {label} | {c['mean_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {c['wins']}/{c['ties']}/{c['losses']} | {'通过' if c['passes_frozen_effect_gate'] else '未通过'} |")
    mechanisms=['| 场景 | 图分支 | 接受边 / 可判定 | 可判定正确率 | 错边 | 同帧否决数 | 最终合并新增冲突对 |',
                '|---|---|---|---:|---:|---:|---:|']
    gains=['| 场景 | M 对比 | 纠正对象 | 退步对象 |','|---|---|---:|---:|']
    for r in rows:
        diagnostic=read(RUN/'evaluation'/r['scene']/'graph_diagnostics.json')
        for key,v in diagnostic['variants'].items():
            rate=f"{v['joined_precision']:.2%}" if v['joined_precision'] is not None else 'N/A'
            mechanisms.append(f"| {r['scene']} | {key} | {v['joined']}/{v['judgable_joined']} | {rate} | {v['wrong_joined']} | {v['coframe_rejections']} | {v['post_native_merge_conflict_pairs']} |")
        for key,v in r['paired_category_changes'].items():gains.append(f"| {r['scene']} | {key} | {v['gains']} | {v['regressions']} |")
    main=s['methods']['M'];cg=s['methods']['B0'];area=s['methods']['B3'];noveto=s['methods']['A2']
    interpretation=(f"相对 CG，实例 PQ×100 从 {cg['instance_pq100']:.4f} 到 {main['instance_pq100']:.4f}，"
        f"只增加 {main['instance_pq100']-cg['instance_pq100']:.4f} 个百分点；这是微小改善，不能写成显著精度跃升。"
        f"相对面积加权，类别 PQ×100 从 {area['category_pq100']:.4f} 到 {main['category_pq100']:.4f}，"
        "本轮对应一个匹配对象的类别纠正。M 与不做一致性筛选的平方根面积 A3 在三个场景的 pooled 类别 PQ 全部相同，"
        "因此没有一致性筛选独立精度增益的证据。"
        f"同帧否决的消融更有区分度：M 的实例 PQ×100 比去否决 A2 高 {main['instance_pq100']-noveto['instance_pq100']:.4f}，"
        "但 041 场景实例分数退步，且 A2 的类别 pooled PQ 均值更高，必须保留这个权衡。"
        f"M 在实例评价域仅匹配 {main['tp']}/{main['tp']+main['fn']} 个可见 GT 对象；召回和绝对精度仍限制当前论文主张。")
    contrast_table='\n'.join(contrasts);mechanism_table='\n'.join(mechanisms);gain_table='\n'.join(gains)
    main_table='\n'.join(table); all_scenes='\n'.join(scene_table); domain_table='\n'.join(domains); cost_table='\n'.join(cost)
    md=f'''# omni_slam：最后一轮固定 benchmark

预选六个固定场景；五个场景完成全部预测，共 25 份几何、40 行预测；其中 {n} 个场景完成八行评分，共 {n*8} 行。六场景 48 个预定方法/场景条目均有成绩或明确缺失原因。未据新场景分数调参、换模型或替换场景。完整执行目录：`{RUN}`。

## 结论

关联主张：**{assoc}**。类别相对面积和无筛选平方根面积的独立增量：**{cat}**。这只说明冻结协议下的效果，不等同于统计显著或官方排行榜领先。

**{coverage}** 数值效果与样本覆盖是两道独立门槛，下表的数值通过不能代替覆盖通过。

{contrast_table}

{interpretation}

{main_table}

![方法比较](comparison.png)

![预定主要对比](paired_deltas.png)

## 公平性与边界

全部方法复用 SAM2.1 Hiera L / SigLIP384 学习前端、给定位姿与观测支持；每场景 400 几何帧、200 语义帧。CG 来源消费保护共用于所有 CG 分支。OVO 是共享输入原生跟踪核＋至少三不同帧保留，不能称为原版完整 OVO。

类别比较固定各几何分支的均值前景保留；M/B2/B3/A3 共用相同几何及点分配。历史图是离线最终整理，最后原生合并不保证同帧互斥。本文没有定位提升、动态目标跟踪或机器人端实时性结论。

PQ 属于 5cm 观测有效域、严格 IoU>0.5 匹配。pooled 类别 PQ 与逐类宏 PQ 分开报告。未知预测计错；未知 GT 仅按冻结规则剔除，不能把有效域指标写成整图覆盖。场景配对 bootstrap 仅有 {n} 个样本，不以对象数扩大统计样本量。

## 所有场景与方法

{all_scenes}

## 评价域

{domain_table}

## 标注异常与缺失处理

036 的 1,621,485 个唯一标注坐标中，仅一个同时存在 0 和 14 标签，统一记为 void；打乱顺序的验证通过。038、041 的 CVPR18 坐标与原实例网格逐点精确对应。

045、047 已完成全部方法预测，但 CVPR18 标注与原实例网格不对应。已核对官方 NYU 网格、原彩色网格、配套轨迹和 HDF5 坐标通道，仍无可靠对应依据，保留预测并列 N/A。049 官方 ONI 缺失，官方 ZIP 也不含原始 RGB-D。三个场景均未被替换。

适配修订在任何方法分数产生前封存。只增加冲突坐标置 void 和缺失统计；没有修改模型、几何、阈值或预测，也没有通过预测与 GT 拟合获取坐标变换。旧执行清单、源码、失败日志及原预测封存文件保留。旧 025 的参考域、已有指标以及主方法输出逐项复核相同。

[标注适配修订与证据哈希](annotation_adapter_amendment.json)。官方数据来源：[SceneNN CVPR18](https://hkust-vgd.github.io/scenenn/cvpr18/)、[官方 NYU 标注网格](https://hkust-vgd.ust.hk/scenenn/contrib/nyu_class/)、[官方彩色网格](https://hkust-vgd.ust.hk/scenenn/contrib/color_ply/)。

## 机制诊断与类别得失

{mechanism_table}

历史边仅在预测完成后附加 GT 标签。正确率只在可判定边上计算，同时列出覆盖量；少支持、混合及背景观测单列，不用诊断重筛边。错并形态与碎片统计见完整场景表，原始重叠计数随每场景 JSON 保存。

{gain_table}

## 成本

{cost_table}

完整 JSON 同时包含前端各阶段、冷启动/验证开销、逐帧中位数/P95、CPU RSS、GPU 分配与保留显存、磁盘缓存量。CG 因果部分共享，四个最终分支在同一进程测量，RSS 是进程累计高水位；不能当作独立部署分支的精确内存差。原生最终整理还包含沿用的审计读出开销，未做有利于主方法的扣除。GPU 调度采样记录另存，不把离线速度宣称为全系统在线 FPS。

{timing_table}

[前端逐帧计时及原记录哈希](frontend_timing.json) 覆盖五个已预测场景。masks/triplets/features 各有 200 个原运行计时记录，包含该帧文件输出及同步；因果建图逐帧中位数/P95 另见每场景 metrics。

原几何支持生成器只记录阶段总时长，已按协议允许的计时复测补齐每场景 400 帧。复测调用未修改的冻结函数，在帧边界同步 GPU；所有缓存数组及 manifest 内容与原运行相同，临时输出删除。表中几何逐帧分位数来自复测，阶段总时长及方法总成本仍沿用原运行，二者不混算。036 首次复测因 JSON 字节排版检查错误而中止验收（内容已相同），修复检查后第二次通过；其余场景各一次，均在最多三次上限内，未按耗时选择成绩。失败记录保留，无模型参数或评分变化。补测初始化与最终导出单列，不据此宣称在线全系统延迟。

## 贡献决策与停止条件

按预先固定的对比判断保留哪些论文主张，负结果完整保留。本轮之后转入论文写作；没有根据结果自动启动下一方法分支。机器人全链路演示与实时强前端接入另列工程任务。

[原始汇总](summary.json) · [执行清单](execution_manifest.json) · [完成核查](completion_audit.json) · [冻结协议](../../paper/frozen_v1/evaluation_protocol.md)
'''
    (DEST/'report.md').write_text(md)
    # Markdown tables become plain HTML without a document dependency.
    def html_table(lines):
        cells=[[x.strip() for x in line.strip('|').split('|')] for line in lines]
        return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in cells[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in cells[2:])+'</tbody></table>'
    body=f'''<h1>omni_slam · 固定 benchmark</h1><p>预选六场景 / 五个完成预测 / {n} 个可评分 · 八行对照</p>
<div class="note"><b>关联：</b>{assoc}。<br><b>类别独立增量：</b>{cat}。<br><b>覆盖：</b>{coverage}</div>
<p>{interpretation}</p>
<h2>预定主要对比</h2>{html_table(contrasts)}
<img src="comparison.png"><img src="paired_deltas.png"><h2>方法汇总</h2>{html_table(table)}
<h2>完整场景结果</h2>{html_table(scene_table)}<h2>评价覆盖</h2>{html_table(domains)}
<p>036 一个冲突坐标置为 void；045/047 标注坐标无可靠对应，049 缺 RGB-D，均列 N/A。修订先于评分，全部预测不变，未替换场景。</p>
<h2>历史关联机制诊断</h2>{html_table(mechanisms)}<h2>类别逐对象得失</h2>{html_table(gains)}
<h2>运行成本</h2>{html_table(cost)}<p>观测域诊断，不是官方榜单；类别与实例评价域分开。RSS 为共享进程高水位，实时机器人性能尚未验证。</p>
<details><summary>前端逐帧计时及固定输入复测</summary>{html_table(timing_rows)}<p>几何逐帧分位数来自逐数组验证相同的计时复测；总耗时沿用原运行，两者不混算。详见完整说明与 frontend_timing.json。</p></details>
<p><a href="report.md">完整说明</a> · <a href="summary.json">原始汇总</a> · <a href="completion_audit.json">完成审计</a></p>'''
    (DEST/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam 固定 benchmark</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#182c42;max-width:1180px;margin:45px auto;padding:0 24px}h1{font-size:34px}h2{margin-top:36px}.note{background:#edf7f5;border-left:5px solid #007f77;padding:20px}img{width:100%;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;text-align:right;border-bottom:1px solid #d8e0e7}th{background:#edf2f6}th:first-child,td:first-child{text-align:left}a{color:#007f77}</style>'+body+'</html>')
    for name in ['summary.json','execution_manifest.json','completion_audit.json','annotation_adapter_amendment.json']:
        (DEST/name).write_bytes((RUN/name).read_bytes())
    for r in rows:
        target=DEST/'evaluation'/r['scene'];target.mkdir(parents=True,exist_ok=True)
        for name in ['metrics.json','graph_diagnostics.json','category_reference.json']:
            (target/name).write_bytes((RUN/'evaluation'/r['scene']/name).read_bytes())
    print('REPORT',DEST/'index.html')


if __name__=='__main__':report()
