"""Report the completed full-map diagnosis and local-support supervision export."""
import html
from pathlib import Path
import re
import numpy as np
from audit_full_object_maps import OUT,REPORT,STAGES,protocol,read,dump,sha

NAMES={'raw_geometry':'原始几何','mask_support':'掩码支持','accepted_observation_support':'接受观测的原始支持',
       'native_fused_geometry':'原生融合几何','final_v1_all_geometry':'最终全部几何','final_v1_foreground':'前景读出'}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=protocol();maps=read(OUT/'map_summary.json');data=read(OUT/'supervision_summary.json');audit=read(OUT/'audit.json')
    assert audit['complete'] and audit['map_summary_sha256']==sha(OUT/'map_summary.json')
    assert audit['supervision_summary_sha256']==sha(OUT/'supervision_summary.json')
    REPORT.mkdir(parents=True,exist_ok=False)
    values=[sum(s['stages'][stage]['reachable_gt_over_half'] for s in maps['scenes']) for stage in STAGES]
    fig,ax=plt.subplots(figsize=(11,4.8))
    labels=['Raw geometry','Mask support','Accepted support','Fused geometry','Final geometry','Foreground output','Actual matches']
    values.append(maps['matched'])
    bars=ax.barh(labels,values,color=['#8fa5b5']*6+['#245b78'])
    ax.invert_yaxis();ax.set_xlim(0,maps['gt_instances']+12);ax.set_xlabel('GT objects in the same four full observed maps')
    for bar,v in zip(bars,values):ax.text(v+.8,bar.get_y()+bar.get_height()/2,str(v),va='center')
    ax.set_title('Coverage eligibility is not successful object reconstruction')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(REPORT/f'coverage.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    train=data['totals_by_role']['train'];validation=data['totals_by_role']['validation'];evaluation=data['totals_by_role']['evaluation']
    total_v1={k:sum(s['actual_v1'][k] for s in maps['scenes']) for k in ['tp','fp','fn']}
    total_all={k:sum(s['actual_all_regions_control'][k] for s in maps['scenes']) for k in ['tp','fp','fn']}
    lines=['# omni_slam 主线：从完整对象地图定位损失，补齐局部支持监督','',
        '2026-11-12。对象级语义建图为主贡献，LIVO 提供定位与三维支持。本轮复用已有完整建图，检查四个开发场景的全部真值对象，并为原划分的十一场景导出局部 DINO 支持监督。没有修改旧算法、重跑建图或训练新模型。','',
        '## 已完成与当前结论','',
        f"- 全地图覆盖 **{maps['gt_instances']} 个 GT 对象**，v1 实际匹配 **{maps['matched']} 个**；评价不再局限于 32 次可判定候选事件。对象数和候选事件数是不同单位，不能直接比较。",
        '- 26 个未匹配对象在全部检查点都有超过一半的几何覆盖，但仍因实例归属／污染等问题未匹配。这部分值得研究对象内部的局部支持归属。',
        '- 其余未匹配对象有覆盖损失：首个低于匹配必要条件的检查点分别为掩码支持 15、原生融合几何 4、最终全部几何 8、前景读出 8。它们是诊断分组，不是隔离实验得出的唯一因果。',
        f"- 取消背景过滤使 TP 从 {total_v1['tp']} 到 {total_all['tp']}，却使 FP 从 {total_v1['fp']} 到 {total_all['fp']}，四场景 PQ 都下降。因此不能通过保留全部区域解决问题。",
        f"- 训练划分已导出 **{train['observations']:,} 个观测、{train['tokens']:,} 个局部 token**，其中 {train['tokens_with_foreground']:,} 个有前景支持，{train['tokens_multiple_foreground_instances']:,} 个混合多个前景实例。保留完整分布及未知支持，不再因整对象不够纯就丢弃监督。",
        '- 这些是相关的多视图、稀疏实测支持监督，不是同等数量的独立样本，也不是完整像素级标注。它们还没有带来新的算法分数。','',
        '## 全地图证据链','', '![分阶段覆盖与实际对象匹配](coverage.png)','',
        '| 场景 | GT | 原始几何 | 掩码支持 | 接受支持 | 融合几何 | 最终几何 | 前景读出 | 实际匹配 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in maps['scenes']:
        lines.append(f"| {r['key']} | {r['gt_instances']} | "+' | '.join(str(r['stages'][k]['reachable_gt_over_half']) for k in STAGES)+f" | {r['actual_v1']['tp']} |")
    lines += ['', '前六列计数表示：该对象在固定有效参考域内，超过 50% 的参考体素距相应几何不超过 5cm。这只是达到严格 IoU>0.5 的几何必要条件，不是可达到的 TP/PQ，更不是 oracle 算法性能。参考域本来就来自观测几何，原始几何覆盖高不能说明传感器观测到完整物体。', '',
        '不同阶段有下采样、抖动及去噪，覆盖并不严格单调。例如 SceneNN 015 的掩码支持有 13 个对象过半，融合后有 14 个；不能把每一列差值简单解释为对象被删除。全部正反变化和逐对象覆盖均保存在原始审计记录。最终前景有 69 个对象满足覆盖必要条件，其中实际匹配 42 个，另外 27 个仍未匹配；“全部检查点都过半”的失败对象为 26 个，两者定义不同。', '',
        '## 一个完整地图读出控制：保留全部区域','',
        '| 场景 | v1 PQ×100 | 全部区域 PQ×100 | v1 TP / FP / FN | 全部区域 TP / FP / FN | 新增 / 丢失匹配 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in maps['scenes']:
        a,b=r['actual_v1'],r['actual_all_regions_control'];t=r['readout_transition']
        lines.append(f"| {r['key']} | {a['observed_voxel_instance_pq']*100:.3f} | {b['observed_voxel_instance_pq']*100:.3f} | {a['tp']}/{a['fp']}/{a['fn']} | {b['tp']}/{b['fp']}/{b['fn']} | {len(t['gained'])}/{len(t['lost'])} |")
    lines += ['', '控制只取消最后的预测背景类别过滤，几何、实例归属、ID、域和匹配规则不变，没有用 GT 决定保留哪些预测。它说明粗粒度读出存在取舍，不构成新的论文方法。', '',
        '## 局部监督数据','',
        '| 划分 | 观测 | 局部 token | 有有效支持 | 有前景支持 | 含多个前景实例 |',
        '|---|---:|---:|---:|---:|---:|']
    for role in ['train','validation','evaluation']:
        r=data['totals_by_role'][role]
        lines.append('| '+role+' | '+' | '.join(f"{r[k]:,}" for k in ['observations','tokens','tokens_with_valid_support','tokens_with_foreground','tokens_multiple_foreground_instances'])+' |')
    lines += ['', '沿用六训练场景、一验证场景、四评测场景的隔离。此前 office_4 没有任何合格的整对象关联训练事件，现在仍能保留 54,043 个局部 token，其中 17,373 个含前景支持；这些监督来自已有三维参考，而非模型自标注。', '',
        '每个 token 保存原 DINO 特征、patch 索引、所属观测、原始点 ID、逐实例计数。未知为 -1、背景为 0、正数为场景内实例 ID。混合 token 保留完整计数，未被强行分成一个“正确实例”；背景 0 也不能当成全场景同一个物体。', '',
        '所有特征直接来自既有 DINO 缓存，并逐数组核对一致；投影分桶及采样顺序与原始池化代码一致。相机投影的支持可能稀疏或存在标注／配准误差，因此目标表达的是实测支持分布，不是整块图像 patch 的真实像素比例。', '',
        '## 下一步方法应验证的唯一问题','',
        '**局部支持的前景和实例归属能否被学好，并在完整建图中减少错误吸收和对象丢失？**', '',
        '研究输入应为局部 DINO 特征、实测支持及跨视角来源；监督应保留混合实例分布和未知区域。候选方法先只改变支持归属或加权，保留定位、几何候选和其余融合流程，避免再一次同时替换多个环节。对象记忆与学习式注意力可用于读取这些局部证据，但其价值必须由完整地图验证。', '',
        '下一次验收必须统计所有有效 GT 对象的 PQ、TP/FP/FN、错并、碎片、覆盖及计算成本；不以局部损失下降或少量关联准确率代替建图效果。新旧模型都不能接触评测标签来选择 checkpoint。由于这些场景已被查看，后续仍需新的独立数据支撑论文泛化主张。', '',
        '**本轮完成的是全地图诊断与监督基础设施，尚未实现／训练这个局部归属模型，尚无新算法涨点，也没有完成机器人实机验证。**', '',
        '逐对象记录：'+' · '.join(f"[{r['key']}](maps/{r['key']}/objects.json)" for r in maps['scenes']), '',
        '全量核验保留全部 token，逐一检查特征、支持来源和标签分布。原始逐 token 核验因重复解压和成员查找过慢而中止，最终用独立向量化实现完成等价的全量检查，没有抽样或放宽条件。[核验执行记录](verification_runtime.json)。', '',
        '[协议](protocol.json) · [全地图结果](map_summary.json) · [监督清单](supervision_summary.json) · [复核](audit.json) · [数据格式与使用](supervision_schema.md)', '']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n')
    schema=f'''# 局部支持监督格式

根目录：`{OUT/'supervision'}`。按 train / validation / evaluation 分目录。每个场景的实例 ID 仅在该场景内有效；不得跨场景直接比较数值 ID。当前所有场景曾用于项目开发，隔离表示本轮拟合隔离，不是全新测试。

每场景 `manifest.json` 列出所有帧、输入来源与哈希。源几何、相机参数、给定位姿路径见 `protocol.json` 对应 scene 的 `cache`。`point_targets.npz` 的 `instance` 对应该缓存 `geometry.npz` 中的点顺序，-1 为未知，0 为背景。

每帧 NPZ：

| 字段 | 定义 |
|---|---|
| features | 与旧缓存 local_features 数组一致的局部 DINO 特征；不是训练后的特征 |
| offsets / observation_ids | 每个原始观测的 token 范围与不可变观测 ID |
| patch_bins | 原始 resize/14 网格中的行优先 patch 索引 |
| point_ids / point_offsets | 每个 token 的原始 cache geometry 点 ID，CSR 存储 |
| target_instance_ids / target_counts / target_offsets | 每个 token 的完整实例标签直方图，CSR 存储 |
| target_stats | 每个 token 的 [总支持数，有效支持数，前景支持数，前景实例数] |

每帧 JSON 保留所有观测的完整支持分布和源特征文件哈希。没有按 purity 删除混合对象。

训练时仅打开 train；validation 用于选择配置／检查点；evaluation 只能在冻结预测后评分。未知支持没有负标签；背景 0 不是一个可关联的对象；跨视角重复支持高度相关，不能当作独立统计样本。局部实例亲和监督应在同场景、正实例标签之间构造，不把场景内 ID 当作跨场景的固定分类标签。

本次没有启动新模型训练。目标是投影实测支持分布，不是稠密图像分割真值。
'''
    (REPORT/'supervision_schema.md').write_text(schema)
    body=[];table=False;ul=False
    for line in lines:
        if line.startswith('|'):
            if not table:body.append('<table>');table=True
            cells=[v.strip() for v in line.strip('|').split('|')]
            if all(set(c)<=set('-: ') for c in cells):continue
            body.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>');continue
        if table:body.append('</table>');table=False
        if line.startswith('- '):
            if not ul:body.append('<ul>');ul=True
            body.append('<li>'+html.escape(line[2:]).replace('**','')+'</li>');continue
        if ul:body.append('</ul>');ul=False
        if line.startswith('!['):body.append('<img src="coverage.png" alt="全地图覆盖和实际匹配" style="width:100%">')
        elif line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line:body.append('<p>'+html.escape(line).replace('**','')+'</p>')
    object_links=' · '.join(f'<a href="maps/{r["key"]}/objects.json">{r["key"]} 逐对象记录</a>' for r in maps['scenes'])
    (REPORT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>omni_slam 全地图主线</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#172d3b}table{border-collapse:collapse;width:100%;font-size:14px}td{border:1px solid #cbd5df;padding:8px}tr:first-child{background:#e8f0f5;font-weight:bold}h1{font-size:28px}h2{margin-top:36px}</style>'+''.join(body)+'<p>'+object_links+'</p><p><a href="report.md">Markdown</a> · <a href="map_summary.json">全地图结果</a> · <a href="supervision_summary.json">监督数据</a> · <a href="supervision_schema.md">格式</a> · <a href="audit.json">复核</a></p></html>')
    for name in ['protocol.json','map_summary.json','supervision_summary.json','audit.json','verification_runtime.json']:(REPORT/name).write_bytes((OUT/name).read_bytes())
    for r in maps['scenes']:
        dest=REPORT/'maps'/r['key'];dest.mkdir(parents=True)
        for name in ['objects.json','summary.json']:(dest/name).write_bytes((OUT/'maps'/r['key']/name).read_bytes())
    for link in re.findall(r'(?:href|src)="([^"]+)"',(REPORT/'index.html').read_text()):assert (REPORT/link).is_file()
    for link in re.findall(r'\]\(([^)]+)\)',(REPORT/'report.md').read_text()):assert (REPORT/link).is_file()
    dump(OUT/'delivery_audit.json',dict(complete=True,tests_passed=4,report_links_checked=True,
        report_artifact_sha256={str(f.relative_to(REPORT)):sha(f) for f in REPORT.rglob('*') if f.is_file()},
        report_source_sha256=sha(Path(__file__)),audit_sha256=sha(OUT/'audit.json')))
    print('REPORT',REPORT,flush=True)


if __name__=='__main__':main()
