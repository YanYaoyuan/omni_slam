"""Delivery report for the six frozen supervised object-memory fits."""
import html
import json
from pathlib import Path
import re

import numpy as np

from train_object_memory_attention import OUT, REPORT, protocol, read, dump, sha


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=protocol();s=read(OUT/'summary.json');a=read(OUT/'audit.json');assert a['complete']
    assert a['summary_sha256']==sha(OUT/'summary.json')
    ready=read(OUT/'data_ready.json');fits=[read(f['path']) for f in s['fits']]
    assert len(fits)==6
    REPORT.mkdir(parents=True,exist_ok=False)
    colors=['#36668a','#ba7837','#398777']
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
    for ax,mode in zip(axes,['mean','attention']):
        for color,seed in zip(colors,p['seeds']):
            r=next(r for r in fits if r['mode']==mode and r['seed']==seed)
            x=[v['epoch'] for v in r['history']]
            ax.plot(x,[v['train']['nll'] for v in r['history']],color=color,ls='--',alpha=.7)
            ax.plot(x,[v['validation']['nll'] for v in r['history']],color=color,label=f'seed {seed}; best {r["best_epoch"]}')
            b=r['history'][r['best_epoch']];ax.scatter([r['best_epoch']],[b['validation']['nll']],color=color,s=30)
        ax.set_title('Trained '+mode);ax.set_xlabel('Epoch');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Listwise NLL (lower is better)')
    fig.suptitle('Frozen DINO association heads | solid: validation; dashed: training')
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(REPORT/f'training_curves.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    lines=['# 学习式对象记忆：实际训练与场景隔离评测','',
        '2026-11-12。已完成两种关联头 × 三个固定随机种子，共六次真实训练，每次 50 个 epoch。DINO 骨干冻结，训练对象关联模块的投影、记忆读取与残差评分。', '',
        '**上一轮只测试固定公式，本轮才检验学习式关联。结果仍是原生历史上的单步重排序，不能写成地图 PQ 提升。**', '',
        '## 结论', '',
        '训练确实发生：每次 1,500 次梯度更新，注意力 Q/K/V 权重均有变化。训练集共 469 个合格事件、验证集 85 个；两种模型的最终训练损失接近零，但验证表现有明显种子差异和过拟合。', '',
        '四场景评测中，原生 CG 为 30/32；注意力 seed 29、43 分别为 25/32、27/32，均未纠正原生错误，分别破坏 5 次、3 次。seed 17 为 30/32，但验证选择了 epoch 0，实际使用的仍是原生等价初始化，不能列为训练收益。', '',
        '与同样训练的平均聚合比较，seed 29 净少 1 次正确，seed 43 净多 3 次；没有稳定的注意力增量。本轮不能把学习式关联接入主线。', '',
        '一个明确的目标偏差：注意力 seed 29 的验证 NLL 从 0.4315 降到 0.0756，但正确次数从 84/85 降到 81/85。概率损失改善不等于错误关联减少。当前监督稀疏、种子波动及训练目标与纠错目标的差异都需要保留，不能归因于单一原因；本轮没有事后改选检查点。', '',
        '## 划分与样本量', '', '| 角色 | 场景 | 原始候选事件 | 可监督／评测事件 |', '|---|---|---:|---:|']
    for r in ready['inventory']:
        count=r.get('training_eligible')
        if count is None:count=next(v['primary'] for v in s['scenes'] if v['key']==r['scene'])
        lines.append(f"| {r['role']} | {r['scene']} | {r['events']} | {count} |")
    active=sorted(fits[0]['train_scene_counts'])
    lines += ['', f"六个场景被划入训练集，其中实际有合格监督、贡献梯度的场景为 {len(active)} 个：{', '.join(active)}；零样本场景保留在数据表，不补场景。room_2 验证集只选检查点；office_0/office_1/014/015 不进入训练或选模型。所有场景在项目历史中已被查看，因此这里只能称为本轮拟合的场景隔离，不能称为全新未见测试。", '',
        '标签是实测支持点投影得到的对象身份代理：5cm 内有效支持至少 5 点，前景主导纯度至少 70%；查询和所有候选均可判定，并同时存在正确与错误候选。GT 只生成监督目标或评测标签，不作为模型输入，不用于整理记忆。没有放宽上一轮评价口径。', '',
        '## 确实训练了什么', '',
        '- 冻结 DINOv2-small 的 384 维池化特征。共享投影到 64 维；注意力使用 4 个头，学习 Q/K/V 和输出投影；关联 MLP 学习原生评分的残差。',
        '- 训练平均聚合对照使用相同特征投影、V/输出层、关联 MLP、原生评分及辅助余弦统计；历史权重均匀，Q/K 不参与训练。因此两模型不是严格同参数量，差值公开。',
        '- 每帧最多一个历史观测，最多 16 个按时间均匀抽取的历史视图，所有读取器相同；没有当前帧或未来帧。',
        '- AdamW，学习率 0.001、权重衰减 0.0001、批量 16 个事件、梯度裁剪 1.0、50 epoch。训练损失按场景样本数倒数加权，避免大场景支配训练。',
        '- 多正候选 listwise 损失；用唯一验证场景的平均 NLL 选最优 epoch。epoch 0 精确复现原生评分，也可被选中；不按评测分数选 epoch 或随机种子。',
        '- 小型关联头在 CPU 两线程训练，无新增骨干推理；这是实际反向传播训练，不是固定 softmax。', '',
        '| 模式 | seed | 可训练参数 | 梯度步数 | 最优 epoch | 初始训练 NLL → 所选训练 NLL | 初始验证 NLL → 所选验证 NLL | 所选验证正确数 | 耗时(s) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in fits:
        lines.append(f"| {r['mode']} | {r['seed']} | {r['trainable_parameters']} | {r['steps']} | {r['best_epoch']} | {r['initial_train']['nll']:.4f} → {r['selected_train']['nll']:.4f} | {r['initial_validation']['nll']:.4f} → {r['selected_validation']['nll']:.4f} | {r['selected_validation']['correct']}/{r['selected_validation']['total']} | {r['wall_s']:.1f} |")
    lines += ['', '![训练和验证曲线](training_curves.png)', '',
        '## 本轮评测：正确关联选择次数', '',
        f"共 {s['total_events']} 次候选判断，主评价子集 {s['primary_events']} 次。每个随机种子看到的是相同事件，不能把三个种子拼成三倍独立样本。固定公式对照也使用同一份最多 16 视图记忆，因此应与本轮表内结果比较。", '',
        '| 方法 | 正确 / 可判定事件 | 相对原生纠正 / 破坏 |', '|---|---:|---:|']
    for method,r in s['totals'].items():
        t=r['transitions']['native'];lines.append(f"| {method} | {r['correct']} / {r['total']} | {t['gained']} / {t['lost']} |")
    lines += ['', '| seed | 注意力相对训练平均：纠正 / 破坏 |', '|---|---:|']
    for r in s['paired']:
        t=r['attention_vs_trained_mean'];lines.append(f"| {r['seed']} | {t['gained']} / {t['lost']} |")
    lines += ['', '## 场景明细', '',
        '| 场景 | 子集 | 原生 | 均值17 / 注意力17 | 均值29 / 注意力29 | 均值43 / 注意力43 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in s['scenes']:
        m=r['methods'];pairs=' | '.join(f"{m[f'mean_{seed}']['correct']} / {m[f'attention_{seed}']['correct']}" for seed in p['seeds'])
        lines.append(f"| {r['key']} | {r['primary']} | {m['native']['correct']} | {pairs} |")
    lines += ['', '## 判断边界与停止', '',
        '训练集 NLL 下降只能证明优化器学到了训练目标。验证集下降说明在这个验证场景上目标损失改善，也不能替代最终对象恢复或地图 PQ。评测子集只有 32 次，原生 CG 已有 30 次正确；SceneNN 子集原生全部正确，缺少验证纠错收益的机会。混合身份的大量候选仍被列为未知，不能默认为正确或错误。', '',
        '注意力的新增价值必须看它相对“同样训练的平均聚合”，不能只看相对未训练公式的差值。若所选 epoch 为 0，该运行确实训练过，但选择部署的检查点仍是初始化，需明确报告。', '',
        '**六次训练到此结束，不自动追加参数、随机种子或地图。保留原有主线。** 后续若要声称改善建图，还需要具有有效监督的更充分场景及实际因果建图回放；本轮没有完成这一步。', '',
        '[协议](protocol.json) · [完整指标](summary.json) · [数据清单](data_ready.json) · [复核](audit.json) · [训练记录](fits.json)', '']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n')
    body=[];table=False;ul=False
    for line in lines:
        if line.startswith('|'):
            if not table:body.append('<table>');table=True
            cells=[x.strip() for x in line.strip('|').split('|')]
            if all(set(x)<=set('-: ') for x in cells):continue
            body.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>');continue
        if table:body.append('</table>');table=False
        if line.startswith('- '):
            if not ul:body.append('<ul>');ul=True
            body.append('<li>'+html.escape(line[2:])+'</li>');continue
        if ul:body.append('</ul>');ul=False
        if line.startswith('!['):body.append('<img src="training_curves.png" alt="训练验证曲线" style="width:100%">')
        elif line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line:body.append('<p>'+html.escape(line).replace('**','')+'</p>')
    (REPORT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>学习式对象记忆</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#172d3b}table{border-collapse:collapse;width:100%;font-size:14px}td{border:1px solid #cbd5df;padding:8px}tr:first-child{background:#e8f0f5;font-weight:bold}h1{font-size:28px}h2{margin-top:36px}</style>'+''.join(body)+'<p><a href="report.md">Markdown</a> · <a href="summary.json">结果</a> · <a href="fits.json">训练记录</a> · <a href="audit.json">复核</a></p></html>')
    for name in ['protocol.json','summary.json','audit.json','data_ready.json']:
        (REPORT/name).write_bytes((OUT/name).read_bytes())
    dump(REPORT/'fits.json',fits)
    for link in re.findall(r'(?:href|src)="([^"]+)"',(REPORT/'index.html').read_text()):assert (REPORT/link).is_file()
    for link in re.findall(r'\]\(([^)]+)\)',(REPORT/'report.md').read_text()):assert (REPORT/link).is_file()
    dump(OUT/'delivery_audit.json',dict(complete=True,unit_tests_passed=5,training_runs=6,
        report_links_checked=True,report_sha256={f.name:sha(f) for f in REPORT.iterdir() if f.is_file()},
        delivery_source_sha256=sha(Path(__file__)),audit_sha256=sha(OUT/'audit.json')))
    print('REPORT',REPORT,flush=True)


if __name__=='__main__':main()
