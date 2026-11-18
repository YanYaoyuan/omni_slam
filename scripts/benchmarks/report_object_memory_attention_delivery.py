"""Add result interpretation and a count-based plot to the sealed probe report.

This delivery helper is separate from frozen prediction/evaluation sources.
It never changes decisions, labels, metrics, gates or historical artifacts.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'build/benchmarks/object_memory_attention_20261112'
REPORT = ROOT / 'docs/reports/object_memory_attention_2026-11-12'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = json.loads((OUT/'summary.json').read_text())
    assert json.loads((OUT/'audit.json').read_text())['complete']
    counts = {f: {m: sum(d['methods'][f][m]['correct'] for d in s['datasets'].values())
                  for m in ['native','mean','nearest','attention']} for f in ['siglip','dino']}
    total=sum(d['total'] for d in s['datasets'].values())
    primary=sum(d['primary'] for d in s['datasets'].values())
    assert (total,primary)==(1050,32)
    assert counts==dict(siglip=dict(native=30,mean=22,nearest=27,attention=24),
                       dino=dict(native=30,mean=26,nearest=27,attention=27))
    unresolved=sum(r['counts']['unresolved'] for r in s['scenes'])
    same=sum(r['counts']['resolved_all_positive'] for r in s['scenes'])
    absent=sum(r['counts']['resolved_no_positive'] for r in s['scenes'])
    assert (unresolved,same,absent)==(727,289,2)
    md='''## 阅读结论：样本覆盖不足，固定注意力没有显示额外收益

四场景全部 1,050 次候选判断已完成；其中 727 次因查询或候选的实例身份代理不够纯而无法可靠判对错，289 次所有候选都是同一身份代理，2 次候选中没有正确身份。真正能比较对象区分能力的共同子集只有 **32 次（3.05%）**。SceneNN 这 10 次中，原生 CG 全部正确，没有可验证的纠错机会；因此预先约定的覆盖门槛未通过。

下表为这 **32 次事件的正确选择次数**，不是地图对象数量、实例 PQ 或总体准确率：

| 特征 | 原生 CG | 平均聚合 | 最近邻检索 | 固定注意力 |
|---|---:|---:|---:|---:|
| SigLIP | 30/32 | 22/32 | 27/32 | 24/32 |
| DINOv2 | 30/32 | 26/32 | 27/32 | 27/32 |

![固定可判定子集：正确关联选择次数](comparison.png)

在这组样本上，保留多视角后做最近邻，比平均聚合多选对一些事件；但固定注意力没有超过最近邻，三种纯外观重排序都没有超过原生 CG。DINO 注意力相对原生共纠正 1 次、破坏 4 次。不能用“比平均高”来证明应替换主线。

注意：下文保留预先约定的场景等权均值。Replica 两场景分别只有 20 和 2 个有效事件，因此等权均值与合并事件计数可能呈现相反方向；不能挑选有利统计。原始分场景计数及纠正／破坏均公开。

**决定：保留现有主线，不部署此读取器，不追加温度搜索或训练。** 本轮得到的是“现有池化特征＋固定注意力在可判定子集上没有额外优势，且评价覆盖不足”，而不是“所有注意力无效”。本试验没有训练可学习的 Q/K/V，也没有测局部 DINO token 注意力。高比例混合／未知候选说明当前诊断不能回答最困难的对象纠错问题；无法判定不等于模型判断错误。

'''
    original=(REPORT/'report.md').read_text()
    assert '## 阅读结论' not in original
    lines=original.splitlines(keepends=True)
    (REPORT/'report.md').write_text(lines[0]+'\n'+md+''.join(lines[1:]))
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    colors=['#34546c','#9bacba','#c68b43','#138f82']
    for ax,f in zip(axes,['siglip','dino']):
        vals=list(counts[f].values())
        ax.bar(['Native CG','Mean','Nearest','Attention'],vals,color=colors,width=.65)
        for i,v in enumerate(vals):ax.text(i,v+.5,f'{v}/32',ha='center',fontsize=10)
        ax.set_title('SigLIP' if f=='siglip' else 'DINOv2-small')
        ax.set_ylim(0,35);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Correct choices in fixed proxy-resolvable subset')
    fig.suptitle('Object-memory probe: only 32 of 1,050 events are discriminative and resolvable',fontsize=12)
    fig.tight_layout()
    for suffix in ['png','svg']:fig.savefig(REPORT/f'comparison.{suffix}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    html=(REPORT/'index.html').read_text()
    table='<table><tr><td>特征</td><td>原生 CG</td><td>平均</td><td>最近邻</td><td>注意力</td></tr>'
    for f in counts:
        table+='<tr><td>'+f+'</td>'+''.join(f'<td>{v}/32</td>' for v in counts[f].values())+'</tr>'
    table+='</table>'
    note='<section><h2>结论：覆盖不足，固定注意力未显示额外收益</h2><p>1,050 次候选判断全部完成；727 次身份代理无法可靠判定，289 次所有候选均同一身份，2 次没有正确候选。主评价子集只有 32 次（3.05%）。以下是正确选择次数，不是 PQ 或总体准确率。</p>'+table+'<img src="comparison.png" alt="固定子集正确选择次数" style="width:100%;margin:20px 0"><p>DINO 注意力比平均多选对 1 次，但与最近邻持平，低于原生 CG；相对原生纠正 1 次、破坏 4 次。保留现有主线，不追加调参或训练。</p><p>下面的预定结果为场景等权均值。Replica 两场景有效事件数为 20 与 2，等权均值与合并计数可能呈相反方向，不能选择有利统计。该试验是固定 Q/K/V 的单步重排序，未验证学习式或局部 token 注意力；混合身份候选的高比例也限制结论。</p></section>'
    pos=html.index('</h1>')+len('</h1>')
    (REPORT/'index.html').write_text(html[:pos]+note+html[pos:])
    for name in ['protocol.json','summary.json','audit.json']:
        assert (REPORT/name).read_bytes()==(OUT/name).read_bytes()
    html=(REPORT/'index.html').read_text()
    for link in re.findall(r'(?:href|src)="([^"]+)"',html):
        assert (REPORT/link).is_file(),link
    for link in re.findall(r'\]\(([^)]+)\)',(REPORT/'report.md').read_text()):
        assert (REPORT/link).is_file(),link
    result=dict(complete=True, prior_frozen_v1_lightweight_check_passed=True,
                unit_tests_passed=7, all_events=total, primary_events=primary,
                sources_and_predictions_unchanged=True, no_new_maps=True,
                report_links_checked=True,
                audit_sha256=sha(OUT/'audit.json'),summary_sha256=sha(OUT/'summary.json'),
                delivery_source_sha256=sha(Path(__file__)),
                report_sha256={p.name:sha(p) for p in REPORT.iterdir() if p.is_file()})
    (OUT/'delivery_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('DELIVERY',REPORT,flush=True)


if __name__=='__main__':main()
