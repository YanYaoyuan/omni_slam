"""Scientific figures and independent rank-metric checks for the sealed audits."""
import html
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from audit_censored_partition_objective import OUT as OBJECTIVE
from audit_mask_hypothesis_information import OUT as INFORMATION
from evaluate_censored_partition import OUT as PARENT

DEST=ROOT/'output/objective_information_audit_v1'
NAMES=dict(hard='Hard winner',alternatives='All native hypotheses',geometry='Geometric distance',categorical='Categorical gain')


def rank_metrics(y,score):
    """Threshold-group AP; exact pairwise rank AUC, without sklearn."""
    y=np.asarray(y,bool);score=np.asarray(score,float)
    if not y.any() or y.all():return dict(ap=None,auc=None)
    order=np.argsort(-score,kind='stable');ranked=y[order];values=score[order]
    end=np.r_[np.flatnonzero(values[:-1]!=values[1:]),len(values)-1]
    tp=np.cumsum(ranked)[end];precision=tp/(end+1)
    ap=float(np.sum(np.diff(np.r_[0,tp])*precision)/y.sum())
    negatives=np.sort(score[~y]);pos=score[y]
    below=np.searchsorted(negatives,pos,side='left');above=np.searchsorted(negatives,pos,side='right')
    auc=float(np.sum(below+.5*(above-below))/(y.sum()*len(negatives)))
    return dict(ap=ap,auc=auc)


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    obj=read(OBJECTIVE/'summary.json');info=read(INFORMATION/'summary.json')
    assert obj['complete'] and info['complete'];assert not DEST.exists()
    for s in [obj,info]:
        for f,h in s['files'].items():assert sha(f)==h,f
    DEST.mkdir(parents=True);strata={};checks=0;parts=[]
    fig,axes=plt.subplots(2,1,figsize=(14,7),constrained_layout=True)
    for ax,(key,result) in zip(axes,obj['results'].items()):
        rows=result['objects'];values=np.array([r['whole_minus_fragments'] for r in rows])
        ax.bar(np.arange(len(rows)),values,color=np.where(values < -1e-8,'#d96a52','#14947e'))
        ax.set_xticks(np.arange(len(rows)),[str(r['gt_id']) for r in rows],rotation=90,fontsize=7)
        ax.axhline(0,c='#355269',lw=.7);ax.set_yscale('symlog',linthresh=1.)
        ax.set_ylabel('Whole minus fragmented score\n(symlog; higher favors whole)')
        ax.set_title(key+' / every reference object on the fixed unanimous-support subset')
    fig.savefig(DEST/'objective.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,(key,result) in zip(axes,info['results'].items()):
        methods=list(NAMES);values=[result['statistics']['all'][m]['roc_auc'] for m in methods]
        ax.bar(np.arange(4),values,color=['#7e98b0','#14947e','#c99535','#bb7388'])
        ax.set_xticks(np.arange(4),[NAMES[m] for m in methods],rotation=15,ha='right',fontsize=9)
        ax.set_ylim(0,1.05);ax.set_ylabel('Same-object pair ROC AUC');ax.set_title(key)
        for i,v in enumerate(values):ax.text(i,v+.02,f'{v:.3f}',ha='center')
        ax.axhline(.5,c='#555',lw=.7,ls='--')
    fig.suptitle('Diagnostic pair ranking, not benchmark AP / correlated, selected edges')
    fig.savefig(DEST/'information.png',dpi=170);plt.close(fig)
    for key,result in info['results'].items():
        q=dict(np.load(INFORMATION/key/'pairs.npz'));d=np.load(PARENT/'prepared'/key/'atoms.npz')
        local=np.isin(q['left']*len(d['cell_ids'])+q['right'],d['local_edges'])
        groups=dict(all=np.ones(len(local),bool),building=q['train_class']==2,car=q['train_class']==13,
                    local=local,query_only=~local)
        strata[key]={};table=''
        for group,take in groups.items():
            y=q['same_gt'][take];entries={}
            for m in NAMES:
                v=rank_metrics(y,q[m][take]);inverse=rank_metrics(~y,-q[m][take])
                if group in result['statistics']:
                    expected=result['statistics'][group][m]
                    np.testing.assert_allclose([v['ap'],v['auc']],
                        [expected['average_precision'],expected['roc_auc']],atol=1e-12,rtol=0);checks+=2
                entries[m]=dict(same_object_ap=v['ap'],roc_auc=v['auc'],different_object_ap=inverse['ap'])
                table+=f'<tr><td>{group}</td><td>{NAMES[m]}</td><td>{int(y.sum())} / {int((~y).sum())}</td>'+''.join(
                    f'<td>{100*x:.3f}</td>' if x is not None else '<td>N/A</td>' for x in [v['ap'],v['auc'],inverse['ap']])+'</tr>'
            strata[key][group]=dict(pairs=len(y),same_object=int(y.sum()),different_object=int((~y).sum()),methods=entries)
        parts.append(f'<h2>{key}</h2><table><tr><th>Edge subset</th><th>Readout</th><th>Same / different</th><th>Same AP %</th><th>AUC %</th><th>Different AP %</th></tr>{table}</table>')
        parts.append(f'<p>Previously exported complete-map visualization; no new map:</p><img src="../censored_object_partition_v1/{key}_rgb.png">')
        rows=''
        for r in obj['results'][key]['objects']:
            rows+=f'<tr><td>{r["gt_id"]}</td><td>{r["atoms"]}</td><td>{r["predicted_fragments"]}</td><td>{r["induced_graph_components"]}</td><td>{r["whole_minus_fragments"]:.6f}</td><td>{r["views_with_multiple_tokens"]} / {r["observed_views"]}</td></tr>'
        parts.append('<details><summary>Every object diagnostic</summary><table><tr><th>Raw target ID</th><th>Atoms</th><th>Fragments</th><th>Graph components</th><th>Whole − fragments</th><th>Multi-token / all views</th></tr>'+rows+'</table></details>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam：对象目标与观测信息复核</title>
<style>body{max-width:1300px;margin:40px auto;padding:0 24px;font:16px/1.6 sans-serif;color:#1e3546}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:6px;border-bottom:1px solid #d8e1e4;text-align:right}td:first-child,th:first-child{text-align:left}.note{background:#fff1d9;padding:18px}details{margin:20px 0;overflow:auto}</style>
<h1>omni_slam：对象划分目标与原始掩码信息复核</h1>
<p class="note">结论：当前类别计数目标与物理对象身份错位。完整对象在许多案例中反而被打低分，不能靠继续优化这个目标解决。保留全部原始掩码候选改善了配对信息，但仍明显落后于同域的几何距离控制；不能直接当作新方法或 benchmark 涨点。强基线不变。</p>
<p>所有比较基于两个已暴露的 KITTI360 开发窗口。参考标签仅用于诊断；没有新地图、新模型推理、训练、机器人或独立场景评分。本轮不改变旧候选、旧主指标和验收门槛。</p>
<h2>1. 完整对象是否获得更好的内部目标？</h2>
<p>仅保留一个单元内全部实测点最近参考标签相同、且与预测类别一致的单元。两个窗口分别纳入 25,032/47,761、18,642/54,683 个单元，其余明确排除。这是参考投影一致的子集，不是对原始三维真值的无误保证。相同观测分别按完整参考对象和预测碎片聚合。负值表示目标偏好碎片。</p>
<img src="objective.png">
<p>只看同一参考对象且候选图内连通的多碎片部分，合回整体的目标在 43/51、34/36 例中下降。把混入多个参考对象的预测区域拆开，43/48、23/27 例的目标也下降。因此，这个目标可能同时奖励错误拆分与错误合并。它不是单纯缺少搜索路径；子集干预仍不能量化全部 AP 下降的因果份额。</p>
<h2>2. 最高分掩码之外还有信息吗？</h2>
<p>固定同一批跨预测区域边、同一批共同可见帧，比较最高分掩码、全部原生候选的分数加权直方图、初始类别合并分数与几何距离。全部候选通过预设的信息保留门槛，但只说明比最高分选择保留了更多区分信息。没有证明超过几何强控制，也没有生成新的语义地图。</p>
<img src="information.png">
<p>此域极不均衡：同对象边占约98%；高配对 AP 不能解释为高建图精度。几何控制很强，但域排除了非目标/混合单元；局部边只有2/4条跨真实对象负例，不能推广到完整背景污染与独立场景。以下保留类别、局部/非局部来源和反向“不同对象”检测 AP，避免隐藏这些限制。局部/非局部分层是事后解释，不改变原门槛。</p>'''+''.join(parts)+'''
<h2>主线决定</h2><p>关闭当前类别计数目标，不继续换先验或调合并阈值。下一机制需要用真实几何连续性与多视图观测解释来判断对象整体，而不能把两个不同掩码直接当作两个物体的证据；原始候选保留只能作为输入选择。必须纳入几何控制，并在完整地图同时检验新对象恢复、错误实例和历史对象损伤，达到强基线门槛后再扩展同版本机器人和独立室外评测。尚未宣称这构成有效新机制或新贡献。</p>
<p>核验：所有聚合目标通过独立特殊函数重算；432个原始掩码读出通过标量核对；配对AP/AUC独立排序实现与原结果一致。原始坐标与评测局部坐标的检查首次遗漏平移，执行立即中止，修正为原job的origin_world并精确验证后完成；首次失败保留，未产生诊断分数或新地图。</p></html>'''
    (DEST/'index.html').write_text(page)
    dump(INFORMATION/'stratified_readout_audit.json',dict(complete=True,posthoc_strata=True,
        results=strata,independent_rank_metric_checks=checks,source_sha256=sha(__file__),
        original_summary_sha256=sha(INFORMATION/'summary.json'),no_gate_changed=True))
    dump(OBJECTIVE/'report_audit.json',dict(complete=True,source_sha256=sha(__file__),
        objective_summary_sha256=sha(OBJECTIVE/'summary.json'),information_summary_sha256=sha(INFORMATION/'summary.json'),
        files={str(f):sha(f) for f in DEST.iterdir()}))
    print('OBJECTIVE AND INFORMATION REPORT',DEST/'index.html','rank checks',checks,flush=True)


if __name__=='__main__':run()
