"""Summarize every frozen corruption/seed without selecting favorable runs."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_final_common import ROOT,read,dump,sha


def main():
    source=ROOT/'build/benchmarks/revision_robustness_20261113'
    out=ROOT/'output/revision_robustness';out.mkdir(parents=True,exist_ok=True)
    summary=read(source/'summary.json');assert summary['complete']
    clean=read(ROOT/'build/benchmarks/mask_recall_revision_20261113/summary.json')['results']
    colors={'simple_ratio':'#65758a','mask_recall':'#008f83'}
    names={'simple_ratio':'Strong ratio baseline','mask_recall':'Experimental mask-recall model'}
    kinds=['clean','mask_boundary','depth_noise','pose_noise'];labels=['Clean','Mask boundary','Depth noise','Pose noise']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained');records=[]
    for ax,(scene,conditions) in zip(axes,summary['results'].items()):
        for mi,method in enumerate(colors):
            values=[]
            for kind in kinds:
                if kind=='clean':
                    m='simple_ratio' if method=='simple_ratio' else 'leave_point_out'
                    v=[100*clean[scene][m]['score']['observed_voxel_instance_pq']]
                else:v=[100*r['methods'][method]['score']['observed_voxel_instance_pq'] for r in conditions[kind]]
                values.append(v)
                records.append(dict(scene=scene,condition=kind,method=method,pq100=v,mean=float(np.mean(v)),std=float(np.std(v,ddof=1)) if len(v)>1 else 0.))
            xs=np.arange(4)+(mi-.5)*.25
            ax.errorbar(xs,[np.mean(v) for v in values],yerr=[np.std(v,ddof=1) if len(v)>1 else 0 for v in values],
                        marker='o',markersize=5,color=colors[method],label=names[method],capsize=4,lw=1.6)
            for x,v in zip(xs,values):
                if len(v)>1:ax.scatter(x+np.linspace(-.035,.035,len(v)),v,s=12,color=colors[method],alpha=.55)
        ax.set_xticks(np.arange(4),labels,rotation=15);ax.set_ylabel('Instance PQ (%)')
        ax.set_title('Replica office_0' if scene.startswith('replica') else 'SceneNN 014',fontweight='bold');ax.grid(axis='y',alpha=.16)
    axes[0].legend(frameon=False,loc='lower left',fontsize=9)
    fig.suptitle('Fixed-map observation sensitivity | frozen parameters | three perturbation seeds',fontsize=12)
    fig.savefig(out/'sensitivity.png',dpi=200);fig.savefig(out/'sensitivity.svg');plt.close(fig)
    dump(out/'results.json',dict(records=records,summary_sha256=sha(source/'summary.json'),
                               protocol_sha256=sha(source/'protocol.json'),error_bars='Sample SD across three seeds, not confidence intervals or independent scenes.'))
    rows=[]
    for scene in summary['results']:
        for kind in kinds:
            a=next(r for r in records if r['scene']==scene and r['condition']==kind and r['method']=='simple_ratio')
            b=next(r for r in records if r['scene']==scene and r['condition']==kind and r['method']=='mask_recall')
            rows.append(f"| {scene} | {kind} | {a['mean']:.3f} ± {a['std']:.3f} | {b['mean']:.3f} ± {b['std']:.3f} | {b['mean']-a['mean']:+.3f} |")
    text='''# 固定地图的观测扰动评测

范围：两个已使用的开发场景；三类扰动、每类三个固定种子、两种方法，共 36 个新后处理地图。原始地图和对象归属不变，仅扰动修订阶段的输入。它不是从受扰传感器重新执行完整 SLAM 的结果。

- 掩码边界：每个原掩码独立腐蚀，半径从 1、2、3 像素均匀采样。
- 深度：有效像素增加标准差 2 cm 的零均值高斯噪声，未知深度保持未知。
- 位姿：相机局部平移各轴标准差 1 cm、旋转向量各轴标准差 1°。
- 统一使用种子 1101、1102、1103。配置预先冻结，两场景全部输出封存后统一评分。

| 场景 | 条件 | 强对照 simple ratio PQ | 候选漏分模型 PQ | 差值 |
|---|---|---:|---:|---:|
'''+ '\n'.join(rows)+'''

数值为百分制均值 ± 种子间样本标准差；clean 只有一个固定结果，±0 不表示统计不确定性为零。三个种子不是三个独立场景。

结论：候选模型在 office_0 的掩码边界扰动下有一致收益，但在 014 不能复现；深度与位姿扰动下整体仍不及比例过滤。office_0 中比例过滤的 PQ 在位姿扰动后反而上升，说明该评分下删除/最近邻重新分配可能受噪声影响；不能据此宣称位姿误差改善了真实几何。所有条件均保留，不将单一有利扰动选作主结论。

这是一份内部机制验证记录，不替代正式论文的新颖性或独立泛化证据。用户已要求优先完成公开 benchmark；下一步按冻结版本执行预留 Hypersim 两场景，车库独立标注暂缓。

图：`output/revision_robustness/sensitivity.png` 与 `.svg`。
权威结果与预测：`build/benchmarks/revision_robustness_20261113/`。
'''
    (ROOT/'docs/reports/revision_robustness_2026-11-13.md').write_text(text)
    print(out/'sensitivity.png')


if __name__=='__main__':main()
