"""Render frozen diagnostic coverage; not method benchmark bars."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_final_common import ROOT,read,sha,dump

root=ROOT/'build/benchmarks/strong_baseline_support_audit_v1'
summary=read(root/'summary.json')
fig,axes=plt.subplots(1,2,figsize=(13,5))
stages=['all_measured','any_native','high_native','final_same_class','best_actual_iou']
labels=['Measured pool','All native masks','High-score masks','Final geometry','Actual IoU > 0.5']
for ax,category in zip(axes,['building','car']):
    largest=0
    for j,(key,result) in enumerate(summary['results'].items()):
        counts=result['counts'][category];values=[counts['thresholds']['0.5'][s] for s in stages]
        largest=max(largest,max(values))
        bars=ax.bar(np.arange(5)+(j-.5)*.36,values,.36,label=key+' / GT '+str(counts['gt_count']),color=['#168f92','#dd9862'][j])
        ax.bar_label(bars,padding=2)
    ax.set_ylim(0,largest*1.35)
    ax.set_xticks(range(5),labels,rotation=25,ha='right');ax.set_ylabel('GT objects');ax.set_title(category)
    ax.legend();ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Coverage inventory (first four stages) vs actual best IoU (last stage)\nCoverage is necessary, not attainable AP or object recall')
fig.tight_layout();dest=ROOT/'output/strong_baseline_support_audit_v1';dest.mkdir(parents=True,exist_ok=True)
fig.savefig(dest/'coverage.png',dpi=150);plt.close(fig)
dump(dest/'manifest.json',dict(source_sha256=sha(__file__),input_sha256=sha(root/'summary.json'),figure_sha256=sha(dest/'coverage.png')))
