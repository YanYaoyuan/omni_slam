"""Report actual measured opportunities separately from accepted detections."""
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from audit_object_observation_opportunities import OUT

REPORT = ROOT/'output/prethreshold_candidates_v1'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = read(OUT/'summary.json'); p = read(OUT/'protocol.json'); assert s['protocol_sha256'] == sha(OUT/'protocol.json')
    REPORT.mkdir(parents=True, exist_ok=True); table = ''; figures = ''
    for key, objects in s['results'].items():
        for obj in objects:
            counts = obj['best_raw_iou_over_half_frames']; table += f'<tr><td>{key}</td><td>{obj["gt_id"]}</td><td>{obj["measured_visible_frames"]}</td><td>{obj["old_pure_deferred_frames"]}</td><td>{counts["high"]}</td><td>{counts["deferred"]}</td><td>{counts["any_positive"]}</td></tr>'
            frames = obj['frames']; x = np.array([r['frame_id'] for r in frames]); visible = np.array([r['visible_sources'] for r in frames]); fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.3, 1]))
            axes[0].bar(x, visible, width=3, color='#6089a4'); axes[0].axhline(16, color='#bd7448', linestyle=':', linewidth=1); axes[0].set_ylabel('Visible sources'); axes[0].set_yscale('symlog', linthresh=16)
            for group, color in [('high', '#1976ad'), ('deferred', '#e79b39'), ('any_positive', '#188c78')]:
                value = [r.get('best_raw_queries', {}).get(group) for r in frames]; y = [a['measured_iou'] if a is not None and r['enough_measured_support'] else np.nan for r,a in zip(frames,value)]
                axes[1].plot(x, y, marker='.', linewidth=1, color=color, label=group)
            axes[1].axhline(.5, color='#888888', linestyle=':', linewidth=1); axes[1].set_ylim(0,1.05); axes[1].set_ylabel('GT-assisted\nmeasured IoU'); axes[1].legend(fontsize=8, ncol=3, loc='lower left')
            values = [r.get('best_raw_queries', {}).get('any_positive') for r in frames]
            scores = [a['score'] if a is not None and r['enough_measured_support'] else np.nan for r,a in zip(frames,values)]
            axes[2].plot(x, scores, marker='.', linewidth=1, color='#188c78'); axes[2].set_yscale('log'); axes[2].axhline(.5,color='#1976ad',linestyle='--',linewidth=1); axes[2].axhline(.1,color='#e79b39',linestyle=':',linewidth=1); axes[2].set_ylabel('Native score of\nGT-best raw query'); axes[2].set_xlabel('Frame ID')
            for ax in axes: ax.grid(alpha=.2)
            fig.suptitle(key+f' | diagnostic GT {obj["gt_id"]} | visibility is not detection count',fontsize=11); fig.tight_layout(); name=f'{key}_opportunities_{obj["gt_id"]}.png'; fig.savefig(REPORT/name,dpi=150,bbox_inches='tight'); plt.close(fig)
            figures += f'<h3>{key} · GT {obj["gt_id"]}</h3><img src="{name}">'
    (REPORT/'observability.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>对象可见机会与候选数量</title><style>body{{font:16px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;color:#193349}}img{{max-width:100%}}td,th{{border:1px solid #d5dfe8;padding:8px;text-align:right}}table{{border-collapse:collapse;width:100%}}.note{{padding:20px;background:#edf5f7;border-left:5px solid #058e85}}</style><h1>纠正判断：少候选不等于少观测</h1><p class="note">0002 的建筑 2011 / 2012 分别有 <b>8 / 7 帧</b>有效实测可见。0.1～0.5、独占像素与连通处理后，事后纯候选只有 1 / 2 帧；这不能表述为对象只被看见一两次。查看不经 0.1 截断的原生非零查询，分别有 <b>4 / 3 帧</b>能找到测量点 IoU >0.5 的同类掩码。</p><table><tr><th>窗口</th><th>GT</th><th>≥16 实测可见点的帧</th><th>旧审计纯低分帧</th><th>高分原始掩码 IoU>0.5</th><th>0.1～0.5 原始掩码 IoU>0.5</th><th>全部非零原始掩码 IoU>0.5</th></tr>{table}</table><p>三列 IoU 均在当前可见且标签可信的实测源点上计算；按原始重叠掩码独立计算，未经像素独占或连通截断。与“旧审计纯低分帧”定义不同，不能直接相减作单一机制因果归因。每帧最佳掩码由 GT 事后选择，表示信息存在，不是可部署的检索结果。原生零分/已抑制查询未计入。</p><h2>全部六个诊断对象的时间序列</h2><p>目标集合按上一审计预定规则确定：原 AP50 未匹配、加入低分后必要连通覆盖超过一半的全部对象。图中保留不可见帧和弱分数。虚线 0.5 / 0.1 仅标出已有筛选边界，没有据此寻找新阈值。</p>{figures}<h2>对下一机制的约束</h2><p>不将旧“0.1 入口＋三帧纯候选”门槛改判通过，也不据此放宽为一帧接纳。合理的下一项核查是：保持三维物理假设，通过实测可见区域读取别的视角的原始查询，是否无需 GT 就能取回这些证据，同时区分背景与重复假设。为此正在实现独立的证据读取模块；它还没有对象接纳、完整地图、机器人或独立 benchmark 成绩。</p><p><a href="opportunities.json">完整逐帧诊断</a> · <a href="index.html">返回筛选前候选报告</a></p></html>''')
    dump(REPORT/'opportunities.json',s)
    print('REPORT',REPORT/'observability.html',flush=True)


if __name__=='__main__':main()
