"""Read-only reporting of frozen Hypersim results; never selects configurations.

Preview mode uses RGB and model predictions only. Full reporting requires the
completed score file and verifies sealed predictions before reading references.
"""
import argparse
import html
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_final_common import ROOT, read, dump, sha
import run_hypersim_benchmark as bench

OUT = ROOT/'output/hypersim_independent'
NAMES = {
    'cg_shared': 'CG (shared frontend)', 'history2': 'History2 (strong baseline)',
    'sor': 'History2 + SOR', 'visibility1': 'Ours (frozen visibility1)',
    'simple_ratio': 'Simple ratio (strong control)', 'mc_point_ratio': 'MC-style point ratio',
    'recall_pooled': 'Candidate: pooled recall', 'recall_per_view': 'Candidate: per-view recall',
    'recall_leave_point_out': 'Candidate: leave-point-out recall',
    'maskclustering_shared': 'MaskClustering (shared frontend)',
}


def color(ids):
    ids = np.asarray(ids)
    rgb = plt.get_cmap('turbo')((ids * .61803398875) % 1)[..., :3]
    rgb[ids <= 0] = [.88, .89, .90]
    return rgb


def previews(protocol):
    """Frames 0/50/99, fixed before scores; per-frame mask colors are not IDs."""
    files = []
    for scene, path in protocol['jobs'].items():
        job = read(path); folder = Path(job['output'])
        status = folder/'masks_run.json'
        if not status.exists() or not read(status)['complete']:
            continue
        state = read(status)
        assert state['job_sha256'] == sha(path)
        frames = {r['frame_id']: r for r in job['frames']}
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), layout='constrained')
        for col, fid in enumerate([0, 50, 99]):
            record = frames[fid]; rgb_path = record['rgb']
            assert sha(rgb_path) == job['input_sha256'][rgb_path]
            image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)/255.
            mask_path = folder/'masks'/f'{fid:06d}.npz'
            assert sha(mask_path) == state['files_sha256'][str(mask_path.relative_to(folder))]
            with np.load(mask_path) as z: seg = z['segmentation'].astype(int)+1
            overlay = image.copy(); valid = seg > 0
            overlay[valid] = .5*image[valid]+.5*color(seg)[valid]
            axes[0, col].imshow(image); axes[0, col].set_title(f'Frame {fid:03d} | RGB')
            axes[1, col].imshow(overlay); axes[1, col].set_title('SAM2.1 predicted masks')
        for ax in axes.flat: ax.axis('off')
        fig.suptitle(f'{scene} | fixed views | mask colors are local to each frame', fontsize=13)
        name = scene+'_inputs.png'; fig.savefig(OUT/name, dpi=160); plt.close(fig)
        files.append(name)
    return files


def write_tables(summary, protocol):
    rows = ['| Method | '+' | '.join(protocol['scenes'])+' | Mean PQ | Δ vs History2 | Δ vs ratio |',
            '|---|'+'---:|'*(len(protocol['scenes'])+3)]
    means = summary['scene_mean_pq100']
    for method in protocol['methods']:
        values = [100*summary['results'][s]['methods'][method]['primary']['observed_voxel_instance_pq'] for s in protocol['scenes']]
        rows.append('| '+NAMES[method]+' | '+' | '.join(f'{v:.3f}' for v in values)+
                    f" | {means[method]:.3f} | {means[method]-means['history2']:+.3f} | {means[method]-means['simple_ratio']:+.3f} |")
    return '\n'.join(rows)


def map_figures(summary, protocol):
    files = []
    methods = protocol['methods']
    for scene in protocol['scenes']:
        ref_path = bench.OUT/'reference'/scene/'reference.npz'
        assert sha(ref_path) == read(ref_path.parent/'manifest.json')['reference_sha256']
        ref = np.load(ref_path)
        xyz = ref['xyz']; truth = ref['instance']
        # Identical world XY projection, sampling, and limits for every method.
        sample = np.arange(0, len(xyz), max(1, int(np.ceil(len(xyz)/100000))))
        sample = sample[np.argsort(xyz[sample, 2], kind='stable')]
        fig, axes = plt.subplots(3, 4, figsize=(17, 12), layout='constrained')
        for ax, method in zip(axes.flat, ['reference']+methods):
            ids = truth if method == 'reference' else np.load(bench.OUT/'evaluation'/scene/(method+'.npz'))['instance']
            background = sample[ids[sample] == 0]; foreground = sample[ids[sample] > 0]
            for subset in [background, foreground]:
                ax.scatter(xyz[subset, 0], xyz[subset, 1], c=color(ids[subset]), s=.7, linewidths=0, rasterized=True)
            title = 'Reference instances' if method == 'reference' else NAMES[method]
            if method != 'reference':
                pq = 100*summary['results'][scene]['methods'][method]['primary']['observed_voxel_instance_pq']
                title += f' | PQ {pq:.2f}'
            ax.set_title(title.replace(' | PQ', '\nPQ'), fontsize=9); ax.set_aspect('equal'); ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)')
        for ax in list(axes.flat)[len(methods)+1:]: ax.axis('off')
        fig.suptitle(f'{scene} | common observed surface | foreground drawn over gray context\nColors denote IDs within each map, not cross-method identity correspondence', fontsize=12)
        name = scene+'_maps.png'; fig.savefig(OUT/name, dpi=180); plt.close(fig); files.append(name)
    return files


def full_report(protocol, previews_list):
    summary = read(bench.OUT/'summary.json'); assert summary['complete']
    bench.gate()
    assert summary['protocol_sha256'] == sha(bench.OUT/'protocol.json')
    verification = read(bench.OUT/'verification.json')
    assert verification['complete'] and verification['summary_sha256'] == sha(bench.OUT/'summary.json')
    maps = map_figures(summary, protocol)
    methods = protocol['methods']; y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    for i, scene in enumerate(protocol['scenes']):
        scores = [100*summary['results'][scene]['methods'][m]['primary']['observed_voxel_instance_pq'] for m in methods]
        ax.barh(y+(i-.5)*.35, scores, height=.33, label=scene)
        for row, score in zip(y+(i-.5)*.35, scores):
            ax.text(score+.12, row, f'{score:.2f}', va='center', fontsize=8, color='#304356')
    ax.set_yticks(y, [NAMES[m] for m in methods]); ax.invert_yaxis(); ax.set_xlabel('Observed-surface instance PQ (%)')
    ax.set_title('Frozen configurations | two independent Hypersim scenes'); ax.legend(); ax.grid(axis='x', alpha=.15)
    fig.savefig(OUT/'comparison.png', dpi=180); fig.savefig(OUT/'comparison.svg'); plt.close(fig)
    details = []; transitions = {}; resources = {}
    for scene in protocol['scenes']:
        details += [f'\n## {scene}', '\n| Method | SQ (%) | RQ (%) | TP / FP / FN | PQ, GT ≥20 voxels | Gains / losses vs History2 | Gains / losses vs ratio |', '|---|---:|---:|---|---:|---|---|']
        transitions[scene] = {}
        for m in methods:
            r = summary['results'][scene]['methods'][m]; p = r['primary']
            def pct(v): return 'N/A' if v is None else f'{100*v:.3f}'
            def change(key):
                t = r[key]; return f"{len(t['gained_matched_gt'])} / {len(t['lost_matched_gt'])}"
            details.append(f"| {NAMES[m]} | {pct(p['sq'])} | {pct(p['rq'])} | {p['tp']} / {p['fp']} / {p['fn']} | {pct(r['minimum20_sensitivity']['observed_voxel_instance_pq'])} | {change('vs_history2')} | {change('vs_simple_ratio')} |")
            transitions[scene][m] = {k: r[k] for k in ['vs_history2', 'vs_simple_ratio']}
        folder = Path(read(protocol['jobs'][scene])['output'])
        resources[scene] = {phase: {k: v for k, v in read(folder/(phase+'_run.json')).items() if k in ['wall_s', 'gpu_peak_allocated_bytes', 'gpu_peak_reserved_bytes', 'device', 'memory_limit_bytes']} for phase in ['masks', 'triplets', 'features', 'geometry']}
        cleanup = read(bench.OUT/'maps'/scene/'cleanup/run.json')
        resources[scene]['cleanup'] = {k: cleanup[k] for k in ['total_seconds', 'methods', 'iterative']}
        cg = read(bench.OUT/'maps'/scene/'cg/run.json')
        resources[scene]['cg'] = {k: v for k, v in cg.items() if k in ['causal_mapping_s', 'frame_median_s', 'frame_p95_s', 'cold_start_s', 'cpu_peak_rss_bytes']}
        mc = read(bench.OUT/'maps'/scene/'maskclustering/run.json')
        resources[scene]['maskclustering'] = {k: v for k, v in mc.items() if k in ['graph_s', 'postprocess_s', 'total_s', 'input_masks', 'frames', 'objects']}
    dump(OUT/'object_transitions.json', transitions)
    dump(OUT/'resources.json', resources)
    interpretation = '''## 本轮结论

第一场景 ai_001_010：冻结 Ours 的 PQ 为 21.022，相对 History2 的 19.531 提高 1.491 个百分点，相对 SOR 的 20.428 提高 0.594 点。正确匹配新增 3 个、丢失 1 个。这个场景支持旧可见性修订存在实际收益。

同条件 MC-style 点比例为 22.305，MaskClustering 共享输入核心为 23.623，均高于旧 Ours。新漏分模型为 21.147–21.228；逐视角版本没有超过统一版本，排除自身统计也没有带来额外匹配。不能把相对 History2 的提升归因于新建模中的所有设计。

第二场景 ai_003_007：十种方法均为 0 PQ。History2 的 38 个最终对象中，37 个被继承的前景规则排除（ceiling/pillar/wall/window），仅保留一株植物；该植物只覆盖对应 GT 的 61/626 个参考体素，IoU 约 0.097，未过匹配阈值。该场景同时暴露类别定义不一致和目标覆盖不足；不是后处理能单独解决的问题。所有零分仍保留在预设两场景宏平均中，不删除场景、不改主指标。

事后诊断保存在 `build/benchmarks/hypersim_independent_20261113/ontology_diagnostic.json`：例如 GT 2 的 3049 个参考体素中，2482 个最近原始支撑来自被排除的 window；GT 6 的 2197 个体素中，2166 个来自被排除的 pillar。这里检查的是全部源点的覆盖，不是另一套调优分数。

因此本轮完成了独立比较，尚未证明新候选优于现有强方法，也不能把两个场景的平均差值当作两个场景均获提升。当前版停止调参；后续室外评测应在推理前统一类别范围、传感器输入与时序协议。按需证据选择仍是待验证的新假设。

'''
    text = '''# Hypersim 冻结配置独立评测

两个预先选定的官方 test 场景，各 100 帧。所有预测封存后才读取 GT。下列数值是本项目的可观测表面实例评测，不是官方排行榜成绩；两个场景不足以单独证明广泛泛化。

**Ours (frozen visibility1)** 是旧稿冻结方法；三个 Candidate 是漏分模型的新候选，不因本次成绩自动替换 Ours。CG、MaskClustering 是共享 SAM2.1 / SigLIP 输入的核心适配，不冒充各自原版完整系统。所有方法的后处理条件见冻结协议。

## 主指标

PQ 为百分制；差值是百分点。均值对两个场景等权，不按体素数加权。

'''+write_tables(summary, protocol)+'\n\n'+interpretation+'\n'+ '\n'.join(details)+'''

## 可视化与逐对象结果

固定使用帧 0、50、99，未按结果挑图。地图图像在同一参考域、同一视角显示各方法标签，灰色表示背景或未分配；不是新增测量点，也不是 RGB 点云。前景覆盖在灰色背景上方绘制，便于观察对象，不代表物理遮挡顺序。不同方法的颜色不表示跨方法身份对应。

所有新增/丢失匹配的 GT ID 见 `object_transitions.json`。原始逐对象 IoU 与误差分类保留在权威目录 `evaluation/`。GT 小于 20 体素的敏感性结果单独列出，不择优替代主指标。前端词表与参考类定义的差异沿用冻结协议，未据此事后移除不利对象。

共享 GPU 阶段耗时和显存见 `resources.json`；它们受其他作业影响，不代表机器人实时速度。后处理时间包含证据构建和全部候选计算，不能拿它作为单一方法延迟。
'''
    for name in ['comparison.png']+previews_list+maps: text += f'\n![{name}]({name})\n'
    (OUT/'report.md').write_text(text)
    dump(OUT/'report_provenance.json', dict(summary_sha256=sha(bench.OUT/'summary.json'), protocol_sha256=sha(bench.OUT/'protocol.json'), verification_sha256=sha(bench.OUT/'verification.json'), ontology_diagnostic_sha256=sha(bench.OUT/'ontology_diagnostic.json'), reporter_sha256=sha(__file__), figures=['comparison.png']+previews_list+maps))
    # Self-contained local HTML index; tables retain exact Markdown alongside figures.
    body = '<h1>Hypersim independent evaluation</h1><p>Frozen Ours, strong controls and experimental candidates. See report.md for exact tables and scope.</p>'
    body += '<pre>'+html.escape(text.split('## 可视化与逐对象结果')[0])+'</pre>'
    body += ''.join(f'<figure><img src="{name}"><figcaption>{html.escape(name)}</figcaption></figure>' for name in ['comparison.png']+previews_list+maps)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Hypersim evaluation</title><style>body{max-width:1300px;margin:40px auto;font:16px system-ui;color:#183248}pre{white-space:pre-wrap;font-size:13px}img{width:100%}figure{margin:32px 0}</style>'+body)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--preview-only', action='store_true'); args = parser.parse_args()
    protocol = bench.verify(); OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    figures = previews(protocol)
    if not args.preview_only: full_report(protocol, figures)
    print(OUT)


if __name__ == '__main__': main()
