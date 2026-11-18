"""Post-score attribution and actual images for the exposed outdoor window."""
from pathlib import Path
import argparse
import html
import cv2
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_final_common import ROOT, read, dump, sha
import evaluate_kitti360_development as evaluation

OUT = ROOT/'output/kitti360_development'


def colors(ids):
    return plt.get_cmap('turbo')((np.asarray(ids)*.61803398875)%1)[..., :3]


def main(connected=False):
    global OUT
    if connected:
        from evaluate_kitti360_connected import configure
        configure()
        OUT = ROOT/'output/kitti360_connected_development'
    protocol = evaluation.verify()
    summary = read(evaluation.OUT/'summary.json')
    assert summary['complete'] and summary['protocol_sha256'] == sha(evaluation.OUT/'protocol.json')
    for name, expected in summary['files_sha256'].items(): assert sha(evaluation.OUT/name) == expected
    OUT.mkdir(exist_ok=True, parents=True)
    job = read(protocol['job'])
    cache = evaluation.SPARSE/'cache' if connected else Path(protocol['cache'])
    manifest = read(cache/'manifest.json')
    reference = np.load(evaluation.OUT/'reference.npz')
    ref, truth, weights = reference['xyz'], reference['train_instance'], reference['confidence']
    measured = np.load(cache/'geometry.npz')
    used = np.zeros(len(measured['xyz']), bool)
    observations = {}
    for rec in manifest['records']:
        path = cache/'frames'/rec['file']
        assert sha(path) == rec['sha256']
        f = np.load(path)
        used[f['point_ids']] = True
        for i in range(rec['masks']):
            observations[rec['observation_offset']+i] = dict(frame_id=rec['frame_id'], mask_index=i,
                support=f['point_ids'][f['offsets'][i]:f['offsets'][i+1]])
    def proximity(points):
        if not len(points): return np.zeros(len(ref), bool)
        return np.isfinite(cKDTree(points).query(ref, workers=1, distance_upper_bound=.2)[0])
    base = Path(protocol['methods']['history2'])
    g = np.load(base/'geometry.npz')
    rows = read(base/'classification.json')['objects']
    targets = [r['object_id'] for r in rows if r['predictions']['mean']['retained']]
    stage = dict(measured=reference['observed_support'], any_mask=proximity(measured['xyz'][used]),
        all_mapped_objects=proximity(g['xyz']), target_class_objects=proximity(g['xyz'][np.isin(g['instance'], targets)]))
    coverage = []
    for row in summary['gt_coverage']:
        domain = truth == row['gt_id']
        total = weights[domain].sum()
        coverage.append(dict(gt_id=row['gt_id'], train_id=row['train_id'], confidence_mass=float(total),
            **{name:float(weights[domain & mask].sum()/total) if total else None for name, mask in stage.items()}))
    # Label source supports only for this post-hoc diagnostic; never feed them
    # back into predictions or count them as independent annotated 2D masks.
    distance, nearest = cKDTree(ref).query(measured['xyz'], workers=1, distance_upper_bound=.2)
    source_gt = np.zeros(len(distance), np.int64)
    valid = np.isfinite(distance)
    source_gt[valid] = truth[nearest[valid]]
    objects = []
    for row in rows:
        if not row['predictions']['mean']['retained']: continue
        obs_rows = []
        for oid in row['observation_ids']:
            obs = observations[oid]
            values = source_gt[obs['support']]
            car = values[values//1000 == 13]
            ids, counts = np.unique(car, return_counts=True)
            order = np.argsort(-counts, kind='stable')
            obs_rows.append(dict(observation_id=oid, frame_id=obs['frame_id'], mask_index=obs['mask_index'],
                support_points=len(values), car_supported_points=len(car),
                car_components=[dict(gt_id=int(ids[i]), points=int(counts[i])) for i in order]))
        objects.append(dict(object_id=row['object_id'], class_name=row['predictions']['mean']['name'], observations=obs_rows))
    dump(OUT/'mechanism_audit.json', dict(complete=True, posthoc_development_only=True,
        summary_sha256=sha(evaluation.OUT/'summary.json'), stage_coverage=coverage, objects=objects,
        support_points=int(used.sum()), measured_points=len(used),
        limitation='Coverage is proximity of any geometry, not correct identity. Mask purity uses nearest static reference within20cm, not independent image annotation.'))
    # Fixed first, middle and final inputs, not a selection of successful views.
    fig, axes = plt.subplots(2, 3, figsize=(16, 3.8), layout='constrained')
    for col, index in enumerate([0, len(job['frames'])//2, len(job['frames'])-1]):
        record = job['frames'][index]
        image = cv2.cvtColor(cv2.imread(record['rgb']), cv2.COLOR_BGR2RGB)/255.
        seg = np.load(Path(job['output'])/'masks'/f"{record['frame_id']:06d}.npz")['segmentation']
        overlay = image.copy()
        keep = seg >= 0
        overlay[keep] = .5*image[keep]+.5*colors(seg[keep]+1)
        axes[0, col].imshow(image)
        axes[0, col].set_title(f"Frame {record['frame_id']} | original RGB", fontsize=10)
        axes[1, col].imshow(overlay)
        axes[1, col].set_title('Actual SAM2.1 masks | frame-local colors', fontsize=10)
    for ax in axes.flat: ax.axis('off')
    fig.savefig(OUT/'inputs.png', dpi=160)
    plt.close(fig)
    sample = np.arange(0, len(ref), max(1, len(ref)//120000))
    views = [('reference', 'Static reference: building / car'), ('history2', 'History2: connected supports' if connected else 'History2: scale adaptation only'),
             ('ours_visibility1', 'Ours: same IDs + visibility revision')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout='constrained')
    for ax, (name, title) in zip(axes, views):
        ids = np.where(np.isin(truth//1000, [2,13]), truth, 0) if name == 'reference' else np.load(evaluation.OUT/(name+'_prediction.npz'))['instance']
        bg = sample[ids[sample] == 0]
        fg = sample[ids[sample] > 0]
        ax.scatter(ref[bg,0], ref[bg,1], c='#dbe0e5', s=.3, linewidths=0, rasterized=True)
        ax.scatter(ref[fg,0], ref[fg,1], c=colors(ids[fg]), s=1.0, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('Local X (m)')
        ax.set_ylabel('Local Y (m)')
    fig.suptitle('Exposed development window 0000 | common reference domain\nPredicted object IDs transferred within20cm; colors are not matched across panels', fontsize=12)
    fig.savefig(OUT/'maps.png', dpi=160)
    plt.close(fig)
    labels = dict(native_indoor_history2='室内配置直接迁移', scale_only_history2='History2（仅尺度适配）',
        cg_sparse='CG（连通支撑+尺度适配）' if connected else 'CG（仅尺度适配）',
        history2='History2（完整车库强基线迁移）' if connected else 'History2（仅尺度适配）',
        sor='History2 + SOR', ours_visibility1='Ours：visibility1', simple_ratio='Simple ratio', mc_point_ratio='MC-style 点比例')
    table = ['| 方法 | 完整静态窗口 AP50 (%) | 可观测支撑诊断 AP50 (%) |', '|---|---:|---:|']
    for name, result in summary['methods'].items():
        table.append(f"| {labels[name]} | {result['full_window']['averages']['allAp50%']*100:.4f} | {result['observed_support']['averages']['allAp50%']*100:.4f} |")
    class_coverage = {}
    for tid, name in [(2,'building'), (13,'car')]:
        selected = [r for r in coverage if r['train_id'] == tid and r['confidence_mass'] >= 100]
        class_coverage[name] = {key:float(np.mean([r[key] for r in selected])) for key in stage}
    dump(OUT/'class_coverage.json', class_coverage)
    counts = summary['methods']['history2']['full_window']['gt_eligible']
    removed = read(evaluation.SPARSE/'maps/cleanup/run.json')['results']
    delta = 100*(summary['methods']['ours_visibility1']['full_window']['averages']['allAp50%']-
                 summary['methods']['history2']['full_window']['averages']['allAp50%'])
    adaptation = ('本版补齐已存在的车库连通支撑步骤：每个掩码保留0.4m半径最大连通测量分量，最少16点；再使用既有尺度适配。此前仅尺度版成绩已经查看，全部属于开发数据，不追认为独立测试。该步骤是既有强基线，不是新贡献。\n\n'
        if connected else '本版只有尺度适配，尚未包含车库的掩码内最大连通支撑，不能称为完整车库强基线。完整版本另见相邻 kitti360_connected_development 报告。\n\n')
    text = ('# KITTI-360 开发窗口：真实输入与对象地图\n\n'
        '这是公开数据上的开发适配结果，不是独立测试或官方榜单成绩。全部输出先封存，再读取开发窗口的静态参考。两个预留验证窗口未推理。\n\n'
        '首轮直接套用室内 CG 配置，262 个片段均未通过三次观测门限。第二版复用既有车库的固定配置：关闭 DBSCAN，关联与最终合并的邻域半径20cm，体素处理仍2.5cm。没有重跑前端或按测试分数搜索配置。\n\n'
        +adaptation+
        f"本版得到{len(rows)}个对象，分类保留{len(targets)}个建筑/车辆对象；完整参考有{counts['building']}个建筑、{counts['car']}辆车达到置信度加权区域门限。Ours 删除{removed['visibility1']['removed']}个目标点，SOR 删除{removed['sor']['removed']}个；Ours 相对 History2 的完整窗口 AP50 变化为{delta:+.4f}个百分点。\n\n"
        +'\n'.join(table)+'\n\n'
        '指标说明：参考置信度参与区域面积与交集，门限100是权重和。复用固定官方 AP/匹配函数，显式适配 NumPy 旧标量名及上游标注构造函数不兼容；完整 AP 的阈值为0.50至0.90另加0.25，主报 AP50。预测按完整地图最近点20cm转移，非目标对象作为零标签参与最近点竞争。可观测诊断只保留距离全部原始测量支撑20cm以内的参考点，与方法预测无关。\n\n'
        '有效雷达像素约4%–5%；缺测保持未知。相机/扫描时刻差已记录，尚未做逐点扫描去畸变。当前使用给定的系统位姿，不是 LIVO 位姿精度评测。\n\n'
        '机制核查：`mechanism_audit.json` 列出每个 GT 对象的测量、掩码支撑、全部预测几何及目标类别几何覆盖；车辆对象另列每个来源掩码所覆盖的 GT 车辆成分。它们是事后诊断，近邻支撑不等于正确身份或独立二维标注。\n\n'
        '另核对上游源码：配置中的 `multi_crop: true` 在当前 SAM2 加载路径没有传入 `crop_n_layers`；实际不能宣称已启用多尺度裁剪。后续若验证局部取证，必须与真正开启的普通裁剪基线比较。\n\n'
        '![真实输入与掩码](inputs.png)\n\n![完整参考与对象地图](maps.png)\n')
    (OUT/'report.md').write_text(text)
    rows_html = ''.join('<tr><td>'+html.escape(labels[name])+'</td><td>'+f"{r['full_window']['averages']['allAp50%']*100:.4f}"+'</td><td>'+f"{r['observed_support']['averages']['allAp50%']*100:.4f}"+'</td></tr>' for name,r in summary['methods'].items())
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>KITTI-360 开发评测</title>'
        '<style>body{font:17px/1.7 system-ui;max-width:1200px;margin:40px auto;color:#193044;padding:0 24px}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;font:inherit}</style>'
        '<h1>KITTI-360：真实室外输入与对象地图</h1><p>开发窗口；两个预留验证窗口未运行。Ours 相对 History2 的完整窗口 AP50 变化：'+f'{delta:+.4f}'+' 个百分点。</p><p>'+html.escape(adaptation)+'</p>'
        '<img src="inputs.png"><table><tr><th>方法</th><th>完整窗口 AP50 (%)</th><th>观测支撑 AP50 (%)</th></tr>'+rows_html+'</table>'
        '<img src="maps.png"><details><summary>协议、限制与机制诊断</summary><pre>'+html.escape(text.split('指标说明：')[1].split('![真实输入')[0])+'</pre></details>')
    dump(OUT/'provenance.json', dict(summary_sha256=sha(evaluation.OUT/'summary.json'), script_sha256=sha(__file__),
        files_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'provenance.json'}))
    print(class_coverage, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connected', action='store_true')
    args = parser.parse_args()
    main(args.connected)
