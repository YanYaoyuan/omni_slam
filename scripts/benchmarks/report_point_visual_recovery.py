"""Full-map scores, source decisions, and actual imagery for the fixed trial."""
import numpy as np
from PIL import Image
from paper_final_common import ROOT, read, dump, sha
from run_point_visual_recovery import OUT, GEOM, PARENT, INPUT, METHODS, TRAINED, verify, seal

REPORT = ROOT/'output/point_visual_recovery_v1'
NAMES = dict(baseline='Strong baseline', nearest_core='Nearest core', intrinsic_geometry='Intrinsic geometry',
             rgb='Geometry + RGB', dino_first='Geometry + first-frame DINO', ours_dino='Ours candidate: multi-view DINO',
             shuffled_dino='Geometry + shuffled DINO')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = verify(True); fs = seal('features_complete'); ts = seal('trained'); seal('predictions_complete')
    summary = read(OUT/'summary.json'); assert sha(OUT/'predictions_complete.json') == summary['prediction_seal_sha256']
    REPORT.mkdir(parents=True, exist_ok=True)
    table = ''; body = ''; decision_rows = ''; diagnostics = {}; images = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    for ax, key in zip(axes, summary['results']):
        for m in TRAINED: ax.plot(np.arange(1, 41), ts['stats'][key+'/'+m]['loss'], label=NAMES[m])
        ax.set_title(key); ax.set_xlabel('Epoch'); ax.set_ylabel('Training BCE'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(REPORT/'training.png', dpi=150); plt.close(fig)
    for key, entry in summary['results'].items():
        rows = entry['methods']; d = np.load(OUT/'features'/key/'candidates.npz'); labels = np.load(PARENT/'training_labels'/key/'TRAINING_ONLY.npz')
        diagnostics[key] = {}; decisions = {}
        for m in METHODS:
            row = rows[m]; metrics = row['metrics']; changes = row['changes_vs_baseline']
            table += f'<tr><td>{key}</td><td>{NAMES[m]}</td><td>{100*metrics["allAp"]:.3f}</td><td>{100*metrics["allAp50%"]:.3f}</td><td>{row["run"]["added_points"]}</td><td>{len(changes["gained"])}</td><td>{len(changes["lost"])}</td></tr>'
            z = np.load(OUT/'maps'/key/m/'decisions.npz'); np.testing.assert_array_equal(z['source_ids'], d['source_ids']); chosen = z['owners']; decisions[m] = chosen
            eligible = labels['eligible']; accepted = chosen > 0
            correct = ((chosen[:, None] == d['owners']) & labels['positive'] & d['valid']).any(1) & accepted
            tp = int((correct & eligible).sum()); fp = int((accepted & ~correct & eligible).sum()); positive = int((labels['positive'].any(1) & eligible).sum())
            diagnostics[key][m] = dict(correct_known_additions=tp, incorrect_known_additions=fp, precision=tp/(tp+fp) if tp+fp else None, recovery_fraction=tp/positive, unknown_accepted=int((accepted & ~eligible).sum()))
            precision = f'{100*tp/(tp+fp):.1f}%' if tp+fp else '—'
            decision_rows += f'<tr><td>{key}</td><td>{NAMES[m]}</td><td>{tp}</td><td>{fp}</td><td>{precision}</td><td>{100*tp/positive:.1f}%</td><td>{diagnostics[key][m]["unknown_accepted"]}</td></tr>'
        manifest = read(INPUT/key/'manifest.json'); source = np.load(manifest['geometry'])['xyz']; base = np.load(OUT/'maps'/key/'baseline/geometry.npz'); n = len(base['xyz'])
        sample = np.linspace(0, len(source)-1, min(50000, len(source)), dtype=int); core_sample = np.linspace(0, n-1, min(40000, n), dtype=int)
        fig, axes = plt.subplots(1, 3, figsize=(9.5, 8), sharex=True, sharey=True)
        for ax, m in zip(axes, ['intrinsic_geometry', 'dino_first', 'ours_dino']):
            g = np.load(OUT/'maps'/key/m/'geometry.npz'); ax.scatter(source[sample, 0], source[sample, 1], s=.3, c='#e0e5eb'); ax.scatter(base['xyz'][core_sample, 0], base['xyz'][core_sample, 1], s=.5, c='#177eab')
            ax.scatter(g['xyz'][n:, 0], g['xyz'][n:, 1], s=1.2, c='#f28b30'); ax.set_aspect('equal'); ax.set_title(NAMES[m].replace('Geometry + ', 'Geometry +\n').replace('Ours candidate: ', 'Ours candidate:\n'), fontsize=9); ax.set_xlabel('x / m')
        axes[0].set_ylabel('y / m'); fig.suptitle(key+' | Blue: fixed core; orange: accepted measured support', fontsize=10)
        fig.tight_layout(); fig.savefig(REPORT/(key+'_maps.png'), dpi=160, bbox_inches='tight'); plt.close(fig)
        added = d['source_ids'][decisions['ours_dino'] > 0]; different = d['source_ids'][decisions['ours_dino'] != decisions['intrinsic_geometry']]
        best = None
        for r in manifest['records']:
            f = np.load(r['file']); score = int(np.isin(f['visible_source_ids'], different).sum())
            if best is None or score > best[0]: best = (score, r)
        _, r = best; f = np.load(r['file']); image = np.asarray(Image.open(r['rgb']).convert('RGB'))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax in axes: ax.imshow(image); ax.axis('off')
        axes[0].set_title('Actual RGB | frame '+str(r['frame_id']))
        for ax, m in zip(axes[1:], ['intrinsic_geometry', 'ours_dino']):
            ids = d['source_ids'][decisions[m] > 0]; pixels = f['pixel_xy'][np.isin(f['visible_source_ids'], ids)]
            if len(pixels): ax.scatter(pixels[:, 0], pixels[:, 1], s=3, c='#ff951b', alpha=.8)
            ax.set_title(NAMES[m]+' | accepted points', fontsize=9)
        fig.tight_layout(); fig.savefig(REPORT/(key+'_rgb.png'), dpi=160, bbox_inches='tight'); plt.close(fig)
        images[key] = dict(frame_id=r['frame_id'], rgb=r['rgb'], sha256=sha(r['rgb']), selection='Maximum visible changed decisions vs intrinsic geometry; earliest tie, no GT.')
        body += f'<h2>{key}: 实际图像与完整地图</h2><img src="{key}_rgb.png"><img src="{key}_maps.png"><p>橙色是实际接受的源点，不是正确性标签。图像按与几何对照决策差异最多的可见帧选择，未用 GT。所有地图共享坐标与比例。</p>'
        per_gt = {}
        for m in METHODS:
            per_gt[m] = {}
            for category, objects in read(OUT/'maps'/key/m/'matches.json')['window']['groundTruth'].items():
                for obj in objects:
                    if obj['instance_id'] < 1000 or obj['vert_count'] < 100: continue
                    best_iou = max((x['intersection']/(obj['vert_count']+x['vert_count']-x['intersection']) for x in obj['matched_pred']), default=0.)
                    per_gt[m][obj['instance_id']] = (category, best_iou)
        lines = ''.join(f'<tr><td>{gid}</td><td>{per_gt["baseline"][gid][0]}</td>'+''.join(f'<td>{per_gt[m][gid][1]:.4f}</td>' for m in METHODS)+'</tr>' for gid in sorted(per_gt['baseline']))
        body += '<details><summary>全部参考对象的最佳同类 IoU</summary><table><tr><th>GT</th><th>类别</th>'+''.join(f'<th>{NAMES[m]}</th>' for m in METHODS)+f'</tr>{lines}</table></details>'
    mean_rows = ''.join(f'<tr><td>{NAMES[m]}</td><td>{100*summary["mean_primary_ap"][m]:.3f}</td></tr>' for m in METHODS)
    gate = '通过开发门槛；下一步才允许机器人扩展，仍需新的独立测试。' if summary['advance_gate'] else '未通过开发门槛；封存本分支，保留原强基线，不启动机器人扩展或阈值搜索。'
    (REPORT/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam | 点级视觉归属验证</title><style>body{{font:16px/1.7 system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#193349}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:7px;border:1px solid #d6e0e8;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}img{{max-width:100%}}.note{{padding:20px;background:#edf5f7;border-left:5px solid #078d81}}code{{overflow-wrap:anywhere}}</style><h1>omni_slam：点级视觉证据能否改善对象归属？</h1><p class="note"><b>{gate}</b><br>研究主线是对象级语义建图。当前模块只验证既有对象的实测表面恢复，不能发现新对象或补出传感器未测量的几何。DINO 作为冻结特征工具，本身不构成新贡献。</p><h2>完整地图结果</h2><table><tr><th>窗口</th><th>方法</th><th>AP50:90 ×100</th><th>AP50 ×100</th><th>新增点</th><th>新增匹配</th><th>丢失匹配</th></tr>{table}</table><h3>两开发窗口均值</h3><table><tr><th>方法</th><th>AP50:90 ×100</th></tr>{mean_rows}</table><p>两个窗口此前已暴露，采用场景交叉训练；目标窗口不参与自身模型拟合。这是开发评测，非独立测试、官方全量协议或榜单成绩。0004 / 0005 未参与本轮。所有预测先封存，随后统一评分。</p><h2>机制与对照</h2><p>只将通过已有实测深度可见性检查的点投影到 RGB。固定对象核心提供外观锚点，排除共享归属的图像块与候选自身图像块；同帧重复测量不能增加证据。比较候选与自身/其他对象核心的局部外观，再按帧汇总。固定几何特征和原对象身份，训练同一 24 维输入、32/32 隐层的归属头。</p><p>RGB 与 DINO 使用相同分辨率、测量支持和汇总方式；单帧只保留最早有效帧；打乱对照在同场景同预测类别内置换多帧视觉证据。四组均为固定种子、40 轮、相同损失和训练数据。没有按目标分数挑模型。老核心、语义分数、候选范围完全不变。</p><h2>源点层诊断（评分后）</h2><table><tr><th>窗口</th><th>方法</th><th>正确新增</th><th>错误新增</th><th>已知新增精度</th><th>可恢复正例召回</th><th>接受但标签未知</th></tr>{decision_rows}</table><p>此表使用已暴露的参考标签，仅统计可信核心归属与已知源点；未知点单列。不能代替完整地图 AP，也不是独立机器人准确率。</p><h2>训练记录</h2><img src="training.png">{body}<h2>复核材料</h2><p><a href="summary.json">评分与门槛</a> · <a href="source_diagnostics.json">源点得失</a> · <a href="images.json">图像来源</a> · <a href="features.json">证据覆盖</a></p><p><code>{OUT/'protocol.json'}</code></p></html>''')
    dump(REPORT/'summary.json', summary); dump(REPORT/'source_diagnostics.json', diagnostics); dump(REPORT/'images.json', images); dump(REPORT/'features.json', fs['stats'])
    print('REPORT', REPORT/'index.html', gate, flush=True)


if __name__ == '__main__': main()
