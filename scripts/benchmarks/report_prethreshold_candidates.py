"""Actual low-confidence masks, measured support and complete diagnostic table."""
import numpy as np
from PIL import Image
from paper_final_common import ROOT, read, dump, sha
from capture_prethreshold_candidates import OUT, FRONT, INPUT, verify
from audit_prethreshold_candidates import AUDIT

REPORT = ROOT/'output/prethreshold_candidates_v1'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = verify(); capture = read(OUT/'predictions_complete.json'); summary = read(AUDIT/'summary.json')
    assert summary['prediction_seal_sha256'] == sha(OUT/'predictions_complete.json')
    for f, h in capture['files'].items(): assert sha(f) == h
    REPORT.mkdir(parents=True, exist_ok=True); table = ''; objects = ''; figures = ''; provenance = {}
    for key, result in summary['results'].items():
        records = capture['results'][key]; c = result['counts']
        table += f'<tr><td>{key}</td><td>{len(records)}</td><td>{sum(r["high"] for r in records)}</td><td>{c["deferred"]}</td><td>{c["deferred_raw16"]}</td><td>{c["deferred_connected16"]}</td><td>{c["deferred_pure16"]}</td></tr>'
        lines = ''
        for o in result['objects']:
            lines += f'<tr><td>{o["gt_id"]}</td><td>{o["category"]}</td><td>{o["already_matched"]}</td>'+''.join(f'<td>{100*o["coverage"][stage]:.1f}%</td>' for stage in ['high_raw', 'high_connected', 'combined_raw', 'combined_connected'])+f'<td>{len(o["diagnostic_deferred_frames"])}</td><td>{o["new_correct_source_mass"]:.1f}</td><td>{o["viable_for_hypothesis"]}</td></tr>'
        objects += f'<h3>{key}</h3><details><summary>全部参考对象的必要覆盖条件与候选证据</summary><table><tr><th>GT</th><th>类别</th><th>原已匹配</th><th>高分原始</th><th>高分连通</th><th>加入低分原始</th><th>加入低分连通</th><th>低分纯候选帧</th><th>新增正确源点权重</th><th>信息门槛</th></tr>{lines}</table></details>'
        manifest = read(INPUT/key/'manifest.json'); by_frame = {r['frame_id']: r for r in manifest['records']}; candidates = {r['frame_id']: r for r in records}; nodes = read(AUDIT/key/'nodes_GT_ASSISTED.json')
        # Display rules are diagnostic and use reference identity after inference.
        # Show the two viable0000 targets and the two0002 targets with newly
        # >50% connected coverage but fewer than three pure deferred frames.
        targets = [o for o in result['objects'] if o['viable_for_hypothesis']] if key == 'kitti360_0000' else [o for o in result['objects'] if not o['already_matched'] and o['coverage']['high_connected']<=.5 and o['coverage']['combined_connected']>.5 and 0<len(o['diagnostic_deferred_frames'])<3]
        for target in targets:
            gid = target['gt_id']; available = [n for n in nodes if n['diagnostic_gt']==gid]; node = max(available, key=lambda n: (n['connected_points'], -n['frame_id'], -n['query_id']))
            frame = by_frame[node['frame_id']]; data = np.load(candidates[node['frame_id']]['file']); image = np.asarray(Image.open(frame['rgb']).convert('RGB')); height, width = image.shape[:2]
            indices = np.r_[np.flatnonzero(data['scores']>.5), node['query_id']]; masks = np.unpackbits(data['masks_packed'][indices], axis=2)[:, :, :width].astype(bool)
            fig, axes = plt.subplots(1, 3, figsize=(15, 2.2))
            for ax in axes: ax.imshow(image); ax.axis('off')
            axes[0].set_title('Actual RGB', fontsize=10); overlay = np.zeros((height, width, 4)); overlay[masks[:-1].any(0)] = [.0, .75, .85, .4]
            axes[1].imshow(overlay); axes[1].set_title('Original accepted masks: score > 0.5', fontsize=9)
            color = np.zeros((height, width, 4)); color[masks[-1]] = [1., .55, .05, .45]; axes[2].imshow(color)
            support = np.load(AUDIT/'supports'/key/f'{node["frame_id"]:06d}.npz'); j = int(np.flatnonzero(support['query_ids']==node['query_id'])[0]); ids = support['point_ids'][support['offsets'][j]:support['offsets'][j+1]]
            visible = np.load(frame['file']); pixels = visible['pixel_xy'][np.isin(visible['visible_source_ids'], ids)]
            if len(pixels): axes[2].scatter(pixels[:, 0], pixels[:, 1], s=.6, c='#00ee88', alpha=.6)
            axes[2].set_title(f'Discarded score {node["score"]:.3f} | {node["connected_points"]} measured points', fontsize=9)
            fig.suptitle(key+f' | GT-assisted example {gid} | frame {node["frame_id"]}', fontsize=10); fig.tight_layout(); name=f'{key}_gt{gid}.png'; fig.savefig(REPORT/name, dpi=160, bbox_inches='tight'); plt.close(fig)
            figures += f'<h3>{key} · 事后诊断对象 {gid}</h3><img src="{name}"><p>候选分数 {node["score"]:.4f}；连通实测点 {node["connected_points"]}，已知标签点 {node["known_points"]}，该参考对象占已知置信度权重 {100*node["diagnostic_purity"]:.1f}%。符合诊断纯度的低分候选有 {len(target["diagnostic_deferred_frames"])} 个不同帧。该对象的必要连通覆盖从 {100*target["coverage"]["high_connected"]:.1f}% 到 {100*target["coverage"]["combined_connected"]:.1f}%；覆盖不是 IoU，也不说明已经正确建图。</p>'
            provenance[key+'/'+str(gid)] = dict(node=node, rgb=frame['rgb'], rgb_sha256=sha(frame['rgb']), selection='Posthoc GT-assisted display; most connected points, earliest frame/query tie. Not predictor input.')
    gate = summary['information_gate']
    assert not gate
    (REPORT/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 筛选前候选信息审计</title><style>body{{font:16px/1.7 system-ui;max-width:1450px;margin:32px auto;padding:0 24px;color:#193349}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:7px;border:1px solid #d4dfe8;text-align:right}}td:first-child{{text-align:left}}.note{{padding:20px;background:#edf5f7;border-left:5px solid #078e82}}</style><h1>筛选前确有对象信息，但尚未形成可验证的新机制</h1><p class="note"><b>108 帧原高置信度掩码、分数、类别及实测连通支持全部精确复现。</b><br>原生实例分数与图像级类别存在分数相乘，低于或等于 0.5 的候选不进入旧建图。本次保存了该筛选前的全部原生查询，没有改模型、提示或原地图。所有新候选封存后才读取开发参考标签。</p><h2>候选逐级保留情况</h2><table><tr><th>窗口</th><th>帧</th><th>原高分</th><th>0.1～0.5 候选</th><th>独占实测点 ≥16</th><th>连通点 ≥16</th><th>诊断纯度 >50%</th></tr>{table}</table><p>大量低分掩码没有独立的实测支持。像素仍按原生分数高低独占分配，低分不能抢走原高分掩码的点。0.1 在提取前固定，只定义一个宽松对照入口；没有试其他阈值。连通处理仍为 0.4 m、至少 16 点。诊断纯度只使用事后参考标签，不能用来过滤预测。</p><h2>找到的信息与停止条件</h2><p>0000 的车辆 13009 和建筑 2001 具有重复的低分测量支持；0002 的建筑 2011 / 2012 在加入低分候选后，必要连通覆盖从零达到约 60.1% / 56.6%，但达到诊断纯度的低分候选只有 1 / 2 帧。预设的“每场景至少一个原漏检对象有三帧可用候选”等条件没有通过。本轮不据此训练、修改三帧门槛或启动完整地图比较。</p><p>早期置信度筛选确实丢弃有用观测，但少量合格候选不等于少量观测机会。后续核查显示，0002 建筑 2011 / 2012 实际有 8 / 7 帧有效实测可见。详见<a href="observability.html">逐帧可见机会核查与解释更正</a>。旧三帧候选门槛仍未通过；真值辅助覆盖和最佳掩码均不是方法 AP。</p><h2>真实图像、原掩码与被筛掉的候选</h2><p>以下图片使用 GT 在评分后选择，明确用于机制解释；绿色点是候选实际连通测量，橙色是未被原阈值接纳的预测掩码，没有手绘轮廓。</p>{figures}<h2>完整对象表</h2>{objects}<h2>研究边界与主线</h2><p><a href="https://arxiv.org/abs/2110.06864">ByteTrack</a> 已利用低置信度检测改善关联，<a href="https://arxiv.org/abs/2610.31476">Point-Level Refinable Semantic Mapping</a> 已研究多视图置信度更新。因此“保留低分”或“累积几帧”本身不能直接作为我们的贡献。若后续提出观测条件下的对象接纳机制，必须与同输入的固定宽松阈值、相同后端和简单确认策略比较，并最终看完整对象地图及独立评测。</p><p>本页是旧开发数据诊断，无新地图/AP、训练或机器人效果结论。omni_slam 的新机制、机器人对照和新增独立评测目标仍未整体完成。</p><p><a href="summary.json">完整诊断</a> · <a href="images.json">图片来源</a> · <a href="capture.json">提取与复现记录</a></p></html>''')
    dump(REPORT/'summary.json', summary); dump(REPORT/'images.json', provenance); dump(REPORT/'capture.json', capture)
    print('REPORT', REPORT/'index.html', 'examples', len(provenance), flush=True)


if __name__ == '__main__': main()
