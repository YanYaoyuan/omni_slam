"""Actual map/image report from frozen predictions and post-score diagnostics."""
from html import escape
import numpy as np
from PIL import Image
from paper_final_common import ROOT, read, dump, sha
from run_signed_query_association import OUT, INPUT, METHODS, pieces

REPORT = ROOT/'output/signed_query_association_v1'
NAMES = dict(baseline='冻结强基线：不变几何', geometry_graph='简单几何关联',
             mc_core='MC-core（共享查询证据适配）', positive_graph='学习关联，仅正边',
             geometry_signed='几何特征、正负边聚合', ours_signed='Ours：跨视角证据、正负边聚合')
CASES = {'kitti360_0000': [13013, 13009, 2001], 'kitti360_0002': [2011]}


def best(rows, gid):
    values = [(g['iou'], r) for r in rows for g in r['overlaps'] if g['gt_id'] == gid]
    return max(values, key=lambda v: v[0]) if values else (0., None)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    REPORT.mkdir(parents=True, exist_ok=True)
    audit = read(OUT/'artifact_audit.json')
    assert audit['complete'] and audit['summary_sha256'] == sha(OUT/'summary.json')
    assert audit['diagnostics_sha256'] == sha(OUT/'object_diagnostics.json')
    summary = read(OUT/'summary.json')
    details = read(OUT/'object_diagnostics.json')['results']
    examples, figures, table, changes, object_tables = [], '', '', '', ''
    keys = list(details)
    for method in METHODS:
        vals = [summary['results'][k]['methods'][method]['metrics']['allAp']*100 for k in keys]
        mean = summary['mean_primary_ap'][method]*100
        table += '<tr'+(' class="ours"' if method == 'ours_signed' else '')+'><td>'+NAMES[method]+'</td>'
        table += ''.join(f'<td>{v:.3f}</td>' for v in vals+[mean, mean-summary['mean_primary_ap']['baseline']*100])+'</tr>'
    for key in keys:
        manifest = read(INPUT/key/'manifest.json')
        source = np.load(manifest['geometry'])['xyz']
        nodes = dict(np.load(OUT/'nodes'/key/'nodes.npz'))
        metadata = read(OUT/'nodes'/key/'nodes.json')
        sets = pieces(nodes['source_ids'], nodes['source_offsets'])
        old_sets = {m['old_object_id']: sets[j] for j, m in enumerate(metadata) if m['old_object_id'] >= 0}
        maps = {m: dict(np.load(OUT/'maps'/key/m/'geometry.npz')) for m in METHODS}
        base = maps['baseline']; n = len(base['xyz'])
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
        background = source[::max(1, len(source)//35000)]
        for ax, method in zip(axes.flat, ['baseline', 'geometry_graph', 'positive_graph', 'ours_signed']):
            g = maps[method]
            ax.scatter(background[:, 0], background[:, 1], s=.15, c='#ccd3d9', rasterized=True)
            old = g['xyz'][:n:3]
            ax.scatter(old[:, 0], old[:, 1], s=.5, c='#317999', rasterized=True)
            changed = g['xyz'][:n][g['instance'][:n] != base['instance']]
            if len(changed): ax.scatter(changed[:, 0], changed[:, 1], s=1, c='#ad4ab2', rasterized=True)
            added = g['xyz'][n:]
            if len(added): ax.scatter(added[::2, 0], added[::2, 1], s=.6, c='#e4783e', rasterized=True)
            ap = summary['results'][key]['methods'][method]['metrics']['allAp']*100
            ax.set_title(f'{method} | AP {ap:.3f}')
            ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_aspect('equal'); ax.grid(alpha=.15)
        fig.suptitle(key+' | blue: old map / purple: changed identity / orange: added measurements')
        fig.tight_layout(); name = key+'_maps.png'; fig.savefig(REPORT/name, dpi=150); plt.close(fig)
        figures += f'<h3>{key} · 完整窗口地图</h3><img src="{name}"><p>所有面板坐标范围一致，灰色为实测背景。颜色表示改动来源，不代表正确性。改变旧 ID 的点为紫色，新增实测点为橙色。</p>'
        for method in METHODS:
            row = summary['results'][key]['methods'][method]
            change, run = row['changes_vs_baseline'], row['run']
            changes += '<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in [key, NAMES[method], change['gained'], change['lost'], run['new_objects'], run['old_ids_merged'], run['added_points']])+'</tr>'
            allrows = ''
            for obj in details[key][method]:
                overlap = obj['overlaps'][0] if obj['overlaps'] else None
                allrows += '<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in [obj['object_id'], 'car' if obj['semantic_id'] == 26 else 'building', obj.get('old_ids', [obj['object_id']]), obj['points'], f'{obj["score"]:.3f}', overlap['gt_id'] if overlap else '—', f'{overlap["iou"]:.3f}' if overlap else '—', f'{obj["reference_mass"]:.1f}'])+'</tr>'
            object_tables += f'<details><summary>{key} / {NAMES[method]}：{len(details[key][method])} 个对象</summary><table><tr><th>ID</th><th>类别</th><th>旧 ID</th><th>点数</th><th>分数</th><th>最佳 GT</th><th>IoU</th><th>参考权重面积</th></tr>{allrows}</table></details>'
        for gid in CASES[key]:
            rows = {m: best(details[key][m], gid) for m in METHODS}
            groupsources = {}
            for method in ['geometry_graph', 'ours_signed']:
                obj = rows[method][1]
                if obj is None:
                    groupsources[method] = np.empty(0, np.int64); continue
                g = maps[method]
                new = g['added_source_ids'][g['instance'][n:] == obj['object_id']]
                groupsources[method] = np.unique(np.concatenate([new]+[old_sets[i] for i in obj['old_ids']]))
            union = np.union1d(*groupsources.values())
            candidates = []
            for frame in manifest['records']:
                v = np.load(frame['file'])
                candidates.append((int(np.isin(v['visible_source_ids'], union).sum()), -frame['frame_id'], frame))
            count, _, frame = max(candidates, key=lambda v: v[:2])
            assert count > 0
            rgb = np.asarray(Image.open(frame['rgb']).convert('RGB'))
            view = dict(np.load(frame['file']))
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            axes[0].imshow(rgb); axes[0].set_title('Actual RGB')
            for ax, method, color in zip(axes[1:], ['geometry_graph', 'ours_signed'], ['#00ce9c', '#ff8a31']):
                ax.imshow(rgb)
                pixels = view['pixel_xy'][np.isin(view['visible_source_ids'], groupsources[method])]
                ax.scatter(pixels[:, 0], pixels[:, 1], s=3, c=color)
                iou, obj = rows[method]
                ax.set_title(f'{method} | object {obj["object_id"] if obj else "none"} | full-map IoU {iou:.3f}')
            for ax in axes: ax.axis('off')
            fig.suptitle(f'{key} / reference {gid} / actual frame {frame["frame_id"]} / measurement provenance')
            fig.tight_layout(); name = f'{key}_gt{gid}.png'; fig.savefig(REPORT/name, dpi=150, bbox_inches='tight'); plt.close(fig)
            values = ' · '.join(f'{NAMES[m]}：{rows[m][0]:.3f}' for m in METHODS)
            figures += f'<h3>{key} / 参考对象 {gid}</h3><p>{values}</p><img src="{name}"><p>图像选择为两方法相关对象实测来源最多的帧（同数取最早）；是评分后的诊断展示。彩点为最终对象的新增测量点及旧对象原始测量来源，不是稠密分割掩码；旧地图点经过体素处理，两者坐标表示不同。上方 IoU 来自完整窗口评测，不是图像裁剪评分。</p>'
            examples.append(dict(scene=key, gt_id=gid, frame=frame['frame_id'], rgb=frame['rgb'], rgb_sha256=sha(frame['rgb']), measured_view=frame['file'], file=name,
                                 best_object={m: dict(iou=iou, object_id=obj['object_id'] if obj else None) for m, (iou, obj) in rows.items()}))
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 跨视角对象关联复核</title>
<style>body{{font:16px/1.7 system-ui;color:#193349;max-width:1400px;margin:32px auto;padding:0 24px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d5dfe8;padding:8px;text-align:right}}td:first-child{{text-align:left}}.ours{{background:#e7f5f1;font-weight:bold}}.note{{background:#f0f4f7;padding:20px;border-left:5px solid #327f93}}details{{margin:12px 0}}summary{{cursor:pointer;font-weight:bold}}.scroll{{overflow:auto}}</style>
<h1>omni_slam：跨视角身份关联与完整地图复核</h1>
<p class="note"><b>本版未超过强对照，封存，不替换主线强基线。</b>Ours 平均 AP50:90 为22.391，强基线22.639，简单几何关联23.134。Ours 新增两个 AP50 正确匹配、丢失一个。所有结果来自两个已暴露室外开发窗口的相机/雷达建图适配，不是官方榜单或独立测试。</p>
<h2>完整地图主指标 · AP50:90 ×100</h2><table><tr><th>方法</th><th>0000</th><th>0002</th><th>均值</th><th>相对强基线</th></tr>{table}</table>
<p>四次固定交叉拟合、十份新地图、两份原图复制；未新增基础模型推理。每个头只使用另一开发窗口的标签及标准化统计。所有地图封存后统一评分。Ours 使用跨视角正负边聚合；所有方法共享实际测量支持与最终导出，不删除旧点。MC-core 为上游聚类核心与共享查询读取的适配，非原版完整 MaskClustering。</p>
<h2>逐窗口得失与实际地图改动</h2><div class="scroll"><table><tr><th>窗口</th><th>方法</th><th>新增正确 GT</th><th>丢失正确 GT</th><th>新增对象</th><th>合并旧 ID 数</th><th>新增点</th></tr>{changes}</table></div>
<h2>机制结论：能补回对象，也会破坏已正确的边界</h2>
<p><b>车辆错误合并：</b>0000 的旧 ID 15、18 分别对应车辆13014、13013。Ours 将二者合并后，13013 IoU 从0.741降到0.264。首次连接这两组的保存事件包含14条正边与18条负边，代价分别为+8.099和−5.652，合计+2.447，因此按既定目标函数发生合并。负边存在仍不足以阻止错误合并；边数不能当作独立视角数。此处是保存轨迹的解释，不是删边反事实实验。</p>
<p><b>建筑恢复与污染：</b>0000 建筑2001达到0.517，说明跨视角几何形成确实能恢复单片段未达标的对象；简单几何关联也达到0.511。0002 建筑2011在简单几何关联下为0.535，Ours却与2012等建筑片段聚合，降至0.201。首次把2011和2012节点连入同组的事件有9条正边、没有负边，总代价+27.861。当前学习关系不是可靠的对象身份概率。</p>
<p><b>决定：</b>保留已证明有用的真实观测与跨视角几何表示；本版学习关联不作为已成立贡献，也不扩大到机器人和新窗口。当前证据指向对象级关联可靠性，而不是继续降低接纳阈值。任何后续机制须先说明如何区分“同一对象的不同部分”和“相邻不同对象”，以及何种独立证据能证明这一点；此报告不自动授权另一个阈值或线性头变体。</p>
<h2>真实图像与地图</h2>{figures}
<h2>全部对象明细（含失败与未达最小面积者）</h2><p>最佳 IoU 不等于该对象必定贡献 AP；评分同时依赖面积、忽略域、置信度排序与重复预测匹配。全部详情与所有 GT 重叠保留于 JSON。</p>{object_tables}
<h2>复现与边界</h2><p>审计已重放训练统计、预测概率、源点独占分配、旧 ID 映射、旧点坐标以及完整正负边合并轨迹；终止时不存在正收益组间合并。评测对象序号通过实际地图投影与置信度面积重新对应，并与保存的匹配记录核对。没有重训练、改地图或改门槛。本轮未执行机器人或新窗口评测，也不以旧版本结果替代。</p>
<p><a href="summary.json">完整评分</a> · <a href="object_diagnostics.json">全部对象诊断</a> · <a href="artifact_audit.json">重放审计</a> · <a href="first_mixing_events.json">首次混合事件</a> · <a href="images.json">图像来源</a> · <a href="../../build/benchmarks/signed_query_association_v1/protocol.json">冻结协议</a></p></html>'''
    (REPORT/'index.html').write_text(page)
    dump(REPORT/'images.json', examples)
    for filename in ['summary.json', 'object_diagnostics.json', 'artifact_audit.json', 'first_mixing_events.json']:
        dump(REPORT/filename, read(OUT/filename))
    dump(REPORT/'manifest.json', dict(generator_sha256=sha(__file__), files={f.name: sha(f) for f in REPORT.iterdir() if f.name != 'manifest.json'}))
    print('REPORT', REPORT/'index.html', flush=True)


if __name__ == '__main__':
    main()
