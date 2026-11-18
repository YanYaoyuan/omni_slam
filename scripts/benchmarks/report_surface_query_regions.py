"""Causal selection audit and all preselected real-image evidence."""
import numpy as np
from PIL import Image
from paper_final_common import ROOT, read, dump, sha
from probe_surface_query_regions import OUT, FRONT, INPUT, POLICIES, verify
from omni_semantic_map.surface_query_regions import SurfaceQueryMemory, crop_grid

REPORT = ROOT/'output/surface_query_regions_probe_v1'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    protocol = verify(); selection = read(OUT/'selection.json'); originals = read(FRONT/'masks_complete.json')
    for f, h in selection['files'].items(): assert sha(f) == h
    REPORT.mkdir(parents=True, exist_ok=True); table = ''; sections = ''; evidence = {}; total = 0; versus_current = 0; versus_geometry = 0
    for scene in protocol['scenes']:
        key = scene['key']; manifest = read(INPUT/key/'manifest.json'); cells = np.load(INPUT/'query_cells'/key/'source_to_cell.npy'); by_frame = {r['frame_id']: r for r in originals['scenes'][key]['records']}; probes = {r['frame_id']: r for r in selection['results'][key]}; memory = SurfaceQueryMemory(); checks = []
        for r in manifest['records']:
            f = np.load(r['file']); ids = f['visible_source_ids']; px = f['pixel_xy']; masks = np.load(by_frame[r['frame_id']]['file']); positive = masks['segmentation'][px[:, 1], px[:, 0]] >= 0
            assert np.all(ids < r['map_points']), 'Future source in visibility'
            memory.observe(r['frame_id'], cells[ids], positive)
            if r['frame_id'] not in probes: continue
            selected = probes[r['frame_id']]; trace = np.load(OUT/'selection'/key/f'{r["frame_id"]:06d}.npz')
            for field, expected in [('source_ids', ids), ('pixel_xy', px), ('cells', cells[ids]), ('covered', positive), ('seen_count', memory.seen[cells[ids]]), ('covered_count', memory.covered[cells[ids]])]: np.testing.assert_array_equal(trace[field], expected)
            replay = memory.rank(r['frame_id'], cells[ids], px, trace['weights'], positive, crop_grid(selected['width'], selected['height']), selected['chosen']['uniform'])
            assert replay['chosen'] == selected['chosen'] and replay['scores'] == selected['scores']
            checks.append(dict(frame_id=r['frame_id'], causal_prefix_exact=True, original_rgb_sha256=sha(r['rgb'])))
            current = selected['chosen']['current'] != selected['chosen']['memory']; geometry = selected['chosen']['geometry'] != selected['chosen']['memory']; total += 1; versus_current += current; versus_geometry += geometry
            table += f'<tr><td>{key}</td><td>{r["frame_id"]}</td>'+''.join(f'<td>{selected["chosen"][method]}</td>' for method in POLICIES)+f'<td>{selected["repeat_visible_cells"]}/{selected["visible_cells"]}</td></tr>'
            image = np.asarray(Image.open(r['rgb']).convert('RGB')); h, w = image.shape[:2]; dense = np.unpackbits(masks['masks_packed'], axis=2)[:, :, :w].astype(bool)
            fig, axes = plt.subplots(1, 3, figsize=(15, 2.2))
            for ax in axes: ax.imshow(image); ax.axis('off')
            axes[0].set_title('Actual RGB', fontsize=10); color = np.zeros((h, w, 4))
            for sem, value in [(26, [.0, .75, .85, .4]), (11, [1., .6, .1, .4])]: color[dense[masks['semantic_ids']==sem].any(0)] = value
            axes[1].imshow(color); axes[1].set_title('Existing masks: cyan car / orange building', fontsize=9)
            for box_id, box in enumerate(selected['boxes']):
                x0, y0, x1, y1 = box; axes[2].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='#dddddd', linewidth=.6))
            box = selected['boxes'][selected['chosen']['memory']]; x0, y0, x1, y1 = box
            axes[2].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='#00e5ab', linewidth=3))
            axes[2].set_title(f'Memory crop {selected["chosen"]["memory"]} | geometry {selected["chosen"]["geometry"]} | current {selected["chosen"]["current"]}', fontsize=9)
            fig.suptitle(key+f' | fixed probe frame {r["frame_id"]}', fontsize=10); fig.tight_layout(); name=f'{key}_{r["frame_id"]:06d}.png'; fig.savefig(REPORT/name, dpi=150, bbox_inches='tight'); plt.close(fig)
            sections += f'<h3>{key} · 帧 {r["frame_id"]}</h3><img src="{name}"><details><summary>六个裁剪候选的实际得分</summary><pre>'+str(selected['scores'])+'</pre></details>'
        evidence[key] = checks
    stopped = not selection['pre_inference_gate']; assert not stopped or not (OUT/'predictions_complete.json').exists()
    result = dict(complete=True, selection_only=True, stopped_before_inference=stopped, frames=total, changed_vs_current=versus_current, changed_vs_geometry=versus_geometry, causal_replay=evidence, selection_sha256=sha(OUT/'selection.json'), no_gt_consumed=True, no_new_gpu_inference=True)
    dump(OUT/'selection_audit.json', result); dump(REPORT/'summary.json', result); dump(REPORT/'selection.json', selection)
    (REPORT/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 新观测查询预检</title><style>body{{font:16px/1.7 system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#193349}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border:1px solid #d5dfe8;text-align:right}}td:first-child{{text-align:left}}.note{{background:#edf5f7;border-left:5px solid #058e85;padding:20px}}pre{{white-space:pre-wrap;font-size:13px}}</style><h1>从未解释实测表面请求新观测：推理前预检</h1><p class="note"><b>预设选择门槛未通过，已停止在 GPU 推理之前。</b><br>8 个预先按索引指定的开发帧中，历史记忆相对纯几何改变 {versus_geometry}/{total} 次选择，相对单帧漏检改变 {versus_current}/{total} 次。没有新掩码、地图、训练、AP 或机器人准确率结果。没有查看 GT 来选择帧或裁剪。</p><h2>全部选择结果</h2><table><tr><th>窗口</th><th>帧</th><th>均匀</th><th>几何</th><th>单帧漏检</th><th>历史记忆</th><th>重复可见单元 / 当前可见单元</th></tr>{table}</table><p>裁剪编号来自共同的六个固定重叠窗口。几何、单帧漏检与记忆三种策略以当前可见实测表面的局部协方差计算竖直面权重；单帧项乘当前未被原掩码覆盖的指示量，记忆项乘跨帧平滑缺失频率。相同单元每帧只投票一次。原全图 SAM3 观测是共同已付出的输入。</p><h2>为什么现在停止</h2><p>这不是完整地图上的负结果，也不证明局部重查询无效。它表明当前记忆项没有相对几何控制改变动作，不能用后续同一裁剪产生的新掩码来证明记忆贡献。按冻结协议，不修改裁剪网格、权重或窗口来寻找收益，不运行尚不具备区分性的完整实验。</p><h2>真实输入与所选区域（全部 8 帧）</h2><p>绿色框为记忆选择；全部图像按固定时间索引列出，未挑选好看样例。掩码来自已经封存的全图预测，非 GT。原图中的未覆盖区域不必然属于目标类别。</p>{sections}<h2>与既有工作及主线的关系</h2><p><a href="https://arxiv.org/html/2603.26541v1">OVI-MAP</a> 已研究对象视角覆盖；<a href="https://arxiv.org/html/2609.25490v1">SAM-V</a> 已将几何上下文引入多视图分割；<a href="https://arxiv.org/abs/2606.16996">ActiveSAM</a> 研究图像条件类别筛选；<a href="https://arxiv.org/abs/2610.18356">Urbex</a> 包含主动空间查询。因此一般的视角选择、裁剪或几何提示都不能直接作为我们的新贡献。本原型只探查持久的测量区域未解释状态是否产生独立查询作用，目前未达到预设条件。</p><p>omni_slam 主线仍是对象级语义地图的完整性和准确性。本次保持原基线与所有既有地图不变；未将机器人或独立评测目标取消。下一机制需首先改变真实对象观测或对象表示，并证明优于同条件简单控制。</p><h2>复核材料</h2><p><a href="summary.json">因果回放核验</a> · <a href="selection.json">完整选择与得分</a></p></html>''')
    print('REPORT', REPORT/'index.html', result, flush=True)


if __name__ == '__main__': main()
