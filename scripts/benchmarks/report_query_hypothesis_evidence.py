"""Actual RGB and sealed query choices; all visible views of fixed cases."""
from pathlib import Path
import html
import numpy as np
from PIL import Image
from paper_final_common import ROOT, read, dump, sha
from audit_query_hypothesis_evidence import OUT as AUDIT
from extract_query_hypothesis_evidence import OUT as REPRESENTATION, sealed
from capture_prethreshold_candidates import OUT as RAW, INPUT

REPORT = ROOT/'output/query_hypothesis_evidence_v1'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit = read(AUDIT/'summary.json'); evidence = sealed('evidence_complete')
    assert audit['evidence_seal_sha256'] == sha(REPRESENTATION/'evidence_complete.json')
    for path,h in audit['files'].items(): assert sha(path)==h
    capture = read(RAW/'predictions_complete.json')
    REPORT.mkdir(parents=True,exist_ok=True)
    groups = ''; figures = ''; examples = ''; provenance = {}
    for key,result in audit['results'].items():
        for name,g in result['groups'].items():
            def value(x): return '—' if x is None else f'{x:.3f}'
            groups += '<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in [key,name,g['hypotheses'],g['same_class_majority'],g['wrong_class_or_mixed'],g['unknown'],g['same_class_three_agreements'],g['wrong_or_mixed_three_agreements'],g['unknown_three_agreements'],g['eligible_seed_view_pairs'],g['iou_over_half_pairs'],g['oracle_iou_over_half_pairs'],value(g['mean_iou'])])+'</tr>'
        manifest = read(INPUT/key/'manifest.json'); measured = {r['frame_id']:r for r in manifest['records']}
        predictions = {r['frame_id']:r for r in capture['results'][key]}
        for name,case in result['fixed_examples'].items():
            # Keep all measured-visible target views, including misses, plus seed.
            rows = [r for r in case['frames'] if r['target_visible_sources']>=16 or r['seed_frame']]
            count = len(rows); cols = 3; grid_rows = (count+cols-1)//cols
            fig,axes = plt.subplots(grid_rows, cols, figsize=(15,grid_rows*2.4),squeeze=False)
            for ax in axes.flat: ax.axis('off')
            shown = []
            for ax,row in zip(axes.flat,rows):
                frame = row['frame_id']; inp = measured[frame]; rec = predictions[frame]
                assert sha(inp['rgb']) == manifest['input_hashes'][inp['rgb']]
                assert sha(rec['file']) == capture['files'][rec['file']]
                rgb = np.asarray(Image.open(inp['rgb']).convert('RGB')); ax.imshow(rgb)
                query = case['fixed_seed']['query_id'] if row['seed_frame'] else row['query_id']
                mask_score = None
                if query>=0:
                    data = np.load(rec['file']); mask = np.unpackbits(data['masks_packed'][query],axis=1)[:,:rgb.shape[1]].astype(bool)
                    overlay = np.zeros((*mask.shape,4),np.float32); overlay[mask]=[1.,.55,.05,.38]
                    ax.imshow(overlay); mask_score=float(data['scores'][query])
                    visible = np.load(inp['file']); pixels = visible['pixel_xy']; selected = mask[pixels[:,1],pixels[:,0]]
                    ax.scatter(pixels[selected,0],pixels[selected,1],s=.35,c='#00eab0',alpha=.55)
                title = f'Frame {frame} | '+('SEED (not confirmation)' if row['seed_frame'] else (f'query {query}' if query>=0 else 'NO CORRESPONDENCE'))
                if row['seed_frame']: subtitle=f'native score {mask_score:.4f}'
                else: subtitle=f'score {row["native_score"]:.4f} | cell IoU {row["cell_agreement"]:.2f} | GT measured IoU {row["measured_iou"]:.2f}'
                ax.set_title(title+'\n'+subtitle,fontsize=9)
                shown.append(dict(frame_id=frame,query_id=query,rgb=inp['rgb'],rgb_sha256=sha(inp['rgb']),seed_frame=row['seed_frame']))
            fig.suptitle(name+' | fixed diagnostic example | all target-visible measured views',fontsize=13)
            fig.tight_layout(); filename=name.replace('/','_')+'.png'
            fig.savefig(REPORT/filename,dpi=150,bbox_inches='tight'); plt.close(fig)
            s=case['summary']
            examples += '<tr>'+''.join(f'<td>{v}</td>' for v in [name,s['other_measured_views'],s['retrieved'],s['iou_over_half'],s['oracle_over_half'],s['weak_good']])+'</tr>'
            figures += f'<h3>{html.escape(name)}</h3><img src="{filename}"><p>橙色：已封存检索所选原始预测掩码；绿色：该掩码内当前可见的真实测量投影。种子帧单独标出，不参与确认。显示全部至少有 16 个目标可信测量点的视图及种子帧，包含失败帧。GT measured IoU 只作事后诊断。</p>'
            provenance[name]=shown
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 物理对象假设的跨视角证据读取</title>
<style>body{{font:16px/1.7 system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#193349}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:7px;border:1px solid #d4dfe8;text-align:right}}.scroll{{overflow:auto}}.note{{padding:20px;background:#edf5f7;border-left:5px solid #078e82}}</style>
<h1>omni_slam：三维对象假设能否找回其他视角的证据？</h1>
<p class="note">本页验证一个证据读取模块，尚未生成新对象地图。算法选择查询时不读取 GT；所有选择封存后才读取旧开发标签。以下纯度、IoU 和正确性均为诊断，不是对象接纳条件或 benchmark AP。重复假设和邻近视图并不独立。</p>
<h2>本轮结论与去留</h2><p><b>物理假设能取回部分被低分筛掉的有效观测，但重复几何支持不能直接接纳对象。</b>固定案例 0002 / 2011 在非种子帧中取回全部三个可用视角，所选掩码分数都≤0.1；2012 只取回两个可用视角中的一个。0002 满足固定“三帧一致”描述量的55个假设中，35个在标签诊断下为错类或混合假设；它们可能重叠，不是35个独立误检对象。</p><p>决定：保留这项证据表示，暂不把简单三帧确认接入地图。下一项应在同一候选池上检验对象接纳与竞争，保留原生分数、简单确认等强对照；尚无新的地图 AP、机器人优势或独立泛化结论。</p><p>执行中修正了一处 NPZ 重复解压造成的内存问题，算法及原协议不变；原代码和失败日志保留，抽样新旧数组及检索结果精确一致，四项测试通过。<a href="../../build/benchmarks/query_hypothesis_evidence_v1/technical_amendment_npz_io.json">技术修订与验证记录</a>。</p><h2>所有假设：有效证据与错误重复支持</h2>
<p>high：原生分数 &gt;0.5；deferred：0.1&lt;分数≤0.5；weak：0&lt;分数≤0.1。只为沿用原分组报告，没有据此调整接纳阈值。同类多数指已知种子点的置信度加权同类实例纯度 &gt;50%；未知点不当作背景。“三帧一致”是固定描述量：至少三个非种子帧的可见假设单元≥16，且所选查询单元 IoU &gt;0.5；不是已实现的建图规则。</p>
<div class="scroll"><table><tr><th>窗口</th><th>分组</th><th>假设数</th><th>同类多数</th><th>错类/混合</th><th>未知</th><th>同类三帧一致</th><th>错类/混合三帧一致</th><th>未知三帧一致</th><th>可评种子-视图对</th><th>检索 IoU&gt;.5</th><th>GT 最佳 IoU&gt;.5</th><th>平均检索 IoU</th></tr>{groups}</table></div>
<p>视图 IoU 在当前可见、标签可信的实测源点上计算，忽略未知标签；可评对要求种子有同类多数参考身份，且该目标在其他视图至少有16个可信实测点。GT 最佳查询仅表示信息上限参考，不是可部署对照。不得将重复种子-视图对的计数解释为正确对象数。</p>
<h2>此前固定四个案例：全部有效可见视图</h2>
<table><tr><th>案例</th><th>其他可见帧</th><th>取得查询</th><th>检索 IoU&gt;.5</th><th>GT 最佳 IoU&gt;.5</th><th>其中分数≤.1的有效检索</th></tr>{examples}</table>
{figures}<h2>与主线的关系</h2>
<p>下一步须判断该表示能否支持对象接纳与去重，而非只找回掩码。完整地图必须与原强基线、固定宽松入口和简单确认规则比较，检查新增正确对象、误检与丢失。当前没有训练、机器人或新增独立窗口成绩；原失败门槛保持不变。</p>
<p><a href="summary.json">全部诊断</a> · <a href="images.json">图像与查询来源</a> · <a href="../prethreshold_candidates_v1/observability.html">先前可见机会核查</a></p></html>'''
    (REPORT/'index.html').write_text(page)
    dump(REPORT/'summary.json',audit); dump(REPORT/'images.json',provenance)
    dump(REPORT/'evidence.json',evidence)
    print('REPORT',REPORT/'index.html',flush=True)


if __name__=='__main__': main()
