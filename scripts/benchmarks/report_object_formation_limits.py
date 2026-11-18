"""Join existing mask diagnostics to the frozen all-object formation audit."""
import html
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/omni_slam_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
from scipy.spatial import cKDTree
from paper_final_common import ROOT, read, dump, sha
from audit_object_formation_limits import OUT, PARENT, SUPPORT, INPUT

REPORT = ROOT/'output/object_formation_limits_v1'
QUERY = ROOT/'build/benchmarks/query_hypothesis_evidence_v1'


def main():
    if (OUT/'query_join.json').exists():
        for f, digest in read(OUT/'query_join.json')['inputs'].items():
            assert sha(f) == digest
    summary = read(OUT/'summary.json')
    protocol = read(OUT/'protocol.json')
    assert summary['complete'] and protocol['source_sha256'] == sha(ROOT/'scripts/benchmarks/audit_object_formation_limits.py')
    for f, digest in protocol['inputs'].items():
        assert sha(f) == digest
    REPORT.mkdir(parents=True, exist_ok=True)
    inputs = {str(OUT/'summary.json'):sha(OUT/'summary.json')}
    joined, captions, cases = {}, [], []
    for scene in read(PARENT/'protocol.json')['scenes']:
        key = scene['key']
        rawpath = QUERY/'readout_diagnostic'/key/'GT_DIAGNOSTIC_ONLY.npz'
        retrievalpath = QUERY/'evidence'/key/'readout.npz'
        for f in [rawpath, retrievalpath]:
            inputs[str(f)] = sha(f)
        d = np.load(rawpath)
        frames = np.load(retrievalpath)['witness_frames']
        supports = {r['gt_id']:r for r in read(SUPPORT/(key+'.json'))['objects']}
        joined[key] = []
        for r in summary['results'][key]['objects']:
            ids = np.flatnonzero(d['anchor_gt'] == r['gt_id'])
            row = dict(gt_id=r['gt_id'], category=r['category'], seeds_with_known_majority=len(ids),
                existing_same_class_geometry_coverage=supports[r['gt_id']]['coverage']['final_same_class'])
            if len(ids):
                oracle = d['oracle_measured_iou'][ids[0]]
                count = d['target_visible_sources'][ids[0]]
                np.testing.assert_array_equal(d['oracle_measured_iou'][ids], np.broadcast_to(oracle, (len(ids),len(oracle))))
                np.testing.assert_array_equal(d['target_visible_sources'][ids], np.broadcast_to(count, (len(ids),len(count))))
                visible = count >= 16
                row.update(measured_views=int(visible.sum()),
                    best_raw_mask_iou_over_half_views=int((visible & (oracle>.5)).sum()),
                    any_seed_retrieves_iou_over_half_views=int((visible & (d['selected_measured_iou'][ids].max(0)>.5)).sum()),
                    maximum_support_frame=int(frames[np.argmax(count)]))
            else:
                row.update(measured_views=None, best_raw_mask_iou_over_half_views=None,
                    any_seed_retrieves_iou_over_half_views=None, maximum_support_frame=None)
            joined[key].append(row)
        # Representative selection is fixed by raw coverage, before inspecting RGB.
        missing = [r for r in summary['results'][key]['objects'] if r['category']=='building' and r['actual_best_iou']<=.5]
        representative = max(missing, key=lambda r:(r['measured_coverage'], -r['gt_id']))
        gid = representative['gt_id']
        row = next(r for r in joined[key] if r['gt_id']==gid)
        manifest = read(INPUT/key/'manifest.json')
        frame = next(r for r in manifest['records'] if r['frame_id']==row['maximum_support_frame'])
        inputs[frame['rgb']] = sha(frame['rgb'])
        ref = np.load(scene['reference'])
        pred = np.load(PARENT/'maps'/key/'baseline/reference_prediction.npz')['instance']
        mask = ref['train_instance'] == gid
        xyz = ref['xyz'][mask]
        source = np.load(manifest['geometry'])['xyz']
        hit = cKDTree(source).query(xyz, distance_upper_bound=.2, workers=1)[0] < .2
        sameids = [o['object_id'] for o in read(PARENT/'maps'/key/'baseline/classification.json')['objects'] if o['predictions']['mean']['train_id']==2]
        present = np.isin(pred[mask], sameids)
        colors = np.where(present[:,None], np.array([.05,.48,.65]),
                          np.where(hit[:,None], np.array([.95,.59,.14]), np.array([.76,.79,.81])))
        keep = np.arange(0,len(xyz), max(1,int(np.ceil(len(xyz)/14000))))
        center = xyz.mean(0)
        q = xyz[keep]-center
        fig = plt.figure(figsize=(12,5.4), facecolor='white')
        ax = fig.add_subplot(121)
        ax.imshow(Image.open(frame['rgb'])); ax.axis('off')
        ax.set_title(f'Actual RGB: {key[-4:]} / frame {frame["frame_id"]}', fontsize=12)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(q[:,0],q[:,1],q[:,2],c=colors[keep],s=1.6,alpha=.9,depthshade=False,rasterized=True)
        span=np.maximum(np.ptp(q,axis=0),1)
        ax2.set_box_aspect(span); ax2.view_init(elev=22,azim=-55)
        for axis in [ax2.xaxis, ax2.yaxis, ax2.zaxis]:
            axis.set_major_locator(MaxNLocator(3))
        ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]'); ax2.set_zlabel('z [m]')
        ax2.set_title(f'GT {gid}: reference surface diagnosis', fontsize=12)
        fig.suptitle('Object support is missing from the map', fontsize=17, fontweight='bold')
        fig.text(.5,.055,'Blue: assigned to building | Orange: measured but no building assignment | Gray: no source within 20 cm',ha='center',fontsize=9)
        fig.text(.5,.02,'GT-assisted diagnostic visualization; not a new prediction. RGB has no GT mask overlay.',ha='center',fontsize=9,color='#586578')
        fig.tight_layout(rect=[0,.09,1,.93]); filename=key+'.png'
        fig.savefig(REPORT/filename,dpi=165); plt.close(fig)
        cases.append(dict(key=key,gt_id=gid,frame_id=frame['frame_id'],selection='Largest measured coverage among unmatched buildings; earliest maximum measured-support frame.',image=filename))
        captions.append(f'<figure><img src="{filename}"><figcaption>{key} / GT {gid}：实测覆盖 {100*representative["measured_coverage"]:.1f}%，现有最佳对象 IoU {representative["actual_best_iou"]:.3f}。{row["measured_views"]} 个具有至少16个可见目标实测点的帧中，{row["best_raw_mask_iou_over_half_views"]} 帧存在局部实测 IoU&gt;0.5 的原始查询。对象与帧均按声明规则选取，GT 仅用于诊断。</figcaption></figure>')
    dump(OUT/'query_join.json', dict(complete=True,diagnostic_only=True,inputs=inputs,results=joined,cases=cases,
        interpretation='Raw mask score is measured-visible-point IoU with known source labels, not dense image IoU or full-map IoU. Any-seed retrieval is another oracle relaxation, not one executable selector. No seed does not imply no view or no raw mask: those entries remain unknown.',
        geometric_bound='Existing same-class geometry coverage re-queries the full same-class cloud, ignores all other geometry, and permits perfect identity and rejection. It is a stronger necessary coverage check than a fixed reference partition. Neither diagnostic is AP.'))
    counts=[];allrows=[]
    for key,result in summary['results'].items():
        look={r['gt_id']:r for r in joined[key]}
        for cls in ['building','car']:
            rs=[r for r in result['objects'] if r['category']==cls]
            c=result['counts'][cls]['thresholds']['0.5']['above']
            geometric=sum(look[r['gt_id']]['existing_same_class_geometry_coverage']>.5 for r in rs)
            need=sum(r['actual_best_iou']<=.5 and look[r['gt_id']]['existing_same_class_geometry_coverage']<=.5 and r['measured_coverage']>.5 for r in rs)
            counts.append(f'<tr><td>{key[-4:]}</td><td>{cls}</td><td>{len(rs)}</td><td>{c["actual_best_iou"]}</td><td>{c["single_rejection_ceiling"]}</td><td>{c["best_union_iou"]}</td><td>{geometric}</td><td>{c["measured_coverage"]}</td><td>{need}</td></tr>')
        for r in result['objects']:
            q=look[r['gt_id']]
            maskviews='未知' if q['measured_views'] is None else f'{q["best_raw_mask_iou_over_half_views"]}/{q["measured_views"]}'
            cells=[key[-4:],r['category'],str(r['gt_id']),f'{r["actual_best_iou"]:.3f}',
                f'{r["best_union_iou"]:.3f}',f'{q["existing_same_class_geometry_coverage"]:.3f}',
                f'{r["measured_coverage"]:.3f}',maskviews]
            allrows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>')
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam 对象形成机制核查</title>
<style>body{max-width:1180px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#23384a}h1{font-size:30px}h2{margin-top:38px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #dce4eb;text-align:right}th{background:#eef5f8}img{width:100%}figure{margin:25px 0}figcaption{color:#52657a}.lead{padding:18px 24px;background:#eef5f8;border-left:4px solid #148c9e}a{color:#087d9a}</style>
<h1>omni_slam：完整对象在哪里丢失</h1>
<p>2026-11-14 · 94个GT对象 · 两个已暴露室外开发窗口 · 现有强基线保持</p>
<div class="lead"><strong>决策：停止把旧地图的删除、合并作为主要恢复机制。</strong> 两窗口没有任何漏检建筑能靠理想旧碎片合并跨过 IoU 0.5。12个漏检建筑的原始测量覆盖仍超过一半，但现有同类地图几何达不到；必须重新形成这些缺失的对象表面。原始查询在部分视图中存在可用掩码，但这不保证联合后的对象纯度、身份或AP。</div>
<h2>固定 IoU 0.5 的诊断计数</h2>
<table><tr><th>窗口</th><th>类别</th><th>GT数</th><th>实际最佳IoU</th><th>理想单对象去污染*</th><th>理想碎片合并*</th><th>现有同类几何覆盖†</th><th>全部测量覆盖†</th><th>需新增支持</th></tr>'''+''.join(counts)+'''</table>
<p>* 真值辅助的参考空间诊断：单对象去污染允许完美删除错误参考标签；合并对每个GT独立求最优旧实例子集。不同GT可能要求同一旧实例，不能同时实现。前者也没有重新计算删点后的空间竞争。† 几何覆盖只是必要条件，忽略误检和实例分配。这些列均不是新方法AP或官方榜单结果。</p>
<h2>实际数据中的缺口</h2>'''+''.join(captions)+'''
<h2>下一机制必须满足什么</h2>
<p>恢复未进入地图的实测表面，既要补全已有对象，也要形成尚无对象锚点的新实例，同时解决新增部分的对象/环境区分和跨实例边界。此前直接扩大弱掩码支持已严重污染地图，因此不能把整块低分掩码接入当作解决办法。</p>
<p>下一次候选应让“局部表面片段与完整对象的关系”成为显式决策，允许多片段构成对象，不能只对旧ID删点或把整个弱掩码视为同一个实例。它仍是待实现、待对照的假设，不宣称创新已成立。必须共享观测和候选，至少比较简单几何/语义接纳、整掩码接纳与该机制，并以完整地图新增正确对象、原正确对象得失和AP验收；新的训练或地图运行前另行冻结唯一版本。</p>
<p>机器人与新增独立评测继续等待同一有效版本。当前诊断不会替代这两项，也不会将已暴露窗口重新称为独立测试。</p>
<h2>全部94个对象</h2><p>可用查询视图数：在≥16个目标实测点可见的帧中，是否存在同类原始查询的局部实测 IoU&gt;0.5；由GT挑选，不能直接作为预测选择器。无已知纯种子时记“未知”，不记零。</p>
<table><tr><th>窗口</th><th>类别</th><th>GT</th><th>实际IoU</th><th>最优合并IoU</th><th>现有同类覆盖</th><th>原始测量覆盖</th><th>可用查询/可见视图</th></tr>'''+''.join(allrows)+'''</table>
<p><a href="../../build/benchmarks/object_formation_limits_v1/summary.json">完整诊断</a> · <a href="../../build/benchmarks/object_formation_limits_v1/query_join.json">查询证据与图像来源</a> · <a href="../../build/benchmarks/object_formation_limits_v1/protocol.json">审计协议</a></p></html>'''
    (REPORT/'index.html').write_text(body)
    for f,digest in inputs.items():
        assert sha(f)==digest
    dump(OUT/'report_manifest.json',dict(complete=True,source_sha256=sha(__file__), files={str(f):sha(f) for f in REPORT.iterdir() if f.is_file()},
        diagnostic_inputs_verified=True,goal_complete=False,new_mechanism_validated=False,robot_same_method=False,new_independent_evaluation=False))
    print('REPORT COMPLETE', REPORT/'index.html', flush=True)


if __name__=='__main__':
    main()
