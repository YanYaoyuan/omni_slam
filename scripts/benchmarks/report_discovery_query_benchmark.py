"""Read-only full-map report; GT diagnostics never feed the frozen policies."""
from html import escape
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_discovery_query_benchmark import OUT,INPUT,METHODS,verify

REPORT=ROOT/'output/discovery_query_benchmark_v1'
NAMES=dict(uniform='Uniform',geometric='Geometry only',residual='Ours candidate: residual memory',all_frames='All frames (larger budget)')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=verify(True);summary=read(OUT/'summary.json');seal=read(OUT/'predictions_complete.json')
    assert summary['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    REPORT.mkdir(parents=True,exist_ok=True);rows='';details='';audits={}
    masks=read(OUT/'masks_complete.json')
    def fmt(v):return '—' if v is None else f'{100*v:.3f}'
    for s in p['scenes']:
        key=s['key'];result=summary['results'][key];manifest=read(INPUT/key/'manifest.json')
        source=np.load(manifest['geometry'])['xyz'];sample=np.linspace(0,len(source)-1,min(60000,len(source)),dtype=int)
        fig,axes=plt.subplots(1,4,figsize=(17,5),sharex=True,sharey=True)
        queries={};best={};object_rows=''
        for ax,method in zip(axes,METHODS):
            folder=OUT/'maps'/key/method;q=read(folder/'queries.json');queries[method]=q
            assert read(folder/'policy_replay.json')['identical_events']
            row=result['methods'][method];metric=row.get('metrics',{})
            rows+=f'<tr class="{"ours" if method=="residual" else ""}"><td>{key}</td><td>{NAMES[method]}</td><td>{q["image_queries"]}</td><td>{row["run"]["objects"]}</td><td>{fmt(metric.get("allAp50%"))}</td><td>{fmt(metric.get("allAp"))}</td><td>{fmt(metric.get("classes",{}).get("car",{}).get("ap50%"))}</td><td>{fmt(metric.get("classes",{}).get("building",{}).get("ap50%"))}</td><td>{row["frontend_seconds"]:.2f}</td><td>{row["run"]["backend_seconds"]:.2f}</td></tr>'
            g=np.load(folder/'geometry.npz');xyz=g['xyz'];labels=g['instance'];classification=read(folder/'classification.json')['objects']
            ax.scatter(source[sample,0],source[sample,1],c='#e0e3e8',s=.5,rasterized=True)
            ix=np.linspace(0,len(xyz)-1,min(60000,len(xyz)),dtype=int) if len(xyz) else np.empty(0,int)
            cars=[o['object_id'] for o in classification if o['predictions']['mean']['semantic_id']==26]
            ax.scatter(xyz[ix,0],xyz[ix,1],c=np.where(np.isin(labels[ix],cars),'#008cba','#ed9b32'),s=1,rasterized=True)
            ax.set_aspect('equal');ax.set_title(NAMES[method]+f'\n{len(classification)} objects / {q["image_queries"]} queries',fontsize=9);ax.set_xlabel('x / m')
            if 'metrics' in row:
                matches=read(folder/'matches.json')['window']['groundTruth'];best[method]={}
                for category,objects in matches.items():
                    for obj in objects:
                        if obj['instance_id']<1000 or obj['vert_count']<100:continue
                        ious=[m['intersection']/(obj['vert_count']+m['vert_count']-m['intersection']) for m in obj['matched_pred']]
                        best[method][obj['instance_id']]=dict(category=category,best_same_class_iou=max(ious,default=0.))
        axes[0].set_ylabel('y / m');fig.suptitle(key+' | Actual final maps: blue car, orange building, grey measured context')
        fig.tight_layout();fig.savefig(REPORT/(key+'_maps.png'),dpi=160);plt.close(fig)
        diffs={c:len(set(queries[c]['selected_frame_ids'])^set(queries['residual']['selected_frame_ids'])) for c in ['uniform','geometric']}
        if best:
            for gid in sorted(best['uniform']):
                object_rows+=f'<tr><td>{gid}</td><td>{best["uniform"][gid]["category"]}</td>'+''.join(f'<td>{best[m][gid]["best_same_class_iou"]:.4f}</td>' for m in METHODS)+'</tr>'
        audits[key]=dict(query_selection_symmetric_difference_vs_ours=diffs,per_gt=best,changes=result['changes_vs_controls'])
        details+=f'<h2>{key}</h2><img src="{key}_maps.png"><p>完整实测窗口的原生 CG 最终地图，统一坐标和比例；显示最多各 60,000 点，评分使用完整参考域。颜色表示预测类别，不表示准确率或跨方法统一的对象 ID。</p><p>Ours 候选与 Uniform / Geometry 的选帧集合对称差：{diffs["uniform"]} / {diffs["geometric"]} 帧。门槛：{result["gate"]}。新增/丢失正确匹配：<code>{escape(str(result["changes_vs_controls"]))}</code></p>'
        if best:details+=f'<details><summary>完整逐对象最佳同类 IoU（评分后诊断）</summary><table><tr><th>GT</th><th>类别</th>'+''.join(f'<th>{NAMES[m]}</th>' for m in METHODS)+f'</tr>{object_rows}</table></details>'
        # Fixed temporal midpoint of residual queries, not selected by score/GT.
        selected=queries['residual']['selected_frame_ids'];frame=selected[len(selected)//2]
        measured=next(r for r in manifest['records'] if r['frame_id']==frame)
        cached=next(r for r in masks['scenes'][key]['records'] if r['frame_id']==frame)
        rgb=np.asarray(Image.open(measured['rgb']).convert('RGB'));z=np.load(cached['file']);seg=z['segmentation']
        overlay=rgb.astype(float).copy();valid=seg>=0;color=plt.get_cmap('tab20')(np.maximum(seg,0)%20)[...,:3]*255
        overlay[valid]=.55*overlay[valid]+.45*color[valid]
        fig,axes=plt.subplots(1,2,figsize=(12,4));axes[0].imshow(rgb);axes[1].imshow(overlay.astype(np.uint8))
        for ax,title in zip(axes,['Original camera image','Shared SAM3 segmentation (not the contribution)']):ax.set_title(title);ax.axis('off')
        fig.suptitle(f'{key} / frame {frame}: temporal midpoint of Ours selected frames');fig.tight_layout();fig.savefig(REPORT/(key+'_image.png'),dpi=150);plt.close(fig)
        details+=f'<img src="{key}_image.png"><p>原始数据与实际分割，按候选选帧列表的时间中点固定取图；车库没有独立标签，不报告语义准确率。</p>'
    dump(REPORT/'audit.json',audits);dump(REPORT/'summary.json',summary)
    passed=summary['advance_gate'];decision='达到预设开发门槛；下一步才可做独立场景验证。' if passed else '未达到预设开发门槛；封存这一固定策略，不追加阈值搜索，也不替换已冻结强基线。'
    html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 对象发现记忆等预算评测</title><style>body{{font:16px/1.65 system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#183147;background:#f8fafc}}h1,h2{{color:#12304a}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}td,th{{border:1px solid #d8e0e8;padding:8px;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}img{{width:100%;background:white;margin-top:18px}}.ours{{background:#e1f4ee}}.note{{padding:20px;background:#e9eef5;border-left:5px solid #138b80}}code{{overflow-wrap:anywhere}}</style>
    <h1>omni_slam：对象发现记忆的完整地图评测</h1><p class="note"><b>{decision}</b><br>主线：持续对象级语义建图；LIVO 提供机器人位姿和实测几何。研究问题是在固定分割预算下，利用尚未被对象解释的实测区域选择观测，减少地图漏检。SAM3 与 SigLIP 是共享工具，不是本轮贡献。</p>
    <h2>相同前端、相同建图后端、相同预算</h2><p>Uniform、Geometry only、Ours candidate 均每四帧查询一帧；全帧对照使用更大预算。固定 car/building 提示、分割阈值 0.5、20 cm 查询单元；复用原生 CG/history2 和相同最终处理。全部 12 份预测先封存，再读取 GT 评分。每份选帧日志均通过只读取已选择观测结果的因果回放。</p>
    <table><tr><th>场景</th><th>方法</th><th>图像查询</th><th>地图对象</th><th>AP50</th><th>AP50:90</th><th>车辆 AP50</th><th>建筑 AP50</th><th>前端 GPU s*</th><th>建图 s</th></tr>{rows}</table>
    <p>AP 均乘 100。两个室外窗口已经用于开发，是本地适配评测，不是独立测试或官方榜单。*前端时间是缓存中被选择帧的实际 GPU forward 时间之和，排除加载与 I/O；不是在线端到端运行时间。全帧缓存用于复用输入，不等于各策略的在线执行成本。原 SAM2 结果不能与此表直接相减后归因于查询记忆。</p>
    <p>预设继续门槛：在每个室外窗口分别超过两个等预算对照至少 2 AP50 点，分别新增至少一个正确 GT 匹配，且不丢失已有正确匹配。对机器人仅检查完整链路和可视化。</p>{details}
    <h2>范围与证据</h2><p>“已解释”仅表示过去查询掩码包含该实测区域，不能保证其真实对象归属；误分割可能抑制后续查询。当前策略是固定启发式，不是学习模型或校准的信息增益。</p><p>冻结协议：<code>{OUT/'protocol.json'}</code><br>预测封存：<code>{OUT/'predictions_complete.json'}</code><br><a href="audit.json">逐对象及选帧审计</a> · <a href="summary.json">完整结果</a></p></html>'''
    (REPORT/'index.html').write_text(html)
    print('REPORT',REPORT/'index.html',decision,flush=True)


if __name__=='__main__':main()
