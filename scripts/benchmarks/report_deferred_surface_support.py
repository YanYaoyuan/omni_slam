"""Verified final maps, real support overlays, complete per-target diagnostics."""
from html import escape
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_deferred_surface_support import OUT,PARENT,INPUT,METHODS,verify

REPORT=ROOT/'output/deferred_surface_support_v1'
NAMES=dict(largest='Largest component',all='All raw supports',pointwise='Pointwise confirmation',component='Ours candidate: component recovery')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=verify(True);summary=read(OUT/'summary.json');seal=read(OUT/'predictions_complete.json');support=read(OUT/'supports_complete.json')
    assert sha(OUT/'predictions_complete.json')==summary['prediction_seal_sha256']
    for f,h in seal['files'].items():assert sha(f)==h,f
    REPORT.mkdir(parents=True,exist_ok=True);table='';body='';audit={}
    def fmt(x):return '—' if x is None else f'{100*x:.3f}'
    for s in p['scenes']:
        key=s['key'];scene=summary['results'][key];manifest=read(INPUT/key/'manifest.json');xyz=np.load(manifest['geometry'])['xyz']
        records=read(PARENT/'masks_complete.json')['scenes'][key]['records'];byframe={r['frame_id']:r for r in manifest['records']}
        source_ix=np.linspace(0,len(xyz)-1,min(60000,len(xyz)),dtype=int);fig,axes=plt.subplots(1,4,figsize=(17,4),sharex=True,sharey=True);gt={};unique={};counts={}
        for ax,method in zip(axes,METHODS):
            folder=OUT/'maps'/key/method;run=scene['methods'][method]['run'];m=scene['methods'][method].get('metrics',{})
            table+=f'<tr class="{"ours" if method=="component" else ""}"><td>{key}</td><td>{NAMES[method]}</td><td>{fmt(m.get("allAp"))}</td><td>{fmt(m.get("allAp50%"))}</td><td>{fmt(m.get("classes",{}).get("car",{}).get("ap"))}</td><td>{fmt(m.get("classes",{}).get("building",{}).get("ap"))}</td><td>{run["objects"]}</td><td>{run["map_points"]}</td><td>{run["backend_seconds"]:.2f}</td></tr>'
            g=np.load(folder/'geometry.npz');points=g['xyz'];labels=g['instance'];rows=read(folder/'classification.json')['objects']
            sample=np.linspace(0,len(points)-1,min(60000,len(points)),dtype=int) if len(points) else np.empty(0,int)
            cars=[r['object_id'] for r in rows if r['predictions']['mean']['semantic_id']==26]
            ax.scatter(xyz[source_ix,0],xyz[source_ix,1],s=.4,c='#dfe4e8',rasterized=True)
            ax.scatter(points[sample,0],points[sample,1],s=1,c=np.where(np.isin(labels[sample],cars),'#008fb7','#d99420'),rasterized=True)
            ax.set_title(NAMES[method]+f'\n{run["objects"]} predicted objects',fontsize=9);ax.set_aspect('equal');ax.set_xlabel('x / m')
            seen=np.zeros(len(xyz),bool);counts[method]=[]
            for r in records:
                z=np.load(OUT/'supports'/key/method/f'{r["frame_id"]:06d}.npz');seen[z['point_ids']]=True;counts[method].append(len(z['point_ids']))
            unique[method]=int(seen.sum())
            if m:
                gt[method]={}
                for category,objects in read(folder/'matches.json')['window']['groundTruth'].items():
                    for obj in objects:
                        if obj['instance_id']<1000 or obj['vert_count']<100:continue
                        gt[method][obj['instance_id']]=dict(category=category,best_iou=max((x['intersection']/(obj['vert_count']+x['vert_count']-x['intersection']) for x in obj['matched_pred']),default=0.))
        axes[0].set_ylabel('y / m');fig.suptitle(key+' | Actual final geometry: blue car / orange building / grey measured context',fontsize=11)
        fig.tight_layout(rect=[0,0,1,.90]);fig.savefig(REPORT/(key+'_maps.png'),dpi=160,bbox_inches='tight');plt.close(fig)
        stats=support['stats'][key];body+=f'<h2>{key}</h2><img src="{key}_maps.png"><p>相同坐标和尺度，显示最多每组 60,000 点；评分使用完整参考域。颜色表示类别，不表示准确率。原始支持阶段的唯一源点数：{escape(str(unique))}；不是最终正确点数。</p><p>候选新增 {stats["added_point_observations"]["component"]:,} 次点—观测支持，恢复计算 {stats["recovery_seconds"]:.3f} s（不含后续 CG 完整重放）；已验证两个真实前缀。原最大连通域对照的地图坐标逐元素、实例标签及分类/关联完全复现。</p><p>相对各对照的 AP50 正确匹配变化：<code>{escape(str(scene["changes_vs_controls"]))}</code></p>'
        if gt:
            rows=''
            for gid in sorted(gt['largest']):rows+=f'<tr><td>{gid}</td><td>{gt["largest"][gid]["category"]}</td>'+''.join(f'<td>{gt[m][gid]["best_iou"]:.4f}</td>' for m in METHODS)+'</tr>'
            body+='<details><summary>全部目标的最佳同类 IoU（评分后诊断）</summary><table><tr><th>GT ID</th><th>类别</th>'+''.join(f'<th>{NAMES[m]}</th>' for m in METHODS)+f'</tr>{rows}</table></details>'
        # Select by executed change magnitude, never by score or GT correctness.
        rank=int(np.argmax(np.asarray(counts['component'])-np.asarray(counts['largest'])));r=records[rank];fid=r['frame_id'];frame=byframe[fid]
        rgb=np.asarray(Image.open(frame['rgb']).convert('RGB'));measured=np.load(frame['file']);visible=measured['visible_source_ids'];pixels=measured['pixel_xy']
        base=np.load(OUT/'supports'/key/'largest'/f'{fid:06d}.npz')['point_ids'];fig,axes=plt.subplots(1,4,figsize=(17,3))
        for ax,method in zip(axes,METHODS):
            selected=np.load(OUT/'supports'/key/method/f'{fid:06d}.npz')['point_ids'];in_base=np.isin(visible,base);extra=np.isin(visible,selected)&~in_base
            ax.imshow(rgb);ax.scatter(pixels[in_base,0],pixels[in_base,1],s=3,c='#00d8b0',alpha=.7);ax.scatter(pixels[extra,0],pixels[extra,1],s=5,c='#ff3e6d',alpha=.8)
            ax.set_title(NAMES[method]+f'\n{len(selected)} measured support observations',fontsize=8);ax.axis('off')
        fig.suptitle(f'{key} / frame {fid} | Historical support after delayed revision: green original, pink recovered',fontsize=10)
        fig.tight_layout();fig.savefig(REPORT/(key+'_support.png'),dpi=160,bbox_inches='tight');plt.close(fig)
        body+=f'<img src="{key}_support.png"><p>实际相机图与原始实测支持；按候选补回点数最多的帧固定取图，不按准确率挑选。粉色是窗口结束后恢复的历史支持，不能解释为当帧即时可得。GT 没有用于恢复或选图。</p>'
        audit[key]=dict(unique_source_points=unique,per_gt=gt,example_frame=fid,example_rgb=frame['rgb'],example_rgb_sha256=sha(frame['rgb']),prefix_checks=stats['prefix_checks'])
    passed=summary['advance_gate'];decision='达到本轮开发门槛；仍需新保留窗口和外部强方法验证。' if passed else '未达到预设开发门槛；封存该候选，不调阈值，不替换原强基线。'
    meanrows=''.join(f'<tr><td>{NAMES[m]}</td><td>{fmt(summary["mean_primary_ap"][m])}</td></tr>' for m in METHODS)
    (REPORT/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 延迟支持恢复</title><style>body{{font:16px/1.65 system-ui;max-width:1500px;margin:30px auto;padding:0 24px;color:#1a3449;background:#f8fafb}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #d5dde4;padding:8px;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}.ours{{background:#dff4ed}}.note{{padding:18px;border-left:5px solid #078e83;background:#eaf2f5}}img{{width:100%;margin:15px 0}}code{{overflow-wrap:anywhere}}</style>
    <h1>omni_slam：跨帧证据确认后的历史支持恢复</h1><p class="note"><b>{decision}</b><br>研究对象是几何筛选丢弃的实际测量支持。SAM3 掩码、SigLIP 特征、全部图像、原生 CG/history2 与最终处理一致；没有训练或新分割推理。本轮是窗口结束后的完整重放，不能声称实时部署。</p>
    <p>同一 20 cm 单元需被不同帧、同类别且描述子余弦相似度至少 0.7 的掩码支持；连通域至少一半不同单元获确认才恢复整个域。保留原最大连通域支持，所有新增点来自原掩码的真实测量。未知不作为反证；重复点不增加票数。Pointwise 是只恢复被确认单元的消融，All raw supports 是不加确认的必要强控制。</p>
    <h2>完整窗口评测</h2><table><tr><th>场景</th><th>方法</th><th>AP50:90</th><th>AP50</th><th>车辆 AP</th><th>建筑 AP</th><th>预测对象</th><th>地图点</th><th>CG 重放 s</th></tr>{table}</table><p>AP 均乘 100。两个室外窗口已用于开发；本地传感器建图适配评测，不是官方榜单或独立测试。车库缺独立标签，仅给链路和地图对照。预测对象数量不等于真实正确对象数量。</p>
    <h2>冻结的继续门槛</h2><p>主要指标在推理前固定为 AP50:90，用于检查几何恢复在不同 IoU 阈值下的质量。候选的两窗口均值须同时比 Largest 和 All 高至少 2 点；两个窗口分别在 AP50:90、AP50 上均不退步且没有丢失正确匹配；至少比 Largest 新增一个正确目标。此外，恢复整个连通域的均值必须高于 Pointwise，才支持该机制的独立收益。原始 AP50 和逐目标得失完整保留，不用平均值掩盖单场景退步。</p><table><tr><th>方法</th><th>两窗口 AP50:90 均值</th></tr>{meanrows}</table>{body}
    <h2>范围与产物</h2><p>此候选不能补回从未进入任何原始掩码的对象表面。反复出现的错误掩码也可能被确认；同一空间单元不是可靠对象身份。开发门槛通过仍不等于新颖性充分。新保留窗口 0004 / 0005 未参与本轮。</p><p>冻结协议：<code>{OUT/'protocol.json'}</code><br>预测封存：<code>{OUT/'predictions_complete.json'}</code><br><a href="summary.json">完整评分</a> · <a href="audit.json">逐对象与显示来源</a></p></html>''')
    dump(REPORT/'summary.json',summary);dump(REPORT/'audit.json',audit);print('REPORT',REPORT/'index.html',decision,flush=True)


if __name__=='__main__':main()
