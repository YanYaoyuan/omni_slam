"""Render actual images/maps and all-object evidence for the frozen candidate."""
import os,html
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni_slam_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_surface_formation import OUT,BASE,INPUT,SEM,METHODS,verify,seal

DEST=ROOT/'output/surface_formation_v1'
NAMES=dict(baseline='Frozen strong baseline',geometry='Shared semantics + geometry',positive_only='Positive connections only',first_view='First valid view + signed regions',ours_surface='Ours candidate: multi-view signed regions')


def main():
    p=verify();seal('predictions_complete');s=read(OUT/'summary.json');audit=read(OUT/'export_audit.json');details=read(OUT/'object_diagnostics.json')['results'];formation=read(OUT/'formation_diagnostics.json');assert audit['complete']
    DEST.mkdir(parents=True,exist_ok=True);keys=[x['key'] for x in p['scenes']];table=[];identity=[];oldtable=[];birthtable=[];images=[];figures=[]
    for m in METHODS:
        delta=100*(s['mean_primary_ap'][m]-s['mean_primary_ap']['baseline'])
        values=[NAMES[m]]+[f'{100*s["results"][k]["methods"][m]["metrics"]["allAp"]:.3f}' for k in keys]+[f'{100*s["mean_primary_ap"][m]:.3f}',f'{delta:+.3f}',str(sum(len(s['results'][k]['methods'][m]['changes_vs_baseline']['gained']) for k in keys)),str(sum(len(s['results'][k]['methods'][m]['changes_vs_baseline']['lost']) for k in keys))]
        table.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    for scene in p['scenes']:
        key=scene['key'];manifest=read(INPUT/key/'manifest.json');base=np.load(BASE/key/'geometry_graph/geometry.npz');size=len(base['xyz']);source=np.load(manifest['geometry'])['xyz'];records=read(SEM/'predictions_complete.json')['results'][key];rec=records[len(records)//2];obs=next(r for r in manifest['records'] if r['frame_id']==rec['frame_id']);v=np.load(obs['file']);capture=np.load(rec['file']);rgb=np.asarray(Image.open(rec['rgb']).convert('RGB'))
        fig,axes=plt.subplots(3,1,figsize=(13,10.5),layout='constrained')
        axes[0].imshow(rgb);axes[0].set_title(f'{key} / frame {rec["frame_id"]}: actual RGB')
        for ax,m in zip(axes[1:],['baseline','ours_surface']):
            ax.imshow(rgb);xy=capture['map_pixel_xy'];ax.scatter(xy[:,0],xy[:,1],s=1,c='#27abc0',alpha=.7)
            if m!='baseline':
                g=np.load(OUT/'maps'/key/m/'geometry.npz');lookup=np.zeros(len(source),bool);lookup[g['added_source_ids']]=True;take=lookup[v['visible_source_ids']];px=v['pixel_xy'][take];ax.scatter(px[:,0],px[:,1],s=1.6,c='#f2a51a',alpha=.8)
            ax.set_title(NAMES[m]+': cyan = old map; amber = added measured points',fontsize=11)
        for ax in axes:ax.axis('off')
        filename=key+'_rgb.png';fig.savefig(DEST/filename,dpi=125,bbox_inches='tight');plt.close(fig)
        images.append(dict(file=filename,key=key,frame=rec['frame_id'],rgb_sha256=sha(rec['rgb']),selection='middle original frame by index; no outcome selection',meaning='Actual depth-supported old map pixels plus original measured-visible newly assigned sources. Window-end offline map, not online video. Colours indicate provenance, not correctness.'))
        figures.append(f'<h3>{key} / 实测投影</h3><img src="{filename}">')
        fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained');context=source[::20];lo=source[:,:2].min(0);hi=source[:,:2].max(0)
        for ax,m in zip(axes,['baseline','geometry','ours_surface']):
            g=np.load(OUT/'maps'/key/m/'geometry.npz');ax.scatter(context[:,0],context[:,1],s=.12,c='#c5cbd0',rasterized=True);ax.scatter(base['xyz'][:,0],base['xyz'][:,1],s=.35,c='#17899c',rasterized=True)
            if m!='baseline':ax.scatter(g['xyz'][size:,0],g['xyz'][size:,1],s=.45,c='#e9a023',rasterized=True)
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.set_title(NAMES[m]+'\n'+f'{len(g["xyz"]):,} points',fontsize=9)
        fig.suptitle(key+': complete maps, common measured extents; amber = additions, not GT correctness',fontsize=11)
        filename=key+'_maps.png';fig.savefig(DEST/filename,dpi=140,bbox_inches='tight');plt.close(fig);images.append(dict(file=filename,key=key,selection='all map points; every20th original source for grey context; common complete source extents'))
        figures.append(f'<h3>{key} / 完整地图</h3><img src="{filename}">')
        lookup={m:{o['object_id']:o for o in details[key][m]} for m in METHODS}
        for baseobj in details[key]['baseline']:
            oid=baseobj['object_id'];gid=baseobj['overlaps'][0]['gt_id'] if baseobj['overlaps'] else None
            values=[key[-4:],str(oid),str(gid) if gid is not None else 'N/A']
            vals=[next((v['iou'] for v in lookup[m][oid]['overlaps'] if v['gt_id']==gid),0.) for m in METHODS]
            values += [f'{v:.4f}' if gid is not None else 'N/A' for v in vals]
            oldtable.append('<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
        rows=read(OUT/'maps'/key/'ours_surface/classification.json')['objects']
        for obj in rows:
            oid=obj['object_id']
            if oid in lookup['baseline']:continue
            d=lookup['ours_surface'][oid];best=d['overlaps'][0] if d['overlaps'] else None
            values=[key[-4:],str(oid),obj['predictions']['mean']['name'],f'{obj["predictions"]["mean"]["score"]:.3f}',f'{d["reference_mass"]:.1f}',str(best['gt_id']) if best else 'N/A',f'{best["iou"]:.4f}' if best else '0.0000']
            birthtable.append('<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
        for control,v in audit['canonical_source_differences'][key].items():
            identity.append(f'<tr><td>{key[-4:]}</td><td>{NAMES[control]}</td><td>{v["changed_source_ownership"]}</td><td>{v["changed_admission"]}</td></tr>')
    data_rows=[]
    for key,methods in formation['counts'].items():
        for method in ['baseline','geometry','ours_surface']:
            b=methods[method]['building'];c=methods[method]['car']
            data_rows.append(f'<tr><td>{key[-4:]}</td><td>{NAMES[method]}</td><td>{b["actual_over_half"]}</td><td>{b["oracle_union_over_half"]}</td><td>{c["actual_over_half"]}</td><td>{c["oracle_union_over_half"]}</td></tr>')
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam 表面形成候选对照</title>
<style>body{{font:16px/1.7 system-ui;color:#23384a;max-width:1300px;margin:35px auto;padding:0 24px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:7px;border:1px solid #dbe5eb}}th{{background:#eff5f8}}img{{width:100%}}.note{{padding:20px;background:#fff1dc}}details{{margin:20px 0}}a{{color:#087e90}}</style>
<h1>omni_slam：多视角表面形成候选</h1><p>2026-11-14 · 已暴露KITTI-360开发窗口 · 10份完整地图 · 5项核心测试</p>
<p class="note"><b>候选未通过：平均AP50:90为8.470，强基线23.134；新增1个AP50正确实例，丢失3个。</b>简单几何控制8.755更高，多视角与首个有效视图同分。本版关闭，不替换强基线、不追加阈值搜索，也不将其扩展成机器人或独立测试主结果。</p>
<h2>本轮实际实现</h2><p>所有修改条件共享原108帧、SAM3原始查询和固定的外部Cityscapes SegFormer。原对象实测来源作为锚点；新的同类实测点划成20cm表面单元。图上使用0.4m内16近邻提案。两个单元有共同有效掩码观测时，按所选掩码在实测点上的IoU产生正负证据；未知视图不作反证。区域合并检查全部边界代价，避免一条正连接覆盖其他负证据。</p>
<p>同一旧对象的锚点强制相连，不合并不同旧ID。无旧锚点的区域满足共同的16点、3可见帧规则后产生新实例。所有旧点、ID和分数原样保留；新增坐标来自真实源点。本版不删除旧污染、不拆旧对象，也没有跨越局部图断连的非局部归属边。外部语义模型与原检测器的分数未跨模型校准，不能把它的增益记为Ours。</p>
<h2>完整地图对照</h2><p>building/car，原置信度加权AP，阈值0.50:0.05:0.90；本地固定适配协议，不是官方榜单或独立泛化测试。</p><table><tr><th>条件</th><th>0000</th><th>0002</th><th>均值</th><th>Δ基线</th><th>新增匹配</th><th>丢失匹配</th></tr>{''.join(table)}</table>
<p>四个修改条件均新增0000车辆GT13030，同时丢失0000车辆13024、0002建筑2011及车辆13022。共同的新增正确实例没有证明累积冲突机制的独立收益。Ours新增100,166 / 139,954个点，创建307 / 244个新ID；数量增加并没有转化为完整对象质量。</p>
<h2>为什么下一步不能只改合并半径</h2><table><tr><th>窗口</th><th>条件</th><th>建筑实际IoU&gt;0.5</th><th>建筑理想合并&gt;0.5</th><th>车辆实际IoU&gt;0.5</th><th>车辆理想合并&gt;0.5</th></tr>{''.join(data_rows)}</table>
<p>GT辅助地挑选每个目标最有利的预测片段组合：0000建筑有4个超过0.5，而实际只有1个，说明存在碎片化；但0002即使这样理想合并仍为0，低于基线的1个。因此“再加跨片段连接就能解决”没有两窗口证据，不能自动开启半径/长程边搜索。这些理想组合按GT独立选择，不一定能共同实现，也不是新地图或AP。</p>
<p>共同的建图前候选点池对6 / 4个建筑仍有超过一半的覆盖，但这是允许完美去污染与身份分配的必要条件，不能当作可实现召回。当前失败也不全是语义硬接纳删掉了测量。只有14.71% / 11.69%的边有至少两帧有效成对观测，约46% / 45%完全没有这样的证据；本版大量依赖几何先验，多视角机制的作用有限。</p>
<h2>同分核查</h2><table><tr><th>窗口</th><th>与Ours比较</th><th>实际源点归属变化</th><th>是否接纳变化</th></tr>{''.join(identity)}</table><p>新ID用所属源点的最小索引规范化后再比较，排除了编号置换。旧图、源点唯一性、合并轨迹、目标函数增量、新生规则与分数均已重放核验。</p>
<h2>真实RGB和完整地图</h2><p>每窗口固定中间帧；青色为旧地图，橙色为新增实测点，不代表正确标签。投影使用该帧原始实测可见性。展示的是离线窗口结束地图，不冒充实时运行。</p>{''.join(figures)}
<h2>全部83个旧对象</h2><p>固定为基线中重叠最大的合格同类GT；无对应记N/A。不能换目标掩盖退步。</p><details><summary>展开逐对象IoU</summary><table><tr><th>窗口</th><th>对象</th><th>固定GT</th>{''.join('<th>'+NAMES[m]+'</th>' for m in METHODS)}</tr>{''.join(oldtable)}</table></details>
<h2>Ours全部551个新对象</h2><p>列出全部新ID，包括参考面积不足100或无合格同类重叠者。没有IoU50重叠不一律称作官方误检：评测还含面积及void忽略规则。</p><details><summary>展开新对象面积、分数和最佳IoU</summary><table><tr><th>窗口</th><th>新ID</th><th>类别</th><th>分数</th><th>参考面积</th><th>最佳GT</th><th>IoU</th></tr>{''.join(birthtable)}</table></details>
<h2>与已有方法的关系及去留</h2><p><a href="https://openaccess.thecvf.com/content/CVPR2024/html/Yin_SAI3D_Segment_Any_Instance_in_3D_Scenes_CVPR_2024_paper.html">SAI3D</a>已使用几何片段与多视角掩码关联；<a href="https://openaccess.thecvf.com/content/CVPR2024/html/Yan_MaskClustering_View_Consensus_based_Mask_Graph_Clustering_for_Open-Vocabulary_3D_CVPR_2024_paper.html">MaskClustering</a>已使用视图共识。标准有符号聚类和表面表示不作为新颖性主张，本轮也不是这两项原方法的复现成绩。候选必须产生共享输入控制无法解释的完整地图收益，本轮没有达到。</p>
<p>关闭这套固定的“语义硬接纳＋局部表面图＋锚点保护”候选。保留原强基线与既有结果；不将新增点数或一个共同新增匹配包装成成功。新机制、同版本机器人对照和新增独立评测的总目标仍未完成。</p>
<p><a href="summary.json">完整指标</a> · <a href="export_audit.json">导出核查</a> · <a href="object_diagnostics.json">全部对象重叠</a> · <a href="formation_diagnostics.json">形成诊断</a> · <a href="../../build/benchmarks/surface_formation_v1/protocol.json">冻结协议</a></p></html>'''
    (DEST/'index.html').write_text(text)
    for name in ['summary.json','export_audit.json','object_diagnostics.json','formation_diagnostics.json','admission_coverage.json','evidence_observation_counts.json']:dump(DEST/name,read(OUT/name))
    dump(DEST/'images.json',images);dump(DEST/'manifest.json',dict(source_sha256=sha(__file__),files={f.name:sha(f) for f in DEST.iterdir() if f.is_file() and f.name!='manifest.json'}))
    print('REPORT READY',DEST/'index.html',flush=True)


if __name__=='__main__':main()
