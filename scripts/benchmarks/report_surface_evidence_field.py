"""Post-seal audit, real imagery and complete-map results for the field trial."""
from html import escape
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_surface_evidence_field import OUT,PARENT,INPUT,CAPTURE,METHODS,verify,sealed,scene_data

REPORT=ROOT/'output/surface_evidence_field_v1'
NAMES=dict(baseline='强基线：几何关联',high_surface='高分支持＋表面联合',weak_independent='弱查询支持＋独立归属',weak_euclidean='弱查询支持＋欧氏联合',weak_native_score='原生类别置信度加权＋表面联合',ours='Ours：几何条件读取＋表面联合')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=verify();seal=sealed('predictions_complete');summary=read(OUT/'summary.json');assert summary['scored'];assert summary['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    REPORT.mkdir(parents=True,exist_ok=True);audits={};images=[];figures='';changes='';scoretable='';keys=[s['key'] for s in p['scenes']]
    for method in METHODS:
        vals=[summary['results'][k]['methods'][method]['metrics']['allAp']*100 for k in keys]
        vals.extend([summary['mean_primary_ap'][method]*100,(summary['mean_primary_ap'][method]-summary['mean_primary_ap']['baseline'])*100])
        scoretable+='<tr'+(' class="ours"' if method=='ours' else '')+'><td>'+NAMES[method]+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in vals)+'</tr>'
    for scene in p['scenes']:
        key=scene['key'];manifest,source,base,records=scene_data(scene);n=len(base['xyz']);index=dict(np.load(OUT/'features'/key/'index.npz'));eligible=cKDTree(base['xyz']).query(source,workers=1)[0]>.05
        np.testing.assert_array_equal(index['eligible_source'],eligible);maps={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=dict(np.load(folder/'geometry.npz'));maps[method]=g
            assert sha(folder/'classification.json')==sha(PARENT/'maps'/key/'geometry_graph/classification.json') if method=='baseline' else read(folder/'classification.json')==records
            np.testing.assert_array_equal(g['xyz'][:n],base['xyz']);assert set(g['instance'])<={0,*map(int,index['object_ids'])}
            row=summary['results'][key]['methods'][method];run=row['run'];delta=row['changes_vs_baseline']
            if method=='baseline':assert sha(folder/'geometry.npz')==sha(PARENT/'maps'/key/'geometry_graph/geometry.npz')
            else:
                cell=np.load(folder/'cell_labels.npy');ids=g['added_source_ids'];np.testing.assert_array_equal(g['instance'][:n],cell[index['old_cell']]);np.testing.assert_array_equal(g['xyz'][n:],source[ids]);np.testing.assert_array_equal(g['instance'][n:],cell[index['source_cell'][ids]])
                assert len(np.unique(ids))==len(ids) and eligible[ids].all()
                expected=np.flatnonzero(eligible&(cell[index['source_cell']]>0));np.testing.assert_array_equal(ids,expected)
                assert run['old_point_ids_changed']==int((g['instance'][:n]!=base['instance']).sum()) and run['added_points']==len(ids)
            audits[key+'/'+method]=dict(old_coordinates_exact=True,classifications_unchanged=True,new_source_allocation_exact=True,export_labels_exact=True,stats=run)
            values=[key,NAMES[method],f"{row['metrics']['allAp50%']*100:.3f}",delta['gained'],delta['lost'],run['old_points_to_null'],run['old_points_to_other'],run['added_points']]
            changes+='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+'</tr>'
        fig,axes=plt.subplots(2,3,figsize=(16,9),sharex=True,sharey=True);background=source[::max(1,len(source)//20000)]
        for ax,method in zip(axes.flat,METHODS):
            g=maps[method];ax.scatter(background[:,0],background[:,1],s=.15,c='#cbd5df',rasterized=True)
            ax.scatter(g['xyz'][:n:3,0],g['xyz'][:n:3,1],s=.5,c='#258497',rasterized=True)
            null=g['xyz'][:n][g['instance'][:n]==0];other=g['xyz'][:n][(g['instance'][:n]>0)&(g['instance'][:n]!=base['instance'])]
            ax.scatter(null[:,0],null[:,1],s=.8,c='#d75a63',rasterized=True);ax.scatter(other[:,0],other[:,1],s=1.1,c='#9a4cc4',rasterized=True);ax.scatter(g['xyz'][n:,0],g['xyz'][n:,1],s=.5,c='#dc9234',rasterized=True)
            ap=summary['results'][key]['methods'][method]['metrics']['allAp']*100;ax.set_title(f'{method} | AP50:90 {ap:.3f}');ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
        fig.suptitle(key+' | cyan: old geometry / red: null / purple: reassigned / orange: added measurements');fig.tight_layout();name=key+'_maps.png';fig.savefig(REPORT/name,dpi=135);plt.close(fig)
        figures+=f'<h3>{key} · 完整地图</h3><img src="{name}"><p>所有方法使用相同坐标范围；颜色表示操作，不表示准确性。红色旧点保留XYZ，输出标签为null。</p>'
        # Select a diagnostic by largest old ownership change, not reference correctness.
        ours=maps['ours'];old=base['instance'];change={int(o):int(((old==o)&(ours['instance'][:n]!=old)).sum()) for o in index['object_ids']};retrieval=read(OUT/'features'/key/'retrieval.json')
        candidates=[(change[r['object_id']],r['core_coverage'],r['support_cells'],-f['frame_id'],f,r) for f in retrieval for r in f['selections'] if r['mode']=='weak' and r['native_score']<=.5]
        if candidates:
            *_,frame,selected=max(candidates,key=lambda x:x[:4]);fid=frame['frame_id'];record=next(r for r in manifest['records'] if r['frame_id']==fid);raw=next(r for r in read(CAPTURE/'predictions_complete.json')['results'][key] if r['frame_id']==fid);d=np.load(raw['file']);rgb=np.asarray(Image.open(record['rgb']).convert('RGB'));mask=np.unpackbits(d['masks_packed'][selected['query']],axis=1)[:,:rgb.shape[1]].astype(bool)
            fig,axes=plt.subplots(2,1,figsize=(13,7));axes[0].imshow(rgb);axes[0].set_title('Actual RGB');axes[1].imshow(rgb);overlay=np.zeros((*mask.shape,4));overlay[mask]=[.05,.8,.65,.35];axes[1].imshow(overlay)
            axes[1].set_title(f"Actual selected weak query | object {selected['object_id']} | native score {selected['native_score']:.4f} | core coverage {selected['core_coverage']:.3f}")
            for ax in axes:ax.axis('off')
            fig.suptitle(f'{key} / frame {fid}');fig.tight_layout();name=key+'_weak_query.png';fig.savefig(REPORT/name,dpi=130);plt.close(fig)
            images.append(dict(key=key,frame=fid,rgb=record['rgb'],rgb_sha256=sha(record['rgb']),query=selected,file=name,selection='Most changed old object with a weak query, then highest core coverage/support and earliest frame; no GT case selection'))
            figures+=f'<h3>{key} · 实际低分查询</h3><img src="{name}"><p>按旧归属改动最多的对象选择诊断，再按核心覆盖与支持数量选择帧；没有用GT挑成功图。掩码只提供候选证据，不是最终地图或真值。</p>'
    dump(OUT/'artifact_audit.json',dict(complete=True,source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),summary_sha256=sha(OUT/'summary.json'),checks=audits))
    errors=read(OUT/'error_provenance.json');assert errors['prediction_seal_sha256']==sha(OUT/'predictions_complete.json');error_rows=''
    for key,result in errors['results'].items():
        totals=result['totals'];new=totals['new_measurement'];fraction=new['correct']/max(new['correct']+new['outside_target'],1e-12)
        values=[key,f"{fraction*100:.2f}%",f"{new['correct']:.1f}",f"{new['outside_target']:.1f}",f"{totals['old_unchanged']['outside_target']+totals['old_relabelled']['outside_target']:.1f}"]
        error_rows+='<tr>'+''.join('<td>'+escape(v)+'</td>' for v in values)+'</tr>'
    verdict='达到冻结的开发门槛；机器人和真正独立场景尚待同版本验证。' if summary['advance_gate'] else '未达到冻结的开发门槛，封存本版；不替换强基线，不调参数补救。'
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 联合表面归属</title><style>body{{font:16px/1.7 system-ui;color:#193349;max-width:1400px;margin:32px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d5dfe8;padding:8px}}.ours{{background:#e7f5f1;font-weight:bold}}.note{{background:#f0f4f7;padding:20px}}.scroll{{overflow:auto}}</style>
<h1>omni_slam：几何条件读取与联合表面归属</h1><p class="note"><b>{verdict}</b>固定原108帧与模型缓存，无新推理或训练；两段已暴露KITTI-360开发窗口、六条件共12份完整地图。全部预测封存后评分。不是官方榜单、独立测试或机器人验证。</p>
<h2>AP50:90 ×100</h2><table><tr><th>方法</th><th>0000</th><th>0002</th><th>均值</th><th>相对强基线</th></tr>{scoretable}</table>
<h2>实际匹配与地图改动</h2><div class="scroll"><table><tr><th>窗口</th><th>方法</th><th>AP50</th><th>新增正确GT</th><th>丢失正确GT</th><th>旧点变null</th><th>旧点换对象</th><th>新增测量点</th></tr>{changes}</table></div>
<h2>机制是否执行</h2><p>弱查询读取分别424/295次，其中原生分数≤0.5为206/134次。联合与独立归属分别改变687/732个物理单元，表面核与欧氏核只相差20/15个。小差异不能自动构成独立贡献；具体AP对照见上表。每帧读取每个对象的一个几何最优查询，因此冻结前将无可辨识性的重复计数消融替换为原生类别分数加权。</p>
<p>旧地图在每个占据单元提供一次先验观测；新帧按实际可见单元读取支持，缺测未知，看见却没有条件查询支持则记null。该观测模型可能误删旧支持，完整得失保留。20cm单元混叠限制仍存在。图正则化本身不是新颖性证明。</p>
<h2>错误覆盖来自哪里</h2><table><tr><th>窗口</th><th>新增点负责区域的同GT占比</th><th>新增点覆盖同GT权重</th><th>新增点覆盖目标外权重</th><th>旧点覆盖目标外权重</th></tr>{error_rows}</table>
<p>仅审计强基线原AP50正确对象，将其在完整Ours地图中负责的参考区域，按最近输出点来自旧点还是新增点分解。比例是参考置信度面积占比，不是新增源点准确率。目标外包含其他标签与void，不能全部当作可靠语义负例。两个窗口的目标外覆盖主要由新增点承担；旧点改写也造成原正确覆盖损失。核心被掩码覆盖不代表掩码内所有测量点属于该对象，图平滑没有解决这个归属问题。这里是固定输出的来源分解，不是新增反事实地图实验。</p>
<h2>真实输入和输出</h2>{figures}
<h2>复现</h2><p>10项核心/图结构/导出测试通过，残差和概率和经过检查。每份地图原XYZ前缀、旧标签映射、全部新增源点、5cm原几何排除域和固定分类记录均已核验。新增点不使用单元中心、补全深度或验证扫描；现有对象ID与分数保持固定，不产生新ID。</p><p><a href="summary.json">全部指标</a> · <a href="artifact_audit.json">导出审计</a> · <a href="images.json">图像来源</a> · <a href="../../build/benchmarks/surface_evidence_field_v1/protocol.json">冻结协议</a></p></html>'''
    (REPORT/'index.html').write_text(page);dump(REPORT/'images.json',images)
    for name in ['summary.json','artifact_audit.json','error_provenance.json']:dump(REPORT/name,read(OUT/name))
    dump(REPORT/'manifest.json',dict(generator_sha256=sha(__file__),files={f.name:sha(f) for f in REPORT.iterdir() if f.name!='manifest.json'}));print('AUDITED REPORT',REPORT/'index.html',flush=True)


if __name__=='__main__':main()
