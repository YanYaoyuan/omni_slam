"""Audit and show the sealed complete-map experiment; never tune predictions."""
from html import escape
import numpy as np
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_partition_revision import OUT,PARENT,INPUT,BASE,METHODS,verify,unpack,pieces

REPORT=ROOT/'output/partition_revision_v1'
NAMES=dict(baseline='强基线：几何关联',ordinary='普通正点重分割',self_validation='提示帧自验证',current_only='只更新当前观测',retrieval_only='仅检索既有划分',ours='Ours：对称提示与历史修订')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocol=verify();seal=read(OUT/'predictions_complete.json');summary=read(OUT/'summary.json');model=read(OUT/'model_outputs_complete.json')
    assert summary['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    assert seal['model_seal_sha256']==sha(OUT/'model_outputs_complete.json')
    for f,h in {**seal['files'],**model['files']}.items():assert sha(f)==h,f
    REPORT.mkdir(parents=True,exist_ok=True);audits={};table='';events='';figures='';images=[]
    keys=[s['key'] for s in protocol['scenes']]
    for method in METHODS:
        values=[summary['results'][key]['methods'][method]['metrics']['allAp']*100 for key in keys]
        values += [summary['mean_primary_ap'][method]*100,(summary['mean_primary_ap'][method]-summary['mean_primary_ap']['baseline'])*100]
        table+='<tr'+(' class="ours"' if method=='ours' else '')+'><td>'+NAMES[method]+'</td>'+''.join(f'<td>{v:.3f}</td>' for v in values)+'</tr>'
    for key in keys:
        base=dict(np.load(OUT/'maps'/key/'baseline/geometry.npz'));n=len(base['xyz']);manifest=read(INPUT/key/'manifest.json');source=np.load(manifest['geometry'])['xyz']
        assert sha(OUT/'maps'/key/'baseline/geometry.npz')==sha(PARENT/'maps'/key/'geometry_graph/geometry.npz')
        assert sha(OUT/'maps'/key/'baseline/classification.json')==sha(PARENT/'maps'/key/'geometry_graph/classification.json')
        nodes=dict(np.load(PARENT/'nodes'/key/'nodes.npz'));sets=pieces(nodes['source_ids'],nodes['source_offsets'])
        protected=np.unique(np.concatenate([base['added_source_ids'],np.load(BASE/'maps'/key/'intrinsic_geometry/geometry.npz')['added_source_ids']]+[sets[i] for i in np.flatnonzero(nodes['old'])]))
        groups=read(OUT/key/'groups.json');node_group={j:g['group'] for g in groups for j in g['node_ids']}
        group_oid={node_group[g['node_ids'][0]]:g['object_id'] for g in read(PARENT/'maps'/key/'geometry_graph/groups.json')['groups']}
        maps={};all_decisions={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=dict(np.load(folder/'geometry.npz'));maps[method]=g
            objects=read(folder/'classification.json')['objects'];row=summary['results'][key]['methods'][method];run=row['run'];change=row['changes_vs_baseline']
            np.testing.assert_array_equal(g['xyz'][:n],base['xyz']);assert set(g['instance'])=={o['object_id'] for o in objects}
            if method!='baseline':
                ids=g['added_source_ids'];assert len(ids)==len(np.unique(ids)) and not np.isin(ids,protected).any()
                np.testing.assert_array_equal(g['xyz'][n:],source[ids])
                decisions=read(folder/'decisions.json');all_decisions[method]=decisions
                allowed=[group_oid[i] for d in decisions if d['applied'] for i in d['groups'] if i in group_oid]
                changed=g['instance'][:n]!=base['instance'];assert np.isin(base['instance'][changed],allowed).all()
                if method=='current_only':assert not changed.any()
                assert int(changed.sum())==run['old_point_ids_changed'] and len(ids)==run['added_points']
                audits[key+'/'+method]=dict(old_xyz_exact=True,append_exact_measured_source=True,unique_sources=True,protected_sources_not_appended=True,unrelated_old_objects_unchanged=True,changed_old_points=int(changed.sum()),added_points=len(ids),same_geometry_and_labels_as_baseline=bool(np.array_equal(g['xyz'],base['xyz']) and np.array_equal(g['instance'],base['instance'])))
                for d in decisions:
                    events+='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [key,NAMES[method],d['groups'],d['decision']['selected'],d['applied'],d.get('skip_reason',''),d.get('old_point_ids_changed',0),d.get('added_points',0)])+'</tr>'
            figures_row=[key,NAMES[method],f"{row['metrics']['allAp50%']*100:.3f}",change['gained'],change['lost'],run.get('old_point_ids_changed',0),run.get('added_points',0)]
            # Full score and map-change rows are retained in the report JSON as well.
            audits.setdefault(key+'/'+method,{})['report_row']=figures_row
        fig,axes=plt.subplots(2,3,figsize=(16,9),sharex=True,sharey=True)
        background=source[::max(1,len(source)//25000)]
        for ax,method in zip(axes.flat,METHODS):
            g=maps[method];ax.scatter(background[:,0],background[:,1],s=.2,c='#cbd5df',rasterized=True)
            ax.scatter(g['xyz'][:n:3,0],g['xyz'][:n:3,1],s=.5,c='#258497',rasterized=True)
            changed=g['xyz'][:n][g['instance'][:n]!=base['instance']]
            ax.scatter(changed[:,0],changed[:,1],s=2,c='#b747ad',rasterized=True)
            ax.scatter(g['xyz'][n:,0],g['xyz'][n:,1],s=.7,c='#ef903b',rasterized=True)
            ap=summary['results'][key]['methods'][method]['metrics']['allAp']*100
            ax.set_title(f'{method} | AP50:90 {ap:.3f}');ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
        fig.suptitle(key+' | cyan: retained points / purple: changed identity / orange: added measurements');fig.tight_layout()
        name=key+'_maps.png';fig.savefig(REPORT/name,dpi=140);plt.close(fig)
        figures+=f'<h3>{key}：完整窗口</h3><img src="{name}"><p>相同坐标范围；颜色表示改动来源，不表示正确性。</p>'
        # Deterministic diagnostic choice: most historical changes, then added points; no GT selection.
        chosen=max(all_decisions['ours'],key=lambda d:(d.get('old_point_ids_changed',0),d.get('added_points',0),-d['groups'][0]))
        candidate=next(r for r in read(OUT/key/'selected.json') if r['groups']==chosen['groups']);action=candidate['policies']['max_visible'];rgb=np.asarray(Image.open(action['rgb']).convert('RGB'))
        fig,axes=plt.subplots(3,1,figsize=(13,11));axes[0].imshow(rgb);axes[0].set_title('Actual RGB / common acquisition view')
        for ax,policy in zip(axes[1:],['ordinary','symmetric']):
            entry=next(r for r in model['instances'] if r['key']==key and r['groups']==chosen['groups'] and r['policy']==policy)
            ax.imshow(rgb)
            for j,color in enumerate([[.05,.8,.65],[1.,.35,.3]]):
                d=dict(np.load(entry['files'][j]));mask=unpack(d)[int(d['selected'])];overlay=np.zeros((*mask.shape,4));overlay[mask]=color+[.35];ax.imshow(overlay)
            ax.set_title(policy+' / first two selected masks (not ground truth)')
        for ax in axes:ax.axis('off')
        fig.suptitle(f"{key} / groups {chosen['groups']} / Ours: {chosen['decision']['selected']} / applied: {chosen['applied']}")
        fig.tight_layout();name=key+'_masks.png';fig.savefig(REPORT/name,dpi=130);plt.close(fig)
        figures+=f'<h3>{key}：真实采集图像与模型掩码</h3><img src="{name}"><p>按 Ours 历史改写点数、再按新增点数选例；掩码不是 GT，也不代表最终三维归属。完整决策表包含全部16对。</p>'
        images.append(dict(key=key,groups=chosen['groups'],frame=action['frame_id'],rgb=action['rgb'],rgb_sha256=sha(action['rgb']),file=name))
    dump(OUT/'artifact_audit.json',dict(complete=True,source_sha256=sha(__file__),summary_sha256=sha(OUT/'summary.json'),protocol_sha256=sha(OUT/'protocol.json'),checks=audits))
    changes=''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in audits[key+'/'+m]['report_row'])+'</tr>' for key in keys for m in METHODS)
    decision='通过开发验收；仍须同版本独立测试。' if summary['advance_gate'] else '未通过开发验收；封存本版，保留强基线，不扩到机器人或新场景追分。'
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 对象划分与历史修订</title>
<style>body{{font:16px/1.7 system-ui;color:#193349;max-width:1400px;margin:32px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d5dfe8;padding:8px}}.ours{{background:#e7f5f1;font-weight:bold}}.note{{background:#f0f4f7;padding:20px}}details{{margin:20px 0}}.scroll{{overflow:auto}}</style>
<h1>omni_slam：对象划分取证与历史地图修订</h1><p class="note"><b>{decision}</b>主线是持续对象级语义建图；本轮检验对称分割提示、独立观测验收和历史归属修订。两段 KITTI-360 已暴露开发窗口、固定16对、六条件共12份完整地图；不是官方榜单或独立测试。</p>
<h2>完整地图 AP50:90 ×100</h2><table><tr><th>方法</th><th>0000</th><th>0002</th><th>均值</th><th>相对强基线</th></tr>{table}</table>
<h2>匹配得失与实际改动</h2><div class="scroll"><table><tr><th>窗口</th><th>方法</th><th>AP50</th><th>新增正确 GT</th><th>丢失正确 GT</th><th>旧点归属改写</th><th>新增实测点</th></tr>{changes}</table></div>
<p>强基线为冻结的几何关联结果；旧 XYZ 保持原值，新点仅来自原测量点池，29个新扫描仅供验收。所有地图封存后统一评分。自验证条件仍共享新视角的出生资格核对，只消融划分选择的验收帧；不能解释为完全去掉新视角。</p>
<h2>为什么同分，以及为什么没有历史改写</h2>
<p>Ours 与自验证消融在两个窗口的 XYZ、实例数组和分类记录均完全一致。只更新当前观测的几何不同，但主指标同分；相同 AP 不等于相同地图。16对候选中，只有1对涉及两个已有对象，另外2对各涉及一个已有对象，其余13对都是待确认对象。Ours 最终提交5对，另外11对未通过共同的对象出生规则。</p>
<p>唯一双旧对象案例为0000车辆 ID15/18：通过实测深度门的排他本对象支持为174/71点，排他对方支持均为0，因此保持原归属；其余合并提交主要补充对象几何。由此本轮没有辨识出历史修订的独立作用。完整地图基线比较有效，但不能据此宣称历史修订机制已被充分验证，也不能把掩码变化当作该机制的收益。本版封存，不放松出生或深度门槛追分。</p>
<h2>真实图像与完整地图</h2>{figures}
<details><summary>全部候选的选择、提交与拒绝</summary><div class="scroll"><table><tr><th>窗口</th><th>方法</th><th>分组</th><th>选择</th><th>提交</th><th>拒绝原因</th><th>改写旧点</th><th>新增点</th></tr>{events}</table></div></details>
<h2>执行和边界</h2><p>实际编码{model['image_encodes']}次、交互解码{model['instance_decodes']}次、类别查询{model['grounding_decodes']}次；模型阶段{model['wall_seconds']:.1f}秒，峰值已分配显存{model['peak_gpu_allocated_bytes']/1024**3:.2f} GiB。普通与对称提示各四次解码，模型缓存复用于各条件。20 cm单元验收存在边界混叠；相邻验证帧并非独立场景。没有训练、阈值搜索或用 GT 挑掩码。当前为窗口末离线实验，尚未验证机器人在线执行。</p>
<p><a href="summary.json">完整评分</a> · <a href="artifact_audit.json">产物审计</a> · <a href="mechanism_audit.json">同分与历史归属复核</a> · <a href="images.json">图像来源</a> · <a href="../../build/benchmarks/partition_revision_v1/protocol.json">冻结协议</a></p></html>'''
    (REPORT/'index.html').write_text(page);dump(REPORT/'images.json',images)
    for filename in ['summary.json','artifact_audit.json','mechanism_audit.json']:dump(REPORT/filename,read(OUT/filename))
    dump(REPORT/'manifest.json',dict(generator_sha256=sha(__file__),files={f.name:sha(f) for f in REPORT.iterdir() if f.name!='manifest.json'}))
    print('AUDIT AND REPORT',REPORT/'index.html',flush=True)


if __name__=='__main__':main()
