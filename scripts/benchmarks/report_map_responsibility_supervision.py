"""Data-driven report of the fixed spatial-supervision loss comparison."""
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_map_responsibility_supervision import OUT,PARENT,INPUT,METHODS,KEYS,verify,seal

DEST=ROOT/'output/map_responsibility_supervision_v1'
NAMES=['Strong baseline','Source NLL','Object-balanced source NLL','Ours: spatial NLL']
CN=['强基线','逐点监督','对象均衡逐点监督','Ours：空间归属监督']


def main():
    verify();seal('predictions_complete');s=read(OUT/'summary.json');audit=read(OUT/'artifact_audit.json');details=read(OUT/'object_diagnostics.json')['results'];training=read(OUT/'trained.json');prep=read(OUT/'prepared.json')
    assert audit['prediction_seal_sha256']==s['prediction_seal_sha256']==sha(OUT/'predictions_complete.json');DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for ax,key in zip(axes,KEYS):
        vals=[100*s['results'][key]['methods'][m]['metrics']['allAp'] for m in METHODS];bars=ax.bar(range(4),vals,color=['#566d80','#a4b5c4','#ccab70','#11988a']);ax.bar_label(bars,fmt='%.3f',fontsize=9)
        ax.set(xticks=range(4),xticklabels=NAMES,ylim=(0,max(vals)*1.2),ylabel='Full-map AP50:90 x100',title=key+' / exposed development');ax.tick_params(axis='x',rotation=20)
    fig.savefig(DEST/'scores.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,3.6),layout='constrained')
    for ax,method,name in zip(axes,METHODS[1:],NAMES[1:]):
        for key in KEYS:
            curve=training['stats'][key+'/'+method]['loss'];ax.plot(np.arange(1,len(curve)+1),curve,label='Predict '+key[-4:])
        ax.set(title=name,xlabel='Fixed full-batch update',ylabel='Training NLL (own domain)');ax.legend(fontsize=8)
    fig.savefig(DEST/'training.png',dpi=150);plt.close(fig)
    score_rows='';change_rows='';objects='';images=[];figures='';thresholds={};domain_rows=''
    domains=read(OUT/'source_domain_diagnostic.json')['rows']
    for key in KEYS:
        for method,name in zip(METHODS[1:],CN[1:]):
            pair={r['role']:r for r in domains if r['target']==key and r['method']==method}
            domain_rows+=f'<tr><td>{key}</td><td>{name}</td><td>{pair["training"]["source_nll"]:.4f}</td><td>{pair["exposed_target"]["source_nll"]:.4f}</td></tr>'
    for m,name in zip(METHODS,CN):
        values=[100*s['results'][k]['methods'][m]['metrics']['allAp'] for k in KEYS]+[100*s['mean_primary_ap'][m],100*(s['mean_primary_ap'][m]-s['mean_primary_ap']['baseline'])]
        score_rows+=('<tr class="ours">' if m.startswith('ours') else '<tr>')+f'<td>{name}</td>'+''.join(f'<td>{v:.3f}</td>' for v in values)+'</tr>'
    for key in KEYS:
        keyed={m:{r['object_id']:r for r in details[key][m]} for m in METHODS};object_rows=''
        for oid,base in keyed['baseline'].items():
            gid=base['overlaps'][0]['gt_id'] if base['overlaps'] else None
            vals=[next((q['iou'] for q in keyed[m][oid]['overlaps'] if q['gt_id']==gid),0.) for m in METHODS]
            object_rows+=f'<tr><td>{oid}</td><td>{gid}</td>'+''.join(f'<td>{v:.4f}</td>' for v in vals)+'</tr>'
        objects+=f'<h2>{key}</h2><table><tr><th>对象ID</th><th>基线最佳GT</th>'+''.join(f'<th>{v}</th>' for v in CN)+f'</tr>{object_rows}</table>'
        thresholds[key]={}
        for method,name in zip(METHODS,CN):
            r=s['results'][key]['methods'][method];c=r['changes_vs_baseline'];change_rows+='<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [key,name,r['run']['added_points'],c['gained'],c['lost']])+'</tr>'
            aa={(o['object_id'],q['gt_id']):q['iou'] for o in details[key]['baseline'] for q in o['overlaps'] if o['reference_mass']>=100}
            bb={(o['object_id'],q['gt_id']):q['iou'] for o in details[key][method] for q in o['overlaps'] if o['reference_mass']>=100};pairs=set(aa)|set(bb)
            thresholds[key][method]=dict(changed_ious=sum(abs(aa.get(v,0)-bb.get(v,0))>1e-12 for v in pairs),
                flips={str(round(float(t),2)):sum((aa.get(v,0)>float(t))!=(bb.get(v,0)>float(t)) for v in pairs) for t in np.arange(.5,.95,.05)})
        xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];base=dict(np.load(OUT/'maps'/key/'baseline/geometry.npz'));n=len(base['xyz']);limits=[(xyz[:,j].min(),xyz[:,j].max()) for j in [0,1]]
        fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
        for ax,m,name in zip(axes,['baseline','balanced_source_nll','ours_spatial_nll'],[NAMES[0],NAMES[2],NAMES[3]]):
            g=np.load(OUT/'maps'/key/m/'geometry.npz');ax.scatter(xyz[::10,0],xyz[::10,1],s=.15,c='#dce3e7');ax.scatter(g['xyz'][:n,0],g['xyz'][:n,1],s=.25,c='#566d80');ax.scatter(g['xyz'][n:,0],g['xyz'][n:,1],s=1,c='#11988a')
            ax.set(title=name,xlim=limits[0],ylim=limits[1],xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal')
        fig.suptitle(key+' / full extent: gray measured scene, blue fixed objects, green additions (not GT)');name=key+'_map.png';fig.savefig(DEST/name,dpi=150);plt.close(fig);figures+=f'<img src="{name}">'
        # Retain the already fixed image selection, independent of new outputs.
        old=next(v for v in read(ROOT/'output/object_context_acceptance_v1/images.json') if v['key']==key);assert sha(old['rgb'])==old['rgb_sha256'];rgb=np.asarray(Image.open(old['rgb']).convert('RGB'))
        context=read(ROOT/'build/benchmarks/semantic_context_observations_v1/predictions_complete.json')['results'][key];rec=next(v for v in context if v['frame_id']==old['frame_id']);obs=np.load(rec['file']);d=np.load(PARENT/'features'/key/'candidates.npz');take=np.isin(obs['source_ids'],d['source_ids']);xy=obs['pixel_xy'][take];ix=np.searchsorted(d['source_ids'],obs['source_ids'][take])
        a=np.load(OUT/'maps'/key/'balanced_source_nll/decisions.npz')['owners'][ix];b=np.load(OUT/'maps'/key/'ours_spatial_nll/decisions.npz')['owners'][ix];color=np.where((a>0)&(a==b),'#16bba1',np.where(b>0,'#3989ef',np.where(a>0,'#ed9a42','#aeb7be')))
        fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
        for ax in axes:ax.imshow(rgb);ax.axis('off')
        axes[0].set_title('Actual RGB');axes[1].scatter(xy[:,0],xy[:,1],s=3,c=color,linewidths=0);axes[1].set_title('Both green | Ours differs blue | source-only orange\nGray: both reject; candidate measurements, not GT',fontsize=10)
        fig.suptitle(f'{key} frame {old["frame_id"]} / previously fixed frame selection');name=key+'_rgb.png';fig.savefig(DEST/name,dpi=150);plt.close(fig);figures+=f'<img src="{name}">';images.append(dict(**old,new_file=name))
    style='<style>body{font:16px/1.7 system-ui;color:#203746;max-width:1400px;margin:30px auto;padding:0 24px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d8e1e6;padding:7px}.ours{background:#e6f4f0;font-weight:bold}.note{padding:18px;background:#fff1d7}a{color:#167a8a}</style>'
    verdict='通过本次开发门槛；独立评测与机器人验证仍待完成。' if s['advance_gate'] else '未通过冻结门槛，封存本版，保留强基线；不追加参数尝试或扩展失败候选。'
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam 空间归属监督</title>{style}<h1>omni_slam：面向完整地图的空间归属监督</h1>
<p class="note"><b>{verdict}</b>6次固定拟合、8份完整地图。Ours 新增AP50匹配 {s['gained']} 个，丢失 {s['lost']} 个。目标仍是对象级语义建图；当前只检验已知对象的测量接纳，不是完整SLAM创新或新对象发现。</p>
<h2>完整地图主指标</h2><table><tr><th>方法</th><th>0000</th><th>0002</th><th>均值</th><th>相对基线</th></tr>{score_rows}</table><img src="scores.png">
<p>AP50:90 ×100，同一冻结评测适配器。两窗口均已反复开发，父级模型还继承跨窗口监督；不是严格嵌套交叉验证、独立泛化或官方榜单成绩。</p>
<h2>同输入、同网络，监督机制对照</h2><p>共同50维对象条件化特征、候选测量、固定旧地图、MLP32/32隐藏层；seed0、Adam1e-3、200次全批量更新。逐点NLL对照改变为对象均衡权重，再改变为完整空间标签边缘概率的NLL。所有预测封存后统一评分，无checkpoint筛选。与上一失败版的概率参数化及训练匹配也有区别，因此机制结论应基于本轮三组同条件对照。</p>
<p>训练场景按基线同类别IoU进行一对一GT匹配，不设匹配IoU门槛；三个条件共用。未匹配预测没有正目标，其新增错误归属受惩罚，但无法删除既有核心。未知语义和零置信度不当作负例。空间损失可能通过已知区域间接监督源点，因此三组共享标注场景，并非相同监督样本数。推理不读取GT或参考域。</p>
<p>0000 / 0002 训练匹配对象数 {prep['stats'][KEYS[0]]['matched_objects']} / {prep['stats'][KEYS[1]]['matched_objects']}，未匹配数 {prep['stats'][KEYS[0]]['unmatched_objects']} / {prep['stats'][KEYS[1]]['unmatched_objects']}。空间概率为独立源点分类选择下的精确边缘概率；NLL不等于期望AP或IoU，也不是相机遮挡渲染。</p>
<h2>逐窗口得失</h2><table><tr><th>窗口</th><th>方法</th><th>追加点</th><th>新增AP50 GT</th><th>丢失AP50 GT</th></tr>{change_rows}</table>
<h2>冻结模型的训练域与另一窗口</h2><table><tr><th>预测窗口</th><th>方法</th><th>训练窗口源点NLL</th><th>目标窗口源点NLL</th></tr>{domain_rows}</table>
<p>这是评分后诊断，没有新增拟合、地图或AP。各组均按本轮共同的一对一目标定义计算源点NLL；对空间组而言，它不是其训练优化的损失。普通逐点模型在训练窗口较低的误差没有保持到另一窗口，说明不能只凭loss下降认为接纳已可靠。该差距不能单独归因于某个特征或某种损失；所有窗口都已暴露，不构成独立泛化评测。下一步不继续追加损失权重或阈值搜索。</p>
<h2>训练与真实场景</h2><img src="training.png"><p>各损失的监督域和权重不同，曲线绝对值不能跨组直接比较；空间损失包含旧地图不可改变部分的常数。曲线不是验证准确率。</p>{figures}
<h2>复核</h2><p>8项针对性测试通过，包含梯度等价、未知区、不可达标签、未匹配对象和一对一训练匹配。所有模型归一化仅来自训练窗口；8份完整地图的旧坐标、ID和分类记录保持一致，追加点均来自唯一实测源点，6组模型推理逐项复现。</p>
<p><a href="objects.html">全部83个对象IoU</a> · <a href="summary.json">完整指标</a> · <a href="artifact_audit.json">导出与推理审计</a> · <a href="thresholds.json">IoU阈值变化</a> · <a href="../../build/benchmarks/map_responsibility_supervision_v1/protocol.json">冻结协议</a></p></html>'''
    (DEST/'index.html').write_text(body);(DEST/'objects.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'+style+'<h1>全部对象：固定基线最佳GT的IoU</h1><p>诊断匹配不是训练匹配，也不一定达到AP匹配门槛；保留全部ID。</p>'+objects+'</html>')
    for name in ['summary.json','artifact_audit.json','object_diagnostics.json','source_domain_diagnostic.json']:dump(DEST/name,read(OUT/name))
    dump(DEST/'thresholds.json',thresholds);dump(DEST/'images.json',images);dump(DEST/'manifest.json',dict(generator_sha256=sha(__file__),files={f.name:sha(f) for f in DEST.iterdir() if f.is_file() and f.name!='manifest.json'}))
    print('REPORT',DEST/'index.html',flush=True)


if __name__=='__main__':main()
