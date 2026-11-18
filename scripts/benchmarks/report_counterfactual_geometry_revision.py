"""Full-window results, shared-input attribution, and actual RGB/map evidence."""
import runpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_counterfactual_geometry_revision import OUT,BASE,SEM,INPUT,METHODS,verify,sealed

DEST=ROOT/'output/counterfactual_geometry_revision_v1'
NAMES=dict(baseline='Frozen strong baseline',semantic_point='SegFormer + point filter',semantic_region='SegFormer + region filter',local_counterfactual='Local counterfactual',ours_counterfactual='Ours — counterfactual revision')


def main():
    p=verify();sealed('predictions_complete');s=read(OUT/'summary.json');audit=read(OUT/'artifact_audit.json');details=read(OUT/'object_diagnostics.json')['results'];capture=read(SEM/'predictions_complete.json');assert audit['complete'];DEST.mkdir(parents=True,exist_ok=True)
    means=s['mean_primary_ap'];table='';identity='';objects='';figures='';images=[]
    keys=[v['key'] for v in p['scenes']]
    for method in METHODS:
        delta=100*(means[method]-means['baseline']);cls='gain' if delta>0 else '';row=[NAMES[method]]+[f'{100*s["results"][k]["methods"][method]["metrics"]["allAp"]:.3f}' for k in keys]+[f'{100*means[method]:.3f}',f'{delta:+.3f}',str(sum(len(s['results'][k]['methods'][method]['changes_vs_baseline']['gained']) for k in keys)),str(sum(len(s['results'][k]['methods'][method]['changes_vs_baseline']['lost']) for k in keys))];table+='<tr class="'+cls+'">'+''.join('<td>'+v+'</td>' for v in row)+'</tr>'
    for key in keys:
        for comparison,row in audit['identifiability'][key].items():
            identity+='<tr>'+''.join(f'<td>{v}</td>' for v in [key,comparison,row['retained_point_decisions_changed'],row['reference_labels_changed'],row['object_gt_iou_pairs_changed'],row['threshold_crossings_50_to_90']])+'</tr>'
        by={m:{r['object_id']:r for r in details[key][m]} for m in METHODS}
        for base in details[key]['baseline']:
            oid=base['object_id'];gt=base['overlaps'][0]['gt_id'] if base['overlaps'] else None;values=[]
            for method in METHODS:values.append(next((v['iou'] for v in by[method][oid]['overlaps'] if v['gt_id']==gt),0.) if gt is not None else None)
            delta=None if gt is None else values[-1]-values[0];objects+='<tr>'+''.join(f'<td>{v}</td>' for v in [key,oid,gt if gt is not None else 'N/A']+['N/A' if v is None else f'{v:.4f}' for v in values]+['N/A' if delta is None else f'{delta:+.4f}'])+'</tr>'
    palette=np.zeros((19,3),np.uint8);seen=set()
    for label in runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['labels']:
        if 0<=label.trainId<19 and label.trainId not in seen:palette[label.trainId]=label.color;seen.add(label.trainId)
    for scene in p['scenes']:
        key=scene['key'];base=np.load(BASE/key/'geometry_graph/geometry.npz');ours=np.load(OUT/'maps'/key/'ours_counterfactual/geometry.npz');keep=np.zeros(len(base['xyz']),bool);keep[ours['retained_base_indices']]=True;records=capture['results'][key]
        for index in [0,len(records)//2,len(records)-1]:
            rec=records[index];d=np.load(rec['file']);rgb=np.asarray(Image.open(rec['rgb']).convert('RGB'));overlay=(.55*rgb+.45*palette[d['class_map']]).astype(np.uint8);ids=d['map_ids'];xy=d['map_pixel_xy'];colors=plt.get_cmap('viridis')((base['instance'][ids]%64)/63.)
            fig,axes=plt.subplots(4,1,figsize=(13,15),layout='constrained')
            for ax in axes:ax.axis('off')
            axes[0].imshow(rgb);axes[0].set_title(f'{key} / frame {rec["frame_id"]}: actual RGB')
            axes[1].imshow(overlay);axes[1].set_title('External SegFormer semantic overlay — no GT')
            axes[2].imshow(rgb);axes[2].scatter(xy[:,0],xy[:,1],s=2,c=colors,alpha=.8);axes[2].set_title('Frozen baseline: actual depth-supported object map points')
            selected=keep[ids];axes[3].imshow(rgb);axes[3].scatter(xy[selected,0],xy[selected,1],s=2,c=colors[selected],alpha=.8);axes[3].set_title('Ours: retained points from the same map and projection; colours = predicted object IDs')
            name=f'{key}_frame{rec["frame_id"]}.png';fig.savefig(DEST/name,dpi=100,bbox_inches='tight',pad_inches=.1);plt.close(fig);figures+=f'<h3>{key} / frame {rec["frame_id"]}</h3><img src="{name}">';images.append(dict(file=name,key=key,frame=rec['frame_id'],rgb=rec['rgb'],rgb_sha256=sha(rec['rgb']),source_prediction_sha256=sha(rec['file']),selection='first,middle,last of original frame list,not outcome selection'))
        fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained');limits=np.array([base['xyz'][:,:2].min(0),base['xyz'][:,:2].max(0)]);span=np.maximum(limits[1]-limits[0],1);colors=plt.get_cmap('viridis')((base['instance']%64)/63.);context=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'][::20]
        for ax,method in zip(axes,['baseline','semantic_region','ours_counterfactual']):
            g=np.load(OUT/'maps'/key/method/'geometry.npz');ids=np.arange(len(base['xyz'])) if method=='baseline' else g['retained_base_indices'];kept=np.zeros(len(base['xyz']),bool);kept[ids]=True;ax.scatter(context[:,0],context[:,1],c='#aaaaaa',s=.15,alpha=.2,rasterized=True);ax.scatter(base['xyz'][ids,0],base['xyz'][ids,1],c=colors[ids],s=.7,rasterized=True);ax.scatter(base['xyz'][~kept,0],base['xyz'][~kept,1],c='#d5224d',s=2,rasterized=True);ax.set_xlim(limits[0,0]-.03*span[0],limits[1,0]+.03*span[0]);ax.set_ylim(limits[0,1]-.03*span[1],limits[1,1]+.03*span[1]);ax.set_aspect('equal');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_title(NAMES[method]+f'\n{len(ids):,} retained points',fontsize=9)
        fig.suptitle(key+': full window, shared axes. Grey = measured context. Red = removed base points, not GT errors.',fontsize=11);name=key+'_whole_map.png';fig.savefig(DEST/name,dpi=130,bbox_inches='tight',pad_inches=.12);plt.close(fig);figures+=f'<h3>{key} / complete map</h3><img src="{name}">';images.append(dict(file=name,key=key,selection='all original map points,common full baseline XY extents;greycontext=every20thoriginalmeasuredsource',geometry_sha256=sha(BASE/key/'geometry_graph/geometry.npz')))
    style='<style>body{font:16px/1.7 system-ui;color:#203746;max-width:1400px;margin:30px auto;padding:0 24px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d8e1e6;padding:7px}img{width:100%}.gain{background:#edf9f2}.note{background:#fff0d8;padding:18px}.scroll{overflow-x:auto}a{color:#137886}</style>'
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam 完整地图反事实修订对照</title>{style}<h1>完整对象地图：独立语义证据与反事实修订</h1>
<p class="note"><b>Ours平均AP50:90从23.134到23.566（+0.433个百分点），但与直接语义区域筛选同分，新增正确匹配0、丢失1。新机制未通过预设门槛。</b>实际涨点保留，不能归因于本轮反事实选择机制。本版封存，原强基线保持；未扩展到机器人或独立场景。</p>
<h2>主线与本轮范围</h2><p>主线是omni_slam对象级语义建图，LIVO提供定位与实测几何。本轮完成一个历史几何修订候选及两窗口×五条件的完整地图对照。没有改位姿、增加推测几何、合并/拆分对象ID或调分类分数；输出只保留原地图点的子集。不是完整SLAM算法创新。</p>
<h2>完整窗口主指标</h2><p>已暴露KITTI-360开发窗口，本地固定适配协议：building/car、置信度加权AP，主指标为0.50:0.05:0.90，不含0.95。不是官方榜单或独立测试。绿色仅表示均值高于原基线。</p>
<table><tr><th>条件</th><th>0000 AP×100</th><th>0002 AP×100</th><th>均值</th><th>相对基线变化</th><th>新增AP50匹配</th><th>丢失AP50匹配</th></tr>{table}</table>
<p>丢失的是0000建筑GT2001：对应对象47的IoU从0.51110降至0.45484。典型正向变化包括0000车辆GT13002：0.78994→0.82372；0002车辆GT13011：0.73053→0.78071。均值提升不能掩盖这次建筑退步。</p>
<h2>新模块实际做了什么</h2><p>从共同的逐像素语义模型，在实测点上得到类别分布。每个20厘米物理单元等总权重。先提出与旧对象类别不一致的连通单元区域；提交前模拟删除它，重新计算所有受影响实测点的最近地图归属，计入邻近单元、其他存活对象和无对象回退。只有这一语义一致性损失严格下降才提交。</p>
<p>对照分别隔离外部分割器、直接逐点筛选、同候选直接删除以及只检查候选单元内部。Ours检查全部受影响的实测表面；局部对照分别接受了9/11次会增加全域模型损失的操作，Ours没有。但模型损失降低不是GT准确率保证，也没有成为额外AP收益。</p>
<p>7项核心测试通过，涉及冗余点与联合区域、局部改善损害邻域、缺测不授权修改、其他对象接管、顺序更新与完整重算一致、最近点平局及距离边界。原始坐标、保留点ID、类别和排序分数均核验一致。</p>
<h2>同分不是同一地图</h2><table><tr><th>窗口</th><th>比较</th><th>保留点决策差异</th><th>参考位置实例标签差异</th><th>对象–GT IoU变化对数</th><th>0.50至0.90阈值跨越数</th></tr>{identity}</table>
<p>Ours与直接区域删除分别有520/639个原点保留决策不同，30/21对IoU改变，但没有跨过本轮AP阈值，因此得分相同。十份地图均不同；没有复用相同预测文件冒充不同方法。</p>
<h2>独立语义对照及成本</h2><p>使用<a href="https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024">NVIDIA SegFormer-B5 Cityscapes权重</a>，固定版本2c6f153e，严格权重加载与哈希校验。它是有监督外部方法，不是Ours。原19类SAM3分数跨类比较在0000/0002的源点污染检出精度为2.65%/87.87%；外部模型均值读出为11.08%/84.42%，并非处处更强，故最终以完整地图对照判断。</p>
<p>所有108个原始图像各一次FP32前向，原1024×1024预处理，采集约{capture['seconds']:.2f}秒，PyTorch峰值{capture['peak_cuda_gib']:.2f}GiB，不含权重加载与后续建图评分。新修订器没有训练或额外GPU推理。模型标签与KITTI-360合并类别别名在推理前做了规范映射检查；未更换图像或重跑模型挑结果。</p>
<h2>真实RGB、语义输入和完整地图</h2><p>每个窗口固定展示第一、中间、最后一帧；点投影只显示通过原实测深度门槛的旧地图点，使用同一物体ID配色。全图显示全部原地图点，红色为删除位置，不是GT错误标签。</p>{figures}
<h2>全部83个对象的得失</h2><p>固定为各对象在原基线中重叠最大的合格同类GT，再比较同一GT的IoU。没有对应保留N/A；不是重新挑一个更容易的目标。</p><div class="scroll"><table><tr><th>窗口</th><th>对象ID</th><th>固定GT</th>{''.join('<th>'+NAMES[m]+'</th>' for m in METHODS)}<th>Ours−基线</th></tr>{objects}</table></div>
<h2>去留决定</h2><p>门槛要求均值至少比基线高1个百分点、超过全部同输入对照、至少新增1个AP50正确匹配且无丢失。本轮+0.433、与简单区域筛选同分、0新增1丢失，因此停止该固定版本，不追加置信度、区域尺度、顺序或损失参数搜索。共同语义输入的增益作为外部对照保留，不升级为已验证创新或生产主线。</p>
<p>下一项方法设计必须针对完整对象的可识别性和覆盖不足，不能继续把类别一致性目标下降当作实例建图收益。新机制有效性、同版本机器人对照和新增独立评测仍未完成。已有窗口、模型和本轮开发观察全部公开记录，不冒充独立泛化。</p>
<p><a href="summary.json">完整指标</a> · <a href="artifact_audit.json">独立产物核查</a> · <a href="object_diagnostics.json">逐对象数据</a> · <a href="../../build/benchmarks/counterfactual_geometry_revision_v1/protocol.json">冻结协议</a> · <a href="../../docs/benchmarks/counterfactual_geometry_revision_2026-11-14.md">方法与执行说明</a></p></html>'''
    (DEST/'index.html').write_text(body)
    for name in ['summary.json','artifact_audit.json','object_diagnostics.json']:dump(DEST/name,read(OUT/name))
    dump(DEST/'images.json',images);dump(DEST/'manifest.json',dict(generator_sha256=sha(__file__),files={f.name:sha(f) for f in DEST.iterdir() if f.is_file() and f.name!='manifest.json'}));print('REPORT',DEST/'index.html',flush=True)


if __name__=='__main__':main()
