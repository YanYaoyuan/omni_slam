"""Visualize actual counterfactual masks and unused-view evidence; no GT."""
import numpy as np
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from analyze_partition_query_probe import OUT,unpack

REPORT=ROOT/'output/partition_query_probe_v1'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=read(OUT/'protocol.json');done=read(OUT/'predictions_complete.json');analysis=read(OUT/'readout.json');inputs=read(OUT/'verification_inputs.json')
    assert analysis['predictions_sha256']==sha(OUT/'predictions_complete.json')
    REPORT.mkdir(parents=True,exist_ok=True)
    acquired={(r['key'],tuple(r['object_ids'])):r for r in done['acquisition']}
    reads={(r['key'],tuple(r['object_ids'])):r for r in analysis['results']}
    views={(r['key'],r['frame_id']):r for r in inputs['records']}
    verification={(r['key'],r['frame_id'],r['semantic_id']):r for r in done['verification']}
    figures='';table='';selection='';provenance=[]
    for row in p['pairs']:
        selection+=f'<tr><td>{row["key"]}</td><td>{row["object_ids"]}</td><td>{row["status"]}</td><td>{row["eligible_views"]}</td></tr>'
        if row['status']!='ready':continue
        key=row['key'];pair=tuple(row['object_ids']);r=reads[(key,pair)];rgb=np.asarray(Image.open(row['acquisition_rgb']).convert('RGB'))
        fig,axes=plt.subplots(2,3,figsize=(18,7));axes=axes.ravel();axes[0].imshow(rgb);axes[0].set_title(f'Actual RGB / frame {row["acquisition_frame"]}')
        for ax,query in zip(axes[1:5],acquired[(key,pair)]['queries']):
            d=dict(np.load(query['file']));mask=unpack(d,query['selected'])[0];ax.imshow(rgb)
            rgba=np.zeros((*mask.shape,4));rgba[mask]=[.05,.8,.65,.45];ax.imshow(rgba)
            for point,label in zip(d['point_coords'],d['point_labels']):
                ax.scatter(*point,c='#00ff4c' if label else '#ff1744',marker='+' if label else 'x',s=85,linewidths=2)
            satisfied=r['prompts'][query['name']]['prompt_labels_satisfied']
            ax.set_title(f'{query["name"]} | native quality {max(query["scores"]):.3f} | follows prompts: {satisfied}')
        # Always first preselected verification frame, not a best-looking view.
        frame=row['verification_frames'][0];vr=views[(key,frame)];vrgb=np.asarray(Image.open(vr['rgb']).convert('RGB'))
        data=dict(np.load(verification[(key,frame,row['semantic_id'])]['file']));masks=unpack(data);record=r['verification'][0]
        axes[5].imshow(vrgb)
        for name,color in [('original_a',[.2,.8,1.,.4]),('original_b',[1.,.55,.1,.4])]:
            q=record['readout'][name]['query']
            if q>=0:
                rgba=np.zeros((*masks[q].shape,4));rgba[masks[q]]=color;axes[5].imshow(rgba)
        axes[5].set_title(f'Unused frame {frame} / class-only retrieved masks')
        for ax in axes:ax.axis('off')
        fig.suptitle(f'{key} / geometry object IDs {pair} | selected masks, no GT selection',fontsize=15)
        fig.tight_layout();name=f'{key}_{pair[0]}_{pair[1]}.png';fig.savefig(REPORT/name,dpi=170,bbox_inches='tight');plt.close(fig)
        figures+=f'<h3>{key} / 对象 {pair}</h3><img src="{name}"><p>前五格为取证帧和四次实例解码，绿色加号为正点、红色叉号为负点；最后一格始终展示预先固定的第一个新视角，不挑选最佳验证画面。最后一格的查询仅输入类别，掩码按实测几何匹配后着色，低原生分数仍明确保留在下表。</p>'
        provenance.append(dict(file=name,acquisition_rgb=row['acquisition_rgb'],acquisition_sha256=sha(row['acquisition_rgb']),verification_rgb=vr['rgb'],verification_sha256=sha(vr['rgb'])))
        for view in r['verification']:
            a,b=view['readout']['original_a'],view['readout']['original_b'];qa=view['original_pair']['query_mask_cell_iou']
            table+='<tr>'+''.join(f'<td>{v}</td>' for v in [pair,view['frame_id'],f'{a["visible_cells"]} / {b["visible_cells"]}',view['original_pair']['measured16_each'],f'{a["native_score"]:.3f} / {b["native_score"]:.3f}',f'{qa:.3f}' if qa is not None else '未知'])+'</tr>'
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 对象划分取证探针</title><style>body{{font:16px/1.7 system-ui;color:#193349;max-width:1500px;margin:32px auto;padding:0 24px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d5dfe8;padding:8px;text-align:left}}.note{{background:#edf5f7;padding:20px;border-left:5px solid #188b90}}</style>
<h1>对象划分取证：实际新掩码与未使用视角</h1>
<p class="note"><b>已得到可区分两种对象解释的新观测，但尚未做完整地图修订或证明 AP 收益。</b>冻结几何关联与旧学习关联存在六组输出归属冲突；三组有足够共同测量可见性，其余三组明确跳过。为验证补取5个原建图未使用的图像/雷达帧，仅下载约14.2 MB。没有读取参考标签、改变旧地图或重训练。</p>
<h2>为什么值得继续研究取证机制</h2>
<p>车辆对15/18：两个“分别分割”候选满足正负点提示；两个“合成一个对象”候选未覆盖全部正点。未使用的223帧中，原两组各有16/19个实测可见单元，分别对应两个不相交的查询掩码，原生分数0.832/0.824。这为保持两车分离提供了新的图像证据，不能把模型质量分数直接当作对象身份概率。228帧无对应实测支持，保持未知。</p>
<p>建筑对37/50：两个“分别分割”候选都未满足排斥点提示，“合成一个对象”候选满足正点提示。363/368帧中，两侧匹配到的原生查询掩码虽编号不同，却分别有0.857/0.930的实测单元 IoU。这说明只比较查询 ID 会误判分离。但这些查询的原生分数仅0.012–0.051，尚不足以据此认定应当合并，也没有参考标签证明本例对象真值。</p>
<p>建筑对33/48：验证帧缺少B侧原测量支持，两次都无法判断两侧关系。分割候选产生了少量新支持，不等于已有充分验证，不能将缺测当作否定证据。</p>
<h2>全部冲突与跳过情况</h2><table><tr><th>窗口</th><th>几何对象 ID</th><th>状态</th><th>共同可见取证帧数</th></tr>{selection}</table>
<p>0002唯一输出冲突没有满足共同可见条件，因此本探针未验证该窗口的方法效果。候选范围仅覆盖两个现成输出划分的分歧，没有覆盖未出生对象或所有原始假设；这是当前候选覆盖限制，不应伪装成完整对象建图算法。</p>
<h2>新视角的原对象支持读取</h2><table><tr><th>对象对</th><th>帧</th><th>A/B可见单元</th><th>双方≥16且有查询</th><th>A/B原生分数</th><th>查询掩码间 IoU</th></tr>{table}</table>
<p>沿用20 cm物理单元、实际新雷达5 cm深度核对和既有无阈值查询读取。未使用图像是同一已暴露开发窗口内的邻近帧，时间上相关；只表示未参与旧模型输入和提示生成，不能称为独立 benchmark 或统计独立样本。固定+1/+6附近姿态可用帧，未按图像内容或验证结果选帧。</p>
<h2>实际分割结果</h2>{figures}
<h2>运行与下一步边界</h2><p>8次图像编码、12次实例解码、5次类别查询，模型加载及推理合计约{done['wall_seconds']:.2f}秒；这不含下载、可见性准备及CPU分析。GPU1峰值分配{done['peak_gpu_allocated_bytes']/1024**3:.2f} GiB，检查点没有缺失或多余权重。没有更换失败版本的结论，也没有机器人或新场景成绩。</p>
<p>本次支持的结论是：原关联分数之外，确实可以获取关于对象划分的额外图像证据。它尚未证明完整方法优势。下一阶段必须把“原始观测、提示候选和独立于提示的验证读取”分开，给普通重分割和不同选视角策略相同预算，真正修订历史归属并评价完整地图；不能仅把本页三个案例的规则硬编码进地图。</p>
<p><a href="protocol.json">冻结探针协议</a> · <a href="readout.json">全部读取与提示一致性</a> · <a href="images.json">真实图像来源</a> · <a href="predictions_complete.json">模型执行记录</a></p></html>'''
    (REPORT/'index.html').write_text(page);dump(REPORT/'images.json',provenance)
    for name in ['protocol.json','readout.json','predictions_complete.json']:dump(REPORT/name,read(OUT/name))
    dump(REPORT/'manifest.json',dict(generator_sha256=sha(__file__),files={f.name:sha(f) for f in REPORT.iterdir() if f.name!='manifest.json'}))
    print('REPORT',REPORT/'index.html',flush=True)


if __name__=='__main__':main()
