"""Render actual probe outputs and the separate frozen-field error audit."""
from pathlib import Path
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT, read, dump, sha
from audit_mask_observation_code import OUT, INPUT, KEYS, METHODS, verify

DEST = ROOT/'output/mask_observation_code_probe_v1'
NAMES = dict(single_mask='Single-mask coverage', mask_cosine='Mask affinity',
             full_code='Full membership code', shuffled_code='Shuffled code', nearest_anchor='Nearest anchor')


def main():
    verify(); seal=read(OUT/'predictions_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    error=read(OUT/'field_error_types.json'); ep=read(OUT/'field_error_type_protocol.json')
    for f,h in {**ep['source_hashes'],**ep['input_hashes']}.items():assert sha(f)==h,f
    assert error['protocol_sha256']==sha(OUT/'field_error_type_protocol.json')
    DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');x=np.arange(len(METHODS));width=.36
    for j,key in enumerate(KEYS):
        values=[100*seal['results'][key]['methods'][m]['source_consensus']['macro'] for m in METHODS]
        bars=ax.bar(x+(j-.5)*width,values,width,label=key[-4:],color=['#205b80','#1a9b90'][j])
        ax.bar_label(bars,fmt='%.2f',fontsize=8)
    ax.set(xticks=x,xticklabels=[NAMES[m].replace(' ','\n',1) for m in METHODS],ylim=(0,111),ylabel='Macro consistency with predicted anchor IDs (%)',
           title='Spatially withheld anchor identities — NOT GT accuracy or map AP')
    ax.legend();fig.savefig(DEST/'anchor_consistency.png',dpi=160);plt.close(fig)

    categories=['same_target','same_class_other','different_known_class','ignored']
    colors=['#159585','#e7b347','#d75252','#9aabb7']
    fig,ax=plt.subplots(figsize=(9,3.2),layout='constrained')
    for j,key in enumerate(KEYS):
        totals=error['results'][key]['totals'];total=sum(totals.values());left=0
        for name,color in zip(categories,colors):
            value=100*totals[name]/total
            ax.barh(j,value,left=left,color=color,label=name.replace('_',' ') if j==0 else None)
            if value>4:ax.text(left+value/2,j,f'{value:.1f}%',ha='center',va='center',fontsize=9)
            left+=value
    ax.set(yticks=[0,1],yticklabels=['0000','0002'],xlim=(0,100),xlabel='Confidence-weighted reference coverage assigned via NEW points (%)',
           title='Frozen failed field: previously AP50-correct objects only')
    ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,-.27),frameon=False)
    fig.savefig(DEST/'error_types.png',dpi=160);plt.close(fig)

    images=[]
    for key in KEYS:
        frames=read(OUT/key/'frames.json');frame=next(f['frame_id'] for f in frames if not f.get('skipped',False))
        manifest=read(INPUT/key/'manifest.json');rec=next(r for r in manifest['records'] if r['frame_id']==frame)
        d=dict(np.load(OUT/key/'frames'/f'{frame:06d}.npz'));v=dict(np.load(rec['file']))
        where=np.searchsorted(v['visible_source_ids'],d['source_ids'])
        np.testing.assert_array_equal(v['visible_source_ids'][where],d['source_ids'])
        xy=v['pixel_xy'][where];rgb=np.asarray(Image.open(rec['rgb']).convert('RGB'))
        fig,axes=plt.subplots(1,3,figsize=(15,3.5),layout='constrained')
        axes[0].imshow(rgb);axes[0].set_title('Actual RGB / first eligible frame')
        for ax,method in zip(axes[1:],['single_mask','full_code']):
            ax.imshow(rgb);right=d[method]==d['pseudo_identity'];abstain=d[method]==0
            color=np.where(abstain,'#edc047',np.where(right,'#0ec6a3','#f03d63'))
            ax.scatter(xy[:,0],xy[:,1],c=color,s=2,alpha=.75,linewidths=0)
            ax.set_title(NAMES[method])
        for ax in axes:ax.axis('off')
        fig.suptitle(f'{key} frame {frame}: held-out source points; green=same pseudo-ID, red=different, yellow=abstain. NOT GT.')
        name=f'{key}_actual.png';fig.savefig(DEST/name,dpi=150);plt.close(fig)
        images.append(dict(key=key,frame=frame,file=name,rgb=rec['rgb'],rgb_sha256=sha(rec['rgb'])))

    rows=''.join('<tr><td>'+html.escape(NAMES[m])+'</td>'+''.join(f'<td>{100*seal["results"][k]["methods"][m]["source_consensus"]["macro"]:.3f}</td>' for k in KEYS)+'</tr>' for m in METHODS)
    known=[]
    for key in KEYS:
        t=error['results'][key]['totals'];bad=sum(t.values())-t['same_target']
        known.append(100*t['different_known_class']/bad)
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam 机制检查</title>
<style>body{{max-width:1120px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#20333f}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 18px;border-bottom:1px solid #ccd5da}}.note{{padding:16px;background:#fff5df}}a{{color:#176f88}}</style>
<h1>omni_slam：先分清“对象身份”与“目标外污染”</h1>
<p class="note">这是两个已暴露室外窗口的机制诊断，不是新地图 AP、官方榜单或独立语义准确率。完整掩码解码未超过简单亲和度，未通过预设筛选，不扩展完整地图。主线强基线保持冻结。</p>
<h2>1. 真实掩码中存在身份信息，但复杂解码没有优势</h2>
<p>108帧，2米空间块隔离建模与验证点；只对同时可见且有至少16个建模单元的对象做辨识。0000覆盖41/47个预测对象、62950个验证源点；0002覆盖35/36个预测对象、51277点。上游地图仍用过这些观测，因此仅为模块内部留出，不是独立测试。完整管线继承既有监督几何模型，不能称为免训练。</p>
<table><tr><th>方法</th><th>0000 一致率%</th><th>0002 一致率%</th></tr>{rows}</table>
<img src="anchor_consistency.png"><p>数值是与预测核心ID的宏平均一致率；核心ID自身可能错误。没有未知背景点的接纳测试，不能据此宣称99%的语义精度。亲和度是共享输入的简化控制，并非 SAI3D/MV3DIS 原版跑分。</p>
<h2>2. 补查旧失败地图：污染主要跨语义类别</h2>
<p>对旧版 surface_evidence_field 的冻结输出，恢复原始语义标签，分开道路等有效类别与忽略区域。此前正确对象的新点所负责的错误参考覆盖中，{known[0]:.2f}% / {known[1]:.2f}% 来自其他已知语义类别；同类其他实例占比很小。该结论仅适用于这份失败输出及上述对象范围，不等于所有方法或场景均如此。</p>
<img src="error_types.png"><p>这是参考置信度面积分解，不是源点准确率，也不是因果干预。没有修改地图、重新评分或用真值选择预测。</p>
<h2>3. 原始图像与真实源点</h2>'''
    for row in images:body+=f'<img src="{row["file"]}">'
    body+='''<h2>4. 下一步约束</h2><p>下一项机制必须针对未归属点的目标/背景判别，并包含简单语义背景排除与原几何强基线，才能判断新机制是否有独立价值。仅提高已有对象身份一致率不足以解决本次污染。尚未建立有效新方法，机器人参考对照已完成，有效方法的机器人与新增独立场景验收仍待完成。</p>
<p>相关工作：<a href="https://arxiv.org/abs/2312.11557">SAI3D</a>已使用多视角掩码关联几何基元；<a href="https://arxiv.org/abs/2604.08916">MV3DIS</a>进一步用3D先验匹配与修正多视角掩码。普通亲和度或Bernoulli解码本身不作为原创贡献。</p>
<p><a href="../../build/benchmarks/mask_observation_code_probe_v1/protocol.json">冻结探针协议</a> · <a href="../../build/benchmarks/mask_observation_code_probe_v1/predictions_complete.json">全部探针结果</a> · <a href="../../build/benchmarks/mask_observation_code_probe_v1/field_error_types.json">语义错误分解</a></p></html>'''
    (DEST/'index.html').write_text(body)
    dump(DEST/'audit.json',dict(complete=True,information_screen_passed=seal['information_screen_passed'],
         probe_seal_sha256=sha(OUT/'predictions_complete.json'),field_error_types_sha256=sha(OUT/'field_error_types.json'),
         images=images,other_known_semantic_fraction_of_wrong_new_area_percent=known,gt_accuracy=False,new_map=False))
    dump(DEST/'manifest.json',dict(generator_sha256=sha(__file__),files={f.name:sha(f) for f in DEST.iterdir() if f.is_file() and f.name!='manifest.json'}))
    print('REPORT',DEST/'index.html',known,flush=True)


if __name__=='__main__':main()
