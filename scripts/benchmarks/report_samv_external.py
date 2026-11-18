"""Report every fixed SAM-V/SAM3 object clip with actual RGB and masks."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni_samv_report_mpl')
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_final_common import ROOT,read,dump,sha
from probe_samv_external import OUT,OLD,METHODS,CONTROL

DEST=ROOT/'output/samv_evidence_comparison_v1'
NAMES=dict(samv_joint_seed='SAM-V / joint, seed prompts',samv_joint_current='SAM-V / joint, current prompts',
           samv_single_current='SAM-V / single, current prompts',image_reprompt='SAM3 / image reprompt',
           video_seed='SAM3 / video seed',video_world_feedback='SAM3 / video feedback')


def fmt(value):return 'N/A' if value is None else f'{value:.4f}'


def main():
    d=read(OUT/'summary.json');p=read(OUT/'protocol.json');seal=read(OUT/'predictions_complete.json')
    assert d['complete'] and seal['complete'] and read(OUT/'native_parity.json')['logits_exact']
    for f,h in seal['files'].items():assert sha(f)==h,f
    DEST.mkdir(parents=True,exist_ok=False);images=[];records=[]
    table=[]
    order=['image_reprompt','video_seed','video_world_feedback','samv_single_current','samv_joint_current','samv_joint_seed']
    for key,s in d['results'].items():
        table.append(f'<tr><td>{key}</td>'+''.join(f'<td>{s["macro_case_f1"][m]:.4f}</td>' for m in order)+f'<td>{s["missing_gt_cases"]}</td></tr>')
        for row in s['cases']:
            case=next(c for c in p['cases'] if c['name']==row['case'])
            j=case['common_frames'][len(case['common_frames'])//2]
            rgb=np.array(Image.open(case['records'][j]['rgb']).convert('RGB'));h,w=rgb.shape[:2]
            fig=plt.figure(figsize=(15,8),layout='constrained');grid=fig.add_gridspec(3,2)
            ax=fig.add_subplot(grid[0,:]);ax.imshow(rgb);ax.axis('off')
            ax.set_title(f'{case["name"]} | actual frame {case["records"][j]["frame_id"]} | fixed target {row["gt_target"]}',loc='left')
            for panel,method in enumerate(['video_seed','samv_joint_seed','samv_single_current','samv_joint_current']):
                root=OUT if method in METHODS else OLD
                data=np.load(root/'cases'/case['name']/(method+'.npz'));assert data['valid'][j]
                mask=np.unpackbits(data['masks_packed'][j],axis=1)[:,:w].astype(bool)
                overlay=rgb.astype(float)/255;overlay[mask]=.6*overlay[mask]+.4*np.array([0,.82,.68])
                ax=fig.add_subplot(grid[1+panel//2,panel%2]);ax.imshow(overlay);ax.axis('off')
                if 'current' in method:
                    xy=np.array(case['prompts'][j]['pixels']);labels=np.array(case['prompts'][j]['labels'])
                    for lab,color in [(1,'#ffe34a'),(0,'#ff4580')]:
                        take=labels==lab;ax.scatter(xy[take,0],xy[take,1],s=22,c=color,edgecolors='black',linewidths=.4)
                metric=next((x['f1'] for x in row['methods'][method].get('frames',[]) if x['frame_index']==j),None)
                ax.set_title(NAMES[method]+' | measured-observation F1 '+fmt(metric),fontsize=10,loc='left')
            fig.suptitle('External-model evidence test; no new Ours map or independent benchmark score',fontsize=12,color='#596976')
            name=case['name']+'.png';fig.savefig(DEST/name,dpi=140);plt.close(fig);images.append(name)
            records.append(dict(case=case['name'],frame_index=j,frame_id=case['records'][j]['frame_id'],rgb_sha256=sha(case['records'][j]['rgb']),selection='Middle entry of frozen common nonseed frames, independent of masks/GT/metrics',file=name))
    detail=[]
    for key,s in d['results'].items():
        for r in s['cases']:
            detail.append('<tr><td>'+r['case']+'</td><td>'+str(r['gt_target'])+'</td>'+''.join('<td>'+fmt(r['methods'][m]['f1'])+'</td>' for m in order)+'</tr>')
    gate='通过信息预检，但尚非新机制或完整地图验收' if d['information_gate'] else '未通过信息预检；不据此启动完整地图分支或继续调参'
    page=f'''<!doctype html><meta charset="utf-8"><title>SAM-V 外部对象分割对照</title>
<style>body{{max-width:1400px;margin:36px auto;padding:0 24px;background:#f5f7f9;color:#20364a;font:16px/1.65 system-ui}}section{{background:white;padding:24px;margin:18px 0;border-radius:12px}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}a{{color:#008e82}}</style>
<h1>omni_slam：训练后多视图分割器的外部对照</h1><p>8个预先固定对象片段 × 11帧；沿用原始测量点提示。所有新掩码封存后才评分。两窗口均已暴露，非独立测试、官方榜单或完整地图AP。</p>
<section><h2>结果：{gate}</h2><p>表中是共同合格非种子帧上的置信度加权测量观测F1，先逐案例求F1，再在同一有参考目标案例集合取均值。缺少固定GT对应的案例保留N/A，不把它记作零或换案例。所有既有SAM3对照的TP/FP/FN及F1重新读取原始掩码，已与旧结果精确核对。</p>
<table><tr><th>窗口</th>{''.join('<th>'+NAMES[m]+'</th>' for m in order)}<th>缺GT案例</th></tr>{''.join(table)}</table></section>
<section><h2>对照含义与成本</h2><p>Joint-current 与 single-current 使用同一个训练模型、同一当前帧提示和SAM图像特征，只改变VGGT能读取的图像上下文。Joint-seed只用首帧提示，但联合读取11帧，包含未来图像；不能称为因果跟踪。SAM3参照包含不同架构、输入尺寸和提示调度，也不是计算量匹配的消融。所有SAM-V行都属于外部方法，未标为Ours。</p>
<p>实测调用：{seal['counts']}；总推理秒数{seal['seconds']:.1f}；峰值已分配显存{seal['peak_cuda_gib']:.2f} GiB。包括首次原生路径重放核验。未训练、未生成新地图。</p>
<p>加载完整SAM/VGGT基础权重时执行严格键匹配，142个发布训练张量逐项核验；首例缓存路径的低分辨率logits和IoU输出与作者原生forward完全一致。1024方形图像缩放沿用作者接口，输出逐帧拆开后再反变换到原图，以免全景插值污染相邻帧。</p></section>
<section><h2>全部对象</h2><table><tr><th>案例</th><th>固定参考目标</th>{''.join('<th>'+NAMES[m]+'</th>' for m in order)}</tr>{''.join(detail)}</table></section>
<section><h2>真实图像与掩码</h2><p>每例固定选择共同非种子帧列表的中间帧；全部8例保留。绿色为预测区域，黄色／粉色点是当前帧正／负提示（仅标在current条件）。图像F1来自可见测量点，不是稠密像素分割GT。</p>{''.join('<img src="'+x+'">' for x in images)}</section>
<section><h2>原始证据与研究边界</h2><p><a href="../../build/benchmarks/samv_evidence_comparison_v1/protocol.json">固定协议</a> · <a href="../../build/benchmarks/samv_evidence_comparison_v1/summary.json">全部读出</a> · <a href="../../build/benchmarks/samv_evidence_comparison_v1/native_parity.json">原生输出一致性</a> · <a href="../../build/benchmarks/samv_external_v1/assets.json">模型哈希</a> · <a href="https://github.com/gong208/SAM-V/tree/33fab24c1d0d21ac56f4e471abf30c6de3e7018b">作者固定版本</a></p>
<p>主线仍是持续对象级语义建图。外部感知能力不能代替地图更新机制的贡献；若开展后续完整地图试验，必须共享相同新增感知输入，单独检验我们的方法。既有公开／机器人结果保留，总目标仍未完成。</p></section>'''
    (DEST/'index.html').write_text(page);dump(DEST/'examples.json',records)
    dump(DEST/'manifest.json',dict(complete=True,files={str(f):sha(f) for f in DEST.iterdir()},source_sha256=sha(__file__),summary_sha256=sha(OUT/'summary.json'),goal_complete=False))
    print('SAMV REPORT',DEST/'index.html',flush=True)


if __name__=='__main__':main()
