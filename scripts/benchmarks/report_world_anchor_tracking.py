"""All preselected clips, real RGB/masks and full diagnostic table."""
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from probe_world_anchor_tracking import OUT,KEYS,METHODS,verify,prepared

DEST=ROOT/'output/world_anchor_tracking_probe_v1'
NAMES=['Image + world prompts','Video seed only','Video + world feedback']
CN=['逐帧三维提示','普通视频传播','视频＋三维反馈']


def main():
    verify();cases=prepared()['cases'];s=read(OUT/'summary.json');seal=read(OUT/'predictions_complete.json');audit=read(OUT/'prompt_audit.json');assert s['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h
    DEST.mkdir(parents=True,exist_ok=True);table='';figures='';images=[];promptrows='';summaryrows='';compliance_rows=''
    for key in KEYS:
        result=s['results'][key];summaryrows+=f'<tr><td>{key}</td>'+''.join(f'<td>{100*result["macro_case_f1"][m]:.2f}</td>' for m in METHODS)+f'<td>{result["missing_gt_cases"]}</td></tr>'
        frames=[f for c in audit['results'][key] for f in c['frames'] if f['frame_index']>0];known=sum(f['known'] for f in frames);wrongp=sum(f['wrong_positive'] for f in frames);wrongn=sum(f['wrong_negative'] for f in frames);unknown=sum(f['unknown'] for f in frames)
        promptrows+=f'<tr><td>{key}</td><td>{known}</td><td>{wrongp}</td><td>{wrongn}</td><td>{unknown}</td></tr>'
        total=sum(f['prompts'] for f in frames);compliance_rows+=f'<tr><td>{key}</td><td>{total}</td>'+''.join(f'<td>{sum(f["compliance"][m] for f in frames)}</td>' for m in METHODS)+'</tr>'
        for row in result['cases']:
            for m,name in zip(METHODS,CN):
                r=row['methods'][m];vals=[row['case'],name,row['gt_target'],row['common_nonseed_frames']]+([f'{100*r[v]:.2f}' for v in ['precision','recall','f1']] if row['gt_target'] is not None else ['N/A']*3)+[r['new_correct_unowned_sources'],r['unknown_selected']]
                table+='<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in vals)+'</tr>'
    for case in cases:
        j=case['common_frames'][len(case['common_frames'])//2];rec=case['records'][j];rgb=np.asarray(Image.open(rec['rgb']).convert('RGB'));q=case['prompts'][j];xy=np.array(q['pixels']);fig,axes=plt.subplots(3,1,figsize=(12,10),layout='constrained')
        for ax,m,name in zip(axes,METHODS,NAMES):
            d=np.load(OUT/'cases'/case['name']/(m+'.npz'));mask=np.unpackbits(d['masks_packed'][j],axis=1)[:,:int(d['width'])].astype(bool);overlay=np.zeros((*mask.shape,4));overlay[mask]=[0,.9,.6,.4];ax.imshow(rgb);ax.imshow(overlay);ax.scatter(xy[:,0],xy[:,1],c=['lime' if k else 'red' for k in q['labels']],s=22);ax.set_title(name);ax.axis('off')
        fig.suptitle(case['name']+f' / frame {rec["frame_id"]}\nMask overlay; green/red = proposed positive/negative world anchors, NOT GT. Seed-only does not consume these current-frame points.',fontsize=10)
        name=case['name']+'.png';fig.savefig(DEST/name,dpi=110);plt.close(fig);figures+=f'<h3>{case["name"]}</h3><img src="{name}">';images.append(dict(case=case['name'],frame_id=rec['frame_id'],rgb=rec['rgb'],rgb_sha256=sha(rec['rgb']),selection='middle common eligible nonseed frame, no outcome selection',file=name))
    style='<style>body{font:16px/1.7 system-ui;color:#203746;max-width:1300px;margin:30px auto;padding:0 24px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d8e1e6;padding:7px}.note{padding:18px;background:#fff1d7}a{color:#167a8a}</style>'
    verdict='通过输入能力筛选，但尚无完整地图收益或新颖性证明。' if s['capability_screen_pass'] else '三维反馈未通过输入能力筛选，不将该版本扩展为完整建图方法。'
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam 对象跟踪证据验证</title>{style}<h1>三维对象提示与视频掩码：输入能力验证</h1>
<p class="note"><b>{verdict}</b>预先固定8个对象片段，每段11帧。不是新benchmark主指标、完整地图结果或独立测试。普通视频传播是已有能力，不更名为Ours。</p>
<h2>共同非种子帧上的实测观测 F1 ×100</h2><table><tr><th>窗口</th>{''.join('<th>'+n+'</th>' for n in CN)}<th>无GT对应案例</th></tr>{summaryrows}</table>
<p>各可对应GT案例等权平均。0002的对象1没有基线同类GT重叠，保留为N/A；其余三个构成该窗口均值，不能描述为四对象完整准确率。只在三组均有三维提示的共同非种子帧比较，共享测量与参考标注；重复出现的源点按观测重复计数，未知参考排除。这不是3D地图AP。</p>
<h2>全部对象、全部条件</h2><table><tr><th>案例</th><th>方法</th><th>GT对应</th><th>共同帧</th><th>Precision</th><th>Recall</th><th>F1</th><th>正确未归属源点</th><th>选中未知观测</th></tr>{table}</table>
<p>“正确未归属源点”是固定GT对应下新掩码覆盖的原未归属实测源点数，尚未写入地图，也不保证完整地图增益。</p>
<h2>代价与复核</h2><p>实际执行 {seal['counts']['encodes']} 次共享图像编码、{seal['counts']['decodes']} 次掩码解码；采集阶段约 {seal['seconds']:.2f} 秒（不含模型加载），PyTorch峰值分配 {seal['peak_cuda_gib']:.2f} GiB。三组种子掩码逐像素一致。三维反馈比普通传播多解码，因此不是等计算预算消融。3项提示选择测试通过，空间留出块没有作为提示。</p>
<p>首次加载器按扩展名只识别JPEG，故在任何图像编码之前报错。第二次增加指向原PNG的.jpg符号链接，原始字节和像素保持不变，没有重压缩、换案例或改算法。所有掩码封存后才读取GT。提示审计首次JSON写入遇到NumPy整数类型，修正序列化后重算同一审计；未重跑模型。</p>
<h2>地图锚点不能当作真值</h2><table><tr><th>窗口</th><th>后续已知提示</th><th>错误正提示</th><th>错误负提示</th><th>未知提示</th></tr>{promptrows}</table>
<p>提示正确性按原实测源点的最近参考标签复核；该表不能单独证明反馈失败的原因，原生当前掩码与记忆交互等因素仍可能影响结果。没有据此修改提示或挑选新的结果。</p>
<table><tr><th>窗口</th><th>后续提示总数</th>{''.join('<th>'+n+' 满足数</th>' for n in CN)}</tr>{compliance_rows}</table>
<p>本轮反馈显式设置use_prev_mem_frame=True，原生add_new_points_or_box接口默认False；它测试了过去帧记忆与当前纠正提示的直接耦合，不是原生默认交互流程。提示满足率下降提示这种耦合本身尚不可靠，不能把本版失败推广为三维提示无效，也未追加另一个开关版本救场。</p>
<h2>真实图像与掩码</h2><p>每个案例固定展示共同非种子帧的中间一帧，全部8个案例均列出；绿色覆盖为预测掩码，不是真值。三组图中展示相同当前三维提示便于比较，普通视频组实际只使用种子帧提示。</p>{figures}
<h2>研究边界</h2><p><a href="https://any3dis.github.io/">Any3DIS</a>已使用3D辅助2D跟踪生成三维候选；<a href="https://arxiv.org/abs/2606.24449">SENTRY</a>研究写入前记忆验证；<a href="https://arxiv.org/abs/2609.28078">LiAM-SAM</a>研究生命周期记忆修正；<a href="https://arxiv.org/abs/2601.08831">3AM</a>将几何模型特征与跟踪记忆结合。当前原型是能力验证，不是这些方法的原版复现或已成立的新贡献。</p>
<p>旧地图来自窗口末状态，部分锚点可能来自后续帧；不宣称在线因果推理。主线仍是对象级语义建图；新机制、同版本机器人对照和新增独立评测尚未完成。</p>
<p><a href="summary.json">完整诊断</a> · <a href="prompt_audit.json">全部提示审计</a> · <a href="../../build/benchmarks/world_anchor_tracking_probe_v1/protocol.json">冻结协议</a> · <a href="../../build/benchmarks/world_anchor_tracking_probe_v1/predictions_complete.json">掩码封存</a></p></html>'''
    (DEST/'index.html').write_text(body)
    for name in ['summary.json','prompt_audit.json','runtime_input_adapter.json']:dump(DEST/name,read(OUT/name))
    dump(DEST/'images.json',images);dump(DEST/'manifest.json',dict(generator_sha256=sha(__file__),files={f.name:sha(f) for f in DEST.iterdir() if f.is_file() and f.name!='manifest.json'}));print('REPORT',DEST/'index.html',flush=True)


if __name__=='__main__':main()
