"""Export fixed-geometry semantic maps and report the frozen development trial."""
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_context_residual_memory import OUT,METHODS,verify

REPORT=ROOT/'output/context_residual_memory_v1'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocol=verify(True);seal=read(OUT/'predictions_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    result=read(OUT/'corrected_summary.json');assert result['indoor_projection_corrected'] and result['original_baseline_reproduced']
    correction=read(OUT/'evaluation_correction.json')
    assert sha(ROOT/'scripts/benchmarks/rescore_context_residual_memory.py')==correction['source_sha256']
    assert sha(OUT/'summary.json')==correction['original_summary_sha256']
    REPORT.mkdir(parents=True,exist_ok=True);exports={}
    labels=dict(learned='原 learned 均值',masked='masked-only',bbox='bbox-only',fixed_context='固定系数 0.5',object_context='各对象独立估计',shared_context='候选：共享上下文记忆')
    for scene in protocol['scenes']:
        key=scene['key'];source=Path(scene['fixed_map']);rows=read(source/'classification.json')['objects']
        new={r['object_id']:r for r in read(OUT/key/'readouts.json')}
        for method in METHODS:
            folder=OUT/key/'exported_maps'/method;folder.mkdir(parents=True,exist_ok=True)
            geom=folder/'geometry.npz'
            if geom.exists():assert sha(geom)==sha(source/'geometry.npz')
            else:shutil.copyfile(source/'geometry.npz',geom)
            updated=[]
            for r in rows:
                row=deepcopy(r);row['predictions']={'mean':new[r['object_id']]['methods'][method]};updated.append(row)
            dump(folder/'classification.json',dict(objects=updated,method=method,
                fixed_geometry=True,fixed_memberships=True,development_only=True,
                prediction_seal_sha256=sha(OUT/'predictions_complete.json')))
            assert all(a['observation_ids']==b['observation_ids'] and a['object_id']==b['object_id'] for a,b in zip(rows,updated))
            exports[str(folder)]=dict(geometry_sha256=sha(geom),classification_sha256=sha(folder/'classification.json'))
    fig,axes=plt.subplots(2,3,figsize=(13,6.4))
    for ax,scene in zip(axes.flat,protocol['scenes']):
        history=read(OUT/scene['key']/'history.json')
        ax.plot([h['frame_id'] for h in history],[h['fit']['beta'] for h in history],lw=1.5,color='#008d85')
        ax.axhline(.5,color='#8f9ba5',ls='--',lw=1);ax.set_ylim(0,1);ax.set_title(scene['key'].replace('2013_05_28_drive_','drive_'),fontsize=10);ax.set_xlabel('Cached frame ID');ax.set_ylabel('Estimated context coefficient')
    axes.flat[-1].axis('off');fig.suptitle('Fixed-membership prefix replay (not causal SLAM)');fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(REPORT/'context_history.png',dpi=150);plt.close(fig)
    # Real robot: show every changed final candidate label, plus first four
    # unchanged baseline-car objects as explicit controls, no GT selection.
    robot=next(s for s in protocol['scenes'] if s['kind']=='robot');cache=Path(robot['cache'])
    m=read(cache/'manifest.json');job=read(ROOT/'build/benchmarks/evidence_retention_20261111/jobs/livo_full_v2.json')
    rows=read(OUT/'rosbag2245/readouts.json');changed=[r for r in rows if r['methods']['learned']['name']!=r['methods']['shared_context']['name']]
    controls=[r for r in rows if r['methods']['learned']['name']=='car' and r['methods']['shared_context']['name']=='car'][:4]
    obs={rec['observation_offset']+j:(i,rec,j) for i,rec in enumerate(m['records']) for j in range(rec['masks'])}
    chosen=changed+controls;fig,axes=plt.subplots((len(chosen)+3)//4,4,figsize=(16,3.6*((len(chosen)+3)//4)),squeeze=False);captions=[]
    for ax,r in zip(axes.flat,chosen):
        oid=r['selected_observation_ids'][0];i,rec,j=obs[oid];raw=job['frames'][i];assert rec['frame_id']==raw['frame_id']
        image=np.array(Image.open(raw['rgb']).convert('RGB'));f=np.load(cache/'frames'/rec['file'])
        mask=np.unpackbits(f['masks_packed'][j],axis=1)[:,:image.shape[1]].astype(bool);ys,xs=np.where(mask)
        x0=max(0,xs.min()-30);x1=min(image.shape[1],xs.max()+31);y0=max(0,ys.min()-30);y1=min(image.shape[0],ys.max()+31)
        crop=image[y0:y1,x0:x1].astype(float);sel=mask[y0:y1,x0:x1];crop[sel]=.72*crop[sel]+.28*np.array([0,210,190])
        picture=Image.fromarray(crop.astype(np.uint8));picture.thumbnail((440,260));canvas=Image.new('RGB',(440,260),'#eef3f6');canvas.paste(picture,((440-picture.width)//2,(260-picture.height)//2));ax.imshow(canvas);ax.axis('off')
        names=r['methods'];ax.set_title(f"Object {r['object_id']} / frame {rec['frame_id']}\nLearned: {names['learned']['name']} | Candidate: {names['shared_context']['name']}\nMasked: {names['masked']['name']} | Box: {names['bbox']['name']}",fontsize=9)
        captions.append(dict(object_id=r['object_id'],frame_id=rec['frame_id'],observation_id=oid,rgb=raw['rgb'],mask_index=j,
            selection='all_changed' if r in changed else 'first_four_unchanged_car_controls'))
    for ax in axes.flat[len(chosen):]:ax.axis('off')
    fig.suptitle('rosbag2245: all candidate class changes + four unchanged car controls');fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(REPORT/'robot_changes.png',dpi=140);plt.close(fig)
    dump(REPORT/'robot_captions.json',captions)
    public='';publicmd='';robothtml='';robotmd=''
    for method in METHODS:
        ap=result['outdoor_pooled'][method]['allAp50%']*100
        pq=result['results']['scenenn_014']['methods'][method]['metrics']['observed_voxel_instance_pq']*100
        public+=f'<tr><td>{labels[method]}</td><td>{ap:.3f}</td><td>{pq:.3f}</td></tr>'
        publicmd+=f'| {labels[method]} | {ap:.3f} | {pq:.3f} |\n'
        rr=result['results']['rosbag2245']['methods'][method]
        robothtml+=f"<tr><td>{labels[method]}</td><td>{rr['foreground_objects']}</td><td>{rr['foreground_points']}</td><td>{len(rr['class_changes'])}</td></tr>"
        robotmd+=f"| {labels[method]} | {rr['foreground_objects']} | {rr['foreground_points']} | {len(rr['class_changes'])} |\n"
    report=f'''# 上下文残差对象记忆：冻结开发实验

**结论：机制已实现、机器人缓存对照与四个公开开发场景已跑完；共享自适应系数没有超过固定系数基线，未通过预设继续门槛。该分支封存，不升级为 Ours、不追加系数搜索。**

本轮是一个有边界的新机制原型验证，不代表主线目标全部完成。

## 机制和对照

保留不可变观测的原 learned 特征与全图上下文。用对象内中心化的多视角变化估计共享系数 beta，每个对象等权，再计算对象均值减去 beta 倍上下文均值。新观测可更新共享系数，因此能改变没有新观测的历史对象表示。beta 限制在 [0,1]；无法估计时为 0；退化残差回退原均值。普通固定效应回归本身不构成已证明的新颖性。

对象身份、观测归属、三维点、类别词表和原有逐帧选样规则全部固定。历史回放以最终预测归属为条件，可能包含未来的关联信息，**不是因果在线 SLAM、误关联撤销或几何修订**。此次检验的是语义证据修订。

## 公开开发集

| 方法 | 三个 KITTI 窗口合并 AP50 × 100 | SceneNN 014 PQ × 100 |
|---|---:|---:|
{publicmd}

室外窗口包括原开发 0000 和已暴露的 0002/0003；与此前两个窗口的 5.56/24.07 不能直接混比。所有场景已经参与开发，不是独立测试或官方榜单。

预设门槛：候选在室外 AP50 或室内 PQ 至少超过最强简单对照 1 个百分点，另一域退步不超过 0.5 点。实际相对最佳简单对照：室外差值 0、室内差值 −0.180 点。不能将相对失配的 learned 基线的大幅提升归因于自适应记忆。

## 机器人对照

真实 rosbag2245 的 330 帧缓存、108 个最终对象，六组语义地图已导出。几何逐字节相同，身份和观测归属不变，比较的是类别及前景读出。

| 方法 | 前景对象数 | 前景点数 | 相对原分类改变的对象数 |
|---|---:|---:|---:|
{robotmd}

候选最终改变 4 个对象类别，其中 3 个原 car 变为 wall/pillar。不能无标注判定改对或改错。回放中还有 5 次未收到新观测的历史对象类别变化，均发生于对象 62 的 wall/door 往返；这是不稳定性证据，不能当作纠错成功。masked-only 增加前景对象也不能自动解释为准确率提高。

内存回放和日志导出单次用时约 2.45 秒，不含图像神经推理、LIVO、载入以外的完整系统工作，不代表在线系统帧率。独立语义标注继续按用户要求暂缓。

## 核验及评测更正

六项模块测试通过：固定效应分离、历史修改与精确撤回、同帧去重与输入隔离、不可辨识上下文回退、对象顺序不变性、非法及退化输入。

先封存五场景全部六组预测，再评分。初版评分把 KITTI 的“完整地图最近点后筛标签”错误复用到 SceneNN；已保留原结果和更正记录，使用 SceneNN 原协议“筛前景点后最近邻”重算六组。更正后 learned 完整复现原 SceneNN PQ=18.001492。预测、原冻结代码及参数未改。**权威结果是 corrected_summary.json，原 summary.json 不用于结论。**

导出 30 份分类地图，均检查几何 SHA-256 与原地图相同、对象 ID 和观测成员守恒。正式输出路径：`build/benchmarks/context_residual_memory_v1/<scene>/exported_maps/<method>/`。

## 主线下一缺口

当前最强简单对照已经明确；仅做共享上下文系数不足以形成有效贡献。下一机制若继续历史归属方向，必须真正改变受影响旧观测的归属，并与只允许合并、同支持最终处理对照比较。不得把本轮语义改类包装成历史对象结构修订。新机制的独立泛化与对应机器人有效性仍未完成；本目标保持进行中。

[原始更正后结果](corrected_summary.json) · [核验及导出清单](verification.json)。
'''
    (REPORT/'report.md').write_text(report);shutil.copyfile(OUT/'corrected_summary.json',REPORT/'corrected_summary.json')
    (REPORT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>上下文残差对象记忆</title><style>body{max-width:1180px;margin:38px auto;padding:0 24px;font:16px/1.7 system-ui;color:#243647}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.note{padding:18px;background:#edf4f6}a{color:#07877e}</style><h1>上下文残差对象记忆：开发验证与机器人对照</h1><p class="note">候选机制、六组对照和真实330帧车库缓存回放已完成。自适应系数未超过固定系数基线，未达到继续门槛；封存分支，不升级为 Ours。仅修订语义，几何与身份固定。</p><h2>公开开发集</h2><table><tr><th>方法</th><th>三个室外窗口 AP50 × 100</th><th>SceneNN 014 PQ × 100</th></tr>'''+public+'''</table><p>均为已暴露开发数据，不是独立或官方榜单成绩。SceneNN 已按原前景转移协议更正，并复现原基线。</p><img src="context_history.png"><h2>机器人对照</h2><p>108 个固定对象，330 个缓存帧；前景计数和类别变化不是准确率。</p><table><tr><th>方法</th><th>前景对象</th><th>前景点</th><th>改类对象</th></tr>'''+robothtml+'''</table><img src="robot_changes.png"><p>上图展示全部候选最终改类对象及四个固定未改类车辆对照。图片为原始 RGB 与预测掩码，不是人工参考标注。历史对象62发生五次 wall/door 切换，不能视为纠错成功。</p><p><a href="report.md">机制、边界、失败门槛与后续主线</a> · <a href="corrected_summary.json">更正后权威结果</a> · <a href="verification.json">30份语义地图导出核验</a></p></html>''')
    dump(REPORT/'verification.json',dict(complete=True,source_sha256=sha(__file__),
        sealed_files_verified=len(seal['files']),exported_maps=exports,
        original_geometry_and_memberships_preserved=True,indoor_baseline_reproduced=True,
        authoritative_summary_sha256=sha(OUT/'corrected_summary.json'),report_sha256=sha(REPORT/'index.html')))
    print(REPORT/'index.html',flush=True)


if __name__=='__main__':main()
