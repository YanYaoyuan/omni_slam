"""Combine retention and semantic-channel diagnostics without changing maps."""
from pathlib import Path
import html
import numpy as np
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from evaluate_kitti360_development import clean_json
from run_kitti360_reserved import OUT,META
from audit_kitti360_object_retention import DEST as RET, STAGES
from audit_kitti360_semantic_channels import DEST as SEM, REPORT, CHANNELS
from omni_semantic_map.kitti360_evaluation import kernel


def rename(value,scene):
    if isinstance(value,list):return [rename(v,scene) for v in value]
    if isinstance(value,dict):return {k:(scene+'/'+v if k=='filename' else rename(v,scene)) for k,v in value.items()}
    return value


def matched_ids(m):
    return {int(g['instance_id']) for rows in m['prediction'].values() for p in rows for g in p['matchedGt']
        if g['vert_count']>=100 and g['intersection']/(p['vert_count']+g['vert_count']-g['intersection'])>.5}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    retention=read(RET/'summary.json');semantic=read(SEM/'summary.json')
    assert retention['complete'] and semantic['complete']
    for directory in [RET,SEM]:
        p=read(directory/'protocol.json'); script='audit_kitti360_object_retention.py' if directory==RET else 'audit_kitti360_semantic_channels.py'
        assert sha(ROOT/'scripts/benchmarks'/script)==p['source_sha256']
        assert read(directory/'summary.json')['protocol_sha256']==sha(directory/'protocol.json')
    seal=read(SEM/'readouts_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h
    args,ns=kernel(META/'evalInstanceLevelSemanticLabeling.py');pooled={};changes={}
    for name in CHANNELS:
        matches={}
        for s in semantic['scenes']:
            scene=s['scene'];entry=read(SEM/scene/(name+'_matches.json'))['window']
            matches[scene]=rename(entry,scene)
            old=matched_ids(read(SEM/scene/'learned_matches.json')['window']);new=matched_ids(entry)
            changes.setdefault(scene,{})[name]=dict(added=sorted(new-old),lost=sorted(old-new),matched=len(new))
            current=clean_json(ns['computeAverages'](ns['evaluateMatches']({'window':entry},args),args))
            assert current==s['scores'][name]['averages']
        filenames=[r['filename'] for m in matches.values() for rows in m['prediction'].values() for r in rows]
        assert len(filenames)==len(set(filenames))
        pooled[name]=clean_json(ns['computeAverages'](ns['evaluateMatches'](matches,args),args))
        dump(SEM/'pooled_matches'/(name+'.json'),matches)
    assert pooled['learned']==read(OUT/'summary.json')['pooled']['history2']['full_window']
    dump(REPORT/'pooled.json',dict(pooled=pooled,matching_changes=changes,posthoc=True,
        verification=dict(source_sha256=sha(__file__),per_window_ap_recomputations=10,pooled_ap_recomputations=5,
            readout_seals_verified=True,original_baseline_reproduced=True)))
    labels={'learned':'原 learned 融合','global':'仅全图特征','masked':'仅掩码裁剪特征','bbox':'仅框裁剪特征','uniform_triplets':'三通道等权融合'}
    table='\n'.join(f"| {labels[n]} | {pooled[n]['allAp50%']*100:.3f} | {pooled[n]['allAp']*100:.3f} | {pooled[n]['classes']['car']['ap50%']*100:.3f} |" for n in CHANNELS)
    stage_labels=['全部测量点','原始掩码支持','连通分量支持','至少三帧历史组','最终对象观测','建筑/车辆类别输出']
    counts={c:[sum(s['totals'][c]['coverage_over_half'][stage] for s in retention['scenes']) for stage in STAGES] for c in ['building','car']}
    stages='\n'.join(f'| {label} | {counts["building"][i]} / 22 | {counts["car"][i]} / 27 |' for i,label in enumerate(stage_labels))
    image_html=''
    for s in semantic['scenes']:
        scene=s['scene'];captions=read(REPORT/(scene+'_captions.json'));root=OUT/scene
        m=read(root/'connected/cache/manifest.json');records={r['frame_id']:r for r in m['records']}
        fig,axes=plt.subplots((len(captions)+3)//4,4,figsize=(16,3.5*((len(captions)+3)//4)),squeeze=False)
        byid={r['object_id']:r for r in s['rows']}
        for ax,caption in zip(axes.flat,captions):
            image=np.array(Image.open(caption['rgb']).convert('RGB'));rec=records[caption['frame_id']]
            f=np.load(root/'connected/cache/frames'/rec['file']);mask=np.unpackbits(f['masks_packed'][caption['mask_index']],axis=1)[:,:image.shape[1]].astype(bool)
            y,x=np.where(mask);y0=max(0,y.min()-30);y1=min(image.shape[0],y.max()+31);x0=max(0,x.min()-30);x1=min(image.shape[1],x.max()+31)
            crop=image[y0:y1,x0:x1].astype(float);region=mask[y0:y1,x0:x1];crop[region]=.72*crop[region]+.28*np.array([0,220,200])
            picture=Image.fromarray(crop.astype(np.uint8));picture.thumbnail((440,260));canvas=Image.new('RGB',(440,260),'#f1f4f6')
            canvas.paste(picture,((440-picture.width)//2,(260-picture.height)//2));ax.imshow(canvas);ax.axis('off')
            c=byid[caption['object_id']]['readouts']
            ax.set_title(f"Object {caption['object_id']} / frame {caption['frame_id']}\nLearned: {c['learned']['name']} | Global: {c['global']['name']}\nMasked: {c['masked']['name']} | Box: {c['bbox']['name']}",fontsize=9)
        for ax in axes.flat[len(captions):]:ax.axis('off')
        fig.suptitle(scene+' | ALL final objects',fontsize=14)
        fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(REPORT/(scene+'_all_objects.png'),dpi=130);plt.close(fig)
        image_html+=f'<h2>{html.escape(scene)}：全部最终对象</h2><p>每个对象固定选择已用观测中面积最大的掩码；类别文字来自多帧对象读出，非单帧标签。青色为实际预测掩码。</p><img src="{scene}_all_objects.png">'
    report=f'''# 室外对象保留与语义证据诊断

**主线：对象级语义建图。此次定位到的是可恢复的对象证据丢失，而非新的删点阈值。**

## 固定几何的读出对照

同一最终几何、对象身份、选定观测、19 类词表；只替换已缓存的语义读出通道。所有对象、所有通道都报告，无新模型推理和调参。五组读出先封存，再评分。

| 读出方式 | 合并 AP50 × 100 | AP50:90 × 100 | 车辆 AP50 × 100 |
|---|---:|---:|---:|
{table}

建筑 AP50 均为 0。合并成绩由带场景前缀的匹配记录重算，不是窗口分数平均。masked 相对 learned 在 0002 新增 {len(changes[semantic['scenes'][0]['scene']]['masked']['added'])} 个正确车辆匹配，丢失 {len(changes[semantic['scenes'][0]['scene']]['masked']['lost'])} 个；0003 的原有车辆匹配保持。

**这些是已查看数据上的事后机制对照，不是独立验证、官方榜单或新方法收益。** 更换单个特征通道属于普通基线措施，不能命名为新的 Ours。原有完整建图结果保持冻结。本实验未将新通道用于关联，不能将读出成绩冒充重新建图成绩。

## 对象在哪一阶段失去足够支持

计数条件：在固定 20 cm 转移范围内，该阶段源点支持能够覆盖真值对象超过 50% 的置信度质量。它授予完美逐点身份和背景拒绝，是几何覆盖的必要条件诊断，**不是可实现的召回率、AP 或正确预测数**。

| 支持阶段 | 建筑 | 车辆 |
|---|---:|---:|
{stages}

车辆的逐级首次损失：2 辆在原始测量覆盖阶段不足，3 辆在掩码支持阶段、2 辆在连通分量阶段、6 辆在至少三帧历史组阶段、11 辆在类别输出阶段跌破覆盖门槛。后三类不能笼统归因于相同模块。至少三帧阶段按 history2 合并后的组计算，不能用融合前节点代替；初版诊断的这一断言失败及日志已留存，修正的是审计逻辑，未修改预测。

建筑有 17/22 在全部测量点域已经覆盖不足。这受固定采样、局部观测范围及完整静态参考对象范围影响；不能把全部建筑失分归咎于语义分类。其余 5 个中，4 个在掩码支持阶段不足，1 个在跨帧保留阶段不足。

## 可检验的机制缺口

0002 有 11 个车辆真值在最终观测支持域仍超过半数覆盖，却被类别筛选丢弃。其中 9 个关联对象的 masked 和 bbox 通道均判为 car，learned 却判为 sidewalk/road；另一个 masked 为 car、learned 为 bus。读出干预恢复 10 个匹配，说明这部分丢失不需要增加几何或重新检测即可恢复。

当前证据支持“对象局部特征包含的信息在融合后被压制”。全图通道多为 sidewalk/road，符合场景上下文干扰的解释，但尚未直接测量执行时的融合权重，不能宣称已证明具体权重故障。

下一机制应保留分开的对象局部证据与场景上下文，允许后续视角修订对象判断；若做历史归属修订，应在该更强语义基线上单独验证，避免把类别恢复算成关联贡献。必须与 masked-only、bbox-only、普通等权融合比较，并证明超出简单通道替换的收益。此报告没有发布或声称这一新机制已完成。

## 三项交付状态

- 新机制：旧可见性修订、历史拆分和未训练对象记忆不能因这次基线涨点被宣称有效；仍需完整实现及有限预算对照。
- 机器人对照：已有 rosbag2245 的 330 帧基线/SOR/旧 Ours 重放；独立语义准确率标注按用户要求暂缓。新机制的机器人对照仍未完成。
- 新增评测：Hypersim 与 KITTI-360 有实际结果；本次两个室外窗口已暴露，后续不能重复称为独立验证。

数据：`build/benchmarks/kitti360_retention_audit_v1/`、`build/benchmarks/kitti360_semantic_channel_audit_v1/`。逐对象证据与实际图片见同目录 HTML；[合并分数与匹配变化](pooled.json)。GT 派生覆盖诊断与无 GT 通道预测分开保存。
'''
    (REPORT/'report.md').write_text(report)
    # Keep HTML self-contained as readable prose and explicit score/coverage tables.
    rows=''.join(f"<tr><td>{labels[n]}</td><td>{pooled[n]['allAp50%']*100:.3f}</td><td>{pooled[n]['allAp']*100:.3f}</td><td>{pooled[n]['classes']['car']['ap50%']*100:.3f}</td></tr>" for n in CHANNELS)
    coverage=''.join(f'<tr><td>{label}</td><td>{counts["building"][i]} / 22</td><td>{counts["car"][i]} / 27</td></tr>' for i,label in enumerate(stage_labels))
    (REPORT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>omni_slam 对象证据诊断</title><style>body{max-width:1250px;margin:36px auto;padding:0 24px;font:16px/1.7 system-ui;color:#203448}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.note{padding:18px;background:#edf6f5}a{color:#08796e}</style><h1>omni_slam：对象还在，语义证据在哪丢了？</h1><p class="note">同一几何与对象身份，仅使用已有掩码裁剪特征读出，合并 AP50 从 5.556 升至 24.074，恢复 10 个车辆匹配。属于事后强基线诊断，不是新 Ours、独立验证或官方榜单成绩。</p><table><tr><th>读出方式</th><th>AP50 × 100</th><th>AP50:90 × 100</th><th>车辆 AP50 × 100</th></tr>'''+rows+'''</table><h2>逐阶段覆盖诊断</h2><p>仅统计支持能覆盖真值超过 50% 置信度质量的对象数；不是实际正确预测数。建筑在原始测量阶段就有 17/22 覆盖不足。</p><table><tr><th>阶段</th><th>建筑</th><th>车辆</th></tr>'''+coverage+'''</table><p>不能再将缺失都归因于删点或关联。语义读出是本轮找到的具体可恢复损失；后续新机制须超过 masked-only 等强对照。<a href="report.md">详细结论与主线交付状态</a> · <a href="pooled.json">匹配变化及重算记录</a></p>'''+image_html+'</html>')
    print('POOLED', {k:round(v['allAp50%']*100,3) for k,v in pooled.items()},flush=True)


if __name__=='__main__':main()
