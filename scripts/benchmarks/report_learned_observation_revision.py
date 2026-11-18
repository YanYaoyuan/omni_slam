"""Verify and report trained whole-observation revision, including robot maps."""
from pathlib import Path
import shutil
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_learned_observation_revision import OUT,METHODS,verify
from omni_semantic_map.learned_observation_revision import rollback

REPORT=ROOT/'output/learned_observation_revision_v1'
NAMES=dict(native_replay='同源点原归属回放',merge_only='学习边：只允许合并',reassign_history='候选：历史观测归属修订')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=verify(True);r=read(OUT/'summary.json');seal=read(OUT/'predictions_complete.json');training=read(OUT/'trained.json')
    assert r['complete'] and sha(OUT/'predictions_complete.json')==r['prediction_seal_sha256']
    for f,h in seal['files'].items():assert sha(f)==h,f
    assert sha(OUT/'pair_model.joblib')==training['model_sha256']
    REPORT.mkdir(parents=True,exist_ok=True);checks={};table='';md='';detail=''
    for scene in p['scenes']:
        if scene['role']=='train':continue
        key=scene['key'];a=np.load(OUT/key/'observations.npz');rows=[];changes={}
        original=a['native'];same_before=original[:,None]==original[None,:]
        for method in METHODS:
            folder=OUT/key/method;labels=np.load(folder/'observation_identity.npy');g=np.load(folder/'geometry.npz');events=read(folder/'events.json')
            np.testing.assert_array_equal(g['xyz'],a['source_xyz']);np.testing.assert_array_equal(g['source_indices'],a['active_source_indices'])
            np.testing.assert_array_equal(g['instance'],labels[a['source_owner']]);assert np.all(g['instance']>0)
            np.testing.assert_array_equal(rollback(labels,events),original)
            same_after=labels[:,None]==labels[None,:]
            removed=int(np.triu(same_before&~same_after,1).sum());added=int(np.triu(~same_before&same_after,1).sum())
            if method=='merge_only':assert removed==0
            run=read(folder/'run.json');changes[method]=dict(objects=len(np.unique(labels)),removed_pair_links=removed,added_pair_links=added,
                reassign_events=sum(e['kind']=='reassign' for e in events),merge_events=sum(e['kind']=='merge' for e in events),
                converged=run['converged'],wall_seconds=run.get('wall_seconds'),exact_source_ownership_render=True,rollback_verified=True)
            stats=r['results'][key]['methods'][method]
            if scene['kind']=='indoor':
                score=f"{stats['observed_voxel_instance_pq']*100:.3f}";extra=f"{stats['tp']} / {stats['fp']} / {stats['fn']}"
            elif scene['kind']=='outdoor':score=f"{stats['allAp50%']*100:.3f}";extra='AP50 × 100'
            else:score='无独立准确率';extra=f"{run['objects']} 个对象"
            table+=f'<tr><td>{key}</td><td>{NAMES[method]}</td><td>{score}</td><td>{extra}</td></tr>'
            md+=f'| {key} | {NAMES[method]} | {score} | {extra} |\n'
        checks[key]=changes
        fig,axes=plt.subplots(1,3,figsize=(14,4),sharex=True,sharey=True)
        src=a['source_xyz'];sample=np.linspace(0,len(src)-1,min(40000,len(src)),dtype=int)
        for ax,method in zip(axes,METHODS):
            labels=np.load(OUT/key/method/'observation_identity.npy')[a['source_owner']]
            ax.scatter(src[sample,0],src[sample,1],c=labels[sample]%20,cmap='tab20',s=.5,rasterized=True)
            ax.set_aspect('equal');ax.set_title(method+f" / {changes[method]['objects']} objects",fontsize=10);ax.set_xlabel('x / m')
        axes[0].set_ylabel('y / m');fig.suptitle(key+' — identical raw source points, revised object IDs');fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(REPORT/(key+'_maps.png'),dpi=145);plt.close(fig)
        detail+=f'<h2>{key}</h2><img src="{key}_maps.png"><p>同一物理源点支持；颜色只表示各图内部实例编号，不代表正确性。归属修订减少 {changes["reassign_history"]["removed_pair_links"]} 个旧共属观测对、增加 {changes["reassign_history"]["added_pair_links"]} 个共属对。是否改善由上表匹配指标判断。</p>'
    dump(REPORT/'verification.json',dict(complete=True,source_sha256=sha(__file__),sealed_files_verified=len(seal['files']),
        source_conservation_and_rollback=checks,summary_sha256=sha(OUT/'summary.json'),model_sha256=training['model_sha256']))
    for name in ['protocol.json','summary.json','trained.json']:shutil.copyfile(OUT/name,REPORT/name)
    verdict='通过冻结开发门槛，仍须独立评测和机器人准确率验证' if r['advance_gate'] else '未通过冻结开发门槛；封存该分支，不继续参数搜索或自动扩大评测'
    trainingrows='\n'.join(f"| {s['scene']} | {s['pairs']} | {s['positives']} |" for s in training['training_scenes'])
    edge=read(OUT/'posthoc_edge_audit.json');assert edge['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    assert sha(ROOT/'scripts/benchmarks/audit_learned_revision_edges.py')==edge['source_sha256']
    shutil.copyfile(OUT/'posthoc_edge_audit.json',REPORT/'posthoc_edge_audit.json')
    report=f'''# 学习观测对证据的历史归属修订

**{verdict}。** 本次确实改变旧观测的对象归属，并将其渲染为完整同支持地图；不同于上轮仅改变类别。整体研究目标尚未因此自动完成。

## 固定机制

用四个指定场景训练一次梯度提升树，从原始支持重叠、相对几何、SigLIP 余弦和帧关系预测同一对象概率。训练场景与三个公开开发场景的 ID 不重叠。模型参数先冻结，无阈值或模型搜索。未把此前 GT oracle 分配作为推理输入。

从原生对象归属开始，以预测概率的有符号边为证据。只合并控制与候选共享相同目标函数、数据、身份复杂度代价及源点表达；候选额外允许整观测移至其他或新对象。每步记录旧/新归属，逐场景验证逆序撤回可精确复原。算法是近似优化，不保证全局最优或概率校准。

## 训练

| 场景 | 训练观测对 | 正对 |
|---|---:|---:|
{trainingrows}

模型拟合一次，耗时 {training['wall_seconds']:.2f} 秒。训练集 AUC {training['training_only_auc']:.4f}，只描述拟合，不能作为泛化证据。GT 仅用于这四个场景的监督标签；开发场景评分在所有方法和机器人预测封存后执行。

## 成绩

| 场景 | 方法 | PQ/AP50 × 100 | TP / FP / FN 或说明 |
|---|---|---:|---|
{md}

室内 PQ 使用与之前关联诊断一致的原始观测支持回放。两个 native replay 成绩与历史诊断逐项复现。不能将这些分数直接减去经过 CG 去噪/下采样的另一套地图分数。室外使用固定系数语义对照保留的前景观测，其他类别的阻挡点保持固定；对象身份修订不重新分类。

office_0 的 merge-only 提升来自假阳性实例从 21 降至 13，正确匹配仍为 12，没有新增正确匹配。历史重分配在 office_0 丢失 2 个原匹配；SceneNN 新增 1 个、丢失 7 个。不能只报告新增的那一个。

## 已有产物的机制复核

仅在可标注且纯度至少 0.8 的观测及已有候选边上复核，混杂/覆盖不足观测被排除，不能外推为全部边的性能。训练 AUC 为 0.945，两个开发场景 AUC 为 0.850 / 0.825，存在明显泛化差距。

office_0 修订后观测对分区准确率从 81.21% 升至 84.72%，但地图 PQ 反而从 35.61 降至 33.10。这直接说明观测对目标改善不能替代完整对象质量。SceneNN 的同对象观测对召回从 86.16% 降至 67.95%，与丢失正确对象匹配相伴。

四场景优化均已按固定预算收敛，目标函数均上升；退步不能简单解释为迭代没跑完。后续如果设计新版本，应将训练和验收对准对象级操作的实际得失，而不是在这版边阈值、身份代价上搜索。此复核未追加建图、训练或参数改变。[完整边诊断](posthoc_edge_audit.json)。

## 原始证据与边界

- 物理点归属在实验前固定为最早帧、再最小原观测 ID；三个分区具有完全相同的源点和参考域前景覆盖，没有删点或增加覆盖来冒充身份收益。
- 观测作为不可分割单元，只改其对象 ID，不能切开混杂掩码。此前已被移除的对象和缺少掩码的区域不在本模块能力内。
- 使用完整保留序列进行离线历史修订，非因果在线 SLAM。已有最终预测分组用于初始化，不是 GT 初始化。
- 室内三个方法使用相同前景支持；物体类别在几何实例主评测中不参与重新筛选。室外与机器人只对原预测类别相同的对象建立有效修订边。
- 机器人使用 rosbag2245 的 330 帧缓存，图中展示真实测量支持。没有独立实例标注，不报告机器人准确率涨点。
- 三个公开开发场景此前已暴露，训练场景不重叠不等于首次独立验证。学习边、可撤销事件和简单图优化本身也不自动证明论文创新性。

## 冻结验收与主线状态

要求候选在两个室内场景各自至少超越 native replay 与 merge-only 2 个 PQ 点、不降低 TP，且室外 AP50 不低于 merge-only。真实结果和判定见 `summary.json`，不事后改变门槛。

本轮交付是归属修订机制、12 份分区地图、训练模型、源点守恒和精确回滚核验、机器人对照及三个公开开发集评分。新机制独立泛化与机器人准确率验证仍不能由这些产物替代，主线目标保持进行中。

可读取地图：`build/benchmarks/learned_observation_revision_v1/<scene>/<method>/geometry.npz`。对应 `observation_identity.npy`、`classification.json`、`events.json` 提供身份及变化追踪。

[核验](verification.json) · [冻结协议](protocol.json) · [完整指标](summary.json) · [训练记录](trained.json) · [事后边诊断](posthoc_edge_audit.json)
'''
    (REPORT/'report.md').write_text(report)
    (REPORT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>历史观测归属修订</title><style>body{max-width:1200px;margin:38px auto;padding:0 24px;font:16px/1.7 system-ui;color:#243647}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}.note{padding:18px;background:#edf4f6}a{color:#07877e}</style><h1>历史观测归属修订：学习证据与同支持对照</h1><p class="note">'''+verdict+'''。只合并与修订共享学习边、源点和后处理；候选确实可重新分配历史观测，所有改动可精确撤回。</p><table><tr><th>场景</th><th>方法</th><th>PQ/AP50 × 100</th><th>TP/FP/FN 或说明</th></tr>'''+table+'''</table><p>均为已暴露开发数据，不是独立或官方榜单。室内为原始同支持回放；不能与经过另一种最终处理的原生地图直接相减。机器人无独立准确率参考。</p>'''+detail+'''<p><a href="report.md">完整机制、训练、验收与边界</a> · <a href="summary.json">指标</a> · <a href="verification.json">源点守恒和回滚核验</a></p></html>''')
    checks_doc=read(REPORT/'verification.json');checks_doc['posthoc_edge_audit_sha256']=sha(OUT/'posthoc_edge_audit.json')
    dump(REPORT/'verification.json',checks_doc)
    print(REPORT/'index.html',flush=True)


if __name__=='__main__':main()
