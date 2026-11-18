"""Report causal identity, unregistered measured surfaces and DINO diagnostics."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_evidence_retention import ROOT,OUT,REPORT


def main():
    root=OUT/'identity_geometry_v1';dest=REPORT/'identity_geometry_v1'
    geom=json.loads((root/'geometry.json').read_text());pairs=json.loads((root/'dino_pairs.json').read_text())
    features=json.loads((OUT/'dino_livo_tokens_v2/manifest.json').read_text())
    rows=[o for o in geom['objects'] if o['comparisons']]
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for ax,row in zip(axes,rows):
        clouds=np.load(root/f'object_{row["alias"]}_visits.npz')
        for i,key in enumerate(clouds.files):
            xyz=clouds[key];ax.scatter(xyz[:,0],xyz[:,2],s=9,c=['#147eac','#dc6437'][i],alpha=.65,label=f'Visit {i+1}: {len(xyz)} points')
        ax.set(xlabel='World X (m)',ylabel='World Z (m)',title=f'Object {row["alias"]} | median NN {row["comparisons"][0]["symmetric_nn_median_m"]:.3f} m')
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Measured support from separate visits, in unchanged LIVO coordinates\nNo registration or pose correction; partial surfaces and visibility affect distances',fontsize=12)
    fig.savefig(dest/'visit_geometry.png',dpi=170);fig.savefig(dest/'visit_geometry.pdf');plt.close(fig)
    text='''# omni_slam：因果身份、跨视角几何与 DINO 首轮诊断

2026-11-11。固定上一轮“连通支持＋关闭 DBSCAN＋20cm 匹配距离”工程候选，未修改原始点云、SAM2、SigLIP 读出或任何旧运行。

**确认返回车辆在因果关联阶段就恢复了旧身份；同时量化了跨次观测表面的不一致。冻结 DINOv2-small 已处理完整 330 个关键帧，但尚未加入关联打分，也没有产生新的地图成绩。**

## 身份恢复发生在哪里

独立重放时，在每个原生 CG integrate 调用后记录因果标签；最终几何和分类与固定候选精确一致。观测 3075 在第 327 帧首次出现时，已被关联到 alias 6，直到第 332 帧仍保持该归属。因此，这个案例的起点/返回连接不是只靠最终离线合并。

这是因果算法顺序的证据，不是实时 ROS 在线部署。图像支持两次观测对应同一黑车，但仍没有独立人工身份标注，不能据此报告全场景 IDF1 或 ID-switch 准确率。

## 跨次观测的几何并未完全对齐

对全部 41 个前景输出，按成员观测时间间隔超过 30 秒划分访问段；仅有 3 个多访问对象。使用每次访问的原始实测支持点，在共同 2.5cm 体素上去重，计算双向最近表面距离。没有 ICP、平移拟合或局部修图。

| 对象 | 各访问段帧数 | 段内相邻观测距离中位数 | 跨访问表面距离中位数 | 跨访问 P90 |
|---|---|---:|---:|---:|
'''
    for row in rows:
        within=np.median([x['symmetric_nn_median_m'] for x in row['adjacent_comparisons'] if x['valid']]);cross=row['comparisons'][0]
        text+=f'| {row["alias"]} | {" / ".join(str(len(v)) for v in row["visits"])} | {within*100:.2f}cm | {cross["symmetric_nn_median_m"]*100:.2f}cm | {cross["symmetric_nn_p90_m"]*100:.2f}cm |\n'
    text+='''
段内列为各相邻帧对中位距离的中位数；跨访问列为两段池化点的距离，两个统计的采样粒度不同。距离受可见表面、遮挡、采样和可能的关联错误影响，**不是 GT 定位误差、全车尺寸精度或独立漂移估计**。

![三组未配准的多访问实测表面](../reports/evidence_retention_2026-11-11/identity_geometry_v1/visit_geometry.png)

黑车跨访问距离中位数约 24.5cm，最近表面匹配的 Z 方向位移中位数约 18.9cm。该位移不是刚体配准结果，不能直接用于修正轨迹。对象 495 的两个方向 20cm 重叠率约 34.8% 与 94.7%，说明部分可见表面导致强烈不对称；不能把所有大距离都归因于定位漂移。

## DINO 已有实际特征，但尚无地图收益结论

冻结 DINOv2-small 使用本地原权重，640×360 图像按比例缩放到 448×252，不裁剪，得到 18×32 patch tokens；按图像时刻变换将实测支持投影到 token，保存支持加权描述子、最多 24 个局部三维锚点和局部特征。没有训练或修改 SigLIP 分类。

''' + f'''完整缓存为 **{features['frames']} 帧、{features['observations']} 个掩码观测槽位**；没有有效支持的槽位保留零描述子，不当作有效特征。提取循环耗时约 {features['wall_s']:.1f}s，峰值 CUDA 分配约 {features['gpu_peak_allocated_bytes']/1024**3:.2f}GiB；不含模型加载、GPU 等待或其他模型推理，不作为系统端到端时延。

''' + '''本轮预先固定六组观测对，再读取特征分数：三个重访同一对象假设，以及三组同帧可见的不同车辆。后两组重访身份仍待独立核实，不能当作可靠正例训练标签。

| 观测对（ID+1） | 证据类型 | SigLIP 余弦 | DINO 支持池化余弦 | DINO 外观互近邻局部余弦 |
|---|---|---:|---:|---:|
'''
    for row in pairs['rows']:
        m=row['metrics'];kind='同车图像支持' if row.get('name')=='start_return_black_car' else '重访身份待核实' if row['group']=='positive_hypotheses' else '同帧不同车辆'
        local=m['mutual_appearance_cosine'];local=f'{local:.3f}' if local is not None else '不足3对'
        text+=f'| {row["a"]+1} / {row["b"]+1} | {kind} | {m["siglip_cosine"]:.3f} | {m["dino_pooled_cosine"]:.3f} | {local} |\n'
    text+='''
黑车重访的 DINO 支持池化余弦为 0.960；三个不同车辆对为 0.684–0.775，而 SigLIP 对这些不同车辆也给出 0.898–0.937。**这提供了 DINO 区分同类实例的案例证据，但不同编码器分数不具有统一标定，六个例子不足以证明总体优势。** 另两个待核实重访的 DINO 分数为 0.652、0.757，与不同车辆范围重叠，因此不从这六组数据挑选硬阈值。

![黑车重访的固定对照](../reports/evidence_retention_2026-11-11/identity_geometry_v1/pair_6_3075.jpg)

![同帧两辆不同车的固定对照](../reports/evidence_retention_2026-11-11/identity_geometry_v1/pair_525_526.jpg)

### 为什么不能直接按三维最近点硬配 DINO token

黑车按外观互近邻找到 12 对局部 token，平均余弦为 0.882；按 20cm 三维互近邻只剩 4 对，平均余弦为 0.414。两者不是同一组匹配，也没有局部对应 GT，不能将差值写成匹配精度提升。它揭示：几何最近锚点未必来自同一视觉部位，几何误差、粗 patch 网格和稀疏支持都可能破坏这种假设。

对象 495 的三维互近邻在 20cm 与 40cm 下都只有 1 对，按协议返回“证据不足”，没有填充高置信度分数。三组不同车辆在这两个距离内没有锚点对应，几何本身已能排除它们，所以这六组不能证明 DINO 在空间歧义候选上的额外贡献。

## 下一步方法与实验约束

保留 SigLIP 负责类别语义，DINO 用于关联验证。接下来应在有空间歧义的候选对上比较：固定工程底座、DINO 支持池化特征、DINO 局部对应；分别记录误合并和碎片化。局部支持不足时必须保留“不确定”，不能强行否决原关联或伪造高置信度。

跨访问几何误差需要与身份判断分开处理：先检验外观对应的几何一致性，再考虑是否构造对象/关键帧修正约束。本轮没有用单辆车的最近点位移去修轨迹，也没有把局部表面对齐当作独立定位 GT。正式论文评测仍需独立对象身份标注与未参与开发的场景。

## 复现与验证

- 因果诊断重放精确复现冻结候选的最终几何与分类。
- 新增 3 项测试覆盖 16:9 图像 token 映射、图像时刻变换/世界锚点、边界/背后点/空支持；全部通过。
- DINO 首次 `from_pretrained` 因本地 Transformers 缺失可选初始化函数失败，未处理图像。失败目录保留；v2 从本地配置实例化并严格加载原 safetensors，全部权重匹配，没有改变环境或训练。
- 完整结果保留全部 41 对象的段内距离、3 个重访对象及六组特征对照；未按分数丢弃案例。

[几何协议与结果](../reports/evidence_retention_2026-11-11/identity_geometry_v1/geometry.json) · [因果身份记录](../reports/evidence_retention_2026-11-11/identity_geometry_v1/causal_identity.json) · [DINO 六组结果](../reports/evidence_retention_2026-11-11/identity_geometry_v1/dino_pairs.json) · [配对选择协议](../reports/evidence_retention_2026-11-11/identity_geometry_v1/dino_pair_protocol.json) · [前一轮尺度对照](livo_scale_2026-11-11.md)

```bash
# 首次运行；所有实验目录独立保存，已有目录不覆盖。
OPENBLAS_NUM_THREADS=2 build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python scripts/benchmarks/audit_livo_identity_geometry.py
CUDA_VISIBLE_DEVICES=0 build/benchmarks/ovo_full_20261110/venv311/bin/python scripts/benchmarks/extract_livo_dino.py
OPENBLAS_NUM_THREADS=2 build/benchmarks/ovo_full_20261110/venv311/bin/python scripts/benchmarks/compare_livo_dino_pairs.py
PYTHONPATH=omni_semantic_map python3 -m pytest -q omni_semantic_map/test/test_livo_dino_support.py
```
'''
    path=ROOT/'docs/benchmarks/livo_identity_geometry_2026-11-11.md';path.write_text(text);print('WROTE',path)


if __name__=='__main__':main()
