"""Render stage-three evidence without replacing frozen historical results."""
import json
import numpy as np
from audit_evidence_retention import REPORT


def update(text):
    path=REPORT/'stage3/cvpr18_domain_summary.json'
    if not path.exists():return text
    d=json.loads(path.read_text());named=json.loads((REPORT/'stage3/named_domain_summary.json').read_text())
    start=text.index('## 阶段 3：');end=text.index('## 复现与记录')
    section='''## 阶段 3：新场景对比完成；真实 LIVO 全包验证进行中

三个冻结场景的强前端、CG 后端及两种读出均已执行，合计 1,200 个几何帧、600 个语义帧。为降低显存占用，SAM2、SigLIP 三分支特征和学习融合器分进程运行，保留原模型、BF16 运算与八掩码批次。旧 office_0 抽样两帧的掩码、特征和支持点完全一致；另用原 CG 运行环境重放完整 office_0，最终几何、成员关系与均值分类完全复现。这些检查分别证明对应范围内的适配一致性。

### 标注口径纠正与新场景结果

首轮评分错误地依据空 XML 名称决定前景范围。原始 014/015/021 分别有 178/247/140 个区域没有名称，其中包含未命名墙面等，不能全部当作已知前景对象。原始结果和协议保留，**8.46/8.44 的首轮宏平均不作为有效的前景实例结论**。

首先补做了“有名称区域”条件诊断，但有效域仅占共同观测体素的 22.6%、42.2%、86.4%，不能代替整场景评测。随后从[官方 CVPR18 发布包](https://hkust-vgd.github.io/scenenn/)按范围提取三个场景的六个 HDF5 文件，不下载完整 9GB 压缩包。[发布代码说明](https://github.com/hkust-vgd/pointwise)明确该数据按 NYU-D v2 的 40 类重新标注。

已验证三个场景的去重标注坐标全部与原实例网格顶点**精确重合**，重复采样的类别也完全一致，无需配准拟合。保留原实例 ID，以补充类别区分墙/地板/天花板、前景和 void；在统一度量坐标中以 5cm 限距转移补充标注。标签 0 和无标注范围作为 void，与背景区分；共同观测体素投票后再限制到有效域。

这是查看首轮分数后的**评测范围纠正**，不是事前注册的新协议；模型、预测几何、成员关系、类别词表和两组参数全部冻结未改。指标仍是本项目的共同观测体素类无关实例 PQ，不是官方排行榜 AP，也不能与 Replica 分数直接横比。CVPR18 采样未覆盖的高度/区域仍需报告为 void，不能声称整张地图的全对象召回。

| 场景 | 有效评价域 | GT 实例 | 均值 PQ×100 | 面积＋去冗余 | 差值 |
|---|---:|---:|---:|---:|---:|
'''
    for r in d['scenes']:
        domain=r['domain'];m=r['methods']['mean']['observed_voxel_instance_pq']*100;c=r['methods']['area_density']['observed_voxel_instance_pq']*100
        section+=f'| {r["scene"]} | {domain["valid_fraction"]:.1%} | {domain["gt_instances"]} | {m:.4f} | {c:.4f} | {c-m:+.4f} |\n'
    section+=f'| 宏平均 | — | — | {d["macro_pq"]["mean"]*100:.4f} | {d["macro_pq"]["area_density"]*100:.4f} | {d["paired_delta_pq100"]:+.4f} |\n'
    section+='''
**决策：保留不同帧均值强基线，不采用面积＋视角去冗余作为新主方法。** 候选在新场景两负一胜，开发集的 +0.1203 未复现为泛化优势。负结果及全部三种评价范围均保留，未按分数丢弃场景或调参。

- [补充标注评价协议](../reports/evidence_retention_2026-11-11/stage3/cvpr18_domain_protocol.json) · [完整分场景结果](../reports/evidence_retention_2026-11-11/stage3/cvpr18_domain_summary.json)
- [有名称区域条件诊断](../reports/evidence_retention_2026-11-11/stage3/named_domain_summary.json) · [首轮原始结果（前景范围有问题）](../reports/evidence_retention_2026-11-11/stage3/heldout_summary.json)
- [冻结坐标修正](../reports/evidence_retention_2026-11-11/stage3/coordinate_protocol.json)：仅由首个位姿构造 W，对输入、位姿和参考几何统一应用；未拟合 GT。抽样输入对齐距离 5.72/7.09/14.52mm 不作为定位误差。

### 实际 LIVO → 强语义前端 → 三维对象

已用 rosbag2245 的 30 秒前缀跑通原生 ROS LIVO、原子图像/点云/位姿、冻结 SAM2/SigLIP/融合器、CG 关联和均值读出。原始 BEST_EFFORT 回放仅采到两帧的失败记录保留；重试收到 16 个有效语义帧，对照记录到 17 个原生视觉关键帧，缺失首帧 ID0。后续采集加入发布者发现等待，不把传输缺帧隐去。

输入投影使用消息中的图像时刻变换 `T_camera_tracking`，相机世界位姿为 `T_world_tracking @ inverse(T_camera_tracking)`；只使用实测雷达点，以逐像素 z-buffer 去掉已测近表面后的点，不合成深度。图像、点云、标定、位姿和原始 CDR 全部留存。

部署词表固定追加车辆、行人、自行车、摩托车、卡车、消防栓、灭火器及室外背景词，原 benchmark 51 类文本向量保持不变；追加词只用于真实数据，不影响上表。两种读出使用同一词表。

原始强基线生成五个最终区域，过滤后保留 car 和 pipe。人工式图像核查由本助手完成，并非独立人工 GT：车辆证据图支持 car 判断；pipe 覆盖的是顶部长条风管/梁状结构，精确类别不确定。**原始车辆框约 9.40×8.71×4.40m，混入远处背景点，几何验收失败。** 不将“识别到车”包装为可靠整车定位。

![车辆原图、SAM 掩码与实测支持点](../reports/evidence_retention_2026-11-11/stage3/livo/object_6_car.jpg)

### 稀疏支持适配及代价

排查发现原 CG 的 0.1m DBSCAN 半径不适配本录包 0.2m 下采样的稀疏雷达点：没有形成密集簇时，原实现会保留全部点。新增**真实数据专用**支持检查：每个互斥掩码保留最大三维连通分量，连接半径固定为配置雷达下采样间距的两倍（0.4m），最少点数沿用 CG 的 16。该规则在运行适配结果前写入协议，不按车型或 GT 调整；公共数据的冻结 CG 配置不变。
'''
    lp=REPORT/'stage3/livo_supportcc/summary.json'
    if lp.exists():
        live=json.loads(lp.read_text());car=next(o for o in live['retained_objects'] if o['name']=='car');extent=np.array(car['bounds_max_m'])-car['bounds_min_m'];center=car['center_m']
        section+=f'\n支持点由 4,041 减至 1,802；最终保留车辆 ID {car["persistent_alias"]}，有 {car["unique_frames"]} 个不同帧支持。已观测表面范围约 **{extent[0]:.2f}×{extent[1]:.2f}×{extent[2]:.2f}m**，表面质心为 `({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})m`，坐标系 `omni_odom`。这不是完整车辆尺寸、真实物体中心或绝对定位精度。细长管线被过滤，是明确的召回代价。\n'
        section+='\n![相同坐标范围下的原始支持与稀疏支持适配对比](../reports/evidence_retention_2026-11-11/stage3/livo_sparse_comparison.png)\n'
        cost=live['processing'];section+=f'\n短片段既有观测的因果 ID 重分配事件为 {live["causal_identity_reassignment_events"]}，不能解释为 GT ID-switch 为零。分阶段每帧耗时之和中位数约 **{cost["measured_per_frame_service_median_s"]:.3f}s**，P95 约 **{cost["measured_per_frame_service_p95_s"]:.3f}s**；模型加载、排队、采集和最终整理不在此数内，因此不称在线端到端延迟。峰值 CUDA 分配约 **{cost["gpu_peak_allocated_bytes"]/1024**3:.2f}GiB**；该轮语义进程主存未采样。\n'
    section+='''
[原始 LIVO 结果与开销](../reports/evidence_retention_2026-11-11/stage3/livo/summary.json) · [原始结果视觉审核](../reports/evidence_retention_2026-11-11/stage3/livo/visual_audit.json) · [稀疏支持适配结果](../reports/evidence_retention_2026-11-11/stage3/livo_supportcc/summary.json)

## 尚未完成的工作

第三阶段尚未整体验收：正在扩展到 rosbag2245 约 589 秒完整录包，检查不同对象、远近距离及缺图情况下的有效支持、身份和查询证据，补充主存和整体处理耗时。首次全时段输入尝试按三帧取一保留图像，预检证明与 5Hz 雷达相位失配：最近过去图像的中位年龄为 73.3ms，超过原有 40ms 门限，前段均退化为 LIO。自己的失败回放已停止并保留记录，正在以原始约 30Hz 全 RGB 帧重新准备输入；短片段原始 30Hz 输入相同预检为 6.64ms、100% 通过。没有通过放宽同步门限掩盖问题。

当前 30 秒实验与分阶段处理不能代替全包或机器人在线实时性。无独立 GT 的位置/身份指标继续明确测量边界。

新方法研究仍聚焦对象证据与背景混淆；本轮已否决的加权候选不继续在测试集上调参。DINO 和 SAM3 保留独立研究边界，不将 SigLIP 强基线成绩归给它们。

'''
    if (REPORT/'stage3/livo_full_visual_review.json').exists():
        from retention_full_livo_report import section as full_section
        section=section[:section.index('## 尚未完成的工作')]+full_section()
        section=section.replace('## 阶段 3：新场景对比完成；真实 LIVO 全包验证进行中','## 阶段 3：新场景与真实 LIVO 全包实验完成')
    return text[:start]+section+text[end:]
