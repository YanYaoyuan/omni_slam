"""Render the completed full-recording validation, including its failures."""
import json
from audit_evidence_retention import OUT, REPORT


def section():
    base = REPORT/'stage3'
    read = lambda p: json.loads(p.read_text())
    raw = read(base/'livo_full_v2/summary.json')
    adapted = read(base/'livo_full_v2_supportcc/summary.json')
    capture = read(base/'livo_full_v2_capture_audit.json')
    review = read(base/'livo_full_visual_review.json')
    status = read(OUT/'livo_full_validation_v2_status.json')
    assert status['complete'] and review['complete']
    text = '''## 完整录包验证：实验完成，部署质量尚不合格

约 589 秒 rosbag2245 已执行原生 LIVO、原子帧留存、冻结强前端、CG/守恒/均值以及稀疏支持适配两条分支。全时段输入保留约 30Hz 原始 RGB、全部 LiDAR/IMU；原来的 40ms 图像时效门限不变。首轮 10Hz 输入相位失配导致仅 LIO 更新的失败记录保留，未通过放宽门限掩盖问题。

采集得到 **330 个语义关键帧，跨度覆盖 99.35% 的录制时段**，十个等时段都有视觉帧；333 个原生关键帧消息全部收到，其中 332 个为视觉关键帧，缺失语义包 ID **0、2**。最长语义帧间隔为 4.20 秒，拒绝包为 0。共有 2,943 次 LIO 和 2,939 次视觉更新。已逐帧核对文件哈希、时间、原生位姿、相机内参和图像时刻投影变换。

这证明采集及处理覆盖，不代表全场景对象召回。地图包括 1,441,478 个原始实测点、3,112 个掩码观测。以下统计全部最终保留对象，没有按效果挑选子集。

| 项目 | 原始稀疏支持 | 最大连通支持适配 |
|---|---:|---:|
'''
    for title, a, b in [
        ('支持点数（跨观测累加）', raw['matched_supports'], adapted['matched_supports']),
        ('没有支持的掩码数', raw['masks_without_support'], adapted['masks_without_support']),
        ('最终区域数（含背景）', raw['final_regions'], adapted['final_regions']),
        ('前景预测数', len(raw['retained_objects']), len(adapted['retained_objects'])),
        ('car 查询返回数', len(raw['queries']['car']), len(adapted['queries']['car'])),
        ('pipe 查询返回数', len(raw['queries']['pipe']), len(adapted['queries']['pipe']))]:
        text += f'| {title} | {a:,} | {b:,} |\n'
    text += '''
![完整空间范围的两种支持与时间覆盖](../reports/evidence_retention_2026-11-11/stage3/livo_full_comparison.png)

上图使用共同坐标范围，未裁剪地图或轨迹；点云仅为绘图等间隔抽稀。三维框代表已观测表面范围，不能解释为完整物体尺寸或定位真值。

### 全部对象证据与明确失败

对原始分支 6 个、适配分支 3 个保留对象，逐一查看最早帧、最大掩码帧、最晚帧的原始图像、掩码和支持投影。审核者为本助手，不是独立人工 GT；三种选择可能落在同一帧。

| 对象 alias | 图像证据结论 | 原始 → 适配结果 |
|---|---|---|
| 6 | 起点黑色轿车，类别有图像支持 | 表面 AABB 9.41×8.70×4.41m → 0.78×4.32×1.12m；背景污染明显减轻 |
| 2647 | 后段白色 SUV，类别有图像支持 | 2.61×5.78×1.47m → 2.36×3.99×1.48m；范围收紧，无绝对精度 GT |
| 2652 | 白色 SUV 后方可见的另一辆白车 | 原始分支保留，适配分支消失；明确损失了有图像证据的目标 |
| 3075 | 返回起点的黑车，视觉上支持与 alias 6 为同车的判断 | 两分支均生成另一实例；适配后仅 11 点、约 0.04×0.18×0.02m，不能代表整车 |
| 10、3078 | 起点与返回时的顶部长条结构 | pipe 精确类别不确定；适配均移除，不计作已验证正确管道 |

![整包后段的白色车辆证据](../reports/evidence_retention_2026-11-11/stage3/livo_full_v2_supportcc_audit/object_002647.jpg)

![被连通支持规则丢失的另一辆车](../reports/evidence_retention_2026-11-11/stage3/livo_full_v2_audit/object_002652.jpg)

![返回起点仍生成新的微小车辆实例](../reports/evidence_retention_2026-11-11/stage3/livo_full_v2_supportcc_audit/object_003075.jpg)

**决策：最大连通支持仅保留为实验性适配，不升级为普遍优于原始分支的默认方案。** 它改善了两辆车的支持范围，但丢失另一辆车，未解决返回时的重复实例。原始 car 查询返回 4 个实例，其中含返回重复候选；适配返回 3 个实例，也不能宣称检测到 3 辆独立车辆。

person、fire extinguisher、door 查询均为空，仅表示没有保留对应类别预测，不能说明场景里没有这些目标。查询是固定预测标签的精确检索，不是自由文本开放词汇查询评测。

### 碎片化与最终去噪的定位

最终处理前，原始分支有 1,516 个对象，其中 **1,501 个只含一个不同帧**；适配分支有 982 个，其中 **974 个只含一个不同帧**。达到至少三个不同帧的分别只有 14 和 7 个。大量观测没有稳定跨帧关联，是当前完整场景保留对象稀少的重要证据；不能通过单独放松保留条件宣称修复了关联。

另做未改变计算的原生后处理旁路记录，重放后最终几何与分类逐项精确复现。适配分支返回车辆 alias 3075 在最终去噪前为 **377 点，去噪后仅 11 点**。现有配置 `obj_min_points=0`、`obj_min_detections=3`，因其累积六次检测仍通过最终过滤。输入最少 16 点不保证最终去噪后仍有 16 点。这是密集 RGB-D 后处理参数用于稀疏 LiDAR 的具体失效，不是新增方法收益。

两分支既有观测的因果 ID 重分配事件均为 0，但出现了返回重复实例，说明该计数不能代替 GT ID-switch 或身份稳定性。所有保留对象的多帧掩码重投影一致性很高，仍同时存在过大框和微小碎片；因此该指标只作为内部一致性诊断，不作为三维准确率。

### 资源与时延边界

'''
    for label, data in [('原始分支', raw), ('适配分支', adapted)]:
        c = data['processing']
        text += f'- {label}：各阶段已有观测每帧服务耗时之和，中位数 {c["measured_per_frame_service_median_s"]:.3f}s，P95 {c["measured_per_frame_service_p95_s"]:.3f}s；CG 后端总耗时 {c["backend_wall_s"]:.2f}s。\n'
    c = raw['processing']
    total = sum(s['wall_s'] for s in status['steps'])
    text += f'''- GPU 峰值分配 {c['gpu_peak_allocated_bytes']/1024**3:.2f}GiB；语义 worker 完成后的峰值 RSS 约 {c['semantic_workers_peak_rss_bytes']/1024**3:.2f}GiB；原生 LIVO 子进程峰值 RSS 约 {c['livo_children_peak_rss_bytes']/1024**3:.2f}GiB，均按记录的独立进程口径，不相加成系统总内存。
- 运行器九个步骤墙钟时间合计 **{total:.1f}s**，包含约 606.7s 的采集步骤、模型加载与等待、两条后端分支和生成报告；不含先前原始视频转码、后续独立审核与诊断重放。单帧服务耗时不包含这些开销。

当前是先采集后分阶段语义处理，**没有测得在线端到端延迟，也没有完成 S100 实机部署**。原生轨迹长约 316m；轨迹首尾差和 Z 范围没有外部 GT，不能报告成定位精度或漂移率。

[采集与标定审计](../reports/evidence_retention_2026-11-11/stage3/livo_full_v2_capture_audit.json) · [原始结果](../reports/evidence_retention_2026-11-11/stage3/livo_full_v2/summary.json) · [适配结果](../reports/evidence_retention_2026-11-11/stage3/livo_full_v2_supportcc/summary.json) · [全部对象视觉审核](../reports/evidence_retention_2026-11-11/stage3/livo_full_visual_review.json) · [后处理精确复现与点数退化](../reports/evidence_retention_2026-11-11/stage3/livo_full_postprocessing_audit.json)

## 本轮三阶段结论与后续边界

逐对象审计、固定输入六种机制对照、三个冻结新场景及实际 LIVO 整包验证均已执行并保留正负结果。本轮实验完成不等于候选方法通过质量验收：**保留 SAM2＋SigLIP＋CG/守恒/均值强基线；否决加权读出作为新主方法；稀疏最大连通支持不设为默认增强。**

下一轮研究优先级由整包失败确定：先解决稀疏支持在原生去噪中的塌缩与关联碎片化，再固定 SigLIP 读出评估 DINO 局部对应及可修正对象关联。另行冻结实验版本与开发/验证划分，不能继续在本轮已查看的新场景调阈值并称为独立测试。DINO、SAM3、新训练、在线常驻接入和实机验证不属于本轮已经完成的成果。

'''
    return text
