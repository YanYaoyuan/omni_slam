# Flat 语义建图开发基准：omni_slam 对象关联

2026-11-09。研究主线：对象级语义建图，定位作为支撑。**本轮使用公开 Flat 数据验证评测和定位关联缺陷；这不是论文终测，也不是 ScanNet 官方榜单成绩。**

## 数据与复现边界

用户提供的 `flat_dataset.zip` 为 392,714,654 字节，SHA256 `029c89b9963944af0608446f87bcb120ad6f1eca84740a714b36225500e2a3b8`；ZIP 全部条目 CRC 通过。额外收到的 `rio_data.zip` 同样通过校验，它只是 RIO 补充数据，尚缺原始 RGB-D 数据，未作为完整 RIO 实验运行。压缩包哈希用于本次复现，未获得官方哈希，不能声称与官方发布文件逐字节交叉验证。

| 序列 | 全部帧数 | 输入时间跨度 | 图像/深度/位姿/预测/真值 |
| --- | ---: | ---: | --- |
| run1 | 400 | 197.255 s | 全部齐全 |
| run2 | 362 | 140.430 s | 全部齐全 |

[官方数据说明](https://projects.asl.ethz.ch/datasets/panoptic-mapping/) · [原版代码](https://github.com/ethz-asl/panoptic_mapping/tree/3926396d92f6e3255748ced61f5519c9b102570f)。下载失败后改由用户提供压缩包，原文件保留在 `build/benchmarks/flat_semantic_20261109/`。

- 相机：640×480，fx=fy=320，cx=320，cy=240；深度是相机 Z 方向的米值，位姿为提供的 camera→world 矩阵。内部方法及统一评价取深度 0.1–5 m，每隔 4 像素采样一次；原版系统以全分辨率输入和原生配置运行。全部图像帧参与，未抽掉困难帧。
- 主实验固定使用数据内的 Detectron 预测；不重新训练、不调用真值分割驱动主实验关联。预测运行与真值评价为两个独立命令，输入文件哈希及逐帧结果保留。
- 复用现有对象关联（1 m 空间门限、0.35 外观距离门限、0.25 歧义差值门限），以 RGB-D 支持点代替稀疏激光。内部几何基线使用同类约束、1 m 门限及匈牙利分配，不使用外观和歧义拒绝。
- 当前表评测每段独立建图。没有声称跑通 run1→run2 的持久地图更新、跨会话重定位或完整 LIVO；真值/提供位姿是共享输入。
- 保留每次观测的局部三维支持点、像素索引及关键帧锚点，支持点位于各运行的 `supports/`。本轮框或表面中心不当作完整物体中心。

## 逐帧序列实例诊断

下表 **PQ×100** 是把整段序列的有效深度采样像素作为统一域，以实例 IoU>0.5 配对后计算 `ΣIoU / (TP + 0.5 FP + 0.5 FN)`。忽略类别进行实例匹配，不等于语义类别准确率、完整 LSTQ、3D mesh PQ 或官方 ScanNet AP。它能惩罚跨帧 ID 拆分，但重复观察较多的表面会有较高权重；因此另报统一体素诊断。

“记录数”为整个运行曾输出的实例记录数，不是独立真值物体数；没有只保留 confirmed 对象来隐藏失败。

| 方法 | run1 PQ×100 | run1 记录数 | run2 PQ×100 | run2 记录数 |
| --- | ---: | ---: | ---: | ---: |
| 仅几何关联（内部基线） | 7.753 | 101 | 12.743 | 83 |
| omni_slam 当前关联 | 0.475 | 1098 | 1.499 | 507 |
| 关闭歧义拒绝（消融） | 4.382 | 147 | 6.558 | 118 |
| Panoptic Mapping 原版算法＋只读记录器 | 1.925 | 208 | 4.705 | 96 |

![记录增殖与序列实例评分](../reports/flat_semantic_benchmark_2026-11-09/comparison.png)

同一检测输入下，三个内部方法覆盖相同的真值物体采样像素：run1 **37.47%**，run2 **32.95%**。该数值是掩码覆盖率，不是对象级召回率，且当前统计覆盖 Flat 的全部 thing 类，和检测器的类别表不完全一致。床头灯、吸顶灯等有大量低覆盖实例；逐对象统计见 [覆盖审计](../reports/flat_semantic_benchmark_2026-11-09/provenance/detection_coverage.json)。

## 统一观察表面的三维诊断

使用同一深度/位姿和渲染真值生成观察表面参考，以 5 cm 世界体素进行标签多数投票；预测缺失的 0 标签也计票，不能删去缺失点。票数相同时选较小 ID。该参考是从本轮 RGB-D 可见部分构建的，不是带实例标签的独立全场景激光真值。数据包内的 `flat_*_gt_10000.ply` 只有 XYZ，可用于几何检查，不能直接拿来计算实例 AP。

| 方法 | run1 观察体素 PQ×100 | run2 观察体素 PQ×100 |
| --- | ---: | ---: |
| 仅几何关联（内部基线） | 11.179 | 19.574 |
| omni_slam 当前关联 | 2.383 | 6.863 |
| 关闭歧义拒绝（消融） | 7.730 | 11.638 |
| Panoptic Mapping 原版算法＋只读记录器 | 4.554 | 8.568 |

run1 固定 85,829 个观察体素，run2 固定 80,893 个。这里比较同域上的实例关联及投票结果，不比较 TSDF 的几何重建质量。灰色为共享的 RGB-D 观察几何，彩色为方法的对象标签；各方法颜色是任意 ID 编码，不能跨图按颜色对应物体。

![run1 对象表面](../reports/flat_semantic_benchmark_2026-11-09/surfaces/run1/instance_maps.png)

![run2 对象表面](../reports/flat_semantic_benchmark_2026-11-09/surfaces/run2/instance_maps.png)

## 原版系统的复现与比较边界

原版 Panoptic Mapping 已完成 run1 **400/400** 帧、run2 **362/362** 帧。逐帧等待处理完成再发送下一帧，输入、处理回执和实例记录文件数量一致；两段原生 `.panmap` 地图均已保存。使用官方容器的 AMD64 manifest `sha256:d6030c671997c9b51a5ad7b1a2235078ab8bdcd9692f8759055a6dcb32d36dba`，下载时逐层验证压缩 SHA256 和解压 diff-ID，导入后的 25 个文件系统层与官方配置一致。

构建前确认容器内 `panoptic_mapper.cpp` 与固定提交源码一致，仅在实例跟踪后、积分前增加只读记录器；不改变跟踪、融合或地图管理逻辑。[记录器补丁](../reports/flat_semantic_benchmark_2026-11-09/provenance/recording_only.patch) · [run1 完整性记录](../reports/flat_semantic_benchmark_2026-11-09/provenance/upstream/run1/run_transport.json) · [run2 完整性记录](../reports/flat_semantic_benchmark_2026-11-09/provenance/upstream/run2/run_transport.json)。

原版采用 `flat_detectron.yaml` 原生算法配置，修改项限于数据路径、可视化和日志输出。**这不是只替换关联器的严格消融**：原版以 640×480 深度融合并采用 2,000 像素的新对象分配门限；内部方法使用步长 4 的深度采样、至少 8 个支持点。两者共享原始观测、提供位姿和评价域，但输入筛选、地图表示和生命周期策略不同。关联器的受控消融应看三个内部方法。

评分使用每帧当时输出的实例 ID，不把后续合并、地图清理或最终地图重渲染追溯到过去。因此这里的原版成绩是本次**因果实例输出诊断**，不是原论文最终地图成绩复现。未评价原版 TSDF 几何误差，也不根据含记录 I/O 的耗时比较速度。对应配置和源码均在报告附件保存。

## 消融揭示的缺陷

当前规则在两个相近候选之间无法明确匹配时，会再新建一个候选。新增候选又使后续观测更容易陷入歧义，形成重复记录增殖。已构造最小回归用例复现此行为。

只把歧义门限从 0.25 改为 0，其他条件保持不变，预测输入下记录数降为 147 / 118，实例评分明显改善。但仍落后于仅几何关联，不能宣布问题已解决，也没有把实验变体设为部署默认。颜色直方图在部分可见、多视角条件下的门控仍需独立分析。

额外提供真值掩码输入，只使用类别与当前掩码提取观测，**不把真值实例 ID 当作跨帧关联答案**。该组是 oracle 诊断，不能与预测掩码成绩混成端到端排名：

| 方法 | run1 PQ×100 / 记录数 | run2 PQ×100 / 记录数 |
| --- | ---: | ---: |
| omni_slam 当前关联 | 2.429 / 1605 | 6.252 / 903 |
| 关闭歧义拒绝（消融） | 31.237 / 177 | 41.958 / 150 |

真值掩码覆盖约 99.9% 的有效 thing 采样点，少量不足 8 个支持点的片段被统一门限剔除。即使输入掩码理想，原关联仍发生严重拆分，证实检测器不是唯一问题。**关闭歧义拒绝只是消融，不是具有新颖性证明的方法。**

## 工程验证与后续

- 新增数据读取、相机投影、独立评分、局部支持点保存及观察表面投票入口。预测模式在完全没有 GT 文件的测试夹具上可以运行，评价阶段才读取真值。
- 语义测试 24 项通过，包含已知拆分/误合并、漏检/背景误检、投影、反射位姿拒绝、数据隔离和投票规则。环境存在 NumPy/SciPy 声明版本警告，正式环境仍需固定兼容依赖。
- 新增歧义参数保持默认 0.25；修改后重跑 run1 全部 400 帧，与冻结旧实现的逐帧实例图、有效域和完整对象文档逐值一致。[回归证据](../reports/flat_semantic_benchmark_2026-11-09/provenance/default_regression.json)。
- 所有当前 Flat 场景已用于诊断，均为开发数据。下一步需要多视角关联方法、独立场景验证及类别映射后的标准评测，不能将当前开发结果直接包装为论文贡献。
- ScanNet 的官方获取流程要求机构邮箱提交使用协议；3RScan 同样提供官方数据申请流程。尚未获取和运行二者，现有 RIO 补充包不能替代原始数据。[ScanNet 官方说明](https://github.com/ScanNet/ScanNet#scannet-data) · [3RScan 官方入口](https://waldjohannau.github.io/RIO/)。

原始实验：`build/benchmarks/flat_semantic_20261109/runs/`；观察表面 NPZ：`build/benchmarks/flat_semantic_20261109/surfaces_all/`；评分、输入哈希、协议、图与源码备份：`docs/reports/flat_semantic_benchmark_2026-11-09/`。新运行与评价输出拒绝覆盖已有结果。共完成 10 个内部完整运行和 2 个原版完整运行，另有默认行为回归及 3 帧冒烟检查。

可用 CloudCompare 打开 [run1 当前关联三维点图](../../build/benchmarks/flat_semantic_20261109/maps/run1_run1_predicted_omni_slam.ply)、[run1 几何基线](../../build/benchmarks/flat_semantic_20261109/maps/run1_run1_predicted_geometry_hungarian.ply)、[run1 原版因果实例投票点图](../../build/benchmarks/flat_semantic_20261109/maps/run1_run1_predicted_panoptic_mapping.ply)。这些 PLY 是共享观察体素上的标签可视化，原版完整融合地图另存为 [run1 map.panmap](../../build/benchmarks/flat_semantic_20261109/upstream/run1/map.panmap) 和 [run2 map.panmap](../../build/benchmarks/flat_semantic_20261109/upstream/run2/map.panmap)。

```bash
PYTHONPATH=omni_semantic_map \
  build/benchmarks/ntu_viral_20261109/venv/bin/python -m omni_semantic_map.flat_benchmark run \
  --sequence build/benchmarks/flat_semantic_20261109/data/flat_dataset/run1 \
  --method omni_slam --output build/benchmarks/flat_new_run

PYTHONPATH=omni_semantic_map \
  build/benchmarks/ntu_viral_20261109/venv/bin/python -m omni_semantic_map.flat_benchmark evaluate \
  --sequence build/benchmarks/flat_semantic_20261109/data/flat_dataset/run1 \
  --run build/benchmarks/flat_new_run --output build/benchmarks/flat_new_metrics.json
```
