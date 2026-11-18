# omni_slam：Replica 独立场景首轮实验

2026-11-09。两个独立场景、各 200 帧、5 种方法共 10 次建图全部完成，分别保存因果输出与最终划分，共 20 组评分。**当前证据修正方法未展示稳定的地图质量优势，不能据此宣称论文方法已优于基线。** 全部关联参数沿用 Flat；本轮未按 Replica 结果调参。

![固定 RGB 视角](../reports/replica_independent_2026-11-09/rgb.png)

## 数据与公平比较

用户提供的 `vmap.zip` 位于 `/media/user/50D2BD30D2BD1AE4/WorkSpace/`，44,787,616,857 字节。完整 SHA256 校验通过：`e70fd6fd72849072d84fc42bd879dc244c11c6fb8c3f6bdb4403af42fb07a00f`。所选文件解压时 CRC 及提取后 SHA256 均通过。原压缩包保留不变。

来源为 [vMAP 作者数据](https://huggingface.co/datasets/kxic/vMAP/tree/c06e746d78f1862a5262198efb26c4474f05ac8b)。使用 office_0 与 room_1 的 imap/00，固定帧号 0、10、…、1990，各 200 帧，不是原轨迹 2000 帧全量。划分和阈值在检测及评分前记录于 protocol.json。

相机采用[作者配置](https://github.com/kxhit/vMAP/blob/master/configs/Replica/config_replica_room0_vMAP.json)：1200×680，fx=fy=600、cx=599.5、cy=339.5；深度毫米转米，0.1–5 m，原分辨率每 8 像素取样。采用提供的 camera-to-world 位姿，因此这里隔离评测语义关联，不是 LIVO 定位测试。

预测输入为冻结 COCO Mask R-CNN ResNet50 FPN v2，内部缩放 min_size=360/max_size=640；检测分数≥0.5、掩码>0.5，重叠处高检测分数优先，支持点至少 8 个。所有方法共用同一预测与深度支持；不读取 GT 掩码生成观测。DINOv2-small 全图缩放到 448×252，不训练；每观测最多 24 个局部 token。关闭外观代价仍保留几何 token 候选，并不等于完整移除特征计算。

主指标是相同观察表面的 5 cm 体素、类无关实例 PQ（IoU>0.5），对 GT 和预测分别做多数投票、平票取小 ID；未预测区域也保留。背景类别沿用 vMAP 列表，并将未定义 0/255 置为背景；有效原始实例 ID=0 通过 +1 保留。**不是官方 Replica 重建分数、ScanNet AP，也不是完整场景网格评测。** COCO 类别覆盖不足会计为漏检，没有按检测器类别删掉难例。

## 最终地图主结果

| 方法 | office_0 PQ×100 | room_1 PQ×100 | 两场景均值 |
| --- | ---: | ---: | ---: |
| 几何 Hungarian | 9.603 | 9.575 | 9.589 |
| 严格共视 Graph | 6.813 | 9.575 | 8.194 |
| 无共视约束 Graph | 9.818 | 6.075 | 7.946 |
| 证据修正（当前方法） | 6.813 | 9.205 | 8.009 |
| 证据修正，关闭外观代价 | 7.257 | 9.205 | 8.231 |

![主指标](../reports/replica_independent_2026-11-09/metrics.png)

office_0：证据修正与严格共视的 PQ 持平；room_1：略降。无约束在 office_0 较好，但在 room_1 明显下降。几何基线在两场景均不差于当前方法。DINO 外观代价仍没有独立、稳定的增益证据。

## 关联精确率与召回率

仅统计预测支持中 ≥70% 属于同一 GT 对象的观测，属于条件诊断，不代表整体检测精度。表中 P/R/F1 均为观测对百分比。

| 场景 | 方法 | Pair P | Pair R | Pair F1 |
| --- | --- | ---: | ---: | ---: |
| office_0 | 几何 Hungarian | 99.67 | 67.64 | 80.59 |
| office_0 | 严格共视 Graph | 100.00 | 68.51 | 81.31 |
| office_0 | 无共视约束 Graph | 76.00 | 81.12 | 78.48 |
| office_0 | 证据修正（当前方法） | 100.00 | 68.74 | 81.48 |
| office_0 | 证据修正，关闭外观代价 | 99.43 | 69.42 | 81.76 |
| room_1 | 几何 Hungarian | 99.04 | 62.58 | 76.69 |
| room_1 | 严格共视 Graph | 93.58 | 62.15 | 74.69 |
| room_1 | 无共视约束 Graph | 83.49 | 63.80 | 72.33 |
| room_1 | 证据修正（当前方法） | 92.89 | 63.69 | 75.57 |
| room_1 | 证据修正，关闭外观代价 | 93.72 | 63.69 | 75.84 |

| 场景 | 全部观测 / 可评分观测 | 体素 GT 实例 | 可评分观测覆盖的 GT 实例 |
| --- | ---: | ---: | ---: |
| office_0 | 625 / 283 | 18 | 12 |
| room_1 | 480 / 195 | 30 | 14 |

| 证据修正相对严格共视（可评分观测子集） | 新连接同 GT 观测对 | 新连接不同 GT 观测对 | 分离同 GT 对 | 分离不同 GT 对 |
| --- | ---: | ---: | ---: | ---: |
| office_0 | 17 | 0 | 2 | 0 |
| room_1 | 58 | 30 | 0 | 7 |

独立数据上出现了可量化的错误合并，说明“表面接触 + 多视角见证”仍不能保证同一身份。所有有效见证均核对没有引用未来帧。下一版应围绕这些错误证据改进，而不是继续降低当前测试集阈值；两场景现在已被观察，后续方法迭代应另留未查看场景验证。

## 地图、因果输出和成本

![地图对比](../reports/replica_independent_2026-11-09/maps.png)

俯视图基于同一观察体素；颜色是任意实例 ID，跨方法同色不代表同一对象。灰色为背景或未预测。可下载的 JSON 保留全部 TP/FP/FN、匹配和条件身份指标。

| 场景 | 方法 | 当时输出 PQ×100 | 最终地图 PQ×100 | 关联回放耗时（秒） |
| --- | --- | ---: | ---: | ---: |
| office_0 | 几何 Hungarian | 9.603 | 9.603 | 0.343 |
| office_0 | 严格共视 Graph | 12.730 | 6.813 | 0.454 |
| office_0 | 无共视约束 Graph | 13.359 | 9.818 | 0.563 |
| office_0 | 证据修正（当前方法） | 12.730 | 6.813 | 0.618 |
| office_0 | 证据修正，关闭外观代价 | 12.506 | 7.257 | 0.711 |
| room_1 | 几何 Hungarian | 9.575 | 9.575 | 0.181 |
| room_1 | 严格共视 Graph | 7.873 | 9.575 | 0.306 |
| room_1 | 无共视约束 Graph | 5.882 | 6.075 | 0.398 |
| room_1 | 证据修正（当前方法） | 7.621 | 9.205 | 0.484 |
| room_1 | 证据修正，关闭外观代价 | 7.706 | 9.205 | 0.556 |

当时输出固定保留每帧产生的 ID；最终地图允许后来的证据改变过去归属，二者不混比。耗时为本机单次 CPU 关联回放，包含输出写盘，不含检测、特征及评分，未据此宣称实时性。两张 5090 分别处理两个场景；固定缓存生成耗时 office_0 35.37 秒、room_1 16.86 秒，运行时有其他 GPU 负载，不宜直接比较。

## 复现与结论边界

代码：[replica_benchmark.py](../../omni_semantic_map/omni_semantic_map/replica_benchmark.py)。冻结配置与源代码、指标、审计及图表：[报告附件](../reports/replica_independent_2026-11-09/)。数据、特征和运行目录：`build/benchmarks/replica_independent_20261109/`。42 项测试通过；环境保留已有 NumPy/SciPy 声明版本警告，未在本轮更换依赖。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=omni_semantic_map \
  build/experiments/large_semantic_20261106/venv/bin/python \
  -m omni_semantic_map.replica_benchmark run \
  --base build/benchmarks/replica_independent_20261109 --scene office_0
```

已有输出目录会拒绝覆盖；复现实验应在新 base 中执行 extract、prepare、run、evaluate 四步，prepare 需要显式指定本地模型、权重和 GPU。

这一轮完成独立场景输入、冻结参数比较和负面结果审计；尚未运行 vMAP 等外部建图系统，因此表格是内部基线与消融，不是 SOTA 对比。两场景规模小、使用采样帧及提供位姿，不能支撑整体泛化或实机实时性结论。真实 LIVO 位姿修正的独立工程检查见[配套报告](real_pose_revision_2026-11-09.md)。
