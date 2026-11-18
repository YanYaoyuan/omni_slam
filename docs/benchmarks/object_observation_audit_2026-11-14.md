# 对象观测复核：低覆盖帧等权与前端识别错误

**这次定位了可复现的输入证据问题，没有启动新建图或调参。** 上一轮联合机制仍封存，原强基线、地图、评分、失败门槛均保持。当前目标仍是完整对象级语义建图、新机制的同版本机器人对照和真正独立评测，尚未完成。

[真实图像、点投影与逐帧证据图](../../output/object_observation_audit_v1/index.html)

## 核查范围与结论

预先固定上一轮全部7个类别发生变化的对象，不新增GT筛选：室外0000对象39、0002对象1/34，车库对象23/24/33/46。逐个读取全部已有观测，每个对象展示可见来源点数最多的帧，同点数时取最早帧。共45条对象-帧记录、9,134次来源点观测；重复来源不冒充独立样本。

1. **没有发现选中案例的投影索引错位。** 室外8,402次观测从完整实测坐标、原相机位姿与内参独立重算；机器人732次观测使用原camera_pose和floor取整约定独立重算。全部整数像素与原缓存精确一致。这只验证实现和约定一致，不证明物理标定、时间同步、遮挡处理绝对无误。
2. **概率读出正确，但概率不一定可靠。** 对7个展示帧，从保存的原生logits独立作双线性插值及softmax，最大绝对差4.50×10⁻⁶，低于预先固定的2×10⁻⁵容差。全部原图、无叠加局部图和语义叠加图已查看；不能将模型置信度等同于识别准确率。
3. **逐帧等权会让很少的像素获得很大影响。** 0000对象39有3帧各仅1个采样像素，最大观测有1,164个不同像素，但每帧权重都为1/8。近处大片白墙被识别为building，低覆盖观测多为vegetation，最终均值为vegetation。此处是原规则真实权重的精确分解，没有试验另一种加权地图。
4. **识别错误不能靠重复融合自动解决。** 0002对象1从远处小幅图像中的building，转成中近处的truck/car；摩托车对象34的最密观测却主要被识别成person。车库暗处车辆也存在building、traffic sign或train等识别错误。后者说明最大像素数也不是可靠性的充分条件。

## 全部案例

“最大/最小”指同对象不同观测帧的不同采样像素数之比，不是实际可见表面积或信息量。

| 对象 | 有效观测帧 | 最大/最小像素数 | 原冻结融合类别 | 最大观测帧的主要类别 |
|---|---:|---:|---|---|
| 0000 / 39 | 8 | 1164.0 | vegetation | building |
| 0002 / 1 | 11 | 543.0 | building | car |
| 0002 / 34 | 6 | 45.0 | building | person |
| rosbag2245 / 23 | 8 | 67.0 | building | building |
| rosbag2245 / 24 | 3 | 1.22 | building | building |
| rosbag2245 / 33 | 3 | 1.64 | building | train |
| rosbag2245 / 46 | 6 | 37.0 | building | car |

逐帧曲线重建了原算法的“先均衡单元、再均衡帧”概率，与封存对象概率在1×10⁻¹¹内一致。展示帧主要类别是其采样像素的直接概率均值，与单元均衡读出分开记录。所有19类、所有观测帧和逐帧移除敏感性均保存；没有根据敏感性挑帧重新输出预测。

此诊断只覆盖预选7个变化案例，不能外推成所有室外/机器人错误都由观测大小导致。机器人仍无独立语义标注，不提供新精度数值。

## 查阅先例后的研究边界

简单增加视图权重不应直接包装成新贡献：

- [GLFS / On the Overconfidence Problem in Semantic 3D Mapping](https://motion.cs.illinois.edu/papers/ICRA2024_Marques_OverconfidenceSemanticMapping.pdf)已研究语义融合校准，并讨论距离、视角和观测权重。
- [OVI-MAP](https://arxiv.org/html/2603.26541v1)已有类无关实例构建、对象覆盖驱动的视图选择和可见像素加权的对象特征更新；其论文评测为ScanNet/Replica。[官方代码入口](https://github.com/OVI-MAP/OVI-MAP)已核查，本轮未声称完成复现。CVF PDF直接打开返回403，改读作者arXiv全文。

据此，下一研究决定应检验**更有辨识力的新观测能否修正历史对象划分、恢复遗漏表面**，并明确与仅改类别、像素加权/最佳视图及只更新新观测的差别。上述先例意味着“加权融合、选择新视图”本身不构成充分的新颖性论据。

在冻结下一方法前，必须具体说明新证据来自哪一幅真实观测、能改变哪些原来错误的对象归属，以及同输入简单控制为什么不足。不能沿用本轮不可靠概率，再自动追加兼容阈值或另一种平均规则。当前尚未宣布新的有效方法，也未启动另一轮地图/AP实验。

## 复现与记录

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/omni_object_audit_mpl \
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
PYTHONPATH=omni_semantic_map:scripts/benchmarks \
build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python \
scripts/benchmarks/audit_object_observations.py
```

以上入口拒绝覆盖已有审计目录；后续影响分解由`audit_observation_influence.py`执行，也拒绝覆盖原协议。首次独立分解遗漏了原算子的逐行概率归一化，严格重算发现2.2×10⁻⁸差异，初次代码/协议/原因已归档。补齐相同归一化后按原1×10⁻¹¹容差通过，没有放宽门槛，也未改预测。

- [核查协议](../../build/benchmarks/object_observation_audit_v1/protocol.json)
- [全部观测与图像证据](../../build/benchmarks/object_observation_audit_v1/summary.json)
- [原权重分解和独立读出](../../build/benchmarks/object_observation_audit_v1/influence.json)
- [交付核验](../../build/benchmarks/object_observation_audit_v1/delivery_audit.json)
