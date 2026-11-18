# 新场景与增量语义建图基线核对

2026-11-10 后续结果：[SceneNN 与 OVO 更新机制实验](scenenn_2026-11-10.md)现已完成。下文保留数据准备阶段记录；当前已跑 OVO 官方更新内核控制，仍未跑完整 OVO 前端/SLAM。

2026-11-10。这里区分官方代码核对、数据准备和实际实验，尚未跑过的方法不列性能分数。

## 最近方法和最接近对照

| 方法 | 核对到的机制 | 当前状态与比较要求 |
| --- | --- | --- |
| OVO，RA-L 2025，后续代码更新 | 在线对象建图；官方仓库已支持回环更新 | 已下载官方源码并锁定 `e0d16f7a459ea5451540c50ac264b00aca3aeb85`，尚未运行完整模型 |
| OVI-MAP，CVPR 2026 | 增量体素实例重建，选择视角提取语义 | 官方代码公开；尚未安装和实跑 |
| ThinkGraphs，ECCV 2026 | 在线概率体素实例地图，后台 VLM 补充语义并处理重复对象 | 已核对论文与项目页面；尚未实跑 |
| OVIP-SG，2026 预印本 | 对称 3D IoU 关联、按面积融合特征、小目标检索 | 已核对论文和公开仓库入口；尚未实跑 |

来源：[OVO 官方仓库](https://github.com/tberriel/OVO)、[OVI-MAP 官方仓库](https://github.com/OVI-MAP/OVI-MAP)、[ThinkGraphs 论文](https://arxiv.org/abs/2606.31471)、[OVIP-SG 论文](https://arxiv.org/abs/2608.17633)。原文的分数与我们的观察表面 5 cm PQ 不是同一协议，不能直接做数字比较。

OVO 本地代码中，`ovo/slam/orbslam.py:update_map` 按更新后的关键帧位姿搬移点云并处理被删除关键帧；`ovo/entities/ovo.py:update_map` 清理不再受支持的实例，再检查对象对并融合，最后更新描述子。在所核对的这个函数里没有把已经错误合并的实例拆回原观测重算的步骤。这是源码观察，不是已经验证的精度优势。后续对照必须检查“修正后出现拆分需求”的数据，并实测最终实例质量、身份一致性和更新延迟。

优先补 OVO，是因为它有实际回环处理。OVI-MAP 提供近期在线实例建图质量对照。二者需要分别报告原版前端结果与固定输入适配结果，不能将替换其输入后的映射核心称为原版完整复现。OVO 的实例融合也不必等于 ConceptGraphs 的全量输出；跨方法应比较统一 GT 指标，不能强制其标签与我们的参考相同。

## 新场景

用户提供的 `vmap.zip` 只包含已经研究过的八个 Replica 场景。重复这些场景或改用其其他轨迹，不构成新场景验证。

选择尚未运行预测和评分的 SceneNN `005 / 011 / 016`，从官方公开服务器下载原始 ONI、相机标定、相机到世界位姿、实例 mesh 与 XML。数据保存到用户指定外置盘下的 `omni_slam_benchmarks/SceneNN`。下载完成需核验 HTTP 长度、分段范围和本地 SHA256；后续先检查 RGB-D 同步、深度单位、标定和点云到 GT mesh 的对齐，再运行冻结配置，不根据这三个场景调整调度阈值。

来源：[SceneNN 官方说明](https://github.com/hkust-vgd/scenenn)、[官方文件源](https://hkust-vgd.ust.hk/scenenn/main/)。本轮下载日志与清单在 `build/benchmarks/adaptive_fusion_20261110/`。**准备数据不等于完成泛化评测。**

ScanNet 和 3RScan 仍是正式实例评测的重要候选，但其官方数据申请流程需要用户完成协议；没有把未获取的数据记为已测试。[ScanNet 官方说明](https://www.scan-net.org/ScanNet/)、[3RScan 官方说明](https://vmnavab26.in.tum.de/3RScan/documentation.php)。

SceneNN 016 已用官方工具解出 1364 对 RGB-D，发布位姿仅覆盖前 1300 帧。后续只使用有位姿覆盖的前缀，末尾 64 帧不插值补位姿。按官方解码序号 `image00001` 对应轨迹第 0 块，均匀抽查 10 帧的深度点到 GT mesh 的距离中位数为约 0.7–3.1 cm，未拟合额外坐标变换或帧偏移。RGB-D 时间差最大约 33 ms，与官方解码器阈值一致。文本位姿的旋转正交误差最大约 1.12e-4，后续输入适配应保留原值并显式用 SVD 投影到最近 SO(3)，记录修正量，不放宽核心位姿校验。

![新场景的图像与输入配准抽查](../reports/adaptive_fusion_2026-11-10/scenenn_016_input_qa.png)

[输入核验记录](../reports/adaptive_fusion_2026-11-10/scenenn_016_input_qa.json)。这是数据质量检查，尚无该场景的算法性能分数。

三个场景的下载现已完成：13 个文件，共 2.965 GB，含三段 ONI、mesh、XML、轨迹和相机内参。[下载来源与 SHA256 清单](../reports/adaptive_fusion_2026-11-10/scenenn_download_manifest.json)。其中 016 已完成解码和上述输入核验；005 / 011 尚未解码，三个场景均尚未跑算法评分。
