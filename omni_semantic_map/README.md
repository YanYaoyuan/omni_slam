# omni_semantic_map

对象级语义建图包，将每次对象观测绑定到关键帧，保存点集、位置与图像。提供 ROS 在线节点、对象查询工具，以及论文方法 `full_authorization` 的离线修订实现。在线节点与论文实验使用各自的推理和关联入口。

## 代码入口

| 范围 | 入口 | 输入与输出 |
| --- | --- | --- |
| ROS 语义节点 | `live_node.py`、`live_core.py` | `LivoSemanticFrame`，输出对象框、地面点与图像证据 |
| 类别融合 | `category_node.py`、`category_stream.py` | 关键帧对象与多帧类别证据 |
| 对象查询 / 位姿重投影 | `cli.py`、`objects.py` | 对象 JSON、关键帧位姿、证据图 |
| 历史身份授权 | `revision_authorization.py` | 跨帧共享 / 分离掩码证据、兼容对象组 |
| 局部点分配 | `batch_stable_transport.py`、同名 `.cpp` | 实测表面图、对象种子和观测帧组 |
| 表面候选与源点追溯 | `mask_cover_revision.py`、`discovery_measured_input.py` | 掩码覆盖、实测点索引与可见性 |
| KITTI-360 评测适配 | `kitti360_evaluation.py`、`kitti360_reference_identity.py` | 固定地图预测、独立参考标注与官方 AP 内核 |

数据、模型和评测协议见[研究说明](../docs/research/README.md)。其他方法见[历史分支索引](RESEARCH_HISTORY.md)。

## 不依赖 ROS 的核心安装与验证

在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/research-core.txt
python -m pip install -e ./omni_semantic_map
python examples/revision_core_demo.py --output build/demo/revision_core.json
bash scripts/test_research_core.sh
```

需要 C++17 编译器 `g++`，图内核按源码哈希编译到调用方指定的缓存目录。演示使用合成小图检查安装、合并判断和点分配。完整历史研究测试还需要 Torch、OpenCV、Pillow、Open3D、OmegaConf 和部分上游仓库。

## S100 试跑入口

[上机准备与启动说明](../docs/deployment/s100_livo_semantic_trial.md)：S100 跑 LIVO，配套电脑跑本地 Mask R-CNN/可选 SegFormer，发布 `/omni/semantic/boxes`、`people`、`ground`、`image` 和 `status`。语义节点采用最新帧队列和会话限制，不修改定位状态。

`ros2 launch omni_semantic_map trial.launch.py weights:=... output_directory:=... ground_model:=...`；权重与新结果目录必须显式给定。使用 `config/trial.rviz` 查看三维框和地面。S100 BPU 部署与实机验证尚未完成。在线节点识别 COCO 类别，模型配置见部署说明。

下文为车库离线建图流程；室外结果见[三维框与地面实验](../docs/experiments/boxes_ground_2026-11-06.md)。

## 输入与对象记录

`FAST_LIVO2/scripts/build_semantic_map.py` 接受冻结 LIVO 的原生关键帧、最终前端轨迹、全量离线回环位姿和原始 MCAP。按图像实际采集时刻插值前端位姿，投影变换为 `T_camera_imu × inverse(T_world_imu(image)) × T_world_imu(keyframe)`；校正图像使用该录包 CameraInfo。

默认使用 Torchvision Mask R-CNN ResNet50 FPN v2 COCO_V1，只接收 car 类；不把行人写入静态地图。掩码内点经深度、连通性和最小点数检查后生成观测。每个观测保留关键帧 ID、局部位置、图像时间、置信度和证据图。关联使用三维距离、掩码颜色分布、一对一分配和歧义拒绝。至少三个关键帧、四秒跨度且表面位置 P90 离散度不超过 0.8 m 才进入默认查询。

`confirmed` 表示达到上述重复观测门槛。遮挡和重复外观仍可能造成重复实例或错误合并，因此对象数不能直接作为车辆计数。查询支持类别与 ID。

## 构建与运行

和项目一起用 colcon 构建 `omni_semantic_map`。离线建图还依赖 FAST_LIVO2 实验环境中的 PyAV、Open3D、OpenCV、MCAP、Torch 与 Torchvision；运行时显式传入官方模型权重，不隐式下载。

```bash
python3 FAST_LIVO2/scripts/build_semantic_map.py \
  --inventory build/experiments/full_20261105/inventory.json \
  --bag rosbag2245 \
  --source build/experiments/z_20261105/si_full/rosbag2245/livo \
  --maps build/experiments/z_20261105/si_maps/rosbag2245 \
  --weights build/experiments/semantic_20261106/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth \
  --detector maskrcnn --output build/experiments/my_semantic_map

ros2 run omni_semantic_map semantic_map query \
  build/experiments/semantic_20261106/rosbag2245_final/objects.json 找车

ros2 run omni_semantic_map semantic_map query \
  build/experiments/semantic_20261106/rosbag2245_final/objects.json car_0001

ros2 run omni_semantic_map semantic_map reproject \
  build/experiments/my_semantic_map/objects.json \
  --poses /path/to/new/keyframe_poses.npz \
  --output build/experiments/my_semantic_map/objects_revision2.json
```

位姿更新必须属于相同关键帧归档，程序检查源索引哈希与时间戳。重算保留对象 ID 和关键帧局部观测，更新地图位置、离散度与查询资格；不会自动重做实例关联。更大的地图修正仍可能需要重新关联。

语义结果使用 `offline_livo_map` 坐标系，位姿文件哈希标识地图版本。接入在线导航前，需要将它与 `omni_map` 对齐。

[官方模型说明](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html) · [车库实验报告](../docs/experiments/livo_semantic_integration_2026-11-06.md)
