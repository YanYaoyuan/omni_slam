# 开发指南

## 修改与实验记录

ROS 包名、消息类型和目录被部署脚本引用，调整时需同步调用方。当前方法及实验入口见[研究说明](docs/research/README.md)。

实验输出放在 `build/`，交付文件放在 `output/`，数据与权重路径通过配置指定。修改算法或输入后，新建运行目录并记录代码、配置和输入哈希；已有实验的 seal 与结果保留原样。

## Python 与 CPU 内核

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/research-core.txt
python -m pip install -e ./omni_semantic_map
bash scripts/test_research_core.sh
python examples/revision_core_demo.py
```

图传播内核从包内 `.cpp` 源码编译到调用方缓存目录。修改打包配置后，构建 wheel 并在独立目录安装，检查原生内核源码是否随包分发。

完整研究测试：

```bash
PYTHONPATH=omni_semantic_map:scripts/benchmarks \
  CUDA_VISIBLE_DEVICES='' python3 -m pytest -q omni_semantic_map/test
```

可选依赖见[研究说明](docs/research/README.md#测试)。记录测试结果时，同时注明通过、失败和跳过数量。

## ROS 构建与测试

加载 ROS Humble 与所需 Omni 依赖 overlay，按[根目录说明](README.md#快速开始)构建后执行：

```bash
source install/livo_humble/setup.bash
colcon --log-base log/livo_humble_test test \
  --base-paths omni_slam_interfaces FAST_LIVO2 omni_semantic_map \
  --build-base build/livo_humble --install-base install/livo_humble \
  --packages-select fast_livo
colcon test-result --test-result-base build/livo_humble --verbose
```

CI 中的 `build-x86.yml`、`build-s100.yml` 和 `build-orin.yml` 检查 LIO 部署构建；`research-core.yml` 检查 CPU 修订内核。LIVO 的 S100 原生构建、跨机 DDS 和实机运行需要另行验证。

## 提交

提交前运行 `git diff --check`，检查暂存文件。仓库接收源码、测试、配置、文档和论文图表；原始录包、点云、数据集、权重、虚拟环境和完整视频放在仓库外。现场标定与示例配置分开保存，凭据不写入代码或配置示例。

保留上游版权与许可证，来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
