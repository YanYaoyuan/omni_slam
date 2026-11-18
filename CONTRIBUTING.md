# 开发与验证

## 目录与版本

- ROS 包保留原目录名、包名和消息类型，避免破坏部署脚本与消费者。
- 当前算法与研究边界见 `docs/research/README.md`。不要用历史分支替换 Ours 的名称。
- 试验输出放在 `build/`，报告交付放在 `output/`，数据与权重单独配置。
- 修改封存实验算法前，创建新运行根目录并记录新的源码/输入哈希。不要重新计算旧 seal 以掩盖实现变更。

## Python 与 CPU 内核

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/research-core.txt
python -m pip install -e ./omni_semantic_map
bash scripts/test_research_core.sh
python examples/revision_core_demo.py
```

图传播内核从包内 `.cpp` 源码编译到调用方缓存目录。修改打包配置时，需要实际构建 wheel 并在独立目录安装，确认原生内核源码没有遗漏。

完整研究回归：

```bash
PYTHONPATH=omni_semantic_map:scripts/benchmarks \
  CUDA_VISIBLE_DEVICES='' python3 -m pytest -q omni_semantic_map/test
```

历史分支所需模型框架和上游评测器见研究说明。不要把未安装可选依赖导致的跳过记作通过。

## ROS 构建与测试

先加载 ROS Humble 与所需 Omni 依赖 overlay，构建方式见根 README。构建后：

```bash
source install/livo_humble/setup.bash
colcon --log-base log/livo_humble_test test \
  --base-paths omni_slam_interfaces FAST_LIVO2 omni_semantic_map \
  --build-base build/livo_humble --install-base install/livo_humble \
  --packages-select fast_livo
colcon test-result --test-result-base build/livo_humble --verbose
```

现有 `.github/workflows/build-x86.yml`、`build-s100.yml`、`build-orin.yml` 维护原 LIO 部署链路，新 `research-core.yml` 单独验证无 GPU 的论文内核。LIVO 的主机 ROS 构建不等于通过 S100 原生编译、跨机 DDS 或实机验收。

## 提交前

检查 `git diff --check` 和暂存文件清单。提交源码、测试、必要配置、文档及当前论文图表。不要提交原始录包、地图点云、数据集、模型权重、虚拟环境、令牌、私钥或生成报告/视频。配置示例与现场标定分开保存。

已有上游版权和许可证不随整理删除。第三方模块来源见 `THIRD_PARTY.md`。
