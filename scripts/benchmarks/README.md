# Benchmark 执行器索引

本目录保存按实验日期和假设演进的执行器。当前主线见 [研究说明](../../docs/research/README.md)，不要按文件名相似程度选择“最新算法”。大部分脚本绑定封存协议、缓存和输出哈希，历史文件名不是通用配置接口。

## 当前主线

- `run_revision_assessment.py`：两个预固定新窗口的完整修订验证。
- `run_revision_authorization.py`：历史身份授权与局部写入的核心实验驱动。
- `review_causal_revision.py`：固定地图输出的机制干预。
- `run_reviewer_extension.py`：四窗口扩展，内部机制对照与外部核心适配。
- `run_revision_robot.py`：机器人车库的同版本离线验证。
- `evaluate_kitti360_development.py`：本地评价适配器。

## 历史与辅助

- `run_visibility_cleanup.py`、`visibility_independent_*`：早期固定身份 V1。
- `run_public_history_graph.py`、共享前端相关脚本：对象图强基线研究。
- `train_*memory*`、`run_vggt_*`、DINO 相关脚本：独立研究分支。
- `prepare_*`、`verify_*`、`audit_*`、`build_*report*`：数据准备、封存检查、诊断与报告。

保留原路径和源码是为了核对已运行实验的哈希。新环境首先运行仓库根目录的核心演示与测试，再按外部输入准备情况适配完整执行器。`build/`、`docs/reports/` 中的缓存和生成报告不随 Git 提交。
