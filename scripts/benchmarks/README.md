# 评测脚本

当前对象地图修订实验使用以下入口。运行条件与数据准备见[研究说明](../../docs/research/README.md)。

| 脚本 | 用途 |
| --- | --- |
| `run_revision_authorization.py` | 开发窗口上的历史合并判断与局部点分配 |
| `run_revision_assessment.py` | 两个预先选定的新窗口评测 |
| `review_causal_revision.py` | 固定地图输出的机制消融 |
| `run_reviewer_extension.py` | 四窗口扩展、内部对照与外部方法核心适配 |
| `run_revision_robot.py` | 车库录包离线评测 |
| `evaluate_kitti360_development.py` | KITTI-360 本地评分适配 |

这些脚本读取对应实验的协议、缓存和文件哈希，部分路径指向原运行环境。先通过 `--help` 查看参数，再准备输入；新数据或新配置使用独立输出目录。`build/` 和 `docs/reports/` 中的缓存、生成报告不随 Git 分发。

## 其他实验

- `run_visibility_cleanup.py`、`visibility_independent_*`：固定身份的 V1 几何修订。
- `run_public_history_graph.py` 及共享前端脚本：对象图基线对比。
- `train_*memory*`、`run_vggt_*` 及 DINO 脚本：学习式记忆与视觉特征实验。
- `prepare_*`、`verify_*`、`audit_*`、`build_*report*`：数据准备、完整性检查、诊断和报告生成。

历史脚本保留原版本，用于核对已有结果。
