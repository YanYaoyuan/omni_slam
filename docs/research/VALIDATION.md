# 源码整理验证（2026-11-18）

本轮验证源码安装、现有回归测试和 ROS 2 Humble 构建，不重新运行 benchmark，不改变已封存的算法结果。

| 检查 | 结果 |
| --- | --- |
| 完整语义模块测试 | 402 passed，15 skipped |
| Manager 回归（加载本机 ROS 与相邻项目 overlay） | 76 passed |
| LIVO Python 测试 | 13 passed |
| 独立 CPU 虚拟环境核心测试 | 24 passed；这些用例已包含在完整语义测试中，不重复计入总数 |
| wheel 安装后的合成演示 | 通过，安装包包含三个原生 C++ 内核源码，运行时编译及调用成功 |
| ROS 2 Humble 构建 | `omni_slam_interfaces`、`fast_livo`、`omni_semantic_map` 三包通过；Release，Livox 关闭 |
| LIVO 原生回归 | 14 个 GoogleTest 用例通过；colcon 计入 CTest 包装后报告 15 tests、0 failures |
| 论文源码清单 | 117 个文件校验通过，25 处活动图表引用可解析 |
| 试跑源码包 | 428 个源文件加 manifest；排除安装元数据、缓存、未使用的 ROS 1 / Python 2 包装层 |

完整语义测试中的跳过项涉及可选依赖或外部输入，不代表相应完整系统已验证。系统 Python 存在 SciPy / NumPy 版本兼容警告，独立 CPU 环境使用 NumPy 1.26.4、SciPy 1.15.3 并通过核心测试；ROS 构建保留上游编译警告。

此次构建环境为本机 x86 Ubuntu / Humble，不构成 S100 实机验证、在线回环验证或真实机器人语义准确率评测。GitHub Actions 的运行状态以远端工作流为准。

命令入口见 [开发指南](../../CONTRIBUTING.md)，研究范围见 [研究说明](README.md)。本机日志存于被 Git 忽略的 `build/repository_cleanup_20261118/`。
