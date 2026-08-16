# Omni SLAM GitHub Actions 跨平台流水线

本流水线不调整现有 ROS 包与源码目录，只增加 GitHub Actions 和 CI 辅助文件。

## 工作流

| 工作流 | 触发条件 | 构建环境 | 产物 |
|---|---|---|---|
| `build-x86.yml` | PR、main、手动、release 调用 | GitHub Ubuntu 22.04 + ROS 2 Humble | x86_64 tar.gz |
| `build-orin.yml` | main、手动、release 调用 | 自托管 Orin NX + jetson-containers | Orin aarch64 tar.gz |
| `build-s100.yml` | main、手动、release 调用 | GitHub x86 + D-Robotics 官方容器与 robot_dev_config | S100 aarch64 tar.gz |
| `release.yml` | `v*` tag、手动 dry-run | 聚合三平台 | GitHub Release |

PR 只执行 x86 构建和测试，不会把不可信 PR 代码发送到 Orin 自托管 Runner。

## Repository variables

在 GitHub 仓库 `Settings -> Secrets and variables -> Actions -> Variables` 配置：

- `OMNI_ENABLE_ORIN_RUNNER=true`：已有标签为 `self-hosted,linux,arm64,orin` 的可信 Orin Runner 后再打开。
- `OMNI_ENABLE_S100_BUILD=true`：确认官方镜像包含匹配 RDK OS 4/TROS Humble 的 S100 sysroot 后再打开。

在 `Actions -> Secrets` 配置：

- `S100_SYSROOT_URL`：与 RDK OS 4.0.x / TROS Humble 匹配的官方 sysroot `tar.gz` 下载地址。建议使用短期签名 URL，不要把 SDK/sysroot 提交到 Git。

未配置时对应平台明确显示为 skipped；正式 `v*` release 会在 preflight 失败，不会发布缺平台的假完整版本。

## Orin NX Runner

1. 在 Orin NX 上安装 JetPack、Docker、NVIDIA Container Runtime 和 GitHub Actions Runner。
2. 给 Runner 增加 `orin` 标签。
3. 确保 Runner 账号可执行 Docker。
4. 流水线固定使用 `dusty-nv/jetson-containers` 提交
   `70c149aea6126594153ef7690ec4890f8f396518`，由 `autotag` 选择与本机 JetPack/L4T 匹配的 `ros:humble-ros-base`。

Orin 流水线在构建前后执行 Docker builder/image prune，并清除 colcon 输出。该 Runner 不应与其他需要长期保留本地 Docker 镜像的项目共用。

## RDK S100

流水线锁定：

- CPU 工具链镜像：`registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0`
- `D-Robotics/robot_dev_config`：`f44b1ad2575b34c189fd470de1db9c5ad48b6886`
- 官方入口：`bash robot_dev_config/build.sh -p S100`

注意：OpenExplorer 算法工具链镜像与 TROS 交叉编译 sysroot 必须版本匹配。首次启用前应在手动 workflow_dispatch 中验证镜像内存在 `s100_build.sh`、`aarch64_toolchainfile.cmake` 以及构建需要的 sysroot。缺失时流水线会失败，不会退回本机编译。

`robot_dev_config` 要求 sysroot 位于工作区的 `cc_ws/sysroot_docker`。流水线通过 `S100_SYSROOT_URL` 下载并解压到该目录，然后才执行官方 `build.sh -p S100`。缺少 sysroot 或目录结构不正确时会直接失败，避免用错误宿主库生成不可运行产物。

## 缓存和存储策略

- 不使用 `actions/cache`。
- 每次构建前、后清理 colcon 与 Docker 构建缓存。
- 普通运行产物保留 3 天，失败日志保留 2 天。
- tag Release 只保留最终 tar.gz 与 SHA-256，不上传 build/log 中间目录。

## 发布

准备好两个平台开关和 Runner 后：

```bash
git tag v0.1.0
git push origin v0.1.0
```

`release.yml` 只有在 x86、Orin NX、S100 三个平台全部成功且校验和通过后才创建不可变版本资产。
