# Omni SLAM GitHub Actions 跨平台流水线

本项目使用 GitHub Actions 构建 x86、Orin NX 和 RDK S100 三个平台产物。

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
- `OMNI_ENABLE_S100_BUILD=true`：启用 S100 交叉编译。

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

- TogetheROS 交叉编译镜像：`pc_tros_ubuntu22.04:v1.0.0`
- 官方镜像包：`http://archive.d-robotics.cc/TogetheROS/cross_compile_docker/pc_tros_ubuntu22.04_v1.0.0.tar.gz`
- 官方镜像包 SHA-256：`ec065c268782373311421f0bb7f2212bfec6584a7070a9737468055a6bce2c88`
- `D-Robotics/robot_dev_config`：`f44b1ad2575b34c189fd470de1db9c5ad48b6886`
- `D-Robotics/sysroot_docker`：`de9fa286f71a72d24c349477cd59f41cc2cc3d8f`
- 官方入口：`bash robot_dev_config/build.sh -p S100`

这与 OpenExplorer 的 `ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0` 是两套不同的工具链。后者用于算法模型转换和 BPU 开发，不是 `robot_dev_config` 文档指定的 ROS C++ 交叉编译环境。官方 TROS 镜像包含 x86_64 主机侧的 ROS 2 Humble、colcon 和 AArch64 编译器，但不包含完整的 S100 目标库。

`robot_dev_config/build.sh -p S100` 会把 `cc_ws/sysroot_docker/usr_s100` 链接为目标 `/usr`，CMake 工具链也把 `cc_ws/sysroot_docker` 设为 `CMAKE_SYSROOT`，因此 sysroot 不能去掉。脚本从官方 `sysroot_docker` 仓库获取并锁定提交；本地模式支持断点续传。项目需要但基础环境缺少的 Jammy ARM64 Qt/VTK 运行库会补入 `usr_s100`，`pcl_ros` 和 `pcl_conversions` 的 ARM64 包会补入容器内的目标 TROS，且不会将 x86_64 宿主库混入目标环境。构建后还会检查 FAST-LIO/ICP 可执行文件及其 ARM64 架构。

### S100 本地验证

本地和 GitHub Actions 共用 `scripts/build_s100_cross.sh`。在当前工程中执行：

```bash
# 只检查 /data 空间、rootless Docker 和必要命令
./scripts/test_s100_local.sh --check-only

# 下载官方 TROS 镜像并执行完整交叉编译
./scripts/test_s100_local.sh
```

官方 TROS 镜像包公开下载，不需要 registry 账号或 GitHub Secrets。首次运行后会缓存在 `/data/docker-s100/images`。本地入口启动独立的 rootless Docker，默认使用：

- Docker 数据：`/data/docker-s100`
- Docker 客户端配置：`/data/docker-s100/config`
- 源码/sysroot/colcon 工作区：`/data/omni-s100-local`
- colcon 安装树：`/data/omni-s100-local/workspace/cc_ws/tros_ws/install`
- 可部署应用包：`/data/omni-s100-local/workspace/cc_ws/tros_ws/runtime`

该 Docker daemon 不读写系统 Docker 的 `/var/lib/docker`，也不会清理其他项目的镜像或容器。至少需要 50 GiB 可用空间。

S100 的 tar.gz 解压后直接得到应用前缀：`bin/` 是执行入口，`config/` 是运行配置，`lib/` 包含应用库以及递归解析得到的 ARM64 ROS/PCL/VTK 等动态依赖，`share/` 保留 ROS ament 索引、launch、rviz 和消息接口。部署到板端后先执行 `source setup.bash`。glibc 等 RDK OS 基础库不会打进包内，以免覆盖板端系统 ABI。S100 Artifact 暂不生成单独的 `.sha256` 文件。

## 缓存和存储策略

- 不使用 `actions/cache`。
- 每次构建前、后清理 colcon 与 Docker 构建缓存。
- 普通运行产物保留 3 天，失败日志保留 2 天。
- tag Release 只保留最终运行包，不上传 build/log 中间目录；S100 当前只上传 tar.gz，不附带独立 SHA-256 文件。

## 发布

准备好两个平台开关和 Runner 后：

```bash
git tag v0.1.0
git push origin v0.1.0
```

`release.yml` 只有在 x86、Orin NX、S100 三个平台全部成功后才创建不可变版本资产；存在校验和文件的平台仍会执行校验。
