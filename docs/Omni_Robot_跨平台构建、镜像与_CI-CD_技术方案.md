# Omni Robot 跨平台构建、镜像与 CI/CD 技术方案

> **文档版本**：v1.0  
> **适用平台**：x86_64 / NVIDIA Jetson Orin NX / D-Robotics RDK S100  
> **操作系统基线**：Ubuntu 22.04  
> **机器人框架**：ROS2 Humble / TogetheROS Humble  
> **CI/CD**：GitHub Actions  
> **容器仓库**：GitHub Container Registry（GHCR）  
> **制品管理**：GitHub Releases / GitHub Actions Artifacts

---

# 1. 文档目的

Omni Robot 的机器人软件需要同时运行于三类计算平台：

- x86_64 Ubuntu 开发机、服务器或工控机
- NVIDIA Jetson Orin NX
- D-Robotics RDK S100

三个平台虽然都可以运行 Linux 和 ROS2，但底层硬件环境存在明显差异。

| 平台 | CPU 架构 | AI 加速 | ROS 环境 | 推荐构建方式 |
|---|---|---|---|---|
| x86 | amd64 | CPU / NVIDIA GPU | ROS2 Humble | GitHub Hosted Runner |
| Orin NX | arm64 | CUDA / TensorRT | ROS2 Humble | Orin Self-hosted Runner |
| S100 | arm64 | BPU | TogetheROS / ROS2 Humble | x86 Docker + S100 Toolchain |

本方案希望建立统一的软件工程体系：

```text
代码提交
   ↓
自动测试
   ↓
多平台构建
   ↓
Docker / 二进制打包
   ↓
版本发布
   ↓
机器人部署
   ↓
运行验证
   ↓
失败回滚
```

最终开发人员只需要：

```bash
git push
```

即可完成持续集成。

正式发布时：

```bash
git tag v1.2.0
git push origin v1.2.0
```

即可自动生成三个平台的发布物。

---

# 2. 总体架构

```text
                         Developer
                             │
                          git push
                             │
                             ▼
                         GitHub Repo
                             │
                             ▼
                       GitHub Actions
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             ▼               ▼                ▼
        x86 Pipeline    Orin Pipeline     S100 Pipeline
             │               │                │
     GitHub Runner      Orin NX Runner     GitHub Runner
     ubuntu-22.04       self-hosted        ubuntu-22.04
             │               │                │
        ROS2 Humble     JetPack/CUDA      S100 Toolchain
             │          TensorRT/ROS2      Sysroot/TROS
             │               │                │
             ▼               ▼                ▼
       x86_64 Build      arm64 Build        arm64 Build
             │               │                │
             ├───────┐       │       ┌────────┘
             │       │       │       │
             ▼       ▼       ▼       ▼
          Docker Images       Binary Packages
             │                      │
             ▼                      ▼
            GHCR               GitHub Release
             │                      │
             └──────────┬───────────┘
                        ▼
                    Deployment
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
          x86          Orin         S100
```

GitHub 官方支持通过 self-hosted runner 标签选择特定机器，因此 Orin 可以直接作为 ARM64 GitHub Actions Runner，而无需 QEMU 模拟整个 Jetson 环境。

---

# 3. 核心设计原则

## 3.1 一套业务代码

禁止维护三套几乎一致的软件：

```text
omni_robot_x86
omni_robot_orin
omni_robot_s100
```

统一维护：

```text
omni_robot
```

只隔离硬件相关代码。

例如：

```text
src/
├── common/
├── navigation/
├── perception/
├── inspection/
├── video/
└── platform/
    ├── x86/
    ├── orin/
    └── s100/
```

---

## 3.2 平台差异只存在于 HAL 层

例如：

```cpp
class VideoDecoder
{
public:
    virtual bool init() = 0;
    virtual bool decode(const uint8_t* data, size_t size) = 0;
};
```

然后：

```text
x86
 └─ FFmpeg / NVDEC

Orin
 └─ Jetson Multimedia / NVDEC / GStreamer

S100
 └─ D-Robotics Multimedia / H265 Hardware Decoder
```

上层模块只依赖：

```text
VideoDecoder
```

而不直接依赖具体硬件 API。

---

# 4. 推荐仓库结构

```text
omni_robot/
│
├── README.md
├── LICENSE
├── VERSION
│
├── src/
│   ├── common/
│   ├── ros2/
│   ├── navigation/
│   ├── slam/
│   ├── docking/
│   ├── perception/
│   ├── inspection/
│   ├── video/
│   ├── network/
│   └── platform/
│       ├── x86/
│       ├── orin/
│       └── s100/
│
├── config/
│   ├── common/
│   ├── x86/
│   ├── orin/
│   └── s100/
│
├── scripts/
│   ├── build.sh
│   ├── package.sh
│   ├── deploy.sh
│   ├── healthcheck.sh
│   └── rollback.sh
│
├── docker/
│   ├── x86/
│   │   └── Dockerfile
│   ├── orin/
│   │   └── Dockerfile
│   └── s100/
│       └── Dockerfile.builder
│
├── toolchains/
│   └── s100/
│
├── tests/
│
├── deploy/
│   ├── systemd/
│   ├── compose/
│   └── config/
│
└── .github/
    └── workflows/
        ├── ci.yml
        ├── build-x86.yml
        ├── build-orin.yml
        ├── build-s100.yml
        └── release.yml
```

---

# 5. 三个平台的构建策略

## 5.1 x86

x86 用于：

- 日常开发
- 仿真
- RViz
- Foxglove
- rosbag
- 单元测试
- 算法验证
- 后端服务
- 非硬件依赖模块

推荐环境：

```text
Ubuntu 22.04
ROS2 Humble
Docker
amd64
```

使用：

```yaml
runs-on: ubuntu-22.04
```

直接由 GitHub Hosted Runner 构建。

---

# 6. NVIDIA Orin NX

Orin 是最不应该简单做普通 ARM 交叉编译的平台。

Orin 环境通常包含：

```text
JetPack
CUDA
cuDNN
TensorRT
NVIDIA Driver
GStreamer
OpenCV CUDA
ROS2 Humble
```

因此推荐：

```text
Orin NX 真机
      │
GitHub Self-hosted Runner
      │
Native ARM64 Build
```

而不是：

```text
x86
 ↓
QEMU
 ↓
模拟 ARM
 ↓
模拟 Jetson
```

Docker 官方也指出，对于复杂 multi-platform 构建，可以使用多个 native build nodes，而不是依赖 QEMU。

---

# 7. S100

S100 和 Orin 又不同。

截至当前 RDK S100 官方资料，S100 最新公开系统软件资源已经进入 4.0.5 系列，并提供对应 BSP、算法工具链和官方 Docker 工具链。

官方 OpenExplorer 3.7.0 提供了：

```text
ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0
ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0
```

可作为 S100 AI 编译环境。

同时，D-Robotics 官方维护：

```text
D-Robotics/robot_dev_config
```

作为 TogetheROS.Bot / ROS2 构建入口。

因此推荐：

```text
GitHub x86 Runner
       │
       ▼
S100 Builder Docker
       │
       ├─ D-Robotics Toolchain
       ├─ S100 Sysroot
       ├─ TROS
       └─ BPU SDK
       │
       ▼
Cross Compile
       │
       ▼
S100 ARM64 Artifact
```

---

# 8. 为什么 S100 不强制 Runtime Docker

S100 上很多功能依赖：

```text
BPU
ISP
Camera
H265 Decoder
Hardware Encoder
MIPI
Multimedia
Device Nodes
Kernel Driver
RDK OS Service
```

因此推荐：

```text
RDK OS
  │
  ├─ 原厂驱动
  ├─ Multimedia
  ├─ BPU
  ├─ Camera
  │
  └─ Omni Robot Application
```

而不是重新封装整个操作系统。

建议 S100 CI 输出：

```text
omni-robot-s100-arm64.tar.gz
```

然后部署到板端。

---

# 9. Docker 镜像规划

建议 GHCR 中存在三个核心镜像：

```text
ghcr.io/<organization>/omni-x86-base
ghcr.io/<organization>/omni-orin-base
ghcr.io/<organization>/omni-s100-builder
```

其中：

```text
omni-x86-base
```

用于 x86 开发和运行。

```text
omni-orin-base
```

用于 Orin 编译和运行。

```text
omni-s100-builder
```

只负责 S100 编译。

---

# 10. x86 Base Image

```dockerfile
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
```

进一步可以安装：

```text
PCL
OpenCV
Eigen
GStreamer
Livox SDK
Foxglove dependencies
Navigation dependencies
```

---

# 11. Orin Base Image

Orin 建议以 JetPack 对应的 NVIDIA/Jetson 环境为基础。

软件层：

```text
JetPack
  │
  ├─ CUDA
  ├─ cuDNN
  ├─ TensorRT
  ├─ GStreamer
  │
  ▼
ROS2 Humble
  │
  ▼
OpenCV / PCL
  │
  ▼
Livox SDK / Navigation / SLAM
  │
  ▼
Omni Robot
```

建议镜像：

```text
omni-orin-base
```

只在基础依赖发生变化时重新构建。

业务代码镜像：

```text
omni-robot-orin
```

则每次发布构建。

---

# 12. S100 Builder Image

S100 推荐封装：

```text
Ubuntu 22.04
+
S100 Toolchain
+
S100 Sysroot
+
TROS
+
OpenExplorer
+
CMake
+
colcon
```

生成：

```text
omni-s100-builder
```

CI 中：

```bash
docker run --rm \
    -v $(pwd):/workspace \
    omni-s100-builder \
    bash /workspace/scripts/build.sh s100
```

---

# 13. CMake 平台配置

统一增加：

```text
OMNI_PLATFORM
```

支持：

```text
X86
ORIN
S100
```

例如：

```cmake
set(OMNI_PLATFORM "X86" CACHE STRING "Target platform")

if(OMNI_PLATFORM STREQUAL "ORIN")
    add_compile_definitions(OMNI_PLATFORM_ORIN)

elseif(OMNI_PLATFORM STREQUAL "S100")
    add_compile_definitions(OMNI_PLATFORM_S100)

elseif(OMNI_PLATFORM STREQUAL "X86")
    add_compile_definitions(OMNI_PLATFORM_X86)

else()
    message(FATAL_ERROR "Unsupported OMNI_PLATFORM")
endif()
```

使用：

```bash
colcon build \
    --cmake-args \
    -DOMNI_PLATFORM=ORIN
```

---

# 14. 统一 build.sh

```bash
#!/usr/bin/env bash

set -euo pipefail

PLATFORM=${1:-}

case "${PLATFORM}" in

x86)

    source /opt/ros/humble/setup.bash

    colcon build \
        --merge-install \
        --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DOMNI_PLATFORM=X86

    ;;

orin)

    source /opt/ros/humble/setup.bash

    colcon build \
        --merge-install \
        --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DOMNI_PLATFORM=ORIN

    ;;

s100)

    source /opt/tros/humble/setup.bash || true

    colcon build \
        --merge-install \
        --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DOMNI_PLATFORM=S100 \
        -DCMAKE_TOOLCHAIN_FILE=/opt/s100/toolchain.cmake

    ;;

*)

    echo "Usage:"
    echo "./scripts/build.sh x86"
    echo "./scripts/build.sh orin"
    echo "./scripts/build.sh s100"

    exit 1
    ;;

esac
```

注意：

```text
/opt/tros/humble/setup.bash
/opt/s100/toolchain.cmake
```

应根据实际 S100 SDK 调整。

---

# 15. GitHub Actions 总体流程

建议拆成：

```text
ci.yml
     │
     ├─ lint
     ├─ unit-test
     └─ x86 build

build-x86.yml

build-orin.yml

build-s100.yml

release.yml
```

不要一开始做成一个巨大的 YAML。

---

# 16. 普通开发流程

开发人员：

```bash
git checkout -b feature/docking
```

开发：

```bash
git add .
git commit -m "feat: add docking controller"
git push
```

提交 PR：

```text
feature/docking
       │
       ▼
Pull Request
       │
       ▼
GitHub Actions
       │
       ├─ x86 compile
       ├─ unit test
       └─ static check
```

PR 阶段暂时不需要每次都构建 Orin 和 S100。

否则构建成本过高。

---

# 17. main 分支流程

合入：

```text
main
```

后：

```text
GitHub Actions
      │
      ├─ x86
      ├─ Orin
      └─ S100
```

三平台全部构建。

---

# 18. CI Workflow

`.github/workflows/ci.yml`

```yaml
name: CI

on:
  pull_request:
  push:
    branches:
      - main

jobs:

  x86:

    runs-on: ubuntu-22.04

    steps:

      - name: Checkout
        uses: actions/checkout@v4

      - name: Build
        run: |
          ./scripts/build.sh x86

      - name: Test
        run: |
          colcon test

      - name: Test Result
        run: |
          colcon test-result --verbose
```

---

# 19. x86 Pipeline

```yaml
name: Build x86

on:
  push:
    branches:
      - main

  workflow_dispatch:

jobs:

  build:

    runs-on: ubuntu-22.04

    steps:

      - name: Checkout
        uses: actions/checkout@v4

      - name: Build
        run: |
          ./scripts/build.sh x86

      - name: Package
        run: |
          tar czf omni-robot-x86_64.tar.gz install/

      - name: Upload
        uses: actions/upload-artifact@v4
        with:
          name: omni-robot-x86
          path: omni-robot-x86_64.tar.gz
```

---

# 20. Orin GitHub Runner

Orin 上注册 GitHub Self-hosted Runner。

Runner 标签建议：

```text
self-hosted
linux
arm64
orin
```

GitHub Actions 可以通过这些标签将任务发送到指定的 ARM64 Orin。

建议将 Runner 配置为 systemd 服务，这也是 GitHub 官方支持的运行方式。

最终：

```text
Orin Boot
   │
   ▼
systemd
   │
   ▼
GitHub Runner
   │
   ▼
等待 Job
```

---

# 21. Orin Pipeline

```yaml
name: Build Orin

on:

  push:
    branches:
      - main

  workflow_dispatch:

jobs:

  build:

    runs-on:
      - self-hosted
      - linux
      - arm64
      - orin

    steps:

      - name: Checkout
        uses: actions/checkout@v4

      - name: Build
        shell: bash
        run: |
          source /opt/ros/humble/setup.bash
          ./scripts/build.sh orin

      - name: Package
        run: |
          tar czf omni-robot-orin-arm64.tar.gz install/

      - name: Upload
        uses: actions/upload-artifact@v4
        with:
          name: omni-robot-orin
          path: omni-robot-orin-arm64.tar.gz
```

---

# 22. Orin Runner 安全注意事项

非常重要：

Self-hosted Runner 本质上可以执行仓库中的代码。

因此：

**不要允许不可信 PR 直接运行 Orin Self-hosted Runner。**

建议：

```text
PR
 ↓
GitHub x86 Runner

main
 ↓
Orin Runner
```

即：

```text
External PR
    ✕
 Orin Runner
```

只允许：

```text
trusted main
trusted tag
manual workflow
```

使用 Orin。

---

# 23. S100 Pipeline

```yaml
name: Build S100

on:

  push:
    branches:
      - main

  workflow_dispatch:

jobs:

  build:

    runs-on: ubuntu-22.04

    steps:

      - name: Checkout
        uses: actions/checkout@v4

      - name: Build Builder
        run: |
          docker build \
            -t omni-s100-builder \
            docker/s100

      - name: Cross Compile
        run: |
          docker run \
            --rm \
            -v "${{ github.workspace }}:/workspace" \
            omni-s100-builder \
            bash /workspace/scripts/build.sh s100

      - name: Package
        run: |
          tar czf omni-robot-s100-arm64.tar.gz install/

      - name: Upload
        uses: actions/upload-artifact@v4
        with:
          name: omni-robot-s100
          path: omni-robot-s100-arm64.tar.gz
```

---

# 24. Builder 镜像不要每次重新构建

上面的写法适合第一版。

正式上线后应该提前构建：

```text
ghcr.io/omni-ai/omni-s100-builder:4.0.5
```

Pipeline 直接：

```bash
docker pull ghcr.io/omni-ai/omni-s100-builder:4.0.5
```

然后：

```bash
docker run ...
```

避免：

```text
每次 CI
 ↓
重新下载 S100 SDK
 ↓
重新安装工具链
```

---

# 25. GHCR

GitHub Container Registry 使用：

```text
ghcr.io
```

GitHub 官方支持通过 GitHub Actions 使用 `GITHUB_TOKEN` 发布容器镜像。

推荐：

```text
ghcr.io/omni-ai/omni-x86-base
ghcr.io/omni-ai/omni-orin-base
ghcr.io/omni-ai/omni-s100-builder

ghcr.io/omni-ai/omni-robot-x86
ghcr.io/omni-ai/omni-robot-orin
```

---

# 26. GHCR 登录

Workflow：

```yaml
permissions:
  contents: read
  packages: write
```

然后：

```yaml
- name: Login GHCR
  uses: docker/login-action@v4
  with:
    registry: ghcr.io
    username: ${{ github.actor }}
    password: ${{ secrets.GITHUB_TOKEN }}
```

GitHub Container Registry 支持 Docker / OCI images。

---

# 27. Docker Buildx

推荐：

```yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v4
```

然后：

```yaml
- name: Build and Push
  uses: docker/build-push-action@v7
  with:
    context: .
    push: true
    tags: ghcr.io/omni-ai/omni-robot-x86:${{ github.sha }}
```

Docker 官方目前提供专门的 GitHub Actions 集成用于 BuildKit / Buildx 和多平台镜像构建。

---

# 28. Docker Cache

推荐：

```yaml
cache-from: type=gha
cache-to: type=gha,mode=max
```

例如：

```yaml
- name: Build
  uses: docker/build-push-action@v7

  with:

    context: .

    push: true

    tags: |
      ghcr.io/omni-ai/omni-robot-x86:latest

    cache-from: type=gha

    cache-to: type=gha,mode=max
```

Docker 官方支持 BuildKit GitHub Actions cache backend。

---

# 29. 镜像版本规范

禁止只有：

```text
latest
```

推荐同时产生：

```text
latest
v1.2.0
v1.2
sha-a28392f
```

例如：

```text
ghcr.io/omni-ai/omni-robot-orin:v1.2.0
ghcr.io/omni-ai/omni-robot-orin:v1.2
ghcr.io/omni-ai/omni-robot-orin:latest
ghcr.io/omni-ai/omni-robot-orin:sha-a28392f
```

---

# 30. 软件版本规范

建议：

```text
MAJOR.MINOR.PATCH
```

例如：

```text
1.2.3
```

解释：

```text
1
│
Breaking Changes

2
│
Feature

3
│
Bugfix
```

---

# 31. Git Tag 发布

正式版本：

```bash
git tag v1.2.0
git push origin v1.2.0
```

触发：

```yaml
on:

  push:

    tags:

      - "v*"
```

---

# 32. Release Pipeline

最终希望：

```text
Git Tag v1.2.0
       │
       ▼
GitHub Actions
       │
       ├─ x86 build
       ├─ Orin build
       ├─ S100 build
       └─ test
       │
       ▼
Release
       │
       ├─ omni-robot-x86_64-v1.2.0.tar.gz
       ├─ omni-robot-orin-arm64-v1.2.0.tar.gz
       └─ omni-robot-s100-arm64-v1.2.0.tar.gz
```

同时：

```text
GHCR

omni-robot-x86:v1.2.0
omni-robot-orin:v1.2.0
```

---

# 33. 发布物目录

建议安装包：

```text
omni-robot-v1.2.0/
│
├── bin/
├── lib/
├── share/
├── config/
├── scripts/
├── VERSION
└── manifest.json
```

---

# 34. manifest.json

例如：

```json
{
  "project": "omni-robot",
  "version": "1.2.0",
  "platform": "orin",
  "arch": "arm64",
  "ros": "humble",
  "commit": "a9281fc",
  "build_type": "release"
}
```

S100：

```json
{
  "project": "omni-robot",
  "version": "1.2.0",
  "platform": "s100",
  "arch": "arm64",
  "ros": "humble",
  "rdk": "4.0.5"
}
```

---

# 35. 平台版本锁定

这是整个方案里非常重要的一点。

不要写：

```text
Orin = ARM64
```

而应该记录：

```text
Orin
├─ JetPack
├─ L4T
├─ CUDA
├─ TensorRT
└─ ROS2
```

S100：

```text
S100
├─ RDK OS
├─ BSP
├─ TROS
├─ OpenExplorer
└─ BPU Runtime
```

否则未来会出现：

```text
编译成功
  ↓
拷贝上板
  ↓
GLIBC / CUDA / TensorRT / BPU ABI 不兼容
```

---

# 36. env.lock

建议仓库增加：

```text
env/
├── x86.lock
├── orin.lock
└── s100.lock
```

例如：

```text
# orin.lock

UBUNTU=22.04
ROS_DISTRO=humble

JETPACK=6.x
CUDA=12.x
TENSORRT=10.x

ARCH=arm64
```

S100：

```text
RDK_OS=4.0.5
OPENEXPLORER=3.7.0
ROS_DISTRO=humble
ARCH=arm64
```

当前 S100 官方 4.0.5 系统软件和 OpenExplorer 3.7.0 工具链属于匹配关系之一，因此锁版本比长期使用 `latest` 更安全。

---

# 37. 发布流程

完整流程：

```text
Developer
    │
    ▼
feature branch
    │
    ▼
Pull Request
    │
    ▼
x86 compile + unit tests
    │
    ▼
Merge
    │
    ▼
main
    │
    ├───────────┬───────────┐
    ▼           ▼           ▼
   x86         Orin        S100
    │           │           │
    └───────────┼───────────┘
                ▼
         Integration Test
                │
                ▼
             Git Tag
                │
                ▼
            Production
```

---

# 38. 部署设计

建议统一：

```bash
./scripts/deploy.sh
```

用法：

```bash
./scripts/deploy.sh orin robot@192.168.1.20
```

或者：

```bash
./scripts/deploy.sh s100 root@192.168.1.30
```

---

# 39. Orin 部署

推荐 Docker：

```text
Orin
 │
 ├─ JetPack Host
 │
 ├─ NVIDIA Runtime
 │
 └─ Docker
      │
      ▼
 omni-robot-orin
```

例如：

```bash
docker pull ghcr.io/omni-ai/omni-robot-orin:v1.2.0
```

运行：

```bash
docker run \
    --runtime nvidia \
    --network host \
    --ipc host \
    --privileged \
    --restart unless-stopped \
    --name omni-robot \
    ghcr.io/omni-ai/omni-robot-orin:v1.2.0
```

实际部署时应尽量减少 `--privileged`，改为明确映射需要使用的：

```text
/dev
USB
serial
camera
lidar
```

设备。

---

# 40. S100 部署

S100 推荐：

```text
RDK OS
  │
  ▼
/opt/omni/
```

目录：

```text
/opt/omni/
├── current
├── releases/
│   ├── v1.1.0
│   └── v1.2.0
└── config/
```

部署：

```bash
scp omni-robot-s100-arm64-v1.2.0.tar.gz root@s100:/tmp/
```

设备端：

```bash
mkdir -p /opt/omni/releases/v1.2.0

tar xf /tmp/omni-robot-s100-arm64-v1.2.0.tar.gz \
    -C /opt/omni/releases/v1.2.0
```

切换：

```bash
ln -sfn \
    /opt/omni/releases/v1.2.0 \
    /opt/omni/current
```

---

# 41. systemd

例如：

```ini
[Unit]

Description=Omni Robot

After=network-online.target

[Service]

Type=simple

ExecStart=/opt/omni/current/bin/start.sh

Restart=always

RestartSec=3

[Install]

WantedBy=multi-user.target
```

启动：

```bash
systemctl enable omni-robot
systemctl restart omni-robot
```

---

# 42. 自动回滚

目录：

```text
/opt/omni/releases/

v1.0.0
v1.1.0
v1.2.0
```

当前：

```text
current -> v1.2.0
```

如果升级后：

```text
service crash
ROS2 node startup fail
healthcheck fail
```

自动：

```text
current -> v1.1.0
```

然后：

```bash
systemctl restart omni-robot
```

---

# 43. Health Check

建议：

```bash
scripts/healthcheck.sh
```

检查：

```text
process alive
ROS2 node alive
ROS2 topic alive
network alive
lidar alive
camera alive
disk
memory
```

例如：

```bash
ros2 node list | grep omni_dog_edge_agent
```

以及：

```bash
ros2 topic hz /front_lidar
```

---

# 44. OTA 升级

后期可以设计：

```text
GitHub Release
      │
      ▼
Omni Cloud
      │
      ▼
Robot Agent
      │
      ▼
Version Check
      │
      ▼
Download
      │
      ▼
Verify SHA256
      │
      ▼
Install
      │
      ▼
Health Check
      │
   ┌──┴──┐
   │     │
Success Fail
   │     │
   ▼     ▼
Commit Rollback
```

---

# 45. 不建议 GitHub Actions 直接 SSH 所有机器人

第一阶段少量机器人可以：

```text
GitHub Actions
 ↓
SSH
 ↓
Robot
```

但机器人规模扩大后不建议。

应该：

```text
GitHub
 ↓
Artifact
 ↓
Omni Cloud
 ↓
Robot主动查询
 ↓
Download
```

机器人主动连接服务器比服务器直接连接现场设备更容易处理：

```text
NAT
4G
5G
物业网络
动态 IP
防火墙
```

---

# 46. Release Channel

建议：

```text
dev
beta
stable
```

例如：

```text
v1.3.0-dev.15
v1.3.0-beta.1
v1.3.0
```

设备：

```text
实验机器狗
 → dev

试点机器狗
 → beta

生产机器狗
 → stable
```

---

# 47. Docker Tag

对应：

```text
omni-robot-orin:dev
omni-robot-orin:beta
omni-robot-orin:stable
```

不要直接把生产设备追踪：

```text
latest
```

---

# 48. 构建缓存

建议缓存：

```text
apt
pip
ccache
colcon
Docker layer
```

C++ 强烈建议：

```text
ccache
```

例如：

```bash
export CC="ccache gcc"
export CXX="ccache g++"
```

---

# 49. CI 触发策略

推荐：

| 操作 | x86 | Orin | S100 |
|---|---:|---:|---:|
| PR | ✅ | ❌ | ❌ |
| main push | ✅ | ✅ | ✅ |
| tag | ✅ | ✅ | ✅ |
| manual | ✅ | ✅ | ✅ |

这样避免每一次 commit 都占用 Orin。

---

# 50. GitHub Concurrency

Orin 通常只有一台 Runner。

建议：

```yaml
concurrency:

  group: orin-build

  cancel-in-progress: true
```

避免：

```text
commit A
commit B
commit C
```

导致三次构建同时排队。

---

# 51. Secrets

禁止写入仓库：

```text
SSH password
Device password
Registry password
API key
Cloud token
```

使用：

```text
GitHub Secrets
```

例如：

```text
DEPLOY_SSH_KEY
REGISTRY_TOKEN
OMNI_CLOUD_TOKEN
```

---

# 52. Docker Secrets

不要：

```dockerfile
ARG API_TOKEN=xxx
```

尤其现代 BuildKit 可以为镜像附加 provenance 信息，Docker 官方明确提醒不要通过普通 build arguments 传递敏感信息。

应使用 BuildKit secret mount。

---

# 53. Artifact 命名

推荐：

```text
omni-robot-{platform}-{arch}-{version}-{commit}
```

例如：

```text
omni-robot-x86-amd64-v1.2.0-a27193.tar.gz

omni-robot-orin-arm64-v1.2.0-a27193.tar.gz

omni-robot-s100-arm64-v1.2.0-a27193.tar.gz
```

---

# 54. 推荐最终 CI/CD 架构

```text
                           GitHub
                              │
                           Source
                              │
          ┌───────────────────┼──────────────────┐
          │                   │                  │
          ▼                   ▼                  ▼
      x86 Builder          Orin NX           S100 Builder
          │              Self-hosted             │
          │                   │                  │
     Ubuntu 22.04          JetPack            Docker x86
     ROS2 Humble             │              D-Robotics SDK
          │             ROS2 Humble               │
          │             CUDA/TensorRT             │
          │                   │                  │
          ▼                   ▼                  ▼
       amd64               arm64               arm64
          │                   │                  │
          ▼                   ▼                  ▼
      Docker Image        Docker Image        tar.gz
          │                   │                  │
          └──────────┬────────┴──────────┬───────┘
                     │                   │
                     ▼                   ▼
                    GHCR            GitHub Release
                     │                   │
                     └─────────┬─────────┘
                               ▼
                           Omni Cloud
                               │
                               ▼
                         Robot OTA Agent
                               │
                ┌──────────────┼─────────────┐
                │              │             │
                ▼              ▼             ▼
               x86           Orin           S100
```

---

# 55. 推荐的最终制品

## x86

Docker：

```text
ghcr.io/omni-ai/omni-robot-x86:v1.2.0
```

Release：

```text
omni-robot-x86-amd64-v1.2.0.tar.gz
```

---

## Orin

Docker：

```text
ghcr.io/omni-ai/omni-robot-orin:v1.2.0
```

Release：

```text
omni-robot-orin-arm64-v1.2.0.tar.gz
```

---

## S100

Builder：

```text
ghcr.io/omni-ai/omni-s100-builder:rdk-4.0.5
```

Release：

```text
omni-robot-s100-arm64-v1.2.0.tar.gz
```

S100 不强制产生 Runtime Docker。

---

# 56. 第一阶段实施顺序

不建议一次性把所有 CI/CD 功能全部实现。

建议按照以下顺序推进。

## Phase 1

首先完成：

```text
统一代码仓库
+
build.sh
+
x86 build
+
Orin build
+
S100 build
```

目标：

```text
三平台均能稳定编译
```

---

## Phase 2

增加：

```text
GitHub Actions
```

完成：

```text
PR
 ↓
x86 CI

main
 ↓
x86 + Orin + S100
```

---

## Phase 3

增加：

```text
GHCR
+
GitHub Release
```

形成：

```text
git tag
 ↓
三平台 Build
 ↓
Artifacts
 ↓
Release
```

---

## Phase 4

增加：

```text
机器人自动部署
+
版本管理
+
Health Check
+
Rollback
```

---

## Phase 5

最后建设：

```text
Omni Robot OTA
```

实现：

```text
Cloud
 ↓
Robot Fleet
 ↓
灰度升级
 ↓
监控
 ↓
自动回滚
```

---

# 57. 推荐第一版落地架构

当前阶段最推荐：

```text
                  GitHub Repository
                         │
                         ▼
                    GitHub Actions
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
            x86         Orin        S100
             │           │           │
      GitHub Runner   Orin Runner  x86 Docker
             │           │        Cross Build
             │           │           │
             ▼           ▼           ▼
           amd64       arm64       arm64
             │           │           │
             └─────┬─────┴─────┬─────┘
                   │           │
                   ▼           ▼
                 GHCR       Release
                   │           │
                   └─────┬─────┘
                         ▼
                      Deploy
```

---

# 58. 最终技术决策

## x86

采用：

```text
GitHub Hosted Runner
+
Docker
+
ROS2 Humble
+
GHCR
```

---

## Orin

采用：

```text
GitHub Self-hosted Runner
+
Native ARM64 Build
+
JetPack
+
CUDA/TensorRT
+
ROS2 Humble
+
Docker Runtime
```

不将 QEMU 作为主要生产构建方案。

---

## S100

采用：

```text
GitHub Hosted x86 Runner
+
D-Robotics 官方工具链
+
Builder Docker
+
Cross Compile
+
Native Binary Deployment
```

S100 保留官方 RDK OS。

截至当前官方资源，RDK S100 已提供对应系统镜像、BSP、3.7.0 算法工具链及 S100/S600 Docker 工具链，因此 CI 中应该尽量复用官方 SDK，而不是自行重新构造整个 S100 Linux 用户态。

---

# 59. 一句话总结

Omni Robot 的 CI/CD 不应该追求：

```text
三个平台
=
一种构建方式
```

而应该追求：

```text
                 One Git Repository
                        +
               One GitHub CI/CD
                        +
                 One Version System
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
        x86           Orin           S100
       Native         Native         Cross
        Build          Build         Build
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                Same Omni Software
```

即：

> **统一代码、统一版本、统一流水线、统一发布体系，但针对不同硬件采用最合适的构建方式。**

这套架构能够同时满足当前 x86、Orin NX、S100 三个平台，也为后续增加新的 ARM64 / Jetson / RDK / 工控机平台保留扩展空间。