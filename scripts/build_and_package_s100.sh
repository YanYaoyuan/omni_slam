#!/usr/bin/env bash
#
# 一键脚本：在 S100 板端原生 colcon 编译并打包为可部署运行包。
#
#   ./scripts/build_and_package_s100.sh
#
# 产物目录结构 (dist/omni_slam_s100_<版本>/)：
#   bin/     可执行文件 (fastlio_mapping / icp_node / transform_publisher)
#   lib/     包自身的 .so + 通过 ldd 解析出的第三方依赖动态库
#   config/  yaml 参数文件
#   launch/  launch 启动文件
#   maps/    先验点云地图 (.pcd)
#   install/ 完整 colcon install 树 (ros2 launch / ament 索引所需)
#   setup_env.sh          运行环境配置 (source ROS + Livox + install + 捆绑 lib)
#   run_mapping.sh        建图启动脚本
#   run_localization.sh   定位(重定位)启动脚本
#
# 常用环境变量 (均有默认值)：
#   ROS_DISTRO_TARGET   目标 ROS 版本            默认 humble
#   LIVOX_SETUP         Livox 驱动 setup.bash    默认 $HOME/ws_livox/install/setup.bash
#   MAP_PATH            打包进去的先验地图        默认 FAST_LIO/PCD/omni_dog_map.pcd
#   MAKE_JOBS           单包编译并行数            默认 2
#   WORKERS             colcon 并行包数           默认 1
#   BUNDLE_ROS_LIBS     是否把 /opt/ros 下的库也打包(0/1) 默认 0 (目标板已装 ROS)
#   SKIP_BUILD          跳过编译只打包(0/1)       默认 0
#   OUT_DIR             输出目录                 默认 dist
#   VERSION             版本号                   默认 时间戳
#
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROS_DISTRO_TARGET=${ROS_DISTRO_TARGET:-humble}
LIVOX_SETUP=${LIVOX_SETUP:-$HOME/ws_livox/install/setup.bash}
MAP_PATH=${MAP_PATH:-$ROOT_DIR/FAST_LIO/PCD/omni_dog_map.pcd}
MAKE_JOBS=${MAKE_JOBS:-2}
WORKERS=${WORKERS:-1}
BUNDLE_ROS_LIBS=${BUNDLE_ROS_LIBS:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
OUT_DIR=${OUT_DIR:-$ROOT_DIR/dist}
VERSION=${VERSION:-$(date +%Y%m%d_%H%M%S)}

INSTALL_DIR="$ROOT_DIR/install"
PKG_NAME="omni_slam_s100_$VERSION"
PKG_DIR="$OUT_DIR/$PKG_NAME"
TARBALL="$OUT_DIR/$PKG_NAME.tar.gz"

log()  { echo -e "[s100] $*"; }
die()  { echo "[s100] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1) 原生 colcon 编译
# ---------------------------------------------------------------------------
if [[ "$SKIP_BUILD" != "1" ]]; then
  [[ -f "/opt/ros/$ROS_DISTRO_TARGET/setup.bash" ]] || \
    die "找不到 /opt/ros/$ROS_DISTRO_TARGET/setup.bash，请确认板端已安装 ROS $ROS_DISTRO_TARGET"

  log "source ROS $ROS_DISTRO_TARGET"
  set +u
  # The ROS distribution is selected at runtime.
  # shellcheck disable=SC1090
  source "/opt/ros/$ROS_DISTRO_TARGET/setup.bash"
  set -u

  if [[ -f "$LIVOX_SETUP" ]]; then
    log "source Livox: $LIVOX_SETUP"
    set +u
    # Users may override the Livox workspace.
    # shellcheck disable=SC1090
    source "$LIVOX_SETUP"
    set -u
  else
    log "警告: 未找到 Livox setup: $LIVOX_SETUP (若已装于其他路径请设置 LIVOX_SETUP)"
  fi

  log "colcon build (fast_lio + icp_relocalization, Release)..."
  cd "$ROOT_DIR"
  # --merge-install: 生成扁平的 install/lib、install/share 布局，便于打包
  MAKEFLAGS="-j$MAKE_JOBS" colcon build \
    --merge-install \
    --parallel-workers "$WORKERS" \
    --packages-select fast_lio icp_relocalization \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
else
  log "SKIP_BUILD=1，跳过编译"
fi

[[ -d "$INSTALL_DIR" ]] || die "缺少 install 目录: $INSTALL_DIR (请先编译)"
FASTLIO_BIN="$INSTALL_DIR/lib/fast_lio/fastlio_mapping"
ICP_BIN="$INSTALL_DIR/lib/icp_relocalization/icp_node"
TF_BIN="$INSTALL_DIR/lib/icp_relocalization/transform_publisher"
[[ -f "$FASTLIO_BIN" ]] || die "缺少可执行文件: $FASTLIO_BIN"
[[ -f "$ICP_BIN" ]]     || die "缺少可执行文件: $ICP_BIN"

# ---------------------------------------------------------------------------
# 2) 组织打包目录 bin/ config/ launch/ lib/ maps/ install/
# ---------------------------------------------------------------------------
log "准备打包目录: $PKG_DIR"
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR"/{bin,lib,config,launch,maps}

# 完整 install 树 (ros2 launch / ament 索引依赖它)
cp -a "$INSTALL_DIR" "$PKG_DIR/install"

# --- 合并 livox_ros_driver2 (关键) -------------------------------------------
# fast_lio/icp 订阅 Livox CustomMsg，运行时会 dlopen
# liblivox_ros_driver2__rosidl_typesupport_*.so。这些库不在 /opt/ros，也不会
# 被 ldd 抓到(dlopen 加载)，必须把 livox 的 install 树合并进包，否则换机器必崩。
LIVOX_PREFIX=${LIVOX_PREFIX:-$(ros2 pkg prefix livox_ros_driver2 2>/dev/null || true)}
if [[ -n "${LIVOX_PREFIX:-}" && -d "$LIVOX_PREFIX/lib" ]]; then
  log "合并 livox_ros_driver2: $LIVOX_PREFIX"
  cp -a "$LIVOX_PREFIX/lib/."   "$PKG_DIR/install/lib/"   2>/dev/null || true
  cp -a "$LIVOX_PREFIX/share/." "$PKG_DIR/install/share/" 2>/dev/null || true
else
  log "警告: 未探测到 livox_ros_driver2 install (LIVOX_PREFIX=${LIVOX_PREFIX:-空})"
  log "      → 打出的包在无 livox 的机器上会因 dlopen typesupport 失败而崩溃"
  log "      → 请先 source livox 工作区，或用 LIVOX_PREFIX=/路径/install/livox_ros_driver2 指定"
fi

# 可执行文件 -> bin/
for b in "$FASTLIO_BIN" "$ICP_BIN" "$TF_BIN"; do
  [[ -f "$b" ]] && cp -a "$b" "$PKG_DIR/bin/"
done

# 配置文件 -> config/
if [[ -d "$INSTALL_DIR/share/fast_lio/config" ]]; then
  cp -a "$INSTALL_DIR/share/fast_lio/config/." "$PKG_DIR/config/"
else
  cp -a "$ROOT_DIR/FAST_LIO/config/." "$PKG_DIR/config/"
fi

# launch 文件 -> launch/
for sd in "$INSTALL_DIR/share/fast_lio/launch" "$INSTALL_DIR/share/icp_relocalization/launch"; do
  if [[ -d "$sd" ]]; then
    cp -a "$sd/." "$PKG_DIR/launch/" 2>/dev/null || true
  fi
done

# 先验地图 -> maps/
if [[ -f "$MAP_PATH" ]]; then
  cp "$MAP_PATH" "$PKG_DIR/maps/omni_dog_map.pcd"
  log "已打包地图: $MAP_PATH"
else
  log "警告: 未找到地图 $MAP_PATH (定位运行前请手动放入 maps/)"
fi

# ---------------------------------------------------------------------------
# 3) 捆绑依赖动态库 (ldd 解析) -> lib/
# ---------------------------------------------------------------------------
log "解析并捆绑依赖动态库..."

# 不打包的库(由目标系统 glibc / 动态加载器提供，捆绑反而易出错)
is_system_lib() {
  case "$(basename "$1")" in
    ld-linux*|linux-vdso*|libc.so.*|libm.so.*|libdl.so.*|librt.so.*|\
    libpthread.so.*|libresolv.so.*|libutil.so.*|libgcc_s.so.*|libstdc++.so.*)
      return 0 ;;
  esac
  return 1
}

# 先把包 install 树里的 .so 收进 lib/ (含已合并的 livox 类型支持库)
while IFS= read -r solib; do
  cp -a "$solib" "$PKG_DIR/lib/" 2>/dev/null || true
done < <(find "$PKG_DIR/install" -name '*.so*' -type f 2>/dev/null)

# 用于 ldd 解析的完整可执行/库集合
mapfile -t SCAN_TARGETS < <(find "$PKG_DIR/bin" "$PKG_DIR/lib" -type f 2>/dev/null)

bundle_deps() {
  local target="$1"
  ldd "$target" 2>/dev/null | while read -r line; do
    # 形如:  libfoo.so.1 => /path/to/libfoo.so.1 (0x00..)
    local path
    path=$(echo "$line" | sed -nE 's/.*=> (\/[^ ]+).*/\1/p')
    [[ -z "$path" ]] && continue
    [[ -f "$path" ]] || continue
    is_system_lib "$path" && continue
    if [[ "$BUNDLE_ROS_LIBS" != "1" && "$path" == /opt/ros/* ]]; then
      continue
    fi
    local name
    name=$(basename "$path")
    [[ -e "$PKG_DIR/lib/$name" ]] && continue
    cp -aL "$path" "$PKG_DIR/lib/$name" 2>/dev/null || true
  done
}

for t in "${SCAN_TARGETS[@]}"; do
  bundle_deps "$t"
done
# 对新捆绑进来的库再解析一轮，覆盖二级依赖
for t in "$PKG_DIR"/lib/*.so*; do
  [[ -f "$t" ]] && bundle_deps "$t"
done

LIB_COUNT=$(find "$PKG_DIR/lib" -maxdepth 1 -name '*.so*' | wc -l | tr -d ' ')
log "已捆绑动态库数量: $LIB_COUNT (BUNDLE_ROS_LIBS=$BUNDLE_ROS_LIBS)"

# ---------------------------------------------------------------------------
# 4) 生成运行环境 + 启动脚本
# ---------------------------------------------------------------------------
cat > "$PKG_DIR/setup_env.sh" <<'EOS'
#!/usr/bin/env bash
# 运行环境配置：source ROS + 本包 install(已内含 livox 消息库)，并加入捆绑库路径
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROS_DISTRO_TARGET=${ROS_DISTRO_TARGET:-humble}
# 可选：外部 Livox 工作区；本包 install 已内含 livox 类型支持库，一般无需设置
LIVOX_SETUP=${LIVOX_SETUP:-}

set +u
source "/opt/ros/$ROS_DISTRO_TARGET/setup.bash"
if [[ -n "$LIVOX_SETUP" && -f "$LIVOX_SETUP" ]]; then
  source "$LIVOX_SETUP"
fi
source "$SCRIPT_DIR/install/setup.bash"
set -u

# 关键：install/lib 内含 livox 类型支持库(dlopen 加载)；lib 为 ldd 捆绑的第三方库
export LD_LIBRARY_PATH="$SCRIPT_DIR/install/lib:$SCRIPT_DIR/lib:${LD_LIBRARY_PATH:-}"
EOS

cat > "$PKG_DIR/run_mapping.sh" <<'EOS'
#!/usr/bin/env bash
# 建图模式 (vbot: Vanjee 机械雷达 + /imu_raw)
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/setup_env.sh"
echo "[run_mapping] 启动建图 (fast_lio vbot.launch.py)"
exec ros2 launch fast_lio vbot.launch.py "$@"
EOS

cat > "$PKG_DIR/run_localization.sh" <<'EOS'
#!/usr/bin/env bash
# 定位(重定位)模式
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/setup_env.sh"
MAP_PATH=${MAP_PATH:-$SCRIPT_DIR/maps/omni_dog_map.pcd}
if [[ ! -f "$MAP_PATH" ]]; then
  echo "[run_localization] ERROR: 未找到地图 $MAP_PATH，请用 MAP_PATH=/路径/xxx.pcd 指定" >&2
  exit 1
fi
echo "[run_localization] 启动定位，地图: $MAP_PATH"
exec ros2 launch fast_lio omni_dog_relocalization.launch.py map_path:="$MAP_PATH" "$@"
EOS

cat > "$PKG_DIR/README.md" <<EOS
# Omni SLAM S100 运行包

目录结构：
- \`bin/\`     可执行文件
- \`lib/\`     包自身库 + 捆绑的第三方依赖动态库
- \`config/\`  yaml 参数
- \`launch/\`  launch 启动文件
- \`maps/\`    先验点云地图
- \`install/\` 完整 colcon install 树 (ros2 launch 依赖)

## 部署运行

\`\`\`bash
tar -xzf $PKG_NAME.tar.gz -C ~/
cd ~/$PKG_NAME

# 建图
./run_mapping.sh

# 定位 (默认用 maps/omni_dog_map.pcd)
./run_localization.sh
# 或指定地图
MAP_PATH=/path/to/your_map.pcd ./run_localization.sh
\`\`\`

若 Livox 驱动不在默认路径：

\`\`\`bash
export LIVOX_SETUP=/path/to/ws_livox/install/setup.bash
\`\`\`

目标板必须已安装 ROS \$ROS_DISTRO_TARGET (默认 humble)。第三方依赖库
(PCL / VTK / Livox SDK 等) 已捆绑于 \`lib/\`，运行时经 LD_LIBRARY_PATH 加载。
EOS

chmod +x "$PKG_DIR"/*.sh

# ---------------------------------------------------------------------------
# 5) 打 tar 包
# ---------------------------------------------------------------------------
log "打 tar.gz 包..."
cd "$OUT_DIR"
tar -czf "$TARBALL" "$PKG_NAME"

log "完成 ✅"
log "  运行包目录: $PKG_DIR"
log "  压缩包:     $TARBALL"
