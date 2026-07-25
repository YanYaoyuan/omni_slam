#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
INSTALL_BASE=${ORIN_INSTALL_BASE:-$ROOT_DIR/install_orin}
OUT_DIR=${OUT_DIR:-$ROOT_DIR/dist}
VERSION=${VERSION:-$(date +%Y%m%d_%H%M%S)}
MAP_PATH=${MAP_PATH:-$ROOT_DIR/FAST_LIO/PCD/omni_dog_map.pcd}
PKG_NAME=${PKG_NAME:-omni_slam_orin_flat_$VERSION}
PKG_DIR="$OUT_DIR/$PKG_NAME"
TARBALL="$OUT_DIR/$PKG_NAME.tar.gz"

die() { echo "[package_orin_flat_runtime] ERROR: $*" >&2; exit 1; }
need_file() { [[ -f "$1" ]] || die "missing file: $1"; }
need_dir() { [[ -d "$1" ]] || die "missing dir: $1"; }

need_dir "$INSTALL_BASE"
need_file "$INSTALL_BASE/setup.bash"
need_file "$INSTALL_BASE/lib/fast_lio/fastlio_mapping"
need_file "$INSTALL_BASE/lib/icp_relocalization/icp_node"
need_file "$INSTALL_BASE/lib/icp_relocalization/transform_publisher"

rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR" "$OUT_DIR"

# Make package root itself a valid ROS 2 install prefix.
for item in \
  setup.bash setup.sh setup.zsh local_setup.bash local_setup.sh local_setup.zsh \
  _local_setup_util_sh.py COLCON_IGNORE .colcon_install_layout \
  lib share include; do
  if [[ -e "$INSTALL_BASE/$item" ]]; then
    cp -a "$INSTALL_BASE/$item" "$PKG_DIR/"
  fi
done

# Convenience top-level layout for deployment inspection and direct exec.
mkdir -p "$PKG_DIR/bin" "$PKG_DIR/config" "$PKG_DIR/launch" "$PKG_DIR/rviz" "$PKG_DIR/maps" "$PKG_DIR/scripts"
cp -a "$INSTALL_BASE/lib/fast_lio/fastlio_mapping" "$PKG_DIR/bin/"
cp -a "$INSTALL_BASE/lib/icp_relocalization/icp_node" "$PKG_DIR/bin/"
cp -a "$INSTALL_BASE/lib/icp_relocalization/transform_publisher" "$PKG_DIR/bin/"
if [[ -f "$INSTALL_BASE/lib/icp_relocalization/sac_ia_gicp" ]]; then
  cp -a "$INSTALL_BASE/lib/icp_relocalization/sac_ia_gicp" "$PKG_DIR/bin/"
fi
cp -a "$INSTALL_BASE/share/fast_lio/config/." "$PKG_DIR/config/"
cp -a "$INSTALL_BASE/share/fast_lio/launch/." "$PKG_DIR/launch/"
cp -a "$INSTALL_BASE/share/icp_relocalization/launch/." "$PKG_DIR/launch/"
if [[ -d "$INSTALL_BASE/share/fast_lio/rviz" ]]; then
  cp -a "$INSTALL_BASE/share/fast_lio/rviz/." "$PKG_DIR/rviz/"
fi
if [[ -f "$MAP_PATH" ]]; then
  cp "$MAP_PATH" "$PKG_DIR/maps/omni_dog_map.pcd"
else
  echo "[package_orin_flat_runtime] WARNING: map not found, package will not include map: $MAP_PATH" >&2
fi

cat > "$PKG_DIR/setup_env.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROS_DISTRO_TARGET=${ROS_DISTRO_TARGET:-humble}
LIVOX_SETUP=${LIVOX_SETUP:-$HOME/ws_livox/install/setup.bash}
set +u
source "/opt/ros/$ROS_DISTRO_TARGET/setup.bash"
set -u
if [[ -f "$LIVOX_SETUP" ]]; then
  set +u
  source "$LIVOX_SETUP"
  set -u
else
  echo "[setup_env] WARNING: Livox setup not found: $LIVOX_SETUP" >&2
fi
set +u
source "$SCRIPT_DIR/setup.bash"
set -u
EOS

cat > "$PKG_DIR/run_mapping.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/setup_env.sh"
exec ros2 launch fast_lio omni_dog.launch.py "$@"
EOS

cat > "$PKG_DIR/run_relocalization.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/setup_env.sh"
MAP_PATH=${MAP_PATH:-$SCRIPT_DIR/maps/omni_dog_map.pcd}
if [[ ! -f "$MAP_PATH" ]]; then
  echo "[run_relocalization] ERROR: map not found: $MAP_PATH" >&2
  echo "[run_relocalization] Set MAP_PATH=/absolute/path/to/map.pcd or put the map at maps/omni_dog_map.pcd" >&2
  exit 1
fi
exec ros2 launch fast_lio omni_dog_relocalization.launch.py map_path:="$MAP_PATH" "$@"
EOS

cat > "$PKG_DIR/README_RUNTIME.md" <<'EOS'
# Omni SLAM Orin Flat Runtime

This package root is also a ROS 2 install prefix. Keep `share/` and `lib/`; `ros2 launch` needs the ament index under `share/ament_index`.

Run on Orin:

```bash
cd ~/omni_slam_orin_flat_*
./run_mapping.sh
./run_relocalization.sh
```

Useful paths:

```text
bin/                 direct executables
config/              copied FAST_LIO YAML configs
launch/              copied launch files
lib/                 ROS libraries and package executables
share/               ROS package index and package data, required
maps/omni_dog_map.pcd default relocalization map, if included
```

If Livox setup is elsewhere:

```bash
export LIVOX_SETUP=/path/to/livox/install/setup.bash
```
EOS

chmod +x "$PKG_DIR"/*.sh "$PKG_DIR/bin/"*
cd "$OUT_DIR"
tar -czf "$TARBALL" "$PKG_NAME"

echo "$TARBALL"
