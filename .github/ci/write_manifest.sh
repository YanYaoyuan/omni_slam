#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <arch> <platform> <version> <output-dir>" >&2
  exit 2
fi

arch=$1
platform=$2
version=$3
output_dir=$4
mkdir -p "${output_dir}"

cat > "${output_dir}/manifest.json" <<EOF
{
  "project": "omni_slam",
  "version": "${version}",
  "git_sha": "${GITHUB_SHA:-unknown}",
  "omni_tf_manager_git_sha": "${OMNI_TF_MANAGER_SHA:-unknown}",
  "platform": "${platform}",
  "architecture": "${arch}",
  "ros_distribution": "humble",
  "built_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
