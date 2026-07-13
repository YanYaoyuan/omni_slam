#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOG_HOST="${DOG_HOST:-dog3}"
DOG_DIR="${DOG_DIR:-/userdata/1_slam}"

SSH_CMD=(ssh)
RSYNC_SSH=(ssh)
if [ -n "${DOG_PASSWORD:-}" ]; then
  export SSHPASS="$DOG_PASSWORD"
  SSH_CMD=(sshpass -e ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no)
  RSYNC_SSH=(sshpass -e ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no)
fi

echo "目标狗子：$DOG_HOST"
echo "目标目录：$DOG_DIR"

"${SSH_CMD[@]}" "$DOG_HOST" "mkdir -p '$DOG_DIR'"

rsync -az --info=progress2 \
  -e "${RSYNC_SSH[*]}" \
  --exclude '/build/' \
  --exclude '/install/' \
  --exclude '/log/' \
  --exclude '/FAST_LIO/Log/' \
  --exclude '/.git/' \
  --exclude '/maps/' \
  --exclude '/.deps/' \
  --exclude '/test_logs/' \
  --exclude '**/__pycache__/' \
  --exclude '*.pyc' \
  "$SCRIPT_DIR"/ "$DOG_HOST":"$DOG_DIR"/

"${SSH_CMD[@]}" "$DOG_HOST" "cd '$DOG_DIR' && bash build_on_dog.sh"

echo
echo "部署完成。狗子上使用："
echo "  cd $DOG_DIR && bash 1_mapping.sh"
echo "  cd $DOG_DIR && bash 2_localizing.sh"
