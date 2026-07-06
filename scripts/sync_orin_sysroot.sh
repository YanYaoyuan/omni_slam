#!/usr/bin/env bash
set -euo pipefail

ORIN_HOST=${1:-}
SYSROOT=${ORIN_SYSROOT:-$HOME/sysroots/orin}
LIVOX_WS_ON_ORIN=${LIVOX_WS_ON_ORIN:-$HOME/ws_livox/install}

die() { echo "[sync_orin_sysroot] ERROR: $*" >&2; exit 1; }

if [[ -z "$ORIN_HOST" ]]; then
  die "usage: ORIN_SYSROOT=$SYSROOT $0 user@orin-host"
fi

mkdir -p "$SYSROOT"

echo "[sync_orin_sysroot] syncing target sysroot from $ORIN_HOST to $SYSROOT"
rsync -aAX --numeric-ids --delete "$ORIN_HOST:/lib" "$SYSROOT/"
rsync -aAX --numeric-ids --delete "$ORIN_HOST:/usr" "$SYSROOT/"
rsync -aAX --numeric-ids --delete "$ORIN_HOST:/opt/ros" "$SYSROOT/opt/"
mkdir -p "$SYSROOT/home/user/ws_livox"
rsync -aAX --numeric-ids --delete "$ORIN_HOST:$LIVOX_WS_ON_ORIN/" "$SYSROOT/home/user/ws_livox/install/"

echo "[sync_orin_sysroot] done"
