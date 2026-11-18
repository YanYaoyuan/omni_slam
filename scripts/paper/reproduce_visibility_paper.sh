#!/usr/bin/env bash
# Regenerate documents from existing results only. Never launch benchmark jobs.
set -euo pipefail
paper_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$paper_root"
paper_mode="${1:-verify}"
case "$paper_mode" in
  verify)
    python3 scripts/paper/verify_visibility_package.py
    ;;
  build)
    paper_python="$paper_root/build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python"
    if [[ ! -x "$paper_python" ]]; then
      echo "Missing recorded paper environment: $paper_python" >&2
      exit 1
    fi
    MPLCONFIGDIR=/tmp/omni_paper_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
      "$paper_python" scripts/paper/build_visibility_manuscript.py
    python3 scripts/paper/verify_visibility_package.py --allow-unreviewed-pdf
    ;;
  *)
    echo 'Usage: bash scripts/paper/reproduce_visibility_paper.sh [verify|build]' >&2
    exit 2
    ;;
esac
