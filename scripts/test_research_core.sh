#!/usr/bin/env bash
# CPU-only regression coverage for the published revision mechanisms.
set -euo pipefail
RESEARCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$RESEARCH_ROOT"
export PYTHONPATH="$RESEARCH_ROOT/omni_semantic_map${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=""
python3 -m pytest -q \
  omni_semantic_map/test/test_revision_authorization.py \
  omni_semantic_map/test/test_batch_stable_transport.py \
  omni_semantic_map/test/test_mask_cover_revision.py \
  omni_semantic_map/test/test_discovery_measured_input.py \
  omni_semantic_map/test/test_measured_query_cells.py \
  omni_semantic_map/test/test_kitti360_reference_identity.py
