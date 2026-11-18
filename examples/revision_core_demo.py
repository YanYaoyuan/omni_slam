"""Run the real revision kernels on synthetic evidence, without ROS or models."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.sparse import csr_matrix

from omni_semantic_map.revision_authorization import (
    compatible_donors,
    margin_after_view_deletion,
)
from omni_semantic_map.batch_stable_transport import (
    certify,
    native_library,
    nearest_groups,
    nearest_owner,
)


def demonstrate(cache_directory):
    supported = margin_after_view_deletion([10, 11, 12], [0.9, 0.8, 0.7], [0.1] * 3)
    conflicting = margin_after_view_deletion([10, 11, 12], [1.0, 0.0, 0.0], [0.0, 0.2, 0.2])
    group = compatible_donors(1, [2, 3], {(1, 2): supported, (1, 3): supported, (2, 3): conflicting})

    graph = csr_matrix([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 4], [0, 0, 4, 0]])
    nodes = np.array([0, 1, 3])
    owners = np.array([1, 1, 2])
    frames = np.array([10, 11, 12])
    library = native_library(cache_directory)
    labels, distances = nearest_groups(graph, nodes, owners, library)
    pairs, tokens = np.unique(np.column_stack([owners, frames]), axis=0, return_inverse=True)
    token_labels, token_distances = nearest_groups(graph, nodes, tokens, library)
    stable = certify(labels, distances, token_labels, token_distances, pairs[:, 0])
    return {
        "scope": "Synthetic mechanism demo; not a benchmark score or semantic-accuracy claim",
        "supported_pair": supported,
        "conflicting_pair": conflicting,
        "compatible_group": group,
        "nearest_object_per_node": nearest_owner(labels, distances).tolist(),
        "certified_object_per_node": stable.tolist(),
        "unknown_label": -1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new JSON path; refuses overwrite")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    with tempfile.TemporaryDirectory(prefix="omni-revision-demo-") as cache:
        result = demonstrate(cache)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as output:
            output.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
