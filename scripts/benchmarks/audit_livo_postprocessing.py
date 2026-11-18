"""Observe native postprocessing without changing it; verify exact final output."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from audit_evidence_retention import ROOT, dump
from omni_semantic_map.flat_benchmark import sha256
from retention_holdout_backend import run


def main():
    p = argparse.ArgumentParser()
    for name in ['cache', 'reference', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(ROOT/'build/benchmarks/expanded_semantic_20261109/concept-graphs'))
    from conceptgraph.slam import utils
    events = []
    def describe(objects):
        return [dict(alias=min(o['observation_ids'])+1, points=len(o['pcd'].points),
                     observations=len(o['observation_ids']), detections=o['num_detections']) for o in objects]
    originals = {name: getattr(utils, name) for name in ['denoise_objects', 'filter_objects']}
    def wrap(name):
        def observed(cfg, objects):
            before = describe(objects)
            result = originals[name](cfg, objects)
            events.append(dict(operation=name, before=before, after=describe(result)))
            return result
        return observed
    for name in originals:
        setattr(utils, name, wrap(name))
    try:
        run(a.cache, a.output)
    finally:
        for name, fn in originals.items():
            setattr(utils, name, fn)
    lhs = np.load(a.output/'geometry.npz'); rhs = np.load(a.reference/'geometry.npz')
    for key in lhs.files:
        np.testing.assert_array_equal(lhs[key], rhs[key])
    assert json.loads((a.output/'classification.json').read_text()) == json.loads((a.reference/'classification.json').read_text())
    dump(a.output/'postprocessing_audit.json', dict(complete=True, source_sha256=sha256(Path(__file__)),
        reference_sha256=sha256(a.reference/'run.json'), exact_geometry_and_classification=True,
        scope=__doc__, events=events))
    print('VERIFIED unchanged geometry and classification', a.output, flush=True)


if __name__ == '__main__':
    main()
