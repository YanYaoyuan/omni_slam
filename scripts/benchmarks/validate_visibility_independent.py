"""Pre-test acceptance: fixed-assignment behavior and compatible GT reader."""
import sys
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,OLD,read,dump,sha
from visibility_independent_evaluate import locked_projection
from omni_semantic_map.mesh_instance_evaluation import transfer_labels

OUT=ROOT/'build/benchmarks/visibility_independent_20261112'


def main():
    sys.path.insert(0,str(OLD/'tools/hdf5_reader'));import h5py
    xyz=np.array([[0,0,0],[.02,0,0.]])
    labels=np.array([1,2]);query=np.array([[.005,0,0.]])
    locked=locked_projection(xyz,labels,query,np.array([1]))
    standard,_=transfer_labels(xyz[1:],labels[1:],query)
    assert locked.tolist()==[0] and standard.tolist()==[2]
    assert locked_projection(xyz,labels,query,np.array([0,1])).tolist()==[1]
    assert locked_projection(xyz,labels,np.array([[10.,0,0]]),np.array([0,1])).tolist()==[0]
    assert locked_projection(xyz,np.zeros(2,int),query,np.array([0,1])).tolist()==[0]
    # Inspect the schema of an already-viewed annotation only, never the new IDs.
    viewed=OLD/'official_annotation_audit/cvpr18/scenenn_seg_014.hdf5'
    with h5py.File(viewed) as f:assert 'label' in f
    dump(OUT/'evaluator_acceptance.json',dict(complete=True,locked_assignment_tests=4,
        hdf5_reader_import_and_known_schema_passed=True,python=sys.version,hdf5_version=h5py.__version__,
        evaluator_sha256=sha(Path(__file__).with_name('visibility_independent_evaluate.py')),
        no_new_scene_labels_read=True))
    print('EVALUATOR ACCEPTANCE PASS',flush=True)


if __name__=='__main__':main()
