"""Guard against GT background deletion and misleading coverage ceilings."""
import numpy as np
from diagnose_association_ceiling import relabel_atoms, coverage_ceiling
from omni_semantic_map.full_ovo_evaluation import voxel_metrics


def test_whole_atoms_merge_without_deleting_background():
    gt=np.array([1,1,1,1,0,0,0,2]);atoms=np.array([0,0,1,1,2,2,2,-1]);native=np.array([4,4,5,5,6,6,6,0])
    oracle,rows=relabel_atoms(gt,atoms,native)
    assert len(np.unique(oracle[:4]))==1
    assert np.all(oracle[4:7]==6) and oracle[7]==0
    assert np.array_equal(oracle>0,native>0)


def test_mixed_atom_cannot_split_into_two_truth_objects():
    gt=np.array([1,1,2]);native=np.ones(3,dtype=int)
    oracle,_=relabel_atoms(gt,np.zeros(3,dtype=int),native)
    assert len(np.unique(oracle))==1


def test_ceiling_strict_half_and_background():
    gt=np.array([1,1,2,2,2,0]);covered=np.array([1,0,1,1,0,1],bool)
    oracle,rows=coverage_ceiling(gt,covered)
    assert oracle.tolist()==[0,0,2,2,0,0]
    m=voxel_metrics(gt,oracle)
    assert (m['tp'],m['fp'],m['fn'])==(1,0,1)
    assert abs(m['observed_voxel_instance_pq']-4/9)<1e-12


def test_ceiling_dominates_exhaustive_assignments_on_small_domain():
    from itertools import product
    gt=np.array([1,1,2,2,0]);covered=np.array([1,1,1,0,1],bool)
    oracle,_=coverage_ceiling(gt,covered);best=voxel_metrics(gt,oracle)['observed_voxel_instance_pq']
    for values in product(range(4),repeat=int(covered.sum())):
        pred=np.zeros_like(gt);pred[covered]=values
        assert voxel_metrics(gt,pred)['observed_voxel_instance_pq']<=best+1e-12
