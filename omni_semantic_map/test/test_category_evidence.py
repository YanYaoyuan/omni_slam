import numpy as np
import pytest
from omni_semantic_map.category_evidence import readout,METHODS


def test_rejects_invalid_exclusivity():
    with pytest.raises(ValueError):readout(np.eye(2),[2,2],[3,1],np.eye(2))


def test_consensus_rejects_outlier_without_using_class_labels():
    f=np.array([[1.,0],[1,0],[0,1]])
    r=readout(f,[1,1,100],[1,1,100],np.eye(2),method='quality_siglip')
    assert np.array_equal(r['selected'],[True,True,False])
    assert r['scores'].argmax()==0
    # Uniformly wrong features cannot be repaired by agreement weighting.
    wrong=readout(np.tile([0.,1],(3,1)),[1,2,3],[1,2,3],np.eye(2),method='quality_siglip')
    assert wrong['scores'].argmax()==1 and wrong['selected'].all()


def test_missing_dino_retains_quality_and_permutation_invariance():
    f=np.array([[1.,0],[.8,.2],[0,1],[.9,.1]])
    args=([4,9,16,25],[4,3,8,25],np.eye(2))
    q=readout(f,*args,method='exclusive_quality')
    r=readout(f,*args,method='quality_dino',dino=np.zeros((4,3)))
    assert r['fallback']=='missing_consensus_feature'
    np.testing.assert_allclose(r['scores'],q['scores'])
    perm=np.array([2,0,3,1])
    for method in METHODS:
        a=readout(f,*args,method=method,dino=f)
        b=readout(f[perm],np.array(args[0])[perm],np.array(args[1])[perm],args[2],method=method,dino=f[perm])
        np.testing.assert_allclose(a['scores'],b['scores'])
        np.testing.assert_allclose(a['weights'][perm],b['weights'])


def test_zero_exclusive_area_and_single_view_have_defined_fallback():
    for method in METHODS:
        r=readout([[1.,0]],[10],[0],np.eye(2),method=method,dino=[[1,0]])
        assert np.isfinite(r['scores']).all() and r['scores'].argmax()==0
        np.testing.assert_allclose(r['weights'],[1])
