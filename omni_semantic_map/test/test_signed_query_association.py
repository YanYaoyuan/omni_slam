import numpy as np
import pytest
from omni_semantic_map.signed_query_association import pair_features,signed_groups,positive_components


def test_negative_evidence_prevents_transitive_bridge_merge():
    a=np.array([0,0,1]);b=np.array([1,2,2]);weights=np.array([4.,-6.,3.])
    groups,trace=signed_groups(3,a,b,weights)
    assert groups==[[0,1],[2]] and trace[0]['gain']==4
    assert positive_components(3,a,b,weights>0)==[[0,1,2]]


def test_group_gains_equal_full_objective_change():
    rng=np.random.RandomState(7);a,b=np.triu_indices(7,1);weights=rng.randn(len(a));groups,trace=signed_groups(7,a,b,weights)
    total=0.
    for merge in trace:
        left=set(merge['left']);right=set(merge['right'])
        expected=sum(w for i,j,w in zip(a,b,weights) if (i in left and j in right) or (j in left and i in right))
        assert abs(expected-merge['gain'])<1e-12 and expected>0
        total+=expected
    owner={i:k for k,g in enumerate(groups) for i in g}
    assert abs(total-sum(w for i,j,w in zip(a,b,weights) if owner[i]==owner[j]))<1e-12
    with pytest.raises(ValueError,match='Duplicate'):signed_groups(3,[0,0],[1,1],[1.,2.])


def test_pair_symmetry_visibility_and_no_self_confirmation():
    g=dict(cells=np.array([20,30]),intersections=np.array([8.]),centers=np.array([[0.,0,0],[1.,2,3]]),rms=np.array([1.,2.]),scores=np.array([.2,.8]),seed_frames=np.array([2,3]),old=np.array([0,0]),semantic_ids=np.array([11,11]))
    r=dict(witness_frames=np.array([1,2,3,4]),visible_cells=np.array([[20,20,20,0],[25,25,25,30]]),query_ids=np.array([[4,5,6,7],[4,8,9,10]]),agreement=np.ones((2,4))*.8,scores=np.ones((2,4))*.3)
    q=np.array([[.9,0.,0.,0.]])
    x,observed=pair_features([0],[1],g,r,q);y,_=pair_features([1],[0],g,r,q)
    np.testing.assert_array_equal(x,y);assert observed.tolist()==[True]
    changed={k:v.copy() for k,v in r.items()};changed['agreement'][:,1:]=.01;changed['scores'][:,1:]=.99
    z,_=pair_features([0],[1],g,changed,q);np.testing.assert_array_equal(x,z)
    changed['visible_cells'][0,0]=0
    z,observed=pair_features([0],[1],g,changed,q)
    assert not observed[0] and np.all(z[:,11:]==0)
