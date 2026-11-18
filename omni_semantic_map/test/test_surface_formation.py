import numpy as np
from omni_semantic_map.surface_formation import partition_surfaces,edge_view_cost


def test_negative_boundary_blocks_positive_bridge():
    a=np.array([0,0,1]);b=np.array([1,2,2]);w=np.array([3.,-5.,2.])
    signed,trace=partition_surfaces(3,a,b,w,[0,0,0])
    positive,_=partition_surfaces(3,a,b,np.maximum(w,0),[0,0,0])
    assert signed[0]==signed[1] and signed[2]!=signed[0]
    assert len(np.unique(positive))==1
    assert trace['objective_gain']==3.


def test_anchor_must_link_and_identity_barrier():
    labels,trace=partition_surfaces(5,[0,1,2,3],[1,2,3,4],[1.,2.,-4.,3.],[7,0,8,0,7])
    assert labels[0]==labels[4]
    assert labels[2]!=labels[0]
    assert trace['maximum_eligible_remaining_gain']<=0


def test_sparse_objective_replay_with_must_links():
    rng=np.random.default_rng(11)
    for _ in range(30):
        n=12;a,b=np.triu_indices(n,1);keep=rng.random(len(a))<.35
        a,b=a[keep],b[keep];w=rng.normal(size=len(a));anchors=np.array([1,1,0,0,2,2,0,0,0,0,0,0])
        labels,trace=partition_surfaces(n,a,b,w,anchors)
        initial=(anchors[a]>0)&(anchors[a]==anchors[b])
        actual=w[labels[a]==labels[b]].sum()-w[initial].sum()
        np.testing.assert_allclose(actual,trace['objective_gain'],atol=1e-10)
        assert all(t[2]>0 for t in trace['merges'])


def test_unknown_never_negative():
    w,valid=edge_view_cost([0,-1,1,0],[1,0,-2,0],[[1.,.1],[.1,1.]])
    np.testing.assert_allclose(w,[-.8,0,0,1.])
    np.testing.assert_array_equal(valid,[True,False,False,True])


def test_empty_disconnected_and_edge_order():
    labels,_=partition_surfaces(3,[],[],[],[0,2,2]);assert labels.tolist()==[0,1,1]
    a=np.array([0,0,1,2]);b=np.array([1,2,3,3]);w=np.array([2.,-3.,4.,1.])
    first,_=partition_surfaces(4,a,b,w,[0]*4)
    second,_=partition_surfaces(4,a[::-1],b[::-1],w[::-1],[0]*4)
    np.testing.assert_array_equal(first,second)
