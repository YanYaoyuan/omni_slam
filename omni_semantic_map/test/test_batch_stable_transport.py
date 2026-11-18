import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from omni_semantic_map.batch_stable_transport import native_library,nearest_groups,nearest_owner,certify


@pytest.fixture(scope='module')
def lib(tmp_path_factory):return native_library(tmp_path_factory.mktemp('transport_native'))


def solve(graph,node,owner,batch,lib):
    labels,dist=nearest_groups(graph,node,owner,lib)
    pairs,tok=np.unique(np.column_stack([owner,batch]),axis=0,return_inverse=True)
    tokens,td=nearest_groups(graph,node,tok,lib)
    return nearest_owner(labels,dist),certify(labels,dist,tokens,td,pairs[:,0]),(labels,dist,tokens,td)


def test_random_graph_against_exhaustive_seed_groups(lib):
    rng=np.random.RandomState(12)
    for _ in range(20):
        a=rng.uniform(.01,2,(17,17));a[rng.rand(17,17)<.8]=0;a=np.maximum(a,a.T);np.fill_diagonal(a,0)
        graph=csr_matrix(a);node=rng.randint(0,17,13);group=rng.randint(0,5,13)
        lab,dist=nearest_groups(graph,node,group,lib)
        allpath=dijkstra(graph,indices=node)
        unique=np.unique(group);cost=np.column_stack([allpath[group==g].min(0) for g in unique])
        for j in range(17):
            order=sorted(range(len(unique)),key=lambda k:(cost[j,k],unique[k]))
            expect=[k for k in order if np.isfinite(cost[j,k])][:2]
            np.testing.assert_allclose(dist[j,:len(expect)],cost[j,expect],atol=1e-12)
            np.testing.assert_array_equal(lab[j,:len(expect)],unique[expect])
            assert np.all(lab[j,len(expect):]==-1)


def test_same_batch_duplicates_do_not_provide_alternative(lib):
    graph=csr_matrix([[0,1,0,0],[1,0,1,0],[0,1,0,4],[0,0,4,0]])
    owner=np.array([1,1,2]);node=np.array([0,1,3])
    near,same,_=solve(graph,node,owner,np.array([5,5,8]),lib)
    _,different,_=solve(graph,node,owner,np.array([5,6,8]),lib)
    assert near[1]==1 and same[1]==-1 and different[1]==1
    _,repeat,_=solve(graph,np.r_[node,0,0],np.r_[owner,1,1],np.array([5,5,8,5,5]),lib)
    np.testing.assert_array_equal(same,repeat)


def test_every_certificate_survives_actual_batch_deletion(lib):
    rng=np.random.RandomState(8)
    for _ in range(20):
        a=rng.uniform(.1,2,(23,23));a[rng.rand(23,23)<.85]=0;a=np.maximum(a,a.T);np.fill_diagonal(a,0)
        graph=csr_matrix(a);node=rng.randint(0,23,20);owner=rng.randint(0,4,20);batch=rng.randint(0,6,20)
        _,accepted,_=solve(graph,node,owner,batch,lib)
        for b in np.unique(batch):
            keep=batch!=b;lab,dist=nearest_groups(graph,node[keep],owner[keep],lib);after=nearest_owner(lab,dist)
            np.testing.assert_array_equal(after[accepted>=0],accepted[accepted>=0])


def test_competitor_tie_and_missing_second_batch_abstain(lib):
    graph=csr_matrix([[0,1,0,0],[1,0,1,0],[0,1,0,0],[0,0,0,0]])
    near,accepted,_=solve(graph,np.array([0,2,3]),np.array([1,2,3]),np.array([0,1,2]),lib)
    assert near[1]==-1 and accepted[1]==-1 and accepted[3]==-1
    lab,dist=nearest_groups(graph,np.array([0]),np.array([1]),lib)
    assert lab[3,0]==-1 and np.isinf(dist[3,0])


def test_seed_order_and_batch_relabel_invariance(lib):
    graph=csr_matrix([[0,1,0],[1,0,1],[0,1,0]])
    node=np.array([0,1,2]);owner=np.array([1,1,2]);batch=np.array([2,4,6])
    a=solve(graph,node,owner,batch,lib)[1]
    b=solve(graph,node[::-1],owner[::-1],100-batch[::-1],lib)[1]
    np.testing.assert_array_equal(a,b)


def test_invalid_negative_edge_rejected(lib):
    with pytest.raises(ValueError):nearest_groups(csr_matrix([[0,-1],[-1,0]]),[0],[1],lib)
