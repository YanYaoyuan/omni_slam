import numpy as np
import pytest
from omni_semantic_map.censored_partition_native import library,partition_native
from omni_semantic_map.censored_object_partition import make_histograms,merge_gain,join_histogram,region_score


@pytest.fixture(scope='module')
def lib(tmp_path_factory):
    return library(tmp_path_factory.mktemp('native_censored'))


def test_native_trace_replays_reference_greedy_objective(lib):
    rng=np.random.default_rng(13)
    for _ in range(30):
        obs=rng.integers(-1,3,(8,4));a,v=np.where(obs>=0);t=obs[a,v]
        w=rng.uniform(.1,1,len(a));k=np.full(4,4)
        left,right=np.triu_indices(8,1);keep=rng.random(len(left))>.3;left,right=left[keep],right[keep]
        result,trace=partition_native(8,k,left,right,a,v,t,w,lib)
        h=make_histograms(8,a,v,t,w,k);live=set(range(8));edges=set(zip(left.tolist(),right.tolist()))
        labels=np.arange(8)
        for x,y,got in trace['merges']:
            choices={(u,z):merge_gain(h[u],h[z],k) for u,z in edges}
            assert abs(got-choices[(x,y)])<1e-9
            assert got>=max(choices.values())-1e-9 and got>0
            h[x]=join_histogram(h[x],h[y]);h[y]={};live.remove(y);labels[labels==y]=x
            edges={tuple(sorted((x if u==y else u,x if z==y else z))) for u,z in edges if not({u,z}=={x,y})}
        np.testing.assert_array_equal(result,labels)
        assert max((merge_gain(h[u],h[z],k) for u,z in edges),default=0)<=1e-9
        assert abs(sum(region_score(h[j],k) for j in live)-trace['final_score'])<1e-8


def test_native_unknown_and_empty_domains(lib):
    empty=np.array([],np.int64)
    labels,trace=partition_native(0,empty,empty,empty,empty,empty,empty,np.array([],float),lib)
    assert len(labels)==0 and trace['regions']==0
    labels,trace=partition_native(2,np.array([2,2]),np.array([0]),np.array([1]),
        np.array([0,1]),np.array([0,1]),np.array([1,1]),np.ones(2),lib)
    np.testing.assert_array_equal(labels,[0,1]);assert not trace['merges']
    with pytest.raises(ValueError):
        partition_native(1,np.array([2]),empty,empty,np.array([0]),np.array([0]),np.array([9]),np.ones(1),lib)
