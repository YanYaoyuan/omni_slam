import itertools
import math
import numpy as np
from omni_semantic_map.censored_object_partition import (
    view_score,merge_gain,join_histogram,region_score,make_histograms,partition)


def test_exact_integral_of_small_categorical_sequences():
    # K=2, uniform Beta(1/2,1/2): evaluate independent polynomial integration.
    for n in range(1,7):
        for sequence in itertools.product([0,1],repeat=n):
            p=sum(sequence);q=n-p
            expected=math.lgamma(p+.5)+math.lgamma(q+.5)-math.lgamma(n+1)-2*math.lgamma(.5)
            assert abs(view_score({0:q,1:p},2)-expected)<1e-12
    assert view_score({},2)==0


def test_missing_view_is_neutral_and_conflicting_known_mask_is_not():
    a={0:{1:8.}};same={0:{1:8.}};different={0:{2:8.}};unknown={1:{2:8.}}
    assert merge_gain(a,same,[3,3])>0
    assert merge_gain(a,different,[3,3])<0
    assert merge_gain(a,unknown,[3,3])==0
    assert merge_gain(a,join_histogram(same,unknown),[3,3])==merge_gain(a,same,[3,3])


def test_single_merged_mask_does_not_override_later_separation():
    # View0 merges two physical objects; later observations separate them.
    h=[{0:{1:1.},1:{1:4.}}, {0:{1:1.},1:{1:4.}},
       {0:{1:1.},1:{2:4.}}, {0:{1:1.},1:{2:4.}}]
    left=np.array([0,0,1,2]);right=np.array([1,2,3,3])
    labels,trace=partition(h,[2,3],left,right)
    assert labels[0]==labels[1] and labels[2]==labels[3] and labels[0]!=labels[2]
    assert trace['score_gain']>0


def test_sparse_counts_and_graph_permutation_replay():
    rng=np.random.default_rng(12)
    for _ in range(20):
        obs=rng.integers(-1,3,(7,4));aa,ff=np.where(obs>=0)
        h=make_histograms(7,aa,ff,obs[aa,ff],np.ones(len(aa)),np.full(4,4))
        l,r=np.triu_indices(7,1);mask=rng.random(len(l))>.3;l,r=l[mask],r[mask]
        out,trace=partition(h,np.full(4,4),l,r)
        order=rng.permutation(len(l));other,_=partition(h,np.full(4,4),l[order],r[order])
        np.testing.assert_array_equal(out,other)
        state=[{f:dict(v) for f,v in x.items()} for x in h]
        for a,b,gain in trace['merges']:
            assert abs(merge_gain(state[a],state[b],np.full(4,4))-gain)<1e-12
            state[a]=join_histogram(state[a],state[b]);state[b]={}
        expected=sum(region_score(s,np.full(4,4)) for s in state)
        assert abs(expected-trace['final_score'])<1e-10
