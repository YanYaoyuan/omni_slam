import numpy as np
from omni_semantic_map.evidence_rerank import rank_accepted,local_correspondence_score


def test_never_add_ineligible_edge_or_force_evidence():
    assert rank_accepted([1.3,-np.inf],[.2,.99])==(0,'unique_candidate')
    assert rank_accepted([-np.inf,-np.inf],[.9,.8])==(None,'new_object')
    assert rank_accepted([1.3,1.4],[.99,np.nan])==(1,'insufficient_evidence')


def test_rank_and_exact_tie_keep_native_order():
    assert rank_accepted([1.3,1.4],[.9,.8])==(0,'evidence_rank')
    assert rank_accepted([1.3,1.4],[.9,.9])==(1,'evidence_rank')
    assert rank_accepted([1.4,1.4],[.9,.9])==(0,'evidence_rank')


def test_local_matches_require_geometry_and_distinct_mutual_pairs():
    p=np.array([[0.,0,0],[1,0,0],[2,0,0]]);f=np.eye(3)
    score,n=local_correspondence_score(p,f,p+[.1,0,0],f)
    assert score==1 and n==3
    score,n=local_correspondence_score(p,f,p+[0,1,0],f)
    assert np.isnan(score) and n==0
    score,n=local_correspondence_score(p[:2],f[:2],p[:2],f[:2])
    assert np.isnan(score) and n==2


def test_rigid_coordinate_change_preserves_local_score():
    p=np.array([[0.,0,0],[1,0,0],[2,0,0]]);q=p+[.1,0,0];f=np.eye(3)
    R=np.array([[0,-1.,0],[1,0,0],[0,0,1]])
    assert local_correspondence_score(p,f,q,f)==local_correspondence_score(p@R+7,f,q@R+7,f)
