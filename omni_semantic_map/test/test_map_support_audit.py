import numpy as np
from omni_semantic_map.map_support_audit import covered_by,object_coverage,token_groups,label_distribution
from omni_semantic_map.livo_dino_support import pool


def test_half_coverage_is_not_enough_and_zero_gt_is_background():
    truth=np.array([0,1,1,2,2,2])
    rows=object_coverage(truth,np.array([1,1,0,1,1,0],bool))
    assert [r['can_exceed_iou_half'] for r in rows]==[False,True]
    assert [r['covered'] for r in rows]==[1,2]


def test_distance_radius_and_empty_geometry():
    ref=np.array([[0,0,0],[.049,0,0],[.051,0,0]])
    np.testing.assert_array_equal(covered_by(ref[:1],ref),[True,True,False])
    assert not covered_by(np.empty((0,3)),ref).any()


def test_token_provenance_matches_actual_pool_order_and_features():
    rng=np.random.default_rng(7);tokens=rng.normal(size=(4,5,8)).astype(np.float32)
    xyz=np.array([[x+.2,y+.2,1.] for y in range(4) for x in range(5)]+[[.3,.3,1.],[0,0,-1],[7,8,1.]])
    bins,groups=token_groups(xyz,np.eye(4),np.eye(3),5,4,5,4,max_tokens=7)
    _,anchors,features=pool(tokens,xyz,np.eye(4),np.eye(3),5,4,max_tokens=7)
    expected=tokens.reshape(-1,8)[bins];expected/=np.linalg.norm(expected,axis=1,keepdims=True)
    np.testing.assert_allclose(features,expected,atol=1e-7)
    np.testing.assert_allclose(anchors,np.array([np.median(xyz[g],0) for g in groups]),atol=1e-7)
    assert sum(len(g) for g in groups)==8  # one selected bin contains duplicate support
    assert not any(21 in g or 22 in g for g in groups)


def test_mixed_and_unknown_supervision_is_preserved():
    d=label_distribution(np.array([-1,0,3,3,7]))
    assert d==dict(ids=[-1,0,3,7],counts=[1,1,2,1],total=5,valid=4,foreground=3,foreground_instances=2)
    assert label_distribution(np.array([],dtype=int))['total']==0
