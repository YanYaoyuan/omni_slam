import numpy as np
from omni_semantic_map.ownership_graph import ownership_evidence,OwnershipGraph


def view(owner,depth=None):
    owner=np.array([owner],np.int32)
    return dict(owner=owner,depth=np.full(owner.shape,2.) if depth is None else np.array([depth]),
                valid=np.ones(owner.shape,bool),stride=1,intrinsic=np.eye(3))


def points(xs,z=2.):
    return np.array([[x*z,0,z] for x in xs],float)


def test_occluded_unknown_and_out_of_view_points_abstain():
    v=view([4,4,4,4],depth=[2,1,2,2]);v['valid'][0,2]=False
    r=ownership_evidence(points([0,1,2,8]),np.eye(4),np.eye(4),v,4)
    assert r['visible']==1 and r['owned']==1 and r['occluded']==1
    assert r['fraction']==1.


def test_visible_other_mask_background_and_free_space_are_not_ownership():
    r=ownership_evidence(points([0,1,2,3]),np.eye(4),np.eye(4),view([4,5,-1,4],[2,2,2,3]),4)
    assert r['visible']==4 and r['owned']==1 and r['free_space']==1


def test_many_points_at_one_target_pixel_do_not_inflate_support():
    r=ownership_evidence(points([0,0,.1]),np.eye(4),np.eye(4),view([4]),4)
    assert r['owned']==1


def test_mixed_parent_mask_cannot_claim_majority_in_both_children():
    parent=view([2]*6);children=view([0]*3+[1]*3)
    p=np.eye(4)
    assert ownership_evidence(points([0,1,2]),p,p,parent,2)['fraction']==1.
    assert ownership_evidence(points(range(6)),p,p,children,0)['fraction']==.5
    assert ownership_evidence(points(range(6)),p,p,children,1)['fraction']==.5


def test_pose_revision_recomputes_visibility_and_roundtrip_restores_edges():
    k=np.array([[100.,0,3],[0,100.,0],[0,0,1]])
    xyz=np.array([[-.02,0,2],[0,0,2],[.02,0,2]])
    views=[];obs=[]
    for i in range(2):
        v=view([i]*7);v['intrinsic']=k;views.append(v)
        obs.append(dict(observation_id=i,keyframe_id=i,label='thing',local_position=[0,0,2],
            local_xyz=xyz,local_features=np.tile([1.,0.],(3,1)),descriptor=np.array([1.,0.]),support_xyz=xyz))
    poses=np.tile(np.eye(4),(2,1,1));t=OwnershipGraph(views)
    t.add_frame(obs[:1],poses);t.add_frame(obs[1:],poses)
    assert t.labels().tolist()==[1,1]
    corrected=poses.copy();corrected[1,2,3]=.2
    changed=t.revise_poses(corrected)
    assert changed.tolist()==[1,2]
    clean=OwnershipGraph(views)
    clean.add_frame(obs[:1],corrected);clean.add_frame(obs[1:],corrected)
    np.testing.assert_array_equal(changed,clean.labels())
    np.testing.assert_array_equal(t.revise_poses(poses),[1,1])
