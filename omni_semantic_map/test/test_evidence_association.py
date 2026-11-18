import copy
import numpy as np
from omni_semantic_map.dino_features import pool_tokens
from omni_semantic_map.evidence_association import Config, MemoryAssociator, RevisableGraph, local_distance


def observation(frame, x=0., descriptor=(1., 0.)):
    return dict(keyframe_id=frame, local_position=[x,0,2], label='chair',
        descriptor=np.array(descriptor), appearance=[1.,0.],
        local_xyz=np.array([[x-.05,0,2],[x,0,2],[x+.05,0,2]]),
        local_features=np.tile(descriptor,(3,1)))


def test_pooling_preserves_tiny_masks_and_weighting():
    tokens = np.array([[[1.,0.],[0.,1.]]])
    descriptor, xyz, features = pool_tokens(tokens,np.array([[0,0],[1,0],[639,479]]),np.array([[0,0,2],[.2,0,2],[1,1,2]]))
    np.testing.assert_allclose(descriptor,np.array([2.,1.])/np.sqrt(5),atol=1e-6)
    np.testing.assert_allclose(xyz,[[.1,0,2],[1,1,2]])
    assert features.shape == (2,2)


def test_missing_local_overlap_is_unknown_not_negative():
    q=observation(0);q['world_tokens']=q['local_xyz']
    distance,n=local_distance(q,q['local_xyz']+10,q['local_features'],Config())
    assert distance is None and n==0


def test_dino_disambiguates_geometric_tie_without_new_objects():
    poses=np.tile(np.eye(4),(2,1,1))
    tracker=MemoryAssociator('dino_mean')
    assert tracker.add_frame([observation(0,0,(1.,0.)),observation(0,0,(0.,1.))],poses)==[1,2]
    assert tracker.add_frame([observation(1,0,(0.,1.)),observation(1,0,(1.,0.))],poses)==[2,1]


def test_graph_prevents_transitive_merge_of_covisible_objects():
    poses=np.tile(np.eye(4),(2,1,1));tracker=RevisableGraph()
    tracker.add_frame([observation(0,-.1),observation(0,.1)],poses)
    tracker.add_frame([observation(1,0)],poses)
    labels=tracker.labels()
    assert labels[0]!=labels[1] and labels[2] in labels[:2]
    assert tracker.events[-1]['co_visibility_conflicts']>0


def test_pose_revision_splits_and_can_reconnect_without_losing_observations():
    poses=np.tile(np.eye(4),(2,1,1));tracker=RevisableGraph()
    observations=[observation(0),observation(1)]; original=copy.deepcopy(observations)
    tracker.add_frame(observations[:1],poses);tracker.add_frame(observations[1:],poses)
    np.testing.assert_array_equal(tracker.labels(),[1,1])
    moved=poses.copy();moved[1,0,3]=2
    np.testing.assert_array_equal(tracker.revise_poses(moved),[1,2])
    assert tracker.events[-1]['reason']=='pose_revision'
    assert tracker.events[-1]['changed']==[dict(observation_id=1,old=1,new=2)]
    np.testing.assert_array_equal(tracker.revise_poses(poses),[1,1])
    for before,after in zip(original,tracker.observations):
        np.testing.assert_array_equal(before['local_xyz'],after['local_xyz'])


def test_later_bridge_revises_past_without_mutating_causal_labels():
    poses=np.tile(np.eye(4),(3,1,1));tracker=RevisableGraph()
    first=tracker.add_frame([observation(0,0)],poses)
    second=tracker.add_frame([observation(1,.3)],poses)
    # No 3 overlapping tokens at distance .3, but middle observation connects both.
    assert first!=second
    tracker.add_frame([observation(2,.15)],poses)
    assert len(set(tracker.labels()))==1
    assert first==[1] and second==[2]
    assert tracker.events[-1]['changed']


def test_zero_appearance_graph_control_is_independent_of_feature_values():
    poses=np.tile(np.eye(4),(3,1,1))
    a,b=RevisableGraph(Config(appearance_weight=0.)),RevisableGraph(Config(appearance_weight=0.))
    for k in range(3):
        a.add_frame([observation(k,.03*k,(1.,0.))],poses)
        b.add_frame([observation(k,.03*k,(-1.,0.) if k%2 else (0.,1.))],poses)
    np.testing.assert_array_equal(a.labels(),b.labels())
    assert a.edges==b.edges
