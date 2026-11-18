import numpy as np
from omni_semantic_map.competitive_graph import CompetitiveGraph


def fixture():
    poses=np.tile(np.eye(4),(5,1,1));poses[:,0,3]=[0,0,.15,.3,.45]
    layouts=[[0,0,0,1,1,1],[2]*6,[3]*6,[4,4,4,5,5,5],[6,6,6,7,7,7]]
    views=[];groups=[]
    for k,layout in enumerate(layouts):
        owner=np.array([layout]);x=poses[k,0,3]
        views.append(dict(owner=owner,depth=np.full((1,6),2.),valid=np.ones((1,6),bool),stride=1,
                          intrinsic=np.array([[100.,0,2.5+50*x],[0,100.,0],[0,0,1]])))
        group=[]
        for oid in sorted(set(layout)):
            us=np.flatnonzero(owner[0]==oid);xyz=np.column_stack([(us-2.5)/50-x,np.zeros(len(us)),np.full(len(us),2.)])
            group.append(dict(observation_id=oid,keyframe_id=k,label='thing',local_position=np.median(xyz,axis=0),
                local_xyz=xyz,local_features=np.tile([1.,0.],(len(us),1)),descriptor=np.array([1.,0.]),support_xyz=xyz))
        groups.append(group)
    return poses,views,groups


def test_joint_coverage_witness_preserves_fragment_merge_then_separation_revokes_it():
    poses,views,groups=fixture();tracker=CompetitiveGraph(views)
    for g in groups[:2]:tracker.add_frame(g,poses)
    assert tracker.labels()[0]!=tracker.labels()[1]
    tracker.add_frame(groups[2],poses)
    # The strict ownership gate would never allow this merge.
    assert tracker.labels()[0]==tracker.labels()[1]
    for g in groups[3:]:tracker.add_frame(g,poses)
    assert tracker.labels()[0]!=tracker.labels()[1]
    assert tracker.events[-1]['separation_vetoes']


def test_pose_revision_equals_clean_replay_and_retains_causal_outputs():
    poses,views,groups=fixture();tracker=CompetitiveGraph(views);causal=[]
    for g in groups:causal.extend(tracker.add_frame(g,poses))
    snapshot=tracker.labels();old_causal=list(causal)
    corrected=poses.copy();corrected[2,0,3]+=.3
    got=tracker.revise_poses(corrected);clean=CompetitiveGraph(views)
    for g in groups:clean.add_frame(g,corrected)
    np.testing.assert_array_equal(got,clean.labels())
    np.testing.assert_array_equal(tracker.revise_poses(poses),snapshot)
    assert causal==old_causal
