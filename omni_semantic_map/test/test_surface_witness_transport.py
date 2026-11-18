import numpy as np
from omni_semantic_map.surface_witness_transport import local_planes,surface_edges,frame_votes


def test_parallel_and_crossing_surfaces_do_not_share_evidence():
    target=np.array([[0.,0.,0.]])
    # Same plane, parallel but separated, and intersecting orthogonal plane.
    witnesses=np.array([[.1,0.,0.],[0.,0.,.1],[.02,0.,0.]])
    row,col=surface_edges(target,[[0,0,1]],[True],witnesses,[[0,0,1],[0,0,1],[1,0,0]],[True]*3)
    np.testing.assert_array_equal(col,[0])
    np.testing.assert_array_equal(frame_votes(1,row,col,[True,False,False]),[1])


def test_conflict_and_missing_measurements_abstain_and_count_once():
    np.testing.assert_array_equal(frame_votes(4,[0,0,1,1,2],[0,1,2,3,4],[True,False,False,False,True]),[0,-1,1,0])
    np.testing.assert_array_equal(frame_votes(1,[0]*100,list(range(100)),[False]*100),[-1])


def test_degenerate_and_off_surface_neighbourhoods_are_unknown():
    line=np.column_stack([np.linspace(-.1,.1,16),np.zeros((16,2))])
    _,valid=local_planes(line,[[0,0,0]])
    assert not valid[0]
    a,b=np.meshgrid(np.linspace(-.08,.08,7),np.linspace(-.08,.08,7)); plane=np.column_stack([a.ravel(),b.ravel(),np.zeros(a.size)])
    _,valid=local_planes(plane,[[0,0,0],[0,0,.08],[2,2,2]])
    np.testing.assert_array_equal(valid,[True,False,False])


def test_transport_and_plane_fit_are_rigid_transform_invariant():
    a,b=np.meshgrid(np.linspace(-.08,.08,7),np.linspace(-.08,.08,7)); plane=np.column_stack([a.ravel(),b.ravel(),np.zeros(a.size)])
    target=np.array([[0,0,0],[.02,.02,0]])
    tn,tv=local_planes(plane,target); wn,wv=local_planes(plane,plane)
    r,c=surface_edges(target,tn,tv,plane,wn,wv)
    angle=.731; rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    translation=np.array([2.7,-5.3,1.9]); moved=plane@rotation.T+translation; query=target@rotation.T+translation
    n,v=local_planes(moved,query); mn,mv=local_planes(moved,moved)
    rr,cc=surface_edges(query,n,v,moved,mn,mv)
    np.testing.assert_array_equal(tv,v); np.testing.assert_array_equal(wv,mv)
    np.testing.assert_array_equal(r,rr); np.testing.assert_array_equal(c,cc)


def test_invalid_surface_fit_never_generates_a_vote():
    r,c=surface_edges([[0,0,0]],[[0,0,1]],[False],[[0,0,0]],[[0,0,1]],[True])
    assert not len(r)
    np.testing.assert_array_equal(frame_votes(1,r,c,[False]),[0])
