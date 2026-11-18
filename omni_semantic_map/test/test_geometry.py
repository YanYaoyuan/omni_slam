import numpy as np
from omni_semantic_map.geometry import local_ground, object_box, observed_bounds


def test_ground_accepts_slope_rejects_wall_and_keeps_elevation():
    x,y=np.meshgrid(np.linspace(-4,4,30),np.linspace(-4,4,30))
    ground=np.c_[x.ravel(),y.ravel(),(.10*x-.12).ravel()]
    wy,wz=np.meshgrid(np.linspace(-4,4,30),np.linspace(0,3,30))
    wall=np.c_[np.full(900,2.),wy.ravel(),wz.ravel()]
    points=np.r_[ground,wall]
    mask,_=local_ground(points,np.ones(len(points),bool),[0,0,0],seed=2)
    assert mask[:900].mean()>.8
    assert mask[900:].mean()<.15
    assert np.ptp(points[mask,2])>.5  # classified points were not flattened


def test_boxes_follow_keyframe_correction():
    pts=np.array([[x,y,z] for x in [-1,1] for y in [-.5,.5] for z in [0,2]])
    ob={'keyframe_id':0,'local_position':[0,0,1],'local_support_bounds':observed_bounds(pts)}
    pose=np.eye(4);a=object_box([ob],[pose],np.array([0,0,1]))
    pose[:3,3]=[10,-4,2];b=object_box([ob],[pose],np.array([10,-4,3]))
    assert np.allclose(np.array(b['center'])-a['center'],[10,-4,2])
    assert np.allclose(a['extent'],b['extent'])
