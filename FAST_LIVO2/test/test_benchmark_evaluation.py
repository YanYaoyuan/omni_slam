import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_ntu_benchmark import rigid_align,interpolate_reference,prism_positions,PRISM_IN_IMU,relative_displacement


def test_prism_offset_rotates_with_body():
    q=Rotation.from_euler('z',90,degrees=True).as_quat()
    track=np.array([[0.,1.,2.,3.,*q]])
    np.testing.assert_allclose(prism_positions(track)[0],np.array([1,2,3])+[-PRISM_IN_IMU[1],PRISM_IN_IMU[0],PRISM_IN_IMU[2]])


def test_rigid_alignment_recovers_pose_but_does_not_hide_scale_error():
    points=np.random.default_rng(8).normal(size=(50,3))
    r=Rotation.from_euler('xyz',[.2,-.4,.5]).as_matrix()
    target=points@r.T+[2,-1,4]
    fit,*_=rigid_align(points,target)
    np.testing.assert_allclose(fit,target,atol=1e-12)
    fit,*_=rigid_align(points*2,target)
    assert np.sqrt(np.mean(np.sum((fit-target)**2,axis=1)))>.5


def test_reference_gaps_and_outside_times_are_not_extrapolated():
    gt=np.array([[0.,0,0,0],[.05,1,0,0],[1.,2,0,0],[1.05,3,0,0]])
    _,valid=interpolate_reference(np.array([-.1,.025,.5,1.025,1.2]),gt)
    assert valid.tolist()==[False,True,False,True,False]


def test_relative_displacement_ignores_constant_translation():
    t=np.arange(0.,5.,.1);p=np.column_stack([t,t**2,np.sin(t)])
    metric=relative_displacement(t,p+[2,3,4],p,1.)
    assert metric['pairs']>20 and metric['rmse_m']<1e-12
