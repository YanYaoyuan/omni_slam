import numpy as np
from omni_semantic_map.deferred_surface_support import recover


def fixture():
    # A connected discarded patch with two 20cm cells, only one reobserved.
    xyz=np.array([[0,0,0],[2,0,0],[2.25,0,0],[2.01,0,0]],float)
    cells=np.array([0,10,11,10]);features=np.array([[1.,0],[1.,0],[0.,1.]])
    observations=[dict(id=0,frame=0,semantic=26,raw=np.array([0,1,2]),baseline=np.array([0])),
                  dict(id=1,frame=5,semantic=26,raw=np.array([3]),baseline=np.array([],int))]
    return xyz,cells,features,observations


def test_component_recovers_historical_measured_patch_with_future_time():
    xyz,cells,f,obs=fixture();r,events=recover(obs,xyz,cells,f)
    assert set(r[0]['component'])=={0,1,2}
    assert set(r[0]['pointwise'])=={0,1}
    assert r[0]['component_times'].tolist()==[0,5,5]
    event=next(e for e in events if e['observation_id']==0)
    assert event['witness_observations']==[1] and event['added_source_ids']==[1,2]


def test_prefix_causality_and_no_self_confirmation():
    xyz,cells,f,obs=fixture();full,_=recover(obs,xyz,cells,f);prefix,_=recover(obs[:1],xyz,cells,f)
    assert full[0]['component'][full[0]['component_times']<=0].tolist()==prefix[0]['component'].tolist()==[0]
    obs[1]['frame']=0;r,_=recover(obs,xyz,cells,f);assert r[0]['component'].tolist()==[0]


def test_semantic_and_descriptor_incompatibility_do_not_confirm():
    xyz,cells,f,obs=fixture();obs[1]['semantic']=11;r,_=recover(obs,xyz,cells,f);assert r[0]['component'].tolist()==[0]
    obs[1]['semantic']=26;obs[1]['id']=2;r,_=recover(obs,xyz,cells,f);assert r[0]['component'].tolist()==[0]


def test_duplicate_points_cannot_manufacture_majority():
    xyz,cells,f,obs=fixture();xyz=np.vstack([xyz,[2.02,0,0],[2.03,0,0]])
    cells=np.r_[cells,[10,10]];obs[0]['raw']=np.array([0,1,2,4,5])
    r,events=recover(obs,xyz,cells,f,fraction=1.)
    assert r[0]['component'].tolist()==[0]
    assert set(r[0]['pointwise'])=={0,1,4,5}
