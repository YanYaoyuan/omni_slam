import numpy as np
from omni_semantic_map.appearance_partition import partition, validate
from omni_semantic_map.reversible_split import EvidenceView


def setup():
    xyz=np.r_[np.tile([-.5,0,0],(20,1)),np.tile([.5,0,0],(20,1))]
    feature=np.r_[np.tile([1.,0,0],(20,1)),np.tile([0,1.,0],(20,1))]
    counts=np.ones(40,int)
    p=partition(xyz,np.arange(40),feature,counts)
    assert p is not None
    p.update(object_id=1,center=xyz.mean(0),camera_center=np.array([0,2.,0]))
    masks=np.array([np.arange(40)<20,np.arange(40)>=20])
    views=[EvidenceView(i,np.array([2*np.sin(a),2*np.cos(a),0]),np.arange(40),masks.copy(),np.array([2*i,2*i+1]))
           for i,a in enumerate([.5,1.,1.5])]
    return p,feature,counts,views


def test_new_views_support_both_parts():
    p,f,c,v=setup()
    d=validate(p,np.ones(40,int),f,c,v,[0,1,2])
    assert d['accepted'] and min(d['independent_mask_support'])>=2


def test_texture_modes_without_instance_masks_are_not_objects():
    p,f,c,v=setup()
    for frame in v:
        frame.membership=np.ones((1,40),bool);frame.observation_ids=np.array([0])
    assert not validate(p,np.ones(40,int),f,c,v,[0,1,2])['accepted']


def test_view_specific_appearance_fails_validation():
    p,f,c,v=setup()
    changed=f[::-1].copy()
    assert not validate(p,np.ones(40,int),changed,c,v,[0,1,2])['accepted']


def test_no_appearance_signal_or_coverage_has_no_proposal():
    xyz=np.zeros((40,3));f=np.ones((40,3))
    assert partition(xyz,np.arange(40),f,np.ones(40)) is None
    assert partition(xyz,np.arange(40),f,np.zeros(40)) is None
