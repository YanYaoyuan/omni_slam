from copy import deepcopy
import numpy as np
import pytest
from omni_semantic_map.category_stream import CategoryStream


def stream():return CategoryStream(np.eye(2),{'names':['thing','background'],'replica_class_ids':[7,0]},'test')
def event(rev=0,feature=None):
    return dict(schema_version=1,session_id='run1',feature_space='test',revision=rev,frame_id=rev,
        observations=[dict(id=rev,frame_id=rev,area=10,feature=feature or [1,0],evidence='frame.png')],
        objects=[dict(id='object1',observation_ids=list(range(rev+1)),retained=True,geometry_revision='geometry-unchanged')])


def test_category_is_not_a_retention_or_geometry_decision():
    s=stream();e=event(feature=[0,1]);before=deepcopy(e);out=s.apply(e)
    assert e==before and out['objects'][0]['category']['name']=='background'
    assert out['objects'][0]['retained'] and out['objects'][0]['geometry_revision']=='geometry-unchanged'


def test_rejected_events_are_atomic_and_retryable():
    s=stream();s.apply(event());bad=event(1);bad['objects'][0]['observation_ids'].append(99)
    with pytest.raises(ValueError):s.apply(bad)
    assert s.revision==0 and set(s.observations)=={0}
    s.apply(event(1));assert s.revision==1
    with pytest.raises(ValueError):s.apply(event(1))
    with pytest.raises(ValueError):s.apply(event(3))
    assert s.revision==1


def test_session_feature_space_and_ownership_are_enforced():
    for modify in [lambda e:e.update(feature_space='other'),lambda e:e['objects'].append(dict(e['objects'][0],id='object2'))]:
        s=stream();e=event();modify(e)
        with pytest.raises(ValueError):s.apply(e)
        assert s.revision==-1
    s=stream();s.apply(event());e=event(1);e['session_id']='new-run'
    with pytest.raises(ValueError):s.apply(e)


def test_membership_merge_and_geometry_revision_do_not_rewrite_evidence():
    s=stream();s.apply(event());e=event(1);e['objects'][0].update(id='merged-object',geometry_revision='pose_revision_2')
    out=s.apply(e);assert out['objects'][0]['id']=='merged-object'
    assert set(out['objects'][0]['category']['selected_observation_ids'])=={0,1}


def test_same_frame_area_ties_use_global_ids_not_membership_order():
    s=stream();e=event()
    e['observations'].append(dict(id=1,frame_id=0,area=10,feature=[0,1]))
    e['objects'][0]['observation_ids']=[1,0]
    result=s.apply(e)['objects'][0]['category']
    assert result['selected_observation_ids']==[0] and result['name']=='thing'
