import numpy as np
from omni_semantic_map.shared_observations import select_views, classify_objects


def test_duplicate_views_do_not_outvote_other_frames():
    frames=np.array([0,0,10,20]);areas=np.array([10,5,8,6])
    assert select_views([0,1,1,2,3],frames,areas)==[0,2,3]
    f=np.array([[1,0],[0,1],[1,0],[0,1]],np.float32)
    rows=classify_objects([(1,[0,1,1,2,3])],f,frames,areas,np.eye(2),dict(names=['a','b'],replica_class_ids=[7,9]))
    assert rows[0]['unique_frames']==3
    assert rows[0]['predictions']['top10_l1_medoid']['replica_class_id']==7
    assert rows[0]['predictions']['all_views_mean']['replica_class_id']==7


def test_top_views_deterministic_and_medoid_is_member():
    f=np.eye(3,dtype=np.float32);frames=np.array([0,10,20]);areas=np.array([10,10,10])
    assert select_views([2,1,0],frames,areas,2)==[0,1]
    rows=classify_objects([(4,[2,0,1])],f,frames,areas,f,dict(names=['a','b','c'],replica_class_ids=[1,2,3]))
    assert rows[0]['object_id']==4 and rows[0]['medoid_observation_id']==0
