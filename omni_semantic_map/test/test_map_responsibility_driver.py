import numpy as np
from run_map_responsibility_supervision import training_assignment


def obj(oid,train):return dict(object_id=oid,predictions=dict(mean=dict(train_id=train)))


def test_matching_is_one_to_one_class_constrained_and_keeps_unmatched():
    truth=np.array([13001]*160+[13002]*120+[2001]*120+[0]*100)
    predicted=np.array([1]*100+[2]*60+[3]*120+[4]*120+[5]*100)
    assignment,rows=training_assignment(predicted,truth,np.ones(len(truth)),
                                        [obj(5,13),obj(4,13),obj(3,13),obj(2,13),obj(1,13)])
    assert assignment=={1:13001,2:-1,3:13002,4:-1,5:-1}
    assert len({r['gt'] for r in rows})==len(rows)==2


def test_unknown_mass_does_not_create_targets():
    truth=np.array([13001]*120+[13002]*80)
    predicted=np.array([1]*120+[2]*80);weight=np.r_[np.zeros(120),np.ones(80)]
    assignment,rows=training_assignment(predicted,truth,weight,[obj(1,13),obj(2,13)])
    assert assignment=={1:-1,2:-1} and rows==[]
