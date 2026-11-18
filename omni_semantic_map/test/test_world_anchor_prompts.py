import numpy as np
from omni_semantic_map.world_anchor_prompts import prompts,spread,calibration_blocks


def test_prompts_do_not_read_withheld_or_conflicting_anchors():
    ids=np.arange(30);xy=np.column_stack([ids,ids%3]);owners=np.r_[np.ones(20,int),np.full(8,2),-1,0];cells=ids;allowed=np.ones(30,bool);allowed[19]=False
    p=prompts(ids,xy,owners,cells,allowed,1);s=np.array(p['source_ids'])
    assert p['eligible'] and p['own_cells']==19 and len(s)==6 and allowed[s].all()
    assert np.all(owners[s[:3]]==1) and np.all(owners[s[3:]]==2)
    changed=owners.copy();changed[19]=99
    assert prompts(ids,xy,changed,cells,allowed,1)==p


def test_duplicate_measurements_do_not_create_cell_eligibility():
    ids=np.arange(40);p=prompts(ids,np.zeros((40,2)),np.ones(40,int),np.zeros(40,int),np.ones(40,bool),1)
    assert not p['eligible'] and p['own_cells']==1


def test_spread_and_block_split_are_deterministic():
    x=np.array([[0.,0.],[1,0],[2,0],[10,0]])
    np.testing.assert_array_equal(spread(x,3),spread(x,3));assert len(np.unique(spread(x,9)))==4
    xyz=np.array([[0.1,0,0],[0.3,0,0],[2.1,0,0]])
    np.testing.assert_array_equal(calibration_blocks(xyz),[True,True,False])
