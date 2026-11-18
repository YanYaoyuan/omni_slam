import numpy as np
import pytest
from omni_semantic_map.reversible_support_graph import (ReversibleSupportGraph,SupportGraphConfig,
    frame_atom_labels,partition_signed,atomize)


def test_merged_same_class_objects_can_split_and_rejoin():
    # Geometry-only evidence; no class ID is available to this implementation.
    g=ReversibleSupportGraph(4,[[0,1],[1,2],[2,3]])
    g.update(0,[0,0,0,0]);assert len(set(g.labels))==1
    for frame in [1,2]:g.update(frame,[10,10,20,20])
    assert g.labels.tolist()==[1,1,3,3] and g.events[-1]['split_lineage']=={'1':[1,3]}
    # Independent replay and later positive evidence can undo that partition.
    g.update(3,[30,30,30,30]);assert len(set(g.labels))==1
    h=ReversibleSupportGraph(4,[[0,1],[1,2],[2,3]])
    for frame,obs in [(0,[0]*4),(1,[10,10,20,20]),(2,[10,10,20,20]),(3,[30]*4)]:h.update(frame,obs)
    np.testing.assert_array_equal(g.labels,h.labels);assert g.events==h.events


def test_ablation_locks_prior_associations_but_keeps_same_evidence():
    a=ReversibleSupportGraph(2,[[0,1]]);b=ReversibleSupportGraph(2,[[0,1]],allow_revoke=False)
    for frame,obs in [(0,[0,0]),(1,[1,2]),(2,[1,2])]:a.update(frame,obs);b.update(frame,obs)
    assert a.labels.tolist()==[1,2] and b.labels.tolist()==[1,1]
    assert b.check_export()==1
    np.testing.assert_array_equal(a.positive,b.positive);np.testing.assert_array_equal(a.negative,b.negative)


def test_occlusion_and_ambiguous_cells_are_not_negative_evidence():
    g=ReversibleSupportGraph(2,[[0,1]]);g.update(0,[0,0])
    for frame in [1,2,3]:g.update(frame,[1,-1])
    assert g.negative.tolist()==[0] and g.labels.tolist()==[1,1]
    lab=frame_atom_labels(np.array([0,0,1]),[0,1,2],[10,20,30],2)
    assert lab.tolist()==[-1,30]


def test_component_wide_conflict_blocks_transitive_merge_and_final_bypass():
    labels=partition_signed([0,1,2],[[0,1],[1,2]],[2.,1.],[[0,2]])
    assert labels.tolist()==[1,1,3]
    g=ReversibleSupportGraph(3,[[0,1],[1,2],[0,2]])
    g.update(0,[0,0,1]);g.update(1,[0,0,1])
    with pytest.raises(ValueError):g.check_export(np.ones(3,np.int64))


def test_support_tokens_exclusive_and_input_immutable():
    xyz=np.array([[0.,0.,0.],[.01,0,0],[.051,0,0]]);copy=xyz.copy()
    _,mapping,edges=atomize(xyz);np.testing.assert_array_equal(xyz,copy)
    assert mapping.tolist()==[0,0,1] and edges.tolist()==[[0,1]]
    with pytest.raises(ValueError):frame_atom_labels(mapping,[0,0],[10,11],2)


def test_budget_failure_and_duplicate_frame_are_transactional():
    g=ReversibleSupportGraph(3,[[0,1],[1,2]],SupportGraphConfig(max_repair_atoms=2))
    g.update(0,[0,0,-1]);old=g.labels.copy();positive=g.positive.copy()
    with pytest.raises(RuntimeError):g.update(1,[0,0,0])
    assert g.frame==0 and g.revision==1
    np.testing.assert_array_equal(g.labels,old);np.testing.assert_array_equal(g.positive,positive)
    with pytest.raises(ValueError):g.update(0,[0,0,-1])
