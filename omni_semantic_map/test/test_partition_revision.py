import numpy as np
import pytest
from omni_semantic_map.partition_revision import partition_fit,select_partition,revise_history,propose_measured_sources


def test_duplicate_query_ids_cannot_fabricate_two_objects():
    a,b=np.arange(20),np.arange(20,40);whole=np.arange(40)
    # Multiple decoder queries with the SAME support remain one object evidence.
    assert partition_fit([a,b],[whole,whole],[1.,1.],whole)==0.
    assert partition_fit([whole],[whole,whole],[1.,1.],whole)==1.
    assert partition_fit([a,b],[a,b],[1.,1.],whole)==1.
    assert partition_fit([whole],[a,b],[1.,1.],whole)==.5


def test_shared_visibility_and_prompt_separation():
    a,b=np.arange(20),np.arange(20,40);whole=np.arange(40)
    candidates={'keep':[a,b],'merge':[whole]}
    view=dict(frame_id=1,visible_cells=whole,queries=[whole],scores=[1.])
    assert select_partition([a,b],candidates,[view])['selected']=='merge'
    assert select_partition([a,b],candidates,[dict(view,queries=[a,b],scores=[1.,1.])])['selected']=='keep'
    # No visibility of B is not evidence against a two-object explanation.
    assert select_partition([a,b],candidates,[dict(view,visible_cells=a,queries=[a])])['reason']=='no_common_measured_validation'
    with pytest.raises(ValueError):select_partition([a,b],candidates,[dict(view,used_for_prompt=True)])
    assert select_partition([a,b],candidates,[view],valid={'keep':True,'merge':False})['selected']=='keep'


def test_hidden_query_support_rejected_and_score_tie_keeps_original():
    with pytest.raises(ValueError):partition_fit([np.arange(20)],[np.arange(21)],[1.],np.arange(20))
    a,b=np.arange(20),np.arange(20,40)
    result=select_partition([a,b],{'keep':[a,b],'split':[a,b]},[
        dict(frame_id=1,visible_cells=np.arange(40),queries=[a,b],scores=[.8,.9])])
    assert result['selected']=='keep'


def test_history_is_reassigned_without_stealing_other_objects_or_unknown_points():
    old=np.array([1,1,2,2,9,1,2])
    a=np.array([0,0,1,0,1,1,0],bool);b=np.array([0,1,0,0,0,1,0],bool)
    actual=revise_history(old,1,2,'split',a,b)
    np.testing.assert_array_equal(actual,[1,2,1,2,9,1,2])
    np.testing.assert_array_equal(old,[1,1,2,2,9,1,2])
    np.testing.assert_array_equal(revise_history(old,1,2,'merge'),[1,1,1,1,9,1,1])


def test_source_proposals_keep_unseen_history_and_abstain_on_new_overlap():
    a,b=propose_measured_sources([0,1],[2,3],[2,4,6,8],[1,5,6,8],'split',[8])
    np.testing.assert_array_equal(a,[0,2,4]);np.testing.assert_array_equal(b,[1,3,5])
    assert not np.intersect1d(a,b).size
    merged=propose_measured_sources([0,1],[2,3],[2,4,6,8],[1,5,6,8],'merge',[8])[0]
    np.testing.assert_array_equal(merged,np.arange(7))


def test_current_observation_ablation_does_not_silently_recover_historical_sources():
    a,b=propose_measured_sources([0,1],[2,3],[2,4,6,8],[1,5,6,8],'split',[8],include_history=False)
    np.testing.assert_array_equal(a,[2,4]);np.testing.assert_array_equal(b,[1,5])
    with pytest.raises(ValueError):propose_measured_sources([-1],[],[],[],'merge')


def test_precomputed_query_overlaps_preserve_exact_assignment_objective():
    parts=[np.arange(0,24),np.arange(19,40)]
    queries=[np.arange(20),np.arange(10,35),np.arange(28,40),np.array([],dtype=int)]
    overlap=np.array([[len(np.intersect1d(a,b))/max(1,len(np.union1d(a,b))) for b in queries] for a in queries])
    scores=[.8,.2,.7,0.]
    assert partition_fit(parts,queries,scores,np.arange(40),overlap)==partition_fit(parts,queries,scores,np.arange(40))


def test_ambiguous_new_partition_winners_do_not_force_a_commit():
    a,b=np.arange(20),np.arange(20,40);whole=np.arange(40)
    result=select_partition([a,b],{'keep':[a,b],'option_a':[whole],'option_b':[whole]},[
        dict(frame_id=1,visible_cells=whole,queries=[whole],scores=[1.])])
    assert result['selected']=='keep'
