import numpy as np
from omni_semantic_map.counterfactual_geometry_revision import CounterfactualRevision,connected_cells,nearest_indices


def test_redundant_point_has_no_independent_benefit_but_joint_region_can():
    xyz=np.array([[0.,0,0],[.05,0,0]]);q=np.array([[0.,0,0]])
    state=CounterfactualRevision(xyz,[0,0],q,[[.1,.9]],[1.])
    assert not state.trial([0])['accepted']
    assert state.trial([0,1])['accepted']
    assert state.nearest[0]==-1 and not state.active.any()
    state.audit()


def test_local_improvement_can_damage_a_neighbouring_observed_surface():
    xyz=np.array([[0.,0,0]]);q=np.array([[0.,0,0],[-.15,0,0]]);p=np.array([[.2,.8],[.99,.01]])
    full=CounterfactualRevision(xyz,[0],q,p,[1.,1.]);local=CounterfactualRevision(xyz,[0],q,p,[1.,1.])
    assert not full.trial([0])['accepted']
    event=local.trial([0],np.array([True,False]));assert event['accepted'] and event['full_gain']<0
    full.audit();local.audit()


def test_unknown_zero_weight_observations_cannot_authorize_revision():
    state=CounterfactualRevision([[0,0,0]],[0],[[0,0,0]],[[.01,.99]],[0.])
    assert not state.trial([0])['accepted'] and state.active[0]


def test_removal_exposes_another_semantic_class_not_automatically_background():
    state=CounterfactualRevision([[0,0,0],[.1,0,0]],[0,1],[[0,0,0]],[[.1,.8,.1]],[1.])
    assert state.trial([0])['accepted']
    assert state.labels(state.nearest)[0]==1 and state.nearest[0]==1
    state.audit()


def test_sequential_updates_equal_full_recomputation_and_never_increase_risk():
    rng=np.random.RandomState(7);xyz=rng.uniform(0,.4,(40,3));q=rng.uniform(0,.4,(200,3));classes=rng.randint(0,2,40);p=rng.dirichlet([1.,1.,1.],len(q));state=CounterfactualRevision(xyz,classes,q,p,np.ones(len(q)))
    for ids in np.array_split(rng.permutation(40),8):
        old=state.loss;state.trial(ids);assert state.loss<=old+1e-9;state.audit()


def test_readout_ties_preserve_original_index_and_radius_is_strict():
    xyz=np.array([[-.1,0,0],[.1,0,0]])
    np.testing.assert_array_equal(nearest_indices(xyz,[[0,0,0],[.3,0,0]]),[0,1])
    np.testing.assert_array_equal(nearest_indices([[0,0,0]],[[.2,0,0]]),[-1])


def test_component_proposals_do_not_depend_on_input_order():
    keys=np.array([[0,0,0],[1,1,1],[9,0,0],[-1,0,0]])
    parts=connected_cells(keys);sets={frozenset(map(tuple,keys[p])) for p in parts}
    order=np.array([3,2,0,1]);other=connected_cells(keys[order]);assert sets=={frozenset(map(tuple,keys[order][p])) for p in other}
    assert len(parts)==2
