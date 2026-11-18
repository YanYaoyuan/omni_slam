import numpy as np
import pytest
from run_surface_evidence_field import export_field,field_labels


def test_export_relabels_history_without_moving_geometry_or_inventing_points():
    base=dict(xyz=np.array([[0,0,0],[1,0,0],[2,0,0]],float),instance=np.array([7,11,7]))
    source=np.array([[0,0,0],[3,0,0],[4,0,0],[5,0,0]],float)
    index=dict(object_ids=np.array([7,11]),source_cell=np.array([0,3,4,5]),old_cell=np.arange(3),eligible_source=np.array([False,True,True,True]))
    probability=np.array([[0,.9,.1],[.1,.8,.1],[.8,.1,.1],[.1,.1,.8],[.5,.5,0],[.7,.2,.1]])
    geometry,stats=export_field(base,source,index,probability)
    np.testing.assert_array_equal(geometry['xyz'][:3],base['xyz']);np.testing.assert_array_equal(geometry['xyz'][3:],source[[1]])
    np.testing.assert_array_equal(geometry['instance'],[7,7,0,11]);np.testing.assert_array_equal(geometry['added_source_ids'],[1])
    assert stats['old_point_ids_changed']==2 and stats['old_points_to_null']==1 and stats['old_points_to_other']==1


def test_ambiguous_and_invalid_posteriors_do_not_silently_claim_an_object():
    np.testing.assert_array_equal(field_labels(np.array([[0,.5,.5],[.5,.5,0],[.3,.4,.3]]),[7,11]),[0,0,0])
    with pytest.raises(ValueError):field_labels(np.array([[.1,.9,.9]]),[7,11])
