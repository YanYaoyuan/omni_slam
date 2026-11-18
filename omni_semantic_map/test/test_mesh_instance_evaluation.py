import numpy as np
import pytest
from omni_semantic_map.mesh_instance_evaluation import ScanNetAP, UPSTREAM, transfer_labels


@pytest.fixture(scope='module')
def ap():
    if not (UPSTREAM/'evaluate_semantic_instance.py').exists():
        pytest.skip('pinned official ScanNet evaluator required')
    return ScanNetAP()


def test_perfect_and_missing_detection(ap):
    gt = np.repeat([1,2], 100)
    assert ap.score({'s':ap.match(gt,[gt==1,gt==2],[.9,.8],'s')})['ap'] == 1
    assert ap.score({'s':ap.match(gt,[gt==1],[.9],'s')})['ap'] == .5
    assert ap.score({'s':ap.match(gt,[],[],'s')})['ap'] == 0


def test_duplicate_is_false_positive_and_confidence_matters(ap):
    gt = np.repeat([1,2], 100)
    perfect = [gt==1,gt==2]
    duplicate = ap.score({'s':ap.match(gt,perfect+[gt==1],[.9,.8,.95],'s')})['ap']
    assert 0 < duplicate < 1
    lower_duplicate = ap.score({'s':ap.match(gt,perfect+[gt==1],[.9,.8,.1],'s')})['ap']
    assert lower_duplicate > duplicate


def test_strict_iou_threshold_and_small_gt(ap):
    gt = np.ones(200, int); half = np.arange(200)<100
    score = ap.score({'s':ap.match(gt,[half],[1.],'s')})
    assert score['ap50'] == 0 and score['ap25'] == 1
    small = np.ones(99,int)
    assert ap.score({'s':ap.match(small,[small>0],[1.],'s')})['ap'] is None


def test_only_void_prediction_is_ignored_without_crash(ap):
    gt=np.repeat([1,0],100)
    assert ap.score({'s':ap.match(gt,[gt==0],[.9],'s')})['ap'] == 0
    assert ap.score({'s':ap.match(gt,[gt==1,gt==0],[.8,.9],'s')})['ap'] == 1


def test_distance_gating_and_no_unassigned_support():
    xyz=np.array([[0.,0,0],[1,0,0],[3,0,0]])
    vertices=np.array([[.02,0,0],[.2,0,0],[1,0,0],[3,0,0]])
    labels,dist=transfer_labels(xyz,[1,2,0],vertices)
    np.testing.assert_array_equal(labels,[1,0,2,0])
    assert dist[-1] == 2
    unbounded,_=transfer_labels(xyz,[1,2,0],vertices,mode='knn5_unbounded')
    # With two neighbors, torch.mode-compatible tie resolution chooses ID 1.
    np.testing.assert_array_equal(unbounded,[1,1,1,1])
    empty,_=transfer_labels(np.empty((0,3)),[],vertices)
    assert not empty.any()


def test_unbounded_five_neighbor_transfer_matches_official_ovo():
    import ast
    from pathlib import Path
    from typing import Tuple
    torch=pytest.importorskip('torch')
    from scipy.spatial import KDTree
    path=Path('build/benchmarks/adaptive_fusion_20261110/upstream_ovo/ovo/utils/eval_utils.py')
    if not path.exists():
        pytest.skip('pinned official OVO source required')
    tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='match_labels_to_vtx']
    env=dict(torch=torch,np=np,KDTree=KDTree,Tuple=Tuple)
    exec(compile(tree,str(path),'exec'),env)
    rng=np.random.default_rng(17);xyz=rng.normal(size=(300,3));vertices=rng.normal(size=(100,3))
    ids=rng.integers(1,12,size=300)
    official=env['match_labels_to_vtx'](torch.tensor(ids),torch.tensor(xyz),torch.tensor(vertices))[0].numpy()
    ours,_=transfer_labels(xyz,ids,vertices,mode='knn5_unbounded')
    np.testing.assert_array_equal(ours,official)
