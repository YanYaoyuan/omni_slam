import itertools
import numpy as np
from omni_semantic_map.support_recovery_bound import optimistic_iou


def test_bound_dominates_all_joint_assignments_and_source_conflicts():
    rng=np.random.RandomState(6)
    for trial in range(12):
        core=rng.uniform(-1,1,(4,3));labels=np.array([1,1,2,2]);sources=rng.uniform(-1,1,(4,3));ref=rng.uniform(-1,1,(14,3));truth=rng.randint(0,3,14);weight=rng.uniform(.2,1,14)
        # Every physical source is proposed by both owners, but can be committed
        # to at most one. Relaxation must dominate all3^4 feasible assignments.
        candidates=np.repeat(sources,2,axis=0);owners=np.tile([1,2],4)
        result=optimistic_iou(core,labels,candidates,owners,ref,truth,weight,radius=1.)
        for choices in itertools.product([0,1,2],repeat=4):
            choices=np.asarray(choices);keep=choices>0;xyz=np.r_[core,sources[keep]];ids=np.r_[labels,choices[keep]];distance=np.linalg.norm(ref[:,None]-xyz[None],axis=2);nearest=distance.argmin(1);pred=np.where(distance[np.arange(len(ref)),nearest]<1.,ids[nearest],0)
            for i,oid in enumerate(result['object_ids']):
                for j,gid in enumerate(result['gt_ids']):
                    hit=weight[(pred==oid)&(truth==gid)].sum();union=weight[(pred==oid)|(truth==gid)].sum()
                    assert hit/union<=result['iou_upper'][i,j]+1e-10


def test_irreducible_false_core_limits_the_bound():
    core=np.array([[0,0,0],[2,0,0.]])
    r=optimistic_iou(core,[1,1],np.array([[0,.1,0]]),[1],core,[1,2],[1,1],.2)
    np.testing.assert_allclose(r['iou_upper'],[[.5,.5]])


def test_other_owner_can_remove_core_error_only_in_relaxation():
    core=np.array([[0,0,0],[2,0,0],[3,0,0.]])
    r=optimistic_iou(core,[1,1,2],np.array([[2,0,0.]]),[2],core,[1,2,2],[1,1,1],.2)
    assert r['iou_upper'][0,0]==1.
