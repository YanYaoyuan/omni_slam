import numpy as np
from omni_semantic_map.hypothesis_admission import feature_row,logistic_objective,select_hypotheses


def test_shape_invariance_and_no_self_or_hidden_confirmation():
    rng=np.random.RandomState(4);points=rng.randn(30,3)
    meta=dict(score=.3,class_logit=1.,presence_logit=0.,source_points=40,cells=20,semantic_id=11,frame_id=2)
    history=dict(visible_cells=np.array([10,20,0]),query_ids=np.array([3,4,5]),agreement=np.array([.7,1.,1.]),scores=np.array([.1,1.,1.]),class_logits=np.array([1.,100.,100.]))
    frames=np.array([1,2,3]);a=feature_row(points,10,meta,history,frames)
    rotation=np.linalg.qr(rng.randn(3,3))[0]
    b=feature_row(points@rotation+np.array([100,-30,1]),10,meta,history,frames)
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    changed={k:v.copy() for k,v in history.items()}
    for k in ['agreement','scores','class_logits']: changed[k][1:]=0
    np.testing.assert_array_equal(a,feature_row(points,10,meta,changed,frames))


def test_quality_gradient_matches_finite_difference():
    rng=np.random.RandomState(5);x=rng.randn(20,4);y=rng.randint(2,size=20);theta=rng.randn(5)
    _,gradient=logistic_objective(theta,x,y)
    numeric=[]
    for j in range(5):
        delta=np.zeros(5);delta[j]=1e-6
        numeric.append((logistic_objective(theta+delta,x,y)[0]-logistic_objective(theta-delta,x,y)[0])/2e-6)
    np.testing.assert_allclose(gradient,numeric,rtol=1e-6,atol=1e-8)


def test_common_competition_keeps_locked_points_and_one_new_owner():
    sources=[np.arange(0,40),np.arange(5,45),np.arange(30,70),np.arange(70,86)]
    cells=[np.arange(0,40),np.arange(5,45),np.arange(30,70),np.arange(70,86)]
    selected,claimed=select_hypotheses(sources,cells,[11,11,26,11],[.9,.8,.7,.5],[True]*4,100,np.arange(10),.5)
    assert selected==[0,2]
    np.testing.assert_array_equal(claimed[0],np.arange(10,40))
    np.testing.assert_array_equal(claimed[1],np.arange(40,70))
    all_ids=np.concatenate(claimed)
    assert len(np.unique(all_ids))==len(all_ids) and not np.isin(all_ids,np.arange(10)).any()
