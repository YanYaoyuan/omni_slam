"""Development model for incomplete masks on depth-visible source geometry.

Per-view mask recall is a nuisance variable estimated jointly with source-point
membership. This is an uncalibrated Bernoulli model, not a confidence guarantee.
Unknown measurements do not vote. Instance IDs and source coordinates stay fixed.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RecallConfig:
    membership_prior: float = .9
    false_positive_rate: float = .05
    recall_prior_mean: float = .8
    recall_prior_strength: float = 2.
    maximum_iterations: int = 20
    convergence: float = 1e-5


def revise(observations, mode='leave_point_out', config=RecallConfig()):
    """Codes: 0 unknown, 1 visible outside own mask, 2 visible inside.

Modes share initialization and priors. 'pooled' estimates a single object-wide
recall; 'per_view' estimates each frame; 'leave_point_out' excludes the evaluated
point from the sufficient statistics used for its own likelihood.
"""
    z=np.asarray(observations)
    if z.ndim!=2 or not np.isin(z,[0,1,2]).all():raise ValueError('Invalid observation matrix')
    if mode not in ('pooled','per_view','leave_point_out'):raise ValueError('Unknown mode')
    if not (0<config.false_positive_rate<config.recall_prior_mean<1 and 0<config.membership_prior<1):raise ValueError('Invalid model probabilities')
    if config.recall_prior_strength<=0 or config.maximum_iterations<1:raise ValueError('Invalid model configuration')
    visible=z>0;positive=z==2
    observed=visible.any(0);p=np.full(z.shape[1],config.membership_prior,float)
    a=config.recall_prior_mean*config.recall_prior_strength;b=config.recall_prior_strength
    f=config.false_positive_rate;prior=np.log(config.membership_prior/(1-config.membership_prior))
    history=[]
    for iteration in range(config.maximum_iterations):
        hits=positive@p;total=visible@p
        if mode=='pooled':r=np.full((len(z),1),(hits.sum()+a)/(total.sum()+b))
        elif mode=='per_view':r=((hits+a)/(total+b))[:,None]
        else:r=(hits[:,None]-positive*p+a)/(total[:,None]-visible*p+b)
        # A view whose mask is no more informative than the fixed nuisance
        # false-positive model cannot supply an inverted classifier.
        r=np.clip(r,f,1-1e-6)
        ll=np.where(positive,np.log(r/f),np.log((1-r)/(1-f)))
        score=prior+np.where(visible,ll,0).sum(0)
        next_p=1/(1+np.exp(-np.clip(score,-60,60)))
        change=float(np.max(np.abs(next_p-p))) if len(p) else 0.
        p=next_p;history.append(change)
        if change<config.convergence:break
    recall=(positive@p+a)/(visible@p+b)
    return dict(reject=observed&(p<.5),membership=p,view_recall=recall,
                iterations=len(history),maximum_change=history[-1],history=history)
