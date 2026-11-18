"""Quality-based admission of measured object hypotheses.

Semantic/shape-only and cross-view scorers share a calibrated linear head and
the same greedy suppression/allocation. Neither GT nor world coordinates are
input features. This is a window-end prototype, not a real-time SLAM claim.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

SEED_FEATURES = ('score', 'class_probability', 'presence_probability',
                 'log_sources', 'log_cells', 'log_shape_small',
                 'log_shape_middle', 'log_shape_large', 'is_car')
HISTORY_FEATURES = ('log_visible_views', 'log_measured16_views', 'log_matches',
                    'mean_visible_fraction', 'mean_agreement', 'max_agreement',
                    'top3_agreement', 'mean_score', 'max_score',
                    'mean_class_probability', 'max_class_probability',
                    'fraction_strong_agreement', 'mean_score_agreement',
                    'max_class_agreement')
FEATURES = SEED_FEATURES + HISTORY_FEATURES


def feature_row(points, cell_count, meta, history, frames):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Finite measured Nx3 geometry required')
    centered = points-points.mean(0) if len(points) else points
    cov = centered.T@centered/max(1,len(points))
    shape = np.sqrt(np.maximum(np.linalg.eigvalsh(cov),0))
    seed = [meta['score'],expit(meta['class_logit']),expit(meta['presence_logit']),
            np.log1p(len(points)),np.log1p(cell_count),*np.log1p(shape),float(meta['semantic_id']==26)]
    other = np.asarray(frames)!=meta['frame_id']
    visible = (history['visible_cells']>0)&other
    measured16 = (history['visible_cells']>=16)&other
    matched = (history['query_ids']>=0)&visible
    agreement = np.where(matched,history['agreement'],0.)
    scores = np.where(matched,history['scores'],0.)
    probability = np.where(matched,expit(history['class_logits']),0.)
    def mean(values,mask): return float(np.mean(values[mask])) if mask.any() else 0.
    top = np.sort(agreement)[-3:]
    top3 = float(top.sum()/3.)
    temporal = [np.log1p(visible.sum()),np.log1p(measured16.sum()),np.log1p(matched.sum()),
        mean(history['visible_cells']/max(1,meta['cells']),visible),
        mean(agreement,visible),float(agreement.max(initial=0)),top3,
        mean(scores,matched),float(scores.max(initial=0)),
        mean(probability,matched),float(probability.max(initial=0)),
        float(((agreement>.5)&measured16).sum()/max(1,visible.sum())),
        mean(scores*agreement,visible),float((probability*agreement).max(initial=0))]
    result=np.asarray(seed+temporal,np.float64)
    if result.shape!=(len(FEATURES),) or not np.isfinite(result).all():
        raise ValueError('Invalid feature row')
    return result


def logistic_objective(theta, x, y, l2=.001):
    logits=x@theta[:-1]+theta[-1]
    loss=np.mean(np.logaddexp(0.,logits)-y*logits)+.5*l2*np.dot(theta,theta)
    residual=(expit(logits)-y)/len(y)
    gradient=np.r_[x.T@residual,residual.sum()]+l2*theta
    return float(loss),gradient


def fit_quality(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=2 or y.shape!=(len(x),) or not len(y) or not np.isin(y,[0,1]).all():
        raise ValueError('Nonempty aligned binary supervision required')
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1.
    normalized=(x-mean)/scale
    result=minimize(logistic_objective,np.zeros(x.shape[1]+1),args=(normalized,y),jac=True,
                    method='L-BFGS-B',options=dict(maxiter=300,gtol=1e-6,ftol=1e-10))
    if not result.success: raise RuntimeError('Quality optimization failed: '+result.message)
    return dict(mean=mean,scale=scale,theta=result.x,iterations=int(result.nit),
                training_rows=len(y),positive_rows=int(y.sum()),loss=float(result.fun))


def predict_quality(model,x):
    return expit(((np.asarray(x,float)-model['mean'])/model['scale'])@model['theta'][:-1]+model['theta'][-1])


def select_hypotheses(source_sets, cell_sets, semantic_ids, scores, eligible,
                      source_count, locked_sources, threshold):
    """Common same-class IoU NMS then exclusive real-source allocation.

    Descending score, stable original hypothesis-index ties. Original candidate
    cells define suppression; allocation may remove already claimed sources.
    Baseline provenance is locked regardless of semantic class. No GT access.
    """
    count=len(source_sets);scores=np.asarray(scores,float);eligible=np.asarray(eligible,bool)
    semantic_ids=np.asarray(semantic_ids)
    if not (len(cell_sets)==len(semantic_ids)==len(scores)==len(eligible)==count):
        raise ValueError('Aligned hypothesis arrays required')
    if not np.isfinite(scores).all(): raise ValueError('Finite scores required')
    used=np.zeros(source_count,bool);used[np.asarray(locked_sources,np.int64)]=True
    order=np.flatnonzero(eligible&(scores>threshold));order=order[np.argsort(-scores[order],kind='stable')]
    selected=[];claimed=[];classes=[];kept_cells=[]
    for j in order:
        ids=np.unique(np.asarray(source_sets[j],np.int64));cells=np.unique(np.asarray(cell_sets[j],np.int64))
        if np.any(ids<0) or np.any(ids>=source_count): raise ValueError('Invalid source IDs')
        duplicate=False
        for sem,other in zip(classes,kept_cells):
            if sem!=semantic_ids[j]: continue
            intersection=len(np.intersect1d(cells,other,assume_unique=True))
            if intersection/max(1,len(cells)+len(other)-intersection)>.5:
                duplicate=True;break
        if duplicate: continue
        free=ids[~used[ids]]
        if len(free)<16: continue
        selected.append(int(j));claimed.append(free);classes.append(semantic_ids[j]);kept_cells.append(cells)
        used[free]=True
    return selected,claimed
