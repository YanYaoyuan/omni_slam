"""Whole-surface explanations by unions of partial observed masks.

Coverage utility, not a calibrated identity posterior or lossless MDL code.
Each measured cell contributes unit visible mass. Outside-mask leakage is
charged on the same visible measurement domain; hidden space has no penalty.
"""
import numpy as np


def explain_surface(masks, weights, member, costs):
    """Return best singleton and greedy union, sharing exact visible inputs.

    Utility = covered member mass - covered nonmember mass - query costs.
    Query costs are paid once; overlap is counted once. Missing regions/views
    contribute no utility. Positive-gain greedy search is not global set cover.
    """
    masks=np.asarray(masks,bool);weights=np.asarray(weights,float)
    member=np.asarray(member,bool);costs=np.asarray(costs,float)
    if masks.ndim!=2 or weights.shape!=(masks.shape[1],) or member.shape!=weights.shape or costs.shape!=(masks.shape[0],):
        raise ValueError('Aligned masks, visible weights, membership and costs required')
    if not np.isfinite(weights).all() or np.any(weights<=0) or not np.isfinite(costs).all() or np.any(costs<0):
        raise ValueError('Positive finite visible mass and nonnegative finite costs required')
    if not masks.shape[0] or not member.any():
        empty=dict(queries=[],utility=0.,inside=0.,outside=0.,cost=0.)
        return dict(single=empty.copy(),multi=empty.copy(),trace=[])
    signed=np.where(member,weights,-weights)
    initial=masks@signed-costs
    winner=int(np.argmax(initial));single=[] if initial[winner]<=1e-10 else [winner]
    used=[];union=np.zeros(len(weights),bool);trace=[]
    while len(used)<len(costs):
        gain=(masks & ~union)@signed-costs
        if used:gain[used]=-np.inf
        q=int(np.argmax(gain))
        if gain[q]<=1e-10:break
        trace.append(dict(query=q,gain=float(gain[q])))
        used.append(q);union |= masks[q]
    def result(selected):
        covered=np.any(masks[selected],axis=0) if selected else np.zeros(len(weights),bool)
        inside=float(weights[covered&member].sum());outside=float(weights[covered&~member].sum())
        cost=float(costs[selected].sum())
        return dict(queries=selected,utility=inside-outside-cost,inside=inside,outside=outside,cost=cost)
    s,m=result(single),result(used)
    assert m['utility']>=s['utility']-1e-8
    assert abs(sum(t['gain'] for t in trace)-m['utility'])<1e-8
    return dict(single=s,multi=m,trace=trace)
