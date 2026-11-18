"""Compare object partitions using observations not used to prompt them.

Native query scores are weights, not calibrated identity probabilities.
Duplicate query masks do not count as evidence for two distinct objects.
Missing visibility is unknown; all candidates use the same eligible views.
"""
import numpy as np


def partition_fit(parts, queries, query_scores, visible_cells, query_overlap=None):
    """One/two-object partition fit on the current measured-visible domain.

    Match one object to one raw instance mask. For a two-object explanation,
    enumerate distinct assignments and discount overlapping query supports.
    This deliberately does not sum scores over repeated/duplicate queries.
    """
    if len(parts) not in (1, 2):
        raise ValueError('One or two object parts required')
    visible = np.unique(np.asarray(visible_cells, np.int64))
    query_sets = [np.unique(np.asarray(q, np.int64)) for q in queries]
    scores = np.asarray(query_scores, float)
    if scores.shape != (len(queries),) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError('Aligned native scores in [0,1] required')
    if any(not np.isin(q, visible).all() for q in query_sets):
        raise ValueError('Query contains cells without measured visibility')
    sets = [np.intersect1d(np.unique(np.asarray(p, np.int64)), visible) for p in parts]
    fit = np.zeros((len(parts), len(queries)))
    for i, part in enumerate(sets):
        for j, query in enumerate(query_sets):
            intersection = len(np.intersect1d(part, query, assume_unique=True))
            fit[i,j] = scores[j]*intersection/max(1,len(part)+len(query)-intersection)
    if not len(queries): return 0.
    if len(parts)==1: return float(fit.max(initial=0))
    if query_overlap is not None:
        overlap=np.asarray(query_overlap,float)
        if overlap.shape!=(len(queries),len(queries)) or not np.isfinite(overlap).all() or np.any((overlap<0)|(overlap>1)):
            raise ValueError('Aligned query-mask overlaps in [0,1] required')
        values=.5*(fit[0,:,None]+fit[1,None,:])*(1-overlap)
        np.fill_diagonal(values,0.)
        return float(values.max(initial=0))
    result = 0.
    for a, qa in enumerate(query_sets):
        for b, qb in enumerate(query_sets):
            if a==b: continue
            common = len(np.intersect1d(qa,qb,assume_unique=True))
            overlap = common/max(1,len(qa)+len(qb)-common)
            result = max(result, .5*(fit[0,a]+fit[1,b])*(1-overlap))
    return float(result)


def select_partition(original_parts, candidates, views, valid=None):
    """Compare keep/split/merge with identical views and no GT or fitted head.

    Each side of the ORIGINAL dispute needs >=16 measured-visible cells.
    Candidate-dependent visibility cannot select favorable validation views.
    Invalid prompt candidates are excluded. Ties retain the original state.
    """
    if len(original_parts)!=2 or 'keep' not in candidates:
        raise ValueError('Original pair and keep candidate required')
    valid = {k:True for k in candidates} if valid is None else valid
    if not valid.get('keep',False): raise ValueError('Keeping the original must remain valid')
    records=[]
    for view in views:
        if view.get('used_for_prompt',False):
            raise ValueError('Prompt observation cannot validate its own revision')
        visible=np.unique(np.asarray(view['visible_cells'],np.int64))
        counts=[len(np.intersect1d(p,visible)) for p in original_parts]
        if min(counts)<16: continue
        scores={k:partition_fit(parts,view['queries'],view['scores'],visible,view.get('query_overlap'))
                for k,parts in candidates.items() if valid.get(k,False)}
        records.append(dict(frame_id=int(view['frame_id']),original_visible=counts,scores=scores))
    if not records: return dict(selected='keep',reason='no_common_measured_validation',views=[],scores={})
    means={k:float(np.mean([v['scores'][k] for v in records])) for k in records[0]['scores']}
    best=max(means,key=lambda k:(means[k],k=='keep'))
    # Numerical tie protection, not a tuned gain threshold.
    if means[best]<=means['keep']+1e-9 or sum(abs(v-means[best])<=1e-9 for v in means.values())>1: best='keep'
    return dict(selected=best,reason='strict_shared_view_gain' if best!='keep' else 'no_gain',views=records,scores=means)


def revise_history(instance, object_a, object_b, operation, mask_a=None, mask_b=None):
    """Change labels only; caller retains exact original point coordinates.

    Split masks refer to depth-verified historical map-point projections.
    A point with two masks or no mask remains unchanged. A global identity
    merge maps both IDs to the smaller ID after independent-view selection.
    """
    labels=np.asarray(instance)
    if labels.ndim!=1 or not np.issubdtype(labels.dtype,np.integer) or object_a==object_b:
        raise ValueError('Integer point labels and distinct object IDs required')
    out=labels.copy(); affected=(labels==object_a)|(labels==object_b)
    if operation=='keep': return out
    if operation=='merge':out[affected]=min(object_a,object_b);return out
    if operation!='split':raise ValueError('Unknown revision operation')
    a,b=np.asarray(mask_a,bool),np.asarray(mask_b,bool)
    if a.shape!=labels.shape or b.shape!=labels.shape:raise ValueError('Aligned measured mask supports required')
    out[affected&a&~b]=object_a;out[affected&b&~a]=object_b
    return out


def propose_measured_sources(source_a, source_b, mask_a, mask_b, operation,
                             protected_sources=(), include_history=True):
    """Build disjoint source-ID proposals; never synthesize coordinates.

    Existing shared sources prefer A (caller supplies a fixed priority order).
    Exclusive new mask support can revise that ownership. Unseen historical
    support persists; a new source in both split masks remains unassigned.
    Protected sources belonging to the retained map are never appended again.
    Global allocation across separate disputes remains the caller's job.
    """
    arrays=[np.unique(np.asarray(v,np.int64)) for v in
            [source_a,source_b,mask_a,mask_b,protected_sources]]
    if any(v.ndim!=1 or np.any(v<0) for v in arrays):raise ValueError('Nonnegative source IDs required')
    a,b,ma,mb,protected=arrays
    if operation not in ['split','merge']:raise ValueError('Only actual revision proposals accepted')
    old=np.union1d(a,b) if include_history else np.empty(0,np.int64)
    domain=np.union1d(old,np.union1d(ma,mb))
    domain=np.setdiff1d(domain,protected,assume_unique=True)
    if operation=='merge':return [domain]
    labels=np.zeros(len(domain),np.int8)
    if include_history:
        labels[np.isin(domain,b)]=2
        labels[np.isin(domain,a)]=1
    ina,inb=np.isin(domain,ma),np.isin(domain,mb)
    labels[ina&~inb]=1;labels[inb&~ina]=2
    return [domain[labels==1],domain[labels==2]]
