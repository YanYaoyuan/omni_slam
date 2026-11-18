"""Map-level credit for categorical measured-source acceptance.

The ordered arm exactly marginalizes first-accepted-neighbor labels under
independent source choices. The averaging control uses the SAME source
choices and reference supervision but removes nearest-neighbor competition.
Moment ratios are soft-IoU surrogates, not expected IoU or expected AP.
"""
import numpy as np
import torch


def compile_moments(adjacency, truth, weight, target_by_owner, device='cpu'):
    """Compile immutable training-only reference geometry into sparse moments.

Queries without any nearer candidate contribute constant old-map moments.
Unknown annotation is represented by zero weight, never a negative class.
"""
    offsets=np.asarray(adjacency['offsets'],np.int64);source=np.asarray(adjacency['sources'],np.int64)
    fallback=np.asarray(adjacency['fallback'],np.int64);truth=np.asarray(truth,np.int64)
    w=np.asarray(weight,np.float64);target=np.asarray(target_by_owner,np.int64)
    if offsets.shape!=(len(fallback)+1,) or truth.shape!=fallback.shape or w.shape!=truth.shape:
        raise ValueError('Misaligned reference readout')
    if offsets[0]!=0 or offsets[-1]!=len(source) or np.any(np.diff(offsets)<0):raise ValueError('Invalid CSR')
    if np.any(source<0) or np.any(source>=int(adjacency['candidate_count'])):raise ValueError('Invalid source index')
    if not np.isfinite(w).all() or np.any(w<0) or np.any(fallback<0) or np.any(fallback>=len(target)):
        raise ValueError('Invalid labels or weights')
    length=np.diff(offsets);active=np.flatnonzero((length>0)&(w>0));constant=np.ones(len(w),bool);constant[active]=False
    mass=np.bincount(fallback[constant],weights=w[constant],minlength=len(target))
    intersection=np.bincount(fallback[constant],weights=w[constant]*(truth[constant]==target[fallback[constant]]),minlength=len(target))
    sizes=length[active];width=int(sizes.max(initial=0))
    padded=np.zeros((len(active),width),np.int64);valid=np.zeros_like(padded,bool)
    for i,q in enumerate(active):
        ids=source[offsets[q]:offsets[q+1]];padded[i,:len(ids)]=ids;valid[i,:len(ids)]=True
    rows,slots=np.nonzero(valid)
    tensor=lambda a,dtype:torch.as_tensor(a,dtype=dtype,device=device)
    return dict(indices=tensor(padded,torch.long),valid=tensor(valid,torch.bool),rows=tensor(rows,torch.long),
        slots=tensor(slots,torch.long),source=tensor(padded[rows,slots],torch.long),
        fallback=tensor(fallback[active],torch.long),truth=tensor(truth[active],torch.long),
        weight=tensor(w[active],torch.float64),length=tensor(sizes,torch.float64),
        target=tensor(target,torch.long),constant_mass=tensor(mass,torch.float64),
        constant_intersection=tensor(intersection,torch.float64),candidate_count=int(adjacency['candidate_count']))


def moments(compiled, probabilities, owners, mode='ordered'):
    """Return exact class-label mass/intersection, or the mean-neighbor control."""
    if mode not in ['ordered','mean_neighbor']:raise ValueError('Unknown readout mode')
    c=compiled;p=probabilities.to(torch.float64);owner=torch.as_tensor(owners,dtype=torch.long,device=p.device)
    if p.ndim!=2 or owner.shape!=p.shape or len(p)!=c['candidate_count']:raise ValueError('Misaligned choices')
    if bool(torch.any(owner<0)) or bool(torch.any(owner>=len(c['target']))):raise ValueError('Invalid owner')
    # Padded owner=0 must carry zero probability; callers provide a masked softmax.
    mass=c['constant_mass'].clone()+p.sum()*0;inter=c['constant_intersection'].clone()+p.sum()*0
    if not len(c['weight']):return mass,inter
    acceptance=p.sum(1);reject=1-acceptance
    if mode=='ordered':
        rejected=torch.where(c['valid'],reject[c['indices']],torch.ones_like(c['indices'],dtype=p.dtype))
        cumulative=torch.cumprod(rejected,dim=1)
        prefix=torch.cat([torch.ones_like(cumulative[:,:1]),cumulative[:,:-1]],dim=1)
        exposure=c['weight'][c['rows']]*prefix[c['rows'],c['slots']]
        fallback_weight=c['weight']*cumulative[:,-1]
    else:
        exposure=c['weight'][c['rows']]/c['length'][c['rows']]
        fallback_weight=torch.zeros_like(c['weight']).index_add(0,c['rows'],exposure*reject[c['source']])
    contribution=exposure[:,None]*p[c['source']]
    labels=owner[c['source']];truth=c['truth'][c['rows']]
    mass=mass.index_add(0,labels.reshape(-1),contribution.reshape(-1))
    inter=inter.index_add(0,labels.reshape(-1),(contribution*(truth[:,None]==c['target'][labels])).reshape(-1))
    mass=mass.index_add(0,c['fallback'],fallback_weight)
    inter=inter.index_add(0,c['fallback'],fallback_weight*(c['truth']==c['target'][c['fallback']]))
    return mass,inter


def object_loss(mass, intersection, object_ids, target_mass, unmatched_scale):
    """Equal-object soft-IoU, with explicit unmatched-object area penalty.

target_mass is zero for unmatched predictions; their penalty is M/(M+s),
where s is the training scene's mean eligible GT mass of the same class.
No target assignment or GT is needed at inference.
"""
    ids=torch.as_tensor(object_ids,dtype=torch.long,device=mass.device)
    gt=torch.as_tensor(target_mass,dtype=mass.dtype,device=mass.device)[ids]
    scale=torch.as_tensor(unmatched_scale,dtype=mass.dtype,device=mass.device)[ids]
    m,i=mass[ids],intersection[ids];matched=gt>0
    loss=torch.where(matched,1-i/(gt+m-i).clamp_min(1e-12),m/(m+scale).clamp_min(1e-12))
    return loss.mean()
