"""Real-data moment cross-check against the independently tested dense reader."""
from pathlib import Path
import numpy as np
import torch
from paper_final_common import read,dump,sha
from run_map_credit_training import OUT,verify
from omni_semantic_map.map_acceptance_objective import compile_moments,moments
from omni_semantic_map.spatial_acceptance_readout import expected_labels,hard_readout


def main():
    p=verify();assert not (OUT/'input_audit.json').exists();results={};torch.set_num_threads(2)
    for c in p['contexts']:
        root=OUT/'inputs'/c['key'];d=np.load(root/'candidates.npz');y=np.load(root/'TRAINING_ONLY.npz');a=dict(np.load(root/'adjacency.npz'))
        active=np.flatnonzero(np.diff(a['offsets'])>0);inactive=np.flatnonzero(np.diff(a['offsets'])==0)
        selected=np.unique(np.r_[active[np.linspace(0,len(active)-1,min(1024,len(active)),dtype=int)],
                                  inactive[np.linspace(0,len(inactive)-1,min(128,len(inactive)),dtype=int)]])
        sets=[a['sources'][a['offsets'][i]:a['offsets'][i+1]] for i in selected]
        sub=dict(offsets=np.r_[0,np.cumsum([len(s) for s in sets])],sources=np.concatenate(sets),fallback=a['fallback'][selected],candidate_count=a['candidate_count'])
        owners=d['owners'];valid=d['valid'];prob=torch.tensor(valid/(1+valid.sum(1)[:,None]),dtype=torch.float64)
        compiled=compile_moments(sub,y['truth'][selected],y['weight'][selected],y['target'])
        mass,inter=moments(compiled,prob,owners)
        dense=expected_labels(sub,prob,owners,len(y['target']))
        weight=torch.tensor(y['weight'][selected]);truth=torch.tensor(y['truth'][selected]);target=torch.tensor(y['target'])
        want_mass=(dense*weight[:,None]).sum(0)
        want_inter=(dense*weight[:,None]*(truth[:,None]==target[None])).sum(0)
        torch.testing.assert_close(mass,want_mass,rtol=1e-12,atol=1e-10)
        torch.testing.assert_close(inter,want_inter,rtol=1e-12,atol=1e-10)
        np.testing.assert_array_equal(hard_readout(a,np.zeros(len(owners),np.int64)),a['fallback'])
        assigned=y['target'][y['target']>=0];assert len(np.unique(assigned))==len(assigned)
        n_unknown=int((y['point_weight']==0).sum());assert np.all(y['balanced_weight'][y['point_weight']==0]==0)
        results[c['key']]=dict(reference_queries_checked=len(selected),full_baseline_fallback_queries=len(a['fallback']),
             max_mass_error=float((mass-want_mass).abs().max()),max_intersection_error=float((inter-want_inter).abs().max()),
             source_groups=len(owners),source_groups_without_point_labels=n_unknown,matching_one_to_one=True,
             candidate_objects_registered=True)
        print('MAP CREDIT REAL INPUT AUDIT',c['key'],results[c['key']],flush=True)
    dump(OUT/'input_audit.json',dict(complete=True,results=results,source_sha256=sha(__file__),new_maps=0,new_fits=0))


if __name__=='__main__':main()
