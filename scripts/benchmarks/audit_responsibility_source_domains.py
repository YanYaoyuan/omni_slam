"""Post-score source-label diagnostics in training and exposed target windows.

No fitting, parameter selection, new complete map or AP evaluation. Labels are
the new shared training assignment, not the older core-majority assignment.
"""
import numpy as np
import torch
from paper_final_common import read,dump,sha
from run_map_responsibility_supervision import OUT,PARENT,KEYS,METHODS,verify,seal
from run_object_context_acceptance import logits
from omni_semantic_map.surface_ownership import allocate
from omni_semantic_map.map_responsibility_supervision import categorical,source_labels,marginal_nll


def main():
    verify();seal('trained');seal('prepared');seal('predictions_complete');assert read(OUT/'summary.json')['complete'];torch.set_num_threads(1);rows=[]
    for target in KEYS:
        for method in METHODS[1:]:
            m=torch.load(OUT/'models'/target/method/'model.pt',map_location='cpu',weights_only=False)
            for key in KEYS:
                d=dict(np.load(PARENT/'features'/key/'candidates.npz'));y=dict(np.load(OUT/'training_only'/key/'labels.npz'));z=logits(m,d['object_features']);owner,_=allocate(z,d['valid'],d['owners']);label_count=max(int(d['owners'].max()),int(y['source_target'].max()))+1
                prob=categorical(torch.tensor(z),torch.tensor(d['valid']));dist=source_labels(prob,torch.tensor(d['owners']),label_count);truth=torch.tensor(y['source_target']);w=y['source_weight'];known=w>0;accepted=known&(owner>0);correct=owner==y['source_target']
                stat=dict(target=target,method=method,input=key,role='training' if key in m['training'] else 'exposed_target',
                          source_nll=float(marginal_nll(dist,truth,torch.tensor(y['source_plain'],dtype=torch.float32))),
                          balanced_source_nll=float(marginal_nll(dist,truth,torch.tensor(y['source_balanced'],dtype=torch.float32))),
                          known_sources=int(known.sum()),known_accepted=int(accepted.sum()),unknown_accepted=int(((~known)&(owner>0)).sum()),
                          weighted_source_accuracy=float(w[correct].sum()/w.sum()),
                          known_acceptance_precision=float(w[accepted&correct].sum()/w[accepted].sum()) if accepted.any() else None)
                if key==target:np.testing.assert_array_equal(owner,np.load(OUT/'maps'/key/method/'decisions.npz')['owners'])
                rows.append(stat);print('SOURCE DOMAIN',stat,flush=True)
    dump(OUT/'source_domain_diagnostic.json',dict(complete=True,source_sha256=sha(__file__),rows=rows,new_fits=0,new_maps=0,new_ap=0,
         scope='Post-score diagnostic against this trial common one-to-one training target assignment. Source-domain loss, not spatial loss/AP. Unknown nearest source labels excluded. Training vs exposed opposite-window comparison; inherited parent supervision still prevents independent generalization claims. No result used to change a model.'))


if __name__=='__main__':main()
