"""Post-scoring GT diagnostics; never consumed by prediction or training."""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score,log_loss
from paper_final_common import read,dump,sha
from run_learned_observation_revision import OUT,METHODS,verify


def rates(truth,predicted):
    tp=int((truth&predicted).sum());fp=int((~truth&predicted).sum());fn=int((truth&~predicted).sum());tn=int((~truth&~predicted).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,accuracy=(tp+tn)/len(truth) if len(truth) else None)


def main():
    p=verify();seal=read(OUT/'predictions_complete.json');assert read(OUT/'summary.json')['complete']
    for f,h in seal['files'].items():assert sha(f)==h,f
    result={}
    for s in p['scenes']:
        if s['role']!='development' or s['kind']!='indoor':continue
        dest=OUT/s['key'];a=np.load(dest/'observations.npz');data=np.load(dest/'pairs.npz')
        ref=np.load(s['reference']);valid=ref['valid'] if 'valid' in ref else np.ones(len(ref['instance']),bool)
        ids=a['active_source_indices'];source_xyz=a['source_xyz'];support_ids=a['source_point_ids'];offsets=a['offsets']
        tree=cKDTree(ref['xyz'][valid]);gt=ref['instance'][valid]
        point_truth=np.full(int(ids.max())+1,-1,np.int64)
        for start in range(0,len(ids),200000):
            distance,nearest=tree.query(source_xyz[start:start+200000],distance_upper_bound=.05,workers=1);known=np.isfinite(distance)
            point_truth[ids[start:start+200000][known]]=gt[nearest[known]]
        target=np.full(len(a['native']),-1,np.int64)
        for i,(lo,hi) in enumerate(zip(offsets[:-1],offsets[1:])):
            y=point_truth[support_ids[lo:hi]];known=y>=0
            if known.sum()<16 or known.mean()<.5:continue
            u,c=np.unique(y[known],return_counts=True);best=int(c.argmax())
            if c[best]/known.sum()>=.8:target[i]=u[best]
        pairs=data['pairs'];eligible=(target[pairs[:,0]]>=0)&(target[pairs[:,1]]>=0)
        pairs=pairs[eligible];truth=(target[pairs[:,0]]==target[pairs[:,1]])&(target[pairs[:,0]]>0)
        prob=np.load(dest/'pair_predictions.npz')['probability'][eligible]
        partitions={}
        for m in METHODS:
            labels=np.load(dest/m/'observation_identity.npy');partitions[m]=rates(truth,labels[pairs[:,0]]==labels[pairs[:,1]])
        bins=[]
        for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
            hit=(prob>=lo)&((prob<hi) if hi<1 else (prob<=hi))
            if hit.any():bins.append(dict(low=float(lo),high=float(hi),pairs=int(hit.sum()),mean_probability=float(prob[hit].mean()),same_object_fraction=float(truth[hit].mean())))
        row=dict(scene=s['key'],eligible_observations=int((target>=0).sum()),all_observations=len(target),eligible_pairs=len(truth),
            model=dict(auc=float(roc_auc_score(truth,prob)),log_loss=float(log_loss(truth,prob)),brier=float(((prob-truth)**2).mean()),
                threshold_half=rates(truth,prob>=.5),reliability_bins=bins),partitions=partitions)
        result[s['key']]=row;print(s['key'],row['model']['auc'],row['model']['threshold_half'],partitions,flush=True)
    dump(OUT/'posthoc_edge_audit.json',dict(complete=True,gt_diagnostic_only=True,source_sha256=sha(__file__),
        prediction_seal_sha256=sha(OUT/'predictions_complete.json'),scenes=result,
        scope='Known high-purity whole observations and existing candidate pairs only. Pair accuracy is not map PQ; mixed/uncovered observations excluded. No further training or parameter change.'))


if __name__=='__main__':main()
