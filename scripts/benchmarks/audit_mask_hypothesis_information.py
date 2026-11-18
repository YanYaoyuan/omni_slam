"""Information-retention probe on cached native masks, not a new map method."""
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from paper_final_common import ROOT,read,dump,sha
from evaluate_censored_partition import verify_maps,OUT as PARENT
from audit_censored_partition_objective import OUT as ASSIGNMENT

OUT=ROOT/'build/benchmarks/mask_hypothesis_information_v1'
RAW=ROOT/'build/benchmarks/prethreshold_candidates_v1/predictions_complete.json'


def metrics(truth,scores):
    result=dict(pairs=len(truth),same_object=int(truth.sum()),different_object=int((~truth).sum()))
    if not truth.any() or truth.all():return dict(**result,average_precision=None,roc_auc=None)
    return dict(**result,average_precision=float(average_precision_score(truth,scores)),
                roc_auc=float(roc_auc_score(truth,scores)))


def run():
    parent=verify_maps();audit=read(ASSIGNMENT/'summary.json');assert audit['complete']
    assert not OUT.exists();rawseal=read(RAW)
    inputs={str(RAW):sha(RAW),str(ASSIGNMENT/'summary.json'):sha(ASSIGNMENT/'summary.json')}
    for f,h in audit['files'].items():assert sha(f)==h,f
    for c in parent['contexts']:
        key=c['key']
        files=[PARENT/'prepared'/key/'atoms.npz',PARENT/'prepared'/key/'views.json',
            PARENT/'partitions'/key/'region/partition.npz',PARENT/'partitions'/key/'primitive_gain.npy',
            ASSIGNMENT/key/'reference_assignment.npz',c['field']]
        files += [r['file'] for r in rawseal['results'][key]]
        files += [r['file'] for r in read(PARENT/'prepared'/key/'views.json')]
        inputs.update({str(f):sha(f) for f in files})
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),source_sha256=sha(__file__),inputs=inputs,
        scope='Post-hoc cached-information diagnostic, no trained or new mapping method. Two exposed windows. No heldout generalization, benchmark AP, robot or independence claim.',
        population='Every original proposal edge crossing two final predicted regions, with both atoms passing the already-frozen unanimous-reference/class-consistency criterion. Reference identities ONLY label same/different for diagnostic scoring. Scores have no reference access.',
        common_domain='Same-class endpoints with at least one frame where BOTH have hard-winner positive mask support. All methods scored on EXACT same pairs. Report exclusions and each class separately. Correlated edges are not independent samples; no significance claims.',
        hard='Per atom/view, histogram of original highest-native-score covering query at each measured-visible pixel, normalized to sum1. Frame affinity=sum_q min(p_left,q,p_right,q). Average only over jointly hard-observed frames; absent frames neutral.',
        alternatives='Use ALL positive-native-score same-class raw masks on EXACT same measured pixels whose dense semantic class equals the atom class. Count covered pixels per atom/query, multiply by native query score, normalize to sum1. Same histogram intersection, same joint frames. No new score threshold, tuning, calibrated-probability or novelty claim.',
        controls='Original hard-winner histogram intersection; original initial categorical merge gain; negative Euclidean cell-center distance. Same pairs, classes, views and no fitting.',
        decision='Only consider preserving alternative mask evidence a useful next direction if its pair-ranking average precision exceeds hard-winner in BOTH windows, and neither class regresses in either window. This admits only further method design, not a full-map or publication-success claim. If false, do not sweep normalization, thresholds or query weights.',
        budget='One cached-mask pass over108frames, no model calls, training, new maps or benchmark scoring. Fixed protocol before this probe.'))
    results={};all_gate=True
    for c in parent['contexts']:
        start=time.monotonic();key=c['key'];dest=OUT/key;dest.mkdir()
        d=dict(np.load(PARENT/'prepared'/key/'atoms.npz'));n=len(d['cell_ids'])
        gt=np.load(ASSIGNMENT/key/'reference_assignment.npz')['atom_gt']
        region=np.load(PARENT/'partitions'/key/'region/partition.npz')['root']
        left,right=d['left'],d['right']
        chosen=np.flatnonzero((gt[left]>0)&(gt[right]>0)&(region[left]!=region[right]))
        a,b=left[chosen],right[chosen];assert np.all(d['train_class'][a]==d['train_class'][b])
        truth=gt[a]==gt[b];cls=d['train_class'][a]
        lookup=np.full(len(d['seen']),-1,np.int32);lookup[d['cell_ids']]=np.arange(n)
        sums={m:np.zeros(len(a),float) for m in ['hard','alternatives']};frames=np.zeros(len(a),np.int32)
        rawby={r['frame_id']:r for r in rawseal['results'][key]}
        views=read(PARENT/'prepared'/key/'views.json');scalar_checks=0
        for rank,rec in enumerate(views):
            v=dict(np.load(rec['file']));raw=dict(np.load(rawby[rec['frame_id']]['file']))
            atom=lookup[v['cells']];scores=raw['scores'];packed=raw['masks_packed']
            for tid,rid in [(2,11),(13,26)]:
                points=np.flatnonzero((atom>=0)&(v['semantic']==tid))
                points=points[d['train_class'][atom[points]]==tid]
                if not len(points):continue
                active,inverse=np.unique(atom[points],return_inverse=True)
                local=np.full(n,-1,np.int32);local[active]=np.arange(len(active))
                queries=np.flatnonzero((raw['semantic_ids']==rid)&(scores>0))
                if not len(queries):continue
                hard=np.zeros((len(active),len(queries)),float);soft=np.zeros_like(hard)
                query_lookup=np.full(len(scores),-1,np.int32);query_lookup[queries]=np.arange(len(queries))
                winner=v['query_rows'][points];valid=winner>=0
                np.add.at(hard,(inverse[valid],query_lookup[winner[valid]]),1.)
                assert np.all(query_lookup[winner[valid]]>=0)
                px=v['pixel_xy'][points]
                for j,q in enumerate(queries):
                    bits=(packed[q,px[:,1],px[:,0]//8]>>(7-px[:,0]%8))&1
                    soft[:,j]=np.bincount(inverse,weights=bits,minlength=len(active))*scores[q]
                total=hard.sum(1);normalizer=soft.sum(1)
                assert np.all(normalizer[total>0]>0)
                hard/=np.maximum(total[:,None],1.);soft/=np.maximum(normalizer[:,None],np.finfo(float).tiny)
                edge=np.flatnonzero((cls==tid)&(local[a]>=0)&(local[b]>=0))
                edge=edge[(total[local[a[edge]]]>0)&(total[local[b[edge]]]>0)]
                frames[edge]+=1
                for method,matrix in [('hard',hard),('alternatives',soft)]:
                    for lo in range(0,len(edge),4096):
                        ii=edge[lo:lo+4096]
                        sums[method][ii]+=np.minimum(matrix[local[a[ii]]],matrix[local[b[ii]]]).sum(1)
                # Independent scalar packed-mask + normalization check on fixed
                # evenly-spaced active atoms, without GT/score-based selection.
                for loc in np.linspace(0,len(active)-1,min(2,len(active)),dtype=int):
                    pick=np.flatnonzero(inverse==loc);expected=np.zeros(len(queries))
                    for j,q in enumerate(queries):
                        expected[j]=sum(bool(packed[q,int(px[t,1]),int(px[t,0])//8] &
                            (1 << (7-int(px[t,0])%8))) for t in pick)*float(scores[q])
                    expected/=max(expected.sum(),np.finfo(float).tiny)
                    np.testing.assert_allclose(expected,soft[loc],atol=1e-12,rtol=1e-12);scalar_checks+=1
            if (rank+1)%12==0:print('ALTERNATIVE MASK EVIDENCE',key,rank+1,'/',len(views),flush=True)
        take=frames>0;assert take.any()
        values={m:v[take]/frames[take] for m,v in sums.items()}
        centers=np.load(c['field'])['centers'][d['cell_ids']]
        values['geometry']=-np.linalg.norm(centers[a[take]]-centers[b[take]],axis=1)
        values['categorical']=np.load(PARENT/'partitions'/key/'primitive_gain.npy')[chosen[take]]
        stats={name:{m:metrics(truth[take][select],score[select]) for m,score in values.items()}
            for name,select in [('all',np.ones(int(take.sum()),bool)),('building',cls[take]==2),('car',cls[take]==13)]}
        gate=all(stats[x]['alternatives']['average_precision'] is not None and
                 stats[x]['alternatives']['average_precision']>=stats[x]['hard']['average_precision']
                 for x in ['all','building','car'])
        gate &= stats['all']['alternatives']['average_precision']>stats['all']['hard']['average_precision']
        all_gate &= gate
        result=dict(all_cross_region_labelled_edges=len(a),jointly_observed_edges=int(take.sum()),
            excluded_no_joint_mask_view=int((~take).sum()),statistics=stats,window_gate=bool(gate),
            scalar_readout_checks=scalar_checks,seconds=time.monotonic()-start)
        np.savez_compressed(dest/'pairs.npz',left=a[take],right=b[take],same_gt=truth[take],
            train_class=cls[take],joint_frames=frames[take],**values)
        dump(dest/'result.json',result);results[key]=result
        print('HYPOTHESIS INFORMATION RESULT',key,result,flush=True)
    p=read(OUT/'protocol.json');assert sha(__file__)==p['source_sha256']
    for f,h in p['inputs'].items():assert sha(f)==h,f
    dump(OUT/'summary.json',dict(complete=True,results=results,information_gate=bool(all_gate),
        protocol_sha256=sha(OUT/'protocol.json'),files={str(f):sha(f) for f in OUT.glob('*/*') if f.is_file()},
        new_maps=0,new_benchmark_scores=0,new_model_calls=0,new_training=0,
        new_robot_runs=0,new_independent_scenes=0,goal_complete=False))


if __name__=='__main__':run()
