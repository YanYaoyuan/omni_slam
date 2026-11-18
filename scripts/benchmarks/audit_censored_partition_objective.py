"""Post-hoc mechanism audit. Reference labels NEVER enter map inference.

Compare whole-object and fragmented objectives on exactly the same observed
atoms, including an induced-graph-connected comparison. This is neither a new
prediction nor a benchmark improvement. The failed candidate stays frozen.
"""
from pathlib import Path
import time
import runpy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import gammaln
from paper_final_common import ROOT, read, dump, sha
from evaluate_censored_partition import verify_maps, OUT as PARENT
from evaluate_kitti360_development import read_static_ply, transfer
from omni_semantic_map.kitti360_reference_identity import encode_reference
from omni_semantic_map.censored_object_partition import (
    make_histograms, region_score, merge_gain, join_histogram,
)

OUT=ROOT/'build/benchmarks/censored_partition_objective_audit_v1'


def independent_score(hist, categories):
    # Separate vectorized scipy special-function implementation.
    result=0.
    for view, counts in hist.items():
        values=np.asarray(list(counts.values()),float)
        alpha=1./int(categories[view])
        result+=float(-gammaln(1.+values.sum())+(gammaln(alpha+values)-gammaln(alpha)).sum())
    return result


def combine(histograms):
    result={}
    for h in histograms:result=join_histogram(result,h)
    return result


def describe(values):
    values=np.asarray(values,float)
    return dict(count=len(values),positive=int((values>1e-8).sum()),
                negative=int((values < -1e-8).sum()),neutral=int((abs(values)<=1e-8).sum()),
                sum=float(values.sum()),minimum=float(values.min()) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def freeze(contexts):
    assert not OUT.exists()
    files=[PARENT/'prepared.json',PARENT/'map_summary.json',PARENT/'delivery_audit.json',
           ROOT/'build/datasets/kitti360_prepare/labels.py']
    for c in contexts:
        key=c['key'];files += [c['raw_reference'],c['reference'],c['field'],c['job'],
            Path(c['full'])/'geometry.npz',PARENT/'prepared'/key/'atoms.npz',
            PARENT/'partitions'/key/'region/partition.npz']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),inputs={str(f):sha(f) for f in files},
        source_sha256=sha(__file__),
        scope='Post-hoc diagnosis of the sealed failed candidate on two exposed development windows; no new inference, parameters, model calls, training, maps, scores, independent scenes or robot runs.',
        reference='Raw canonical building11/car26 identities; garage aliases void. All nearest-reference coordinates retained. Raw coordinates shifted by the original job origin_world and verified exactly against the cached reference. Zero-confidence labels void. Strict<20cm nearest reference with stable lowest-index ties, matching the existing transfer routine, but source-to-reference direction is diagnostic only.',
        atoms='Use every full measured source in each prepared atom. Accept an atom only if ALL its sources have the same positive target nearest-reference identity and this class equals the predicted atom class. No majority threshold, view/GT selection or new evidence filtering. This is unanimous transferred-reference support, not a guarantee about true unobserved geometry.',
        comparisons='On identical accepted atom observations, compare whole physical-reference object vs its predicted fragments; also compare each same-GT connected component of the induced proposal graph vs its predicted fragments. Compare same-GT adjacent fragment merge gains. Separately measure benefit of splitting a mixed predicted region by reference identity. Unknown and excluded atoms are omitted from ALL compared objectives.',
        interpretation='Whole GT merge gain<0 means this objective prefers fragmentation on the stated diagnostic subset. Connected-component results remove absence of a feasible proposal path within that subset. A pure-subset objective is not the complete inference objective; no claim that these interventions are deployable or exactly explain full AP. Zero shared views are neutral by construction.',
        evidence='Also count observed views containing multiple mask tokens for one reference object. Check every aggregate objective with independent scipy.gammaln, and check categorical distinct-token two-atom merge identity analytically.',
        budget='One audit pass over both full prepared windows; all objects and connected components reported. No parameter sweep or GT-assisted map export. Failed method remains frozen.'))


def run():
    parent=verify_maps();freeze(parent['contexts'])
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    # Different categorical tokens in one common view always penalize merging
    # these unit-mass atoms, independent of K: log Gamma(2)^2/Gamma(3)=-log2.
    for k in [2,7,400]:
        assert abs(merge_gain({0:{0:1.}},{0:{1:1.}},[k])+np.log(2))<1e-12
    results={}
    for c in parent['contexts']:
        started=time.monotonic();key=c['key'];dest=OUT/key;dest.mkdir()
        d=dict(np.load(PARENT/'prepared'/key/'atoms.npz'));n=len(d['cell_ids'])
        labels=np.load(PARENT/'partitions'/key/'region/partition.npz')['root']
        ref=dict(np.load(c['reference']));raw=read_static_ply(c['raw_reference'])
        shifted=np.column_stack([raw[x] for x in ['x','y','z']]).astype(float)-np.asarray(read(c['job'])['origin_world'])
        np.testing.assert_array_equal(shifted,ref['xyz'])
        np.testing.assert_array_equal(raw['confidence'],ref['confidence'])
        truth,_=encode_reference(raw['semantic'],raw['instance'],taxonomy,'target_only')
        truth[ref['confidence']<=0]=0
        field=np.load(c['field']);source_cell=field['source_to_cell']
        lookup=np.full(len(d['seen']),-1,np.int32);lookup[d['cell_ids']]=np.arange(n)
        source_ids=np.flatnonzero(lookup[source_cell]>=0)
        source_atom=lookup[source_cell[source_ids]]
        xyz=np.load(Path(c['full'])/'geometry.npz')['xyz']
        source_truth=transfer(ref['xyz'],truth,xyz[source_ids],.2)
        lo=np.full(n,np.iinfo(np.int64).max,np.int64);hi=np.full(n,-1,np.int64)
        np.minimum.at(lo,source_atom,source_truth);np.maximum.at(hi,source_atom,source_truth)
        count=np.bincount(source_atom,minlength=n)
        assert np.all(count>0)
        unanimous=lo==hi
        target=unanimous & np.isin(lo//1000,[2,13])
        accepted=target & (lo//1000==d['train_class'])
        atom_gt=np.where(accepted,lo,0)
        np.savez_compressed(dest/'reference_assignment.npz',atom_gt=atom_gt,minimum_source_label=lo,
            maximum_source_label=hi,source_count=count,source_ids=source_ids,source_gt=source_truth)
        choose=accepted[d['observation_atom']]
        h=make_histograms(n,d['observation_atom'][choose],d['observation_view'][choose],
            d['observation_token'][choose],d['observation_mass'][choose],d['categories'])
        k=d['categories'];checks=0;max_error=0.
        def score(hist):
            nonlocal checks,max_error
            a=region_score(hist,k);b=independent_score(hist,k)
            error=abs(a-b);max_error=max(max_error,error);checks+=1
            assert error<=1e-8*max(1.,abs(a)),(a,b)
            return a
        ids=np.flatnonzero(accepted);compact=np.full(n,-1,np.int32);compact[ids]=np.arange(len(ids))
        left,right=d['left'],d['right']
        take=accepted[left]&accepted[right]&(atom_gt[left]==atom_gt[right])
        u,v=compact[left[take]],compact[right[take]]
        graph=csr_matrix((np.ones(2*len(u),np.int8),(np.r_[u,v],np.r_[v,u])),shape=(len(ids),len(ids)))
        nc,cc=connected_components(graph,directed=False)
        component=np.full(n,-1,np.int32);component[ids]=cc
        object_rows=[];component_rows=[];mixed_rows=[];boundary_rows=[]
        for gt in np.unique(atom_gt[accepted]):
            members=np.flatnonzero(atom_gt==gt);regions=np.unique(labels[members])
            whole=combine(h[i] for i in members)
            split=[combine(h[i] for i in members[labels[members]==r]) for r in regions]
            delta=score(whole)-sum(score(x) for x in split)
            object_rows.append(dict(gt_id=int(gt),atoms=len(members),source_points=int(count[members].sum()),
                predicted_fragments=len(regions),induced_graph_components=int(len(np.unique(component[members]))),
                whole_minus_fragments=float(delta),observed_views=len(whole),
                views_with_multiple_tokens=sum(len(v)>1 for v in whole.values()),
                weighted_dominant_token_share=float(sum(max(v.values()) for v in whole.values())/
                    sum(sum(v.values()) for v in whole.values()))))
        # Pure graph components keep the feasible-proposal connectivity explicit.
        for comp in range(nc):
            members=ids[cc==comp];regions=np.unique(labels[members]);gt=int(atom_gt[members[0]])
            assert np.all(atom_gt[members]==gt)
            if len(regions)<2:continue
            pieces={int(r):combine(h[i] for i in members[labels[members]==r]) for r in regions}
            whole=combine(pieces.values());delta=score(whole)-sum(score(x) for x in pieces.values())
            component_rows.append(dict(component=comp,gt_id=gt,atoms=len(members),fragments=len(regions),
                whole_minus_fragments=float(delta)))
            edge_take=take&(component[left]==comp)&(labels[left]!=labels[right])
            pairs=np.unique(np.minimum(labels[left[edge_take]],labels[right[edge_take]])*n+
                            np.maximum(labels[left[edge_take]],labels[right[edge_take]]))
            for pair in pairs:
                a,b=map(int,divmod(int(pair),n));ha,hb=pieces[a],pieces[b]
                gain=merge_gain(ha,hb,k)
                assert abs(gain-(score(join_histogram(ha,hb))-score(ha)-score(hb)))<1e-7
                common=set(ha)&set(hb)
                boundary_rows.append(dict(component=comp,gt_id=gt,left_region=a,right_region=b,
                    merge_gain=float(gain),common_views=len(common),
                    common_views_disjoint_tokens=sum(not(set(ha[f])&set(hb[f])) for f in common)))
        for r in np.unique(labels[accepted]):
            members=np.flatnonzero(accepted&(labels==r));gts=np.unique(atom_gt[members])
            if len(gts)<2:continue
            whole=combine(h[i] for i in members)
            split=[combine(h[i] for i in members[atom_gt[members]==gt]) for gt in gts]
            mixed_rows.append(dict(region=int(r),gt_ids=gts.tolist(),atoms=len(members),
                split_minus_mixed=float(sum(score(x) for x in split)-score(whole))))
        result=dict(assignment=dict(all_atoms=n,all_sources_in_atoms=len(source_ids),
            unanimous_target_atoms=int(target.sum()),accepted_atoms=int(accepted.sum()),
            accepted_sources=int(count[accepted].sum()),mixed_or_partly_unlabelled_atoms=int((~unanimous).sum()),
            unanimous_wrong_target_class_atoms=int((target&~accepted).sum()),
            unanimous_other_or_void_atoms=int((unanimous&~target).sum())),
            objects=object_rows,connected_components=component_rows,adjacent_fragment_pairs=boundary_rows,
            mixed_regions=mixed_rows,
            all_fragmented_object_merge=describe([r['whole_minus_fragments'] for r in object_rows if r['predicted_fragments']>1]),
            connected_fragmented_object_merge=describe([r['whole_minus_fragments'] for r in component_rows]),
            adjacent_same_object_merge=describe([r['merge_gain'] for r in boundary_rows]),
            splitting_mixed_regions=describe([r['split_minus_mixed'] for r in mixed_rows]),
            adjacent_pairs_no_common_view=sum(r['common_views']==0 for r in boundary_rows),
            adjacent_pairs_only_disjoint_common_tokens=sum(r['common_views']>0 and r['common_views']==r['common_views_disjoint_tokens'] for r in boundary_rows),
            objective_crosschecks=checks,max_absolute_crosscheck_error=max_error,seconds=time.monotonic()-started)
        dump(dest/'result.json',result);results[key]=result
        print('OBJECTIVE AUDIT',key,{x:result[x] for x in ['assignment','all_fragmented_object_merge',
            'connected_fragmented_object_merge','adjacent_same_object_merge','splitting_mixed_regions',
            'adjacent_pairs_no_common_view','adjacent_pairs_only_disjoint_common_tokens']},flush=True)
    protocol=read(OUT/'protocol.json')
    assert sha(__file__)==protocol['source_sha256']
    for f,expected in protocol['inputs'].items():assert sha(f)==expected,f
    dump(OUT/'summary.json',dict(complete=True,results=results,protocol_sha256=sha(OUT/'protocol.json'),
        files={str(f):sha(f) for f in OUT.glob('*/*') if f.is_file()},
        no_gt_predictions_exported=True,new_maps=0,new_benchmark_scores=0,new_training=0,
        new_robot_runs=0,new_independent_scenes=0,goal_complete=False))


if __name__=='__main__':run()
