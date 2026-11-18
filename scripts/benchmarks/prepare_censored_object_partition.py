"""GT-free, measured-surface observation atoms for a new partition mechanism.

Preparation is separate from prediction/GT capacity auditing. It never reads
reference annotations and does not export a scored object map.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from paper_final_common import ROOT,read,dump,sha

OUT=ROOT/'build/benchmarks/censored_object_partition_v1'
B=ROOT/'build/benchmarks'
RAW=B/'prethreshold_candidates_v1/predictions_complete.json'
SEM=B/'closed_vocabulary_semantics_v1/predictions_complete.json'


def freeze():
    assert not OUT.exists()
    parent=read(B/'map_credit_training_v1/protocol.json')
    contexts=parent['contexts'];raw=read(RAW);sem=read(SEM);inputs={}
    for f in [RAW,SEM,B/'reference_identity_audit_v1/summary.json']:
        inputs[str(f)]=sha(f)
    for c in contexts:
        key=c['key'];c['visibility']=str(B/'full_source_visual_visibility_audit_v1'/key/'manifest.json')
        files=[c['visibility'],c['field'],Path(c['full'])/'geometry.npz',c['job'],
               Path(c['ours'])/'geometry.npz',Path(c['ours'])/'classification.json']
        files += [r['file'] for r in read(c['visibility'])['records']]
        files += [r['file'] for r in raw['results'][key]]
        files += [r['file'] for r in sem['results'][key]]
        for f in files:inputs[str(f)]=sha(f)
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/censored_object_partition.py',
             ROOT/'omni_semantic_map/test/test_censored_object_partition.py']
    dump(OUT/'input_protocol.json',dict(created_epoch=time.time(),contexts=contexts,inputs=inputs,
        sources={str(f):sha(f) for f in sources},
        scope='Mechanism implementation and proposal-space/observation preflight, not a successful method or completed goal. Two exposed outdoor windows, no new training, foundation calls, map scores or independent input.',
        representation='Reuse complete actual LiDAR and fixed20cm cells. Each measured-visible source reads SAME cached native SAM3 positive-score car/building masks and SegFormer19-way argmax. Pick highest native-score covering same-class mask per point, stable row tie. No largest-component clipping or existing-object eligibility gate.',
        categories='Each cell contributes total one semantic vote per measured-visible frame, divided among pointwise19classwinners. Choose largest accumulated semantic vote, stable lowestclass tie. An object atom requires winning classbuilding2/car13 and at leastone same-class raw-mask observation. Other measured classes are retained as background transport seeds; unknown cells remain unknown.',
        observations='For each cell/frame/query record count divided by ALL measured-visible sources in that cell/frame. Thus total identity evidence<=1 per cell/frame. Mask absence/other-class/occlusion does not become a negative mask token. Distinct known same-class masks retain distinct categorical tokens. Frame vocabularies are observed query rows plus one unobserved escape category, minimumK2, fixed before cell-level class filtering.',
        region_objective='Sum over regions and observed views of log Gamma(1)-log Gamma(1+mass)+sum_q(log Gamma(1/K+count_q)-log Gamma(1/K)). Unit-total symmetric Dirichlet compatibility score, not calibrated identity posterior: pixels/cells/frames may be correlated. Unknown/noncommon views cancel exactly. Old object IDs are NOT must-link/cannot-link constraints.',
        proposal_graph='Common graph: inherited strict0.4m16NN cell edges between same-class atoms, plus a deterministic sorted-cell chain connecting every atom pair of consecutive indices sharing a frame/query. Query chains permit nonlocal object proposals but do not impose positive affinity. All merges must improve the whole-region objective, recomputed after every merge. Graph-local greedy optimum only.',
        distinction='Unlike old fixed-ID admission and anchored additive-edge clustering: new surface atoms may form identities; historical IDs may be split/rejoined at export; compatibility depends on complete current-region histograms, not the sum of frozen pair affinities. These are implemented choices, NOT proven novelty or gains. Pairwise same-input control and history-lock ablation required before contribution claims.',
        preflight='Freeze atoms and proposal graph before reference capacity audit. Need nonempty multi-view data and at leastone missed raw physical target per window with potential half coverage within a birth-eligible proposal component. Coverage is an optimistic necessary condition, not a promise of successful optimizer/output. Never use GT IDs or oracle subsets in prediction.',
        delivery='This stage saves measurements, observation provenance and graph; full map export/strong-control protocol follows a passed preflight. Same-version robot comparison and truly unused outdoor assessment remain requirements. Robot must receive the same raw-query interface; existing thresholded robot masks cannot silently substitute.',
        budget='One prepare pass over existing108views; no confidence/radius/prior sweep. No existing map, prediction or result is modified.'))
    print('CENSORED PARTITION INPUTS FROZEN',flush=True)


def verify():
    p=read(OUT/'input_protocol.json')
    for f,h in {**p['inputs'],**p['sources']}.items():assert sha(f)==h,f
    return p


def prepare():
    p=verify();assert not (OUT/'prepared.json').exists()
    rawseal=read(RAW);semseal=read(SEM);files={};results={}
    for c in p['contexts']:
        tick=time.monotonic();key=c['key'];folder=OUT/'prepared'/key
        folder.mkdir(parents=True,exist_ok=False)
        z=np.load(c['field']);cell=z['source_to_cell'];centers=z['centers']
        indptr=z['graph_indptr'];indices=z['graph_indices'];n=len(centers)
        visibility=read(c['visibility'])['records'];rawby={r['frame_id']:r for r in rawseal['results'][key]}
        semby={r['frame_id']:r for r in semseal['results'][key]}
        assert set(rawby)==set(semby)=={r['frame_id'] for r in visibility}
        semantic_votes=np.zeros((n,19),float);seen=np.zeros(n,np.int32)
        records=[];oc=[];ov=[];ot=[];om=[];os=[];vocabulary=[];categories=[]
        for rank,r in enumerate(visibility):
            frame=r['frame_id'];obs=np.load(r['file']);sid=obs['visible_source_ids'];px=obs['pixel_xy']
            assert len(np.unique(sid))==len(sid)
            cc=cell[sid];visible,count=np.unique(cc,return_counts=True);seen[visible]+=1
            denominator=np.zeros(n,np.int32);denominator[visible]=count
            semantic=np.load(semby[frame]['file'])['class_map'][px[:,1],px[:,0]].astype(np.int64)
            assert np.all((semantic>=0)&(semantic<19))
            encoded,counts=np.unique(cc*19+semantic,return_counts=True)
            semantic_votes[encoded//19,encoded%19]+=counts/denominator[encoded//19]
            raw=np.load(rawby[frame]['file']);scores=raw['scores'];packed=raw['masks_packed'];native=raw['semantic_ids']
            query=np.full(len(sid),-1,np.int32)
            for tid,rawid in [(2,11),(13,26)]:
                points=np.flatnonzero(semantic==tid);queries=np.flatnonzero((native==rawid)&(scores>0))
                if len(points) and len(queries):
                    xy=px[points]
                    inside=((packed[queries[:,None],xy[None,:,1],xy[None,:,0]//8]>>(7-xy[None,:,0]%8))&1).astype(bool)
                    values=np.where(inside,scores[queries,None],-1.)
                    winner=values.argmax(0);valid=values[winner,np.arange(len(points))]>0
                    query[points[valid]]=queries[winner[valid]]
                used=np.unique(query[(semantic==tid)&(query>=0)])
                view=2*rank+(tid==13);k=max(2,len(used)+1);categories.append(k)
                vocabulary.append(dict(view=int(view),frame_id=frame,train_class=tid,query_rows=used.tolist(),categories=k))
                take=(semantic==tid)&(query>=0)
                if not take.any():continue
                token=np.searchsorted(used,query[take])+1
                pair,weight=np.unique(cc[take]*k+token,return_counts=True)
                cells=pair//k;tokens=pair%k;mass=weight/denominator[cells]
                oc.append(cells);ov.append(np.full(len(cells),view,np.int32));ot.append(tokens)
                om.append(mass);os.append(np.full(len(cells),tid,np.int16))
            f=folder/'views'/f'{frame:06d}.npz';f.parent.mkdir(exist_ok=True)
            np.savez_compressed(f,source_ids=sid,pixel_xy=px,cells=cc,semantic=semantic,query_rows=query)
            files[str(f)]=sha(f)
            records.append(dict(frame_id=frame,view_rank=rank,file=str(f),measured_sources=len(sid),
                                visible_cells=len(visible),mask_observed_sources=int((query>=0).sum())))
            if (rank+1)%12==0:print('CENSORED OBSERVATIONS',key,rank+1,'/',len(visibility),flush=True)
        np.testing.assert_allclose(semantic_votes.sum(1),seen,atol=1e-10,rtol=0)
        cls=semantic_votes.argmax(1).astype(np.int16);cls[seen==0]=-1
        oc=np.concatenate(oc);ov=np.concatenate(ov);ot=np.concatenate(ot);om=np.concatenate(om);os=np.concatenate(os)
        keep=cls[oc]==os;oc,ov,ot,om=oc[keep],ov[keep],ot[keep],om[keep]
        atoms=np.unique(oc);atom=np.searchsorted(atoms,oc).astype(np.int32);natom=len(atoms)
        lookup=np.full(n,-1,np.int32);lookup[atoms]=np.arange(natom)
        # Cell/class/frame semantics and raw query readout remain replayable.
        frame_mass=np.bincount(atom.astype(np.int64)*len(categories)+ov,weights=om,
                               minlength=natom*len(categories)).reshape(natom,-1)
        assert frame_mass.max(initial=0)<=1+1e-12
        frame_count=(frame_mass>0).sum(1)
        left_full=np.repeat(np.arange(n,dtype=np.int32),np.diff(indptr));right_full=indices
        good=(left_full<right_full)&(lookup[left_full]>=0)&(lookup[right_full]>=0)&(cls[left_full]==cls[right_full])
        local=np.unique(lookup[left_full[good]].astype(np.int64)*natom+lookup[right_full[good]])
        # In a view, tokens of different classes use different view indices.
        stride=max(categories);group=ov.astype(np.int64)*stride+ot
        order=np.lexsort((atom,group));a,b=atom[order[:-1]],atom[order[1:]]
        good=(group[order[:-1]]==group[order[1:]])&(a!=b)
        chains=np.unique(np.minimum(a[good],b[good]).astype(np.int64)*natom+np.maximum(a[good],b[good]))
        edges=np.union1d(local,chains);left,right=edges//natom,edges%natom
        assert np.all(cls[atoms[left]]==cls[atoms[right]])
        graph=csr_matrix((np.ones(2*len(edges),np.int8),(np.r_[left,right],np.r_[right,left])),shape=(natom,natom))
        ncomp,component=connected_components(graph,directed=False)
        component_frames=np.zeros((ncomp,len(visibility)),bool);component_frames[component[atom],ov//2]=True
        source_count=np.bincount(cell,minlength=n)
        component_sources=np.bincount(component,weights=source_count[atoms],minlength=ncomp).astype(np.int64)
        birth=(component_sources>=16)&(component_frames.sum(1)>=3)
        component_class=np.full(ncomp,-1,np.int16);component_class[component]=cls[atoms]
        f=folder/'atoms.npz'
        np.savez_compressed(f,cell_ids=atoms,train_class=cls[atoms],observation_atom=atom,
            observation_view=ov,observation_token=ot,observation_mass=om,categories=np.asarray(categories,np.int32),
            left=left,right=right,local_edges=local,query_edges=chains,component=component,
            component_class=component_class,component_sources=component_sources,
            component_frames=component_frames.sum(1),component_birth_eligible=birth,
            semantic_class_all_cells=cls,semantic_votes=semantic_votes,seen=seen)
        files[str(f)]=sha(f);dump(folder/'vocabularies.json',vocabulary);files[str(folder/'vocabularies.json')]=sha(folder/'vocabularies.json')
        dump(folder/'views.json',records);files[str(folder/'views.json')]=sha(folder/'views.json')
        stats=dict(full_measured_sources=len(cell),full_cells=n,atoms=natom,
            class_atoms={str(t):int((cls[atoms]==t).sum()) for t in [2,13]},
            observation_entries=len(atom),atoms_with_multiple_frames=int((frame_count>=2).sum()),
            maximum_frames_per_atom=int(frame_count.max(initial=0)),proposal_edges=len(edges),
            local_edges=len(local),query_chain_edges=len(chains),nonlocal_query_edges=int(np.setdiff1d(chains,local).size),
            proposal_components=ncomp,birth_eligible_components=int(birth.sum()),
            birth_eligible_by_class={str(t):int((birth&(component_class==t)).sum()) for t in [2,13]},
            seconds=time.monotonic()-tick)
        results[key]=stats;print('CENSORED INPUTS READY',key,stats,flush=True)
    dump(OUT/'prepared.json',dict(complete=True,files=files,results=results,gt_read=False,
        new_maps=0,new_model_calls=0,input_protocol_sha256=sha(OUT/'input_protocol.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare'])
    globals()[parser.parse_args().action]()
