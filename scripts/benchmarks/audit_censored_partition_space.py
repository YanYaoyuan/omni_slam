"""Pre-map capacity and observation audit of the frozen atom/graph space.

The geometric reach sets are optimistic supersets, NOT predicted objects.
GT selects neither atoms, graph edges, seeds, views, nor inference parameters.
"""
import time
from pathlib import Path
import runpy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha
from prepare_censored_object_partition import OUT,verify
from evaluate_kitti360_development import read_static_ply
from omni_semantic_map.kitti360_reference_identity import encode_reference
from omni_semantic_map.censored_object_partition import make_histograms,merge_gain,join_histogram


def run():
    p=verify();prep=read(OUT/'prepared.json');assert prep['complete']
    assert not (OUT/'space_audit_protocol.json').exists()
    for f,h in prep['files'].items():assert sha(f)==h,f
    inputs={str(OUT/'prepared.json'):sha(OUT/'prepared.json')}
    for c in p['contexts']:
        for f in [c['raw_reference'],c['reference'],ROOT/'build/benchmarks/reference_identity_audit_v1/summary.json']:
            inputs[str(f)]=sha(f)
    dump(OUT/'space_audit_protocol.json',dict(created_epoch=time.time(),source_sha256=sha(__file__),inputs=inputs,
        capacity='A future object lies within one same-class query-graph component. For an optimistic geometry-transport superset, allow it to occupy ANY full measured source reachable on the inherited geometry graph without crossing measured semantic background cells. Ignore competing foreground seeds and optimizer/acceptance limits, allow perfect removal of false geometry. Per-GT source coverage of this superset bounds possible intersection; it is not attainable IoU/AP.',
        birth='Only query components with>=16real sources and>=3observedframes can contain a future born object. This is a necessary-component check, not every subregion eligibility.',
        identity='Primary capacity diagnosis uses raw11/26 targets, aliasesvoid, all reference coordinates/weights retained. Original upstream-compatible object coverage also reported. No scoring/prediction/reference labels supplied to optimizer.',
        observations='Recompute2048fixed source-pixel selections per window from scalar unpacked masks, plus all cell/frame mass<=1. Sample2048fixed proposal edges for positive/negative gain;256deterministic region triples for nonadditivity. Fixed stride sampling before labels, not success-selected.',
        gate='Both windows have positive and negative observed merge gains, nonadditive triple evidence, and at leastone currently missed RAW target with optimistic coverage>0.5 in a birth-eligible proposal component. Passing only permits full same-input comparisons; it does not prove efficacy.',
        bounds_do_not='No topological compactness, geometric correctness, AP gain, independence or official-server claim. Larger reach sets can be loose. All oracle numbers stay in this audit.'))
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    prior=read(ROOT/'build/benchmarks/reference_identity_audit_v1/summary.json');results={};all_gate=True
    rawseal=read(ROOT/'build/benchmarks/prethreshold_candidates_v1/predictions_complete.json')
    for c in p['contexts']:
        tick=time.monotonic();key=c['key'];d=dict(np.load(OUT/'prepared'/key/'atoms.npz'))
        views=read(OUT/'prepared'/key/'views.json');rawby={r['frame_id']:r for r in rawseal['results'][key]}
        checked=0
        for rec in [views[0],views[len(views)//3],views[2*len(views)//3],views[-1]]:
            v=dict(np.load(rec['file']));native=dict(np.load(rawby[rec['frame_id']]['file']))
            indices=np.linspace(0,len(v['source_ids'])-1,min(512,len(v['source_ids'])),dtype=int)
            for i in indices:
                cls=int(v['semantic'][i]);expected=-1
                if cls in [2,13]:
                    nativecls=11 if cls==2 else 26;x,y=map(int,v['pixel_xy'][i])
                    q=np.flatnonzero((native['semantic_ids']==nativecls)&(native['scores']>0))
                    inside=[j for j in q if np.unpackbits(native['masks_packed'][j,y])[x]]
                    if inside:expected=max(inside,key=lambda j:(float(native['scores'][j]),-int(j)))
                assert int(v['query_rows'][i])==expected
                checked+=1
        n=len(d['cell_ids']);hist=make_histograms(n,d['observation_atom'],d['observation_view'],
            d['observation_token'],d['observation_mass'],d['categories'])
        chosen=np.linspace(0,len(d['left'])-1,min(2048,len(d['left'])),dtype=int)
        gains=np.array([merge_gain(hist[d['left'][i]],hist[d['right'][i]],d['categories']) for i in chosen])
        neighbor=[[] for _ in range(n)]
        for a,b in zip(d['left'],d['right']):neighbor[a].append(int(b));neighbor[b].append(int(a))
        anchors=np.array([i for i in range(n) if len(neighbor[i])>=2],int)
        triples=anchors[np.linspace(0,len(anchors)-1,min(256,len(anchors)),dtype=int)]
        difference=[]
        for a in triples:
            b,z=sorted(neighbor[a])[:2]
            joint=merge_gain(join_histogram(hist[a],hist[b]),hist[z],d['categories'])
            pair=merge_gain(hist[a],hist[z],d['categories'])+merge_gain(hist[b],hist[z],d['categories'])
            difference.append(joint-pair)
        f=np.load(c['field']);cell=f['source_to_cell'];nc=len(d['seen'])
        bg=(d['seen']>0)&~np.isin(d['semantic_class_all_cells'],[2,13]);allowed=~bg
        graph=csr_matrix((np.ones(len(f['graph_indices']),np.int8),f['graph_indices'],f['graph_indptr']),shape=(nc,nc))
        count,connected=connected_components(graph[allowed][:,allowed],directed=False)
        spatial=np.full(nc,-1,np.int32);spatial[allowed]=connected
        assert np.all(spatial[d['cell_ids']]>=0)
        domains={}
        for component in np.flatnonzero(d['component_birth_eligible']):
            cells=d['cell_ids'][d['component']==component]
            ids=tuple(np.unique(spatial[cells]).tolist());cls=int(d['component_class'][component])
            domains.setdefault((cls,ids),[]).append(int(component))
        full=np.load(Path(c['full'])/'geometry.npz')['xyz'];source_spatial=spatial[cell]
        ref=dict(np.load(c['reference']));raw=read_static_ply(c['raw_reference'])
        physical,mapping=encode_reference(raw['semantic'],raw['instance'],taxonomy,'target_only')
        truth_by_mode={'upstream_train':ref['train_instance'],'target_only':physical}
        tables={}
        for mode,truth in truth_by_mode.items():
            mass=np.bincount(truth,weights=ref['confidence'])
            gt=np.flatnonzero(np.isin(np.arange(len(mass))//1000,[2,13])&(mass>=100))
            matches=set(prior['results'][key]['modes'][mode]['methods']['baseline']['matched_gt_ids'])
            tables[mode]={int(g):dict(gt_id=int(g),mass=float(mass[g]),baseline_matched=int(g) in matches,
                                    direct_cell_coverage=0.,reachable_coverage=0.,direct_component=None,reachable_components=[]) for g in gt}
        # Direct measured cell support for each actual graph component.
        components=[(int(k),int(d['component_class'][k])) for k in np.flatnonzero(d['component_birth_eligible'])]
        search=[]
        cell_component=np.full(nc,-1,np.int32);cell_component[d['cell_ids']]=d['component']
        for k,cls in components:search.append(('direct',[k],cls,cell_component[cell]==k))
        for (cls,ids),comps in domains.items():search.append(('reach',comps,cls,np.isin(source_spatial,ids)))
        domain_rows=[]
        for rank,(kind,comps,cls,take) in enumerate(search):
            xyz=full[take];assert len(xyz)>0;tree=cKDTree(xyz)
            eligible=(ref['train_instance']//1000==cls)
            # Exact bounding box exclusion only; all in-box target points queried.
            eligible &= np.all((ref['xyz']>=xyz.min(0)-.2)&(ref['xyz']<=xyz.max(0)+.2),axis=1)
            positions=np.flatnonzero(eligible);covered=np.zeros(len(physical),bool)
            for lo in range(0,len(positions),100000):
                ii=positions[lo:lo+100000]
                covered[ii]=tree.query(ref['xyz'][ii],distance_upper_bound=.2,workers=1)[0]<.2
            for mode,truth in truth_by_mode.items():
                support=np.bincount(truth[covered],weights=ref['confidence'][covered])
                for gid,row in tables[mode].items():
                    if gid//1000!=cls:continue
                    value=float(support[gid]/row['mass']) if gid<len(support) else 0.
                    field='direct_cell_coverage' if kind=='direct' else 'reachable_coverage'
                    if value>row[field]:
                        row[field]=value
                        row['direct_component' if kind=='direct' else 'reachable_components']=comps[0] if kind=='direct' else comps
            domain_rows.append(dict(kind=kind,components=comps,train_class=cls,source_points=int(take.sum())))
            print('PARTITION CAPACITY DOMAIN',key,rank+1,'/',len(search),kind,cls,int(take.sum()),flush=True)
        for table in tables.values():
            for row in table.values():assert row['direct_cell_coverage']<=row['reachable_coverage']+1e-10
        physical_rows=list(tables['target_only'].values())
        opportunity=[r['gt_id'] for r in physical_rows if not r['baseline_matched'] and r['reachable_coverage']>.5]
        direct=[r['gt_id'] for r in physical_rows if not r['baseline_matched'] and r['direct_cell_coverage']>.5]
        positive=int((gains>1e-10).sum());negative=int((gains<-1e-10).sum());nonlinear=int((np.abs(difference)>1e-8).sum())
        gate=bool(opportunity and positive and negative and nonlinear);all_gate &= gate
        results[key]=dict(scalar_mask_checks=checked,sampled_edges=len(gains),positive_edges=positive,negative_edges=negative,
            neutral_edges=int((abs(gains)<=1e-10).sum()),nonadditive_triples=nonlinear,sampled_triples=len(difference),
            raw_target_potential_new=opportunity,raw_target_direct_potential_new=direct,
            objects={mode:list(rows.values()) for mode,rows in tables.items()},
            proposal_components=len(components),distinct_reach_domains=len(domains),domains=domain_rows,
            necessary_preflight_passed=gate,seconds=time.monotonic()-tick)
        dump(OUT/(key+'_space_audit.json'),results[key])
        print('CENSORED SPACE GATE',key,gate,'direct',direct,'optimistic reach',opportunity,flush=True)
    dump(OUT/'space_audit.json',dict(complete=True,results=results,necessary_preflight_passed=bool(all_gate),
        source_sha256=sha(__file__),new_maps=0,new_scores=0,new_fits=0,gt_diagnostic_only=True,
        not_sufficient_for_method_or_goal_completion=True))


if __name__=='__main__':run()
