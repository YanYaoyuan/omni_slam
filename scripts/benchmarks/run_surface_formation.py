"""Bounded complete-map surface formation experiment, shared external semantics.

No reference labels are loaded by prepare/maps. Score only after every arm is
sealed. The candidate must beat both geometric and positive-only formation.
"""
import argparse,copy,shutil,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha
from capture_closed_vocabulary_semantics import OUT as SEM,BASE,INPUT
from omni_semantic_map.surface_formation import partition_surfaces,edge_view_cost

OUT=ROOT/'build/benchmarks/surface_formation_v1'
RAW=ROOT/'build/benchmarks/prethreshold_candidates_v1'
METHODS=['baseline','geometry','positive_only','first_view','ours_surface']


def freeze():
    assert not OUT.exists()
    parent=read(SEM/'protocol.json');inputs={}
    for path in [SEM/'aggregation_complete.json',RAW/'predictions_complete.json']:
        seal=read(path);assert seal['complete']
        inputs[str(path)]=sha(path)
        for f,h in seal['files'].items():assert sha(f)==h;inputs[f]=h
    for s in parent['scenes']:
        key=s['key'];m=read(INPUT/key/'manifest.json')
        assert len(m['records'])<=64
        for f in [INPUT/key/'manifest.json',m['geometry'],BASE/key/'geometry_graph/geometry.npz',BASE/key/'geometry_graph/classification.json',s['reference']]+[r['file'] for r in m['records']]:inputs[str(f)]=sha(f)
    f=ROOT/'build/benchmarks/signed_query_association_v1/summary.json';inputs[str(f)]=sha(f)
    files=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/surface_formation.py',ROOT/'omni_semantic_map/test/test_surface_formation.py',ROOT/'scripts/benchmarks/evaluate_kitti360_development.py',ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=parent['scenes'],methods=METHODS,input_hashes=inputs,source_hashes={str(f):sha(f) for f in files},
        hypothesis='Shared class evidence admits previously unmapped measured surfaces. Candidate jointly groups surface atoms with all positive and negative cross-region mask costs, instead of letting one positive bridge annex a contaminated region. Unanchored accepted groups can create new identities. Established signed clustering is not itself claimed novel.',
        input='Same108existingframes; external frozen Cityscapes SegFormer mean19classposteriors and SAM3 raw native queries. Original measured source provenance only. Existing anchors preserve their original class; unowned sources enter iff unique mean semantic winner is car13/building2. Conflicting old source provenance excluded. No GT, model fitting, new foundation-model inference or raw-query score tuning.',
        atoms='Unique(class,floor(xyz/0.2m),old_owner_or0), mean XYZ for neighbourhood calculations only. Original source XYZ are exported unchanged. Every old object is must-linked across its anchor atoms; different old IDs cannot merge. This does not fix old split/merge errors or remove old contamination.',
        edges='Union of16nearest-neighbour directed proposals per atom within0.4m, then unique undirected same-class pairs. Distances use mean atom positions. Same edge set in all arms. No normal, radius or degree sweep.',
        observations='Per actual measured-visible source, highest native positive-score same-class raw query containing its pixel; ties smaller raw query index. Atom token only when one query covers strictly more than half its current measured source votes. Otherwise unknown. For a co-token-visible edge, vote=2*IoU(selected query source masks)-1, where mask support uses all currently measured active sources of that class. No GT, full-image dense IoU or all-measured-source visibility claim for an entire voxel.',
        aggregation='One vote per frame per edge, mean over actual co-token-visible frames. Unknown views abstain. Edges with no valid pair evidence retain geometric prior+1. first_view uses first valid vote; ours and positive_only use all valid views. No confidence threshold on query scores besides native>0.',
        partition='Sparse greedy maximum-positive-gain agglomeration, sum of all boundary edge costs; eligible root pairs only if no conflict between old identity anchors. geometry uses+1 for every edge. positive_only clamps mean costs at0. first_view uses signed first evidence. ours_surface uses signed mean evidence. Initial forced old-anchor groups shared. Stop when no eligible positive boundary sum remains; stable node index ties. No claim of global optimum.',
        export='Preserve original fused map XYZ, IDs and classscores exactly. Append each previously unowned admitted actual source once if its group is old-anchored. Unanchored groups create new object if>=16sources and>=3distinct observed frames, shared rule. New score=mean external own-class probability over unique source points. No coordinate synthesis, extra SOR/NMS/postprocessing, or reclassification of old points.',
        scope='Object expansion and birth, not general historical ID revision. Parent includes supervised cross-window geometry and new external Cityscapes semantics. Two previously exposed development windows; neither independent testing nor official leaderboard. Simple/positive controls share both external models and all observations.',
        gate='Ours primary AP50:90 >= baseline eachwindow; mean>=baseline+0.01 and strictly above every control;>=1gained and0lostAP50GT. Candidate must differ in new point ownership from positive_only. Failure closes this version without tuning, robot or independent expansion.',
        budget='2windows x5maps, CPUonly, no threshold/model sweep. Previously failed branches remain frozen. 5coretests passed before freeze.'))
    print('FROZEN SURFACE FORMATION',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    return p


def seal(name):
    d=read(OUT/(name+'.json'))
    for f,h in d['files'].items():assert sha(f)==h,f
    return d


def prepare():
    p=verify();assert not (OUT/'prepared.json').exists();files={};stats={}
    capture=read(RAW/'predictions_complete.json')
    for s in p['scenes']:
        start=time.monotonic();key=s['key'];manifest=read(INPUT/key/'manifest.json')
        xyz=np.load(manifest['geometry'])['xyz'];sem=np.load(SEM/'aggregate'/key/'source.npz')
        owners=sem['owners'];pred=sem['mean_response'];declared=sem['declared_train_id']
        active=(owners>0)|((owners==0)&np.isin(pred,[2,13]));ids=np.flatnonzero(active)
        cls=np.where(owners[ids]>0,declared[ids],pred[ids]);anchor=np.maximum(owners[ids],0)
        assert np.isin(cls,[2,13]).all()
        keys=np.column_stack([cls,np.floor(xyz[ids]/.2).astype(np.int64),anchor])
        atomkeys,inverse=np.unique(keys,axis=0,return_inverse=True);n=len(atomkeys)
        count=np.bincount(inverse,minlength=n);centers=np.column_stack([np.bincount(inverse,weights=xyz[ids,j],minlength=n)/count for j in range(3)])
        full_to_atom=np.full(len(xyz),-1,np.int32);full_to_atom[ids]=inverse
        distances,nn=cKDTree(centers).query(centers,k=min(17,n),distance_upper_bound=.4,workers=1)
        a=np.repeat(np.arange(n),nn.shape[1]);b=nn.ravel();good=np.isfinite(distances.ravel())&(a!=b)&(b<n)
        a,b=a[good],b[good];same=atomkeys[a,0]==atomkeys[b,0];a,b=a[same],b[same]
        encoded=np.unique(np.minimum(a,b)*n+np.maximum(a,b));left,right=encoded//n,encoded%n
        total=np.zeros(len(left));seen=np.zeros(len(left),np.int16);first=np.zeros(len(left));framebits=np.zeros(n,np.uint64)
        rows={r['frame_id']:r for r in manifest['records']}
        for rank,rec in enumerate(capture['results'][key]):
            obs=np.load(rows[rec['frame_id']]['file']);raw=np.load(rec['file']);allids=obs['visible_source_ids'];take=active[allids];sid=allids[take];px=obs['pixel_xy'][take];atom=full_to_atom[sid]
            if not len(sid):continue
            visible=np.unique(atom);framebits[visible]|=np.uint64(1)<<np.uint64(rank)
            pointclass=atomkeys[atom,0];pointquery=np.full(len(sid),-1,np.int32)
            # Full native masks are decoded only at measured active source pixels.
            scores=raw['scores'];semantic=raw['semantic_ids'];packed=raw['masks_packed']
            inside=((packed[:,px[:,1],px[:,0]//8]>>(7-px[:,0]%8)[None,:])&1).astype(bool)
            for tid,native in [(2,11),(13,26)]:
                q=np.flatnonzero((scores>0)&(semantic==native));points=np.flatnonzero(pointclass==tid)
                if not len(q) or not len(points):continue
                values=np.where(inside[np.ix_(q,points)],scores[q,None],-1.)
                choice=np.argmax(values,axis=0);present=values[choice,np.arange(len(points))]>0
                pointquery[points[present]]=q[choice[present]]
            tokens=np.full(n,-1,np.int32);viscount=np.bincount(atom,minlength=n)
            good=pointquery>=0;encoded,countvotes=np.unique(atom[good]*len(scores)+pointquery[good],return_counts=True)
            aa,qq=encoded//len(scores),encoded%len(scores);majority=countvotes>.5*viscount[aa]
            assert len(np.unique(aa[majority]))==majority.sum();tokens[aa[majority]]=qq[majority]
            used=np.unique(tokens[tokens>=0]);lookup=np.full(len(scores),-1,np.int32);lookup[used]=np.arange(len(used));matrix=np.zeros((len(used),len(used)),np.float32)
            for tid,native in [(2,11),(13,26)]:
                qi=np.flatnonzero(semantic[used]==native);points=np.flatnonzero(pointclass==tid)
                m=inside[np.ix_(used[qi],points)].astype(np.float32)
                inter=m@m.T;mass=m.sum(1);union=mass[:,None]+mass[None,:]-inter
                matrix[np.ix_(qi,qi)]=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
            mapped=np.full(n,-1,np.int32);known=tokens>=0;mapped[known]=lookup[tokens[known]]
            cost,valid=edge_view_cost(mapped[left],mapped[right],matrix)
            first[valid&(seen==0)]=cost[valid&(seen==0)];total+=cost;seen+=valid.astype(np.int16)
            if (rank+1)%15==0:print('PREPARE VIEWS',key,rank+1,'/',len(rows),flush=True)
        mean=np.divide(total,seen,out=np.ones_like(total),where=seen>0);first[seen==0]=1.
        prob=sem['mean_probabilities'];confidence=prob[ids,cls]
        folder=OUT/'prepared'/key;folder.mkdir(parents=True,exist_ok=False)
        path=folder/'graph.npz';np.savez_compressed(path,source_ids=ids,source_atoms=inverse,atomkeys=atomkeys,centers=centers,
            left=left,right=right,mean_cost=mean,first_cost=first,evidence_frames=seen,framebits=framebits,confidence=confidence)
        files[str(path)]=sha(path);stats[key]=dict(active_sources=len(ids),new_sources=int((anchor==0).sum()),atoms=n,edges=len(left),
            observed_edges=int((seen>0).sum()),negative_edges=int((mean<0).sum()),zero_edges=int((mean==0).sum()),
            unknown_geometric_prior_edges=int((seen==0).sum()),first_vs_mean_changed=int((abs(first-mean)>1e-8).sum()),seconds=time.monotonic()-start)
        print('GRAPH READY',key,stats[key],flush=True)
    dump(OUT/'prepared.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json'),gt_read=False))


def maps():
    p=verify();seal('prepared');assert not (OUT/'predictions_complete.json').exists();files={};stats={}
    for s in p['scenes']:
        key=s['key'];d=dict(np.load(OUT/'prepared'/key/'graph.npz'));atoms=d['atomkeys'];n=len(atoms)
        base=np.load(BASE/key/'geometry_graph/geometry.npz');classification=read(BASE/key/'geometry_graph/classification.json');oldids=set(o['object_id'] for o in classification['objects'])
        xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];anchor=atoms[:,4];sid=d['source_ids'];sourceatom=d['source_atoms']
        free=anchor[sourceatom]==0;freepos=np.flatnonzero(free);stats[key]={}
        for method in METHODS:
            start=time.monotonic();folder=OUT/'maps'/key/method;folder.mkdir(parents=True,exist_ok=False)
            if method=='baseline':
                for name in ['geometry.npz','classification.json']:shutil.copyfile(BASE/key/'geometry_graph'/name,folder/name)
                run=dict(added_sources=0,new_objects=0,objects=len(oldids),exact_copy=True,seconds=time.monotonic()-start)
            else:
                cost={'geometry':np.ones(len(d['left'])),'positive_only':np.maximum(d['mean_cost'],0),'first_view':d['first_cost'],'ours_surface':d['mean_cost']}[method]
                groups,trace=partition_surfaces(n,d['left'],d['right'],cost,anchor)
                np.savez_compressed(folder/'partition.npz',root=groups)
                dump(folder/'merge_trace.json',trace)
                group_anchor=np.zeros(n,np.int64)
                for a in np.flatnonzero(anchor>0):
                    r=groups[a];assert group_anchor[r] in [0,anchor[a]];group_anchor[r]=anchor[a]
                bits=np.zeros(n,np.uint64);np.bitwise_or.at(bits,groups,d['framebits'])
                counts=np.bincount(groups[sourceatom[free]],minlength=n)
                roots=np.unique(groups[sourceatom[free]]);records=copy.deepcopy(classification);newrecords=[];nextid=max(oldids)+1;bygroup={};assign=np.zeros(n,np.int64)
                for r in roots:
                    if group_anchor[r]:assign[r]=group_anchor[r]
                    elif counts[r]>=16 and int(bits[r]).bit_count()>=3:
                        assign[r]=nextid;nextid+=1
                pointowner=assign[groups[sourceatom[free]]];good=pointowner>0;added=sid[free][good];pointowner=pointowner[good]
                for r in roots:
                    oid=int(assign[r])
                    if not oid:continue
                    positions=freepos[groups[sourceatom[free]]==r];tid=int(atoms[r,0]);assert np.all(atoms[groups==r,0]==tid)
                    bygroup[str(oid)]=dict(root=int(r),sources=int(len(positions)),observed_frames=int(bits[r]).bit_count(),old_anchor=int(group_anchor[r]))
                    if group_anchor[r]:continue
                    score=float(np.mean(d['confidence'][positions]));native=11 if tid==2 else 26
                    newrecords.append(dict(object_id=oid,observation_ids=[],selected_observation_ids=[],source_root=int(r),predictions=dict(mean=dict(name='building' if tid==2 else 'car',semantic_id=native,train_id=tid,score=score,retained=True))))
                records['objects'].extend(newrecords);records['objects'].sort(key=lambda x:x['object_id'])
                assert len(np.unique(added))==len(added)
                np.savez_compressed(folder/'geometry.npz',xyz=np.concatenate([base['xyz'],xyz[added]]),instance=np.r_[base['instance'],pointowner],added_source_ids=added)
                dump(folder/'classification.json',records);dump(folder/'groups.json',bygroup)
                run=dict(added_sources=len(added),new_objects=len(newrecords),objects=len(records['objects']),
                    groups=len(np.unique(groups)),positive_merges=len(trace['merges']),negative_edges=int((cost<0).sum()),old_coordinates_ids_scores_unchanged=True,seconds=time.monotonic()-start)
            dump(folder/'run.json',run)
            for f in folder.iterdir():files[str(f)]=sha(f)
            stats[key][method]=run;print('MAP',key,method,run,flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json'),sealed_before_gt_scoring=True))


def score():
    from evaluate_kitti360_development import transfer,clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p=verify();seal('predictions_complete');assert not (OUT/'summary.json').exists();results={};means={m:[] for m in METHODS};gained=lost=0;gate=True
    evaluator=ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py'
    for s in p['scenes']:
        key=s['key'];ref=np.load(s['reference']);rows={}
        for method in METHODS:
            folder=OUT/'maps'/key/method;g=np.load(folder/'geometry.npz');objects=read(folder/'classification.json')['objects']
            pred=transfer(g['xyz'],g['instance'],ref['xyz'],.2)
            scored=evaluate(ref['train_instance'],ref['confidence'],pred,{o['object_id']:o['predictions']['mean'] for o in objects},evaluator)
            dump(folder/'matches.json',clean_json(scored['matches']));np.savez_compressed(folder/'reference_prediction.npz',instance=pred)
            rows[method]=dict(metrics=clean_json(scored['averages']),matched_gt_ids=sorted(matched_ids(scored['matches'])),run=read(folder/'run.json'))
            means[method].append(float(scored['averages']['allAp']));print('FULL MAP SCORE',key,method,rows[method]['metrics'],flush=True)
        expected=read(ROOT/'build/benchmarks/signed_query_association_v1/summary.json')['results'][key]['methods']['geometry_graph']['metrics']
        assert abs(rows['baseline']['metrics']['allAp']-expected['allAp'])<1e-12
        old=set(rows['baseline']['matched_gt_ids'])
        for row in rows.values():
            ids=set(row['matched_gt_ids']);row['changes_vs_baseline']=dict(gained=sorted(ids-old),lost=sorted(old-ids))
        change=rows['ours_surface']['changes_vs_baseline'];gained+=len(change['gained']);lost+=len(change['lost'])
        gate &= rows['ours_surface']['metrics']['allAp']>=rows['baseline']['metrics']['allAp'];results[key]=dict(methods=rows)
    mean={m:float(np.mean(v)) for m,v in means.items()};gate &= mean['ours_surface']>=mean['baseline']+.01 and mean['ours_surface']>max(v for m,v in mean.items() if m!='ours_surface') and gained>=1 and lost==0
    # Changes in file metadata alone do not establish different ownership.
    different={}
    for key in results:
        a=np.load(OUT/'maps'/key/'positive_only/geometry.npz');b=np.load(OUT/'maps'/key/'ours_surface/geometry.npz')
        different[key]=not(np.array_equal(a['added_source_ids'],b['added_source_ids']) and np.array_equal(a['instance'],b['instance']))
    gate &= any(different.values())
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,gained=gained,lost=lost,
        advance_gate=bool(gate),ownership_differs_vs_positive=different,protocol_sha256=sha(OUT/'protocol.json'),prediction_seal_sha256=sha(OUT/'predictions_complete.json'),official_leaderboard=False,independent_test=False))
    print('ADVANCE GATE',bool(gate),mean,'gained',gained,'lost',lost,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','maps','score']);globals()[ap.parse_args().action]()
