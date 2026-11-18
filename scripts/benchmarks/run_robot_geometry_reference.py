"""Same frozen geometry_graph reference on the real robot; no accuracy claim."""
import argparse,time,shutil
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from geometry_reference_backend import frame_components,export_reference
from omni_semantic_map.query_hypothesis_evidence import membership,retrieve

OUT=ROOT/'build/benchmarks/robot_geometry_reference_v1'
PARENT=ROOT/'build/benchmarks/signed_query_association_v1'
BASE=ROOT/'build/benchmarks/intrinsic_surface_ownership_v1'
OWN=ROOT/'build/benchmarks/surface_ownership_v1'
INPUT=ROOT/'build/benchmarks/discovery_measured_inputs_v1'
FRONT=ROOT/'build/benchmarks/discovery_query_benchmark_v1'
EVIDENCE=ROOT/'build/benchmarks/query_hypothesis_evidence_v1'
KEY='rosbag2245'


def pieces(values,offsets):return [values[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
def ragged(sets):return (np.concatenate(sets) if sets else np.empty(0,np.int64)),np.r_[0,np.cumsum([len(v) for v in sets])]


def golden():
    assert not (OUT/'golden.json').exists();checks={}
    for key in ['kitti360_0000','kitti360_0002']:
        d=dict(np.load(PARENT/'nodes'/key/'nodes.npz'));metadata=read(PARENT/'nodes'/key/'nodes.json');sets=pieces(d['source_ids'],d['source_offsets']);source=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];cells=np.load(INPUT/'query_cells'/key/'source_to_cell.npy');base=dict(np.load(BASE/'maps'/key/'intrinsic_geometry/geometry.npz'));classification=read(BASE/'maps'/key/'intrinsic_geometry/classification.json')
        geometry,records,groups,stat=export_reference(base,classification,source,sets,metadata,cells)
        expected=np.load(PARENT/'maps'/key/'geometry_graph/geometry.npz')
        for field in geometry:np.testing.assert_array_equal(geometry[field],expected[field])
        assert records==read(PARENT/'maps'/key/'geometry_graph/classification.json')
        groups['old_id_assignment']={str(k):v for k,v in groups['old_id_assignment'].items()};assert groups==read(PARENT/'maps'/key/'geometry_graph/groups.json')
        manifest=read(INPUT/key/'manifest.json');raw=read(ROOT/'build/benchmarks/prethreshold_candidates_v1/predictions_complete.json')['results'][key];h=read(EVIDENCE/'index'/key/'hypotheses.json');z=dict(np.load(EVIDENCE/'index'/key/'hypotheses.npz'));supports=pieces(z['source_ids'],z['source_offsets']);samples=[]
        for record in [raw[0],raw[-1]]:
            frame=record['frame_id'];v=dict(np.load(next(r['file'] for r in manifest['records'] if r['frame_id']==frame)));d=dict(np.load(record['file']));active=np.flatnonzero(d['scores']>0);ids=v['visible_source_ids'];xy=v['pixel_xy'];inside=((d['masks_packed'][active[:,None],xy[:,1],xy[:,0]//8]>>(7-xy[:,0]%8)[None,:])&1).astype(bool)
            actual=frame_components(source[ids],inside);old={q:supports[m['index']] for m in h if m['frame_id']==frame for q in m['aliases']}
            for q,component in zip(active,actual):np.testing.assert_array_equal(np.sort(ids[component]),old.get(int(q),np.empty(0,np.int64)))
            samples.append(dict(frame=frame,queries=len(active),all_components_exact=True))
        checks[key]=dict(full_geometry_exact=True,classification_exact=True,groups_exact=True,component_samples=samples,stats=stat)
        print('GOLDEN',key,checks[key],flush=True)
    dump(OUT/'golden.json',dict(complete=True,checks=checks,backend_sha256=sha(ROOT/'scripts/benchmarks/geometry_reference_backend.py'),driver_sha256=sha(__file__),scope='Full old two-window map replay; all native-positive components in first/last frame of each window. No new AP/GT, no retuning.'))


def freeze():
    assert not (OUT/'protocol.json').exists();gold=read(OUT/'golden.json');assert gold['complete'] and gold['backend_sha256']==sha(ROOT/'scripts/benchmarks/geometry_reference_backend.py')
    manifest=read(INPUT/KEY/'manifest.json');capture=read(ROOT/'build/benchmarks/prethreshold_candidates_v1/protocol.json');front=read(FRONT/'protocol.json');scene=next(s for s in front['scenes'] if s['key']==KEY);inputs={str(OUT/'golden.json'):sha(OUT/'golden.json')}
    for f in [INPUT/KEY/'manifest.json',manifest['geometry'],INPUT/'query_cells'/KEY/'source_to_cell.npy',OWN/'features'/KEY/'core_sources.json',BASE/'models'/KEY/'intrinsic_geometry/model.pt',FRONT/'masks_complete.json',PARENT/'protocol.json',PARENT/'summary.json']:
        inputs[str(f)]=sha(f)
    for record in manifest['records']:
        for f in [record['file'],record['rgb']]:inputs[f]=sha(f)
    for record in read(FRONT/'masks_complete.json')['scenes'][KEY]['records']:inputs[record['file']]=sha(record['file'])
    for method in ['baseline','intrinsic_geometry']:
        for name in ['geometry.npz','classification.json']:f=BASE/'maps'/KEY/method/name;inputs[str(f)]=sha(f)
    sources=[Path(__file__),ROOT/'scripts/benchmarks/geometry_reference_backend.py',ROOT/'scripts/benchmarks/capture_prethreshold_candidates.py',ROOT/'omni_semantic_map/omni_semantic_map/query_hypothesis_evidence.py',ROOT/'omni_semantic_map/omni_semantic_map/signed_query_association.py',ROOT/'omni_semantic_map/test/test_geometry_reference_backend.py']
    sources += [Path('/home/user/vbot/sam3/sam3')/f for f in ['model_builder.py','model/sam3_image.py','model/sam3_image_processor.py']]
    p=dict(created_epoch=time.time(),scenes=[scene],methods=['native_core','intrinsic_geometry','geometry_graph_reference'],inputs=inputs,sources={str(f):sha(f) for f in sources},checkpoint=capture['checkpoint'],checkpoint_sha256=capture['checkpoint_sha256'],prompts=['car','building'],semantic_ids=[26,11],
        scope='Reference-chain portability on existing330-frame rosbag2245. No failed candidate is expanded or promoted. No robot GT, benchmark accuracy, real-time S100 or novelty claim. Native/intrinsic maps are explicitly reused old outputs; only geometry_graph reference is newly executed.',
        capture='Reuse exact pinned native raw-capture function in an isolated process, new output/verification scope only. GPU1,18GiBfree,50percent cap,1008BF16. For every330frame require high-score masks/scores/classes exact equality with old frozen frontend. No source files patched, no prompts changed.330encodes+660decodes plus fixed synthetic warmup.',
        representation='Same all-positive native hypotheses,largest0.4m measured component>=16; same-frame/class identical components collapse to largest native score,aliases kept. Shared per-frame radius graph is mathematically equivalent induced-subgraph extraction; tested against original and existing frame artifacts.',
        selection='All native>0.1 anchor hypotheses; one fixed expansion to hypotheses selected by any anchor at another frame with>=16visible cells and best same-class measuredIoU>0.5. Old nodes original provenance, excluding learned additions. No recursive discovery or learned signed associations.',
        map='Exact frozen geometry_graph control: same-class cellIoU>0.5 components; same native3seed-frame birth, source locking, score and ID rules. Full old two-window geometries/classifications/groups reproduced exactly before this scope froze. Core XYZ preserved; measured sources unique.',
        budget='One raw capture and one CPU representation/map pass; two exact reference map copies plus one new reference map. No fits/tuning/new dataset labels. Qualitative report and object-level change provenance only.',
        goal_boundary='Provides missing stable reference robot comparison, not completed real-world validation for a successful new mechanism. Original goal remains incomplete if no effective mechanism or independent assessment exists.')
    dump(OUT/'protocol.json',p);dump(OUT/'raw/protocol.json',p);print('FROZEN ROBOT REFERENCE',len(manifest['records']),flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['inputs'],**p['sources']}.items():assert sha(f)==h,f
    return p


def capture():
    p=verify();assert not (OUT/'raw/execution.json').exists();assert sha(OUT/'raw/protocol.json')==sha(OUT/'protocol.json')
    import capture_prethreshold_candidates as native
    native.OUT=OUT/'raw';native.verify=verify
    dump(OUT/'raw/execution.json',dict(started_epoch=time.time(),physical_gpu=1,protocol_sha256=sha(OUT/'protocol.json'),native_capture_source_sha256=sha(native.__file__)))
    native.predict()


def index():
    verify();assert not (OUT/'index_complete.json').exists();raw=read(OUT/'raw/predictions_complete.json')
    for f,h in raw['files'].items():assert sha(f)==h,f
    manifest=read(INPUT/KEY/'manifest.json');views={r['frame_id']:r for r in manifest['records']};source=np.load(manifest['geometry'])['xyz'];cells=np.load(INPUT/'query_cells'/KEY/'source_to_cell.npy');hypotheses=[];sets=[];frames=[];files={};start=time.monotonic()
    for rank,record in enumerate(raw['results'][KEY]):
        d=dict(np.load(record['file']));v=dict(np.load(views[record['frame_id']]['file']));ids=v['visible_source_ids'];xy=v['pixel_xy'];active=np.flatnonzero(d['scores']>0)
        inside=((d['masks_packed'][active[:,None],xy[:,1],xy[:,0]//8]>>(7-xy[:,0]%8)[None,:])&1).astype(bool);components=frame_components(source[ids],inside);chosen={};query_cells=[]
        for q,mask,component in zip(active,inside,components):
            query_cells.append(np.unique(cells[ids[mask]]))
            if len(component)<16:continue
            support=np.sort(ids[component]);identity=(int(d['semantic_ids'][q]),support.tobytes());score=float(d['scores'][q])
            if identity not in chosen:chosen[identity]=dict(query_id=int(q),score=score,sources=support,aliases=[int(q)])
            else:
                old=chosen[identity];old['aliases'].append(int(q))
                if score>old['score']:old.update(query_id=int(q),score=score)
        for (sem,_),h in sorted(chosen.items(),key=lambda kv:kv[1]['query_id']):
            support=h.pop('sources');sets.append(support);hypotheses.append(dict(index=len(hypotheses),frame_id=record['frame_id'],semantic_id=sem,**h))
        values,offsets=ragged(query_cells);file=OUT/'index'/f'{record["frame_id"]:06d}.npz';file.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(file,query_ids=active,semantic_ids=d['semantic_ids'][active],scores=d['scores'][active],cell_ids=values,offsets=offsets,visible_cells=np.unique(cells[ids]));files[str(file)]=sha(file);frames.append(dict(frame_id=record['frame_id'],file=str(file)))
        if (rank+1)%10==0:print('INDEX',rank+1,len(raw['results'][KEY]),len(hypotheses),flush=True)
    values,offsets=ragged(sets);file=OUT/'index/hypotheses.npz';np.savez_compressed(file,source_ids=values,source_offsets=offsets);files[str(file)]=sha(file);file=OUT/'index/hypotheses.json';dump(file,hypotheses);files[str(file)]=sha(file)
    dump(OUT/'index_complete.json',dict(complete=True,files=files,frames=frames,hypotheses=len(hypotheses),seconds=time.monotonic()-start,raw_seal_sha256=sha(OUT/'raw/predictions_complete.json')))


def maps():
    verify();assert not (OUT/'maps_complete.json').exists();seal=read(OUT/'index_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    meta=read(OUT/'index/hypotheses.json');raw=dict(np.load(OUT/'index/hypotheses.npz'));sets=pieces(raw['source_ids'],raw['source_offsets']);cells=np.load(INPUT/'query_cells'/KEY/'source_to_cell.npy');anchors=np.array([i for i,m in enumerate(meta) if m['score']>.1],np.int64);selected=set(map(int,anchors));lookup={(m['frame_id'],q):i for i,m in enumerate(meta) for q in m['aliases']};h=membership([np.unique(cells[sets[i]]) for i in anchors],int(cells.max())+1);sem=np.array([meta[i]['semantic_id'] for i in anchors]);seeds=np.array([meta[i]['frame_id'] for i in anchors])
    for rank,frame in enumerate(seal['frames']):
        d=dict(np.load(frame['file']));q=membership(pieces(d['cell_ids'],d['offsets']),h.shape[1]);z=retrieve(h,sem,q,d['semantic_ids'],d['scores'],d['visible_cells']);good=(z['visible_cells']>=16)&(z['agreement']>.5)&(z['query']>=0)&(seeds!=frame['frame_id'])
        for qi in np.unique(z['query'][good]):
            token=(frame['frame_id'],int(d['query_ids'][qi]))
            if token in lookup:selected.add(lookup[token])
        if (rank+1)%30==0:print('READOUT',rank+1,len(seal['frames']),'selected',len(selected),flush=True)
    selected=sorted(selected);provenance=read(OWN/'features'/KEY/'core_sources.json');base=dict(np.load(BASE/'maps'/KEY/'intrinsic_geometry/geometry.npz'));classification=read(BASE/'maps'/KEY/'intrinsic_geometry/classification.json');source=np.load(read(INPUT/KEY/'manifest.json')['geometry'])['xyz'];nodes=[];metadata=[]
    for o in sorted(classification['objects'],key=lambda x:x['object_id']):
        nodes.append(np.array(provenance[str(o['object_id'])],np.int64));metadata.append(dict(old_object_id=o['object_id'],hypothesis_index=-1,seed_frame=-1,semantic_id=o['predictions']['mean']['semantic_id'],score=o['predictions']['mean']['score']))
    for i in selected:
        nodes.append(sets[i]);m=meta[i];metadata.append(dict(old_object_id=-1,hypothesis_index=i,seed_frame=m['frame_id'],semantic_id=m['semantic_id'],score=m['score']))
    files={};stats={}
    for target,old in [('native_core','baseline'),('intrinsic_geometry','intrinsic_geometry')]:
        folder=OUT/'maps'/target;folder.mkdir(parents=True,exist_ok=False)
        for name in ['geometry.npz','classification.json']:shutil.copyfile(BASE/'maps'/KEY/old/name,folder/name);files[str(folder/name)]=sha(folder/name)
        stats[target]=dict(reused_frozen=True,source=str(BASE/'maps'/KEY/old))
    started=time.monotonic();g,c,groups,stat=export_reference(base,classification,source,nodes,metadata,cells);folder=OUT/'maps/geometry_graph_reference';folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'geometry.npz',**g);dump(folder/'classification.json',c);dump(folder/'groups.json',groups);values,offsets=ragged(nodes);np.savez_compressed(folder/'nodes.npz',source_ids=values,source_offsets=offsets);dump(folder/'nodes.json',metadata)
    stat.update(seconds=time.monotonic()-started,anchors=len(anchors),weak_nodes=len(selected)-len(anchors),nodes=len(nodes));dump(folder/'run.json',stat)
    for f in folder.iterdir():files[str(f)]=sha(f)
    stats['geometry_graph_reference']=stat;dump(OUT/'maps_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json'),robot_accuracy_available=False,novel_method=False));print('ROBOT REFERENCE COMPLETE',stat,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['golden','freeze','capture','index','maps']);globals()[parser.parse_args().action]()
