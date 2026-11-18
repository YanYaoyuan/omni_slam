"""Finite necessary-condition probe for measured-ray surface continuation.

No new map/AP, no GT-selected geometry and no parameter retries. The purpose
is to test whether a physically witnessed gap discriminates object-boundary
links, before incorporating it into full object formation.
"""
import argparse, runpy, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT, read, dump, sha
from prepare_full_lidar_surface_inputs import OUT as INPUT, verify as verify_input, CALIB
from audit_mask_observation_code import source_anchors
from omni_semantic_map.kitti360_input import load_left_calibration
from omni_semantic_map.surface_gap_witness import MeasuredRayTubes, UNKNOWN, SUPPORTED, OCCLUDED, FREE

OUT=ROOT/'build/benchmarks/surface_gap_witness_v1'


def freeze():
    assert not OUT.exists(); p=verify_input(); seal=read(INPUT/'prepared.json');assert seal['complete']
    inputs={str(INPUT/'prepared.json'):sha(INPUT/'prepared.json'),**seal['files']}
    for f,h in inputs.items():assert sha(f)==h
    previous=read(ROOT/'build/benchmarks/provenance_object_memory_v1/protocol.json')
    inputs.update(previous['input_hashes']);inputs.update(previous['source_hashes'])
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/surface_gap_witness.py',ROOT/'omni_semantic_map/test/test_surface_gap_witness.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=p['scenes'],references=previous['references'],evaluation_scenes=previous['scenes'],
        input_hashes=inputs,source_hashes={str(f):sha(f) for f in sources},
        hypothesis='Can an actual scan supporting both surface endpoints and traversing empty space at their midpoint reject erroneous object-continuation links, beyond naive midpoint carving and the first common observed view?',
        candidates='Full same-scan measured ledger. One first-chronological actual source per20cmworldcell; unowned sources only. Pair to nearest frozen positive sourceanchor with strictdistance>5cm and<=40cm. Exclude conflicting anchors. Deterministically sample up to4096old-camera and4096additional-source candidates perwindow, uniformly in sortedsourceID, before geometry/GT checks.',
        geometry='Both endpoints must have>=6measured neighbours among32nearest within40cm; localPCA residualstd<=5cm,secondprincipalstd>=5cm. Absolute normaldot>=cos15deg and both symmetric point-to-tangent offsets<=5cm. Candidate midpoint is a link query only; never exported as geometry.',
        rays='Actual rawscan rays with calibratedsensororigin. All beams within5cmmetric tube; support if range residual<=5cm; aligned nearerreturn marks occlusion; support overrides occlusion and both override free. Missingbeam unknown. Gap iff endpoints supported andmidpointfree inthesamescan. Only scans at/afterbothendpointsbirth. No interpolation,deskeworfilleddepth.',
        controls='Naive:anymidpointfree;firstcommon:gapflag in firstscanwithall3querystatesknown;ours:anystrictgapwitness acrossavailable scans. All sharegeometricallyqualifiedlinks. No votes/thresholdsweeps.',
        budget='Atmost8192sampledlinks perwindow,onefixedgeometrytest,108existingrawscans,noGPU/newfoundationmodels/training/maps/AP.',
        gate='Necessaryscreen only:inEACHwindow >=10known different-GTforeground-anchor links rejected,with >=90%precision among known foreground-anchor links rejected,and strictlyhigherprecision than naivemidpointfree. Failure closes this physical-gap predicate as an object-boundary candidate without scale/votecount rescue. Passing still requires complete-map mechanism,controls and independent/robot tests.',
        evaluation='After evidence sealed:nearest reference withinstrict20cm,knownnativeclass andpositiveconfidence;anchorendpointmustbelongtoeligiblebuilding/carGT. SameGT is sameobject,otherknownGT/class is different. Unknown/unsupportedreference excludedfromprecision andreported. Gap canoccurinsideoneobject,so aphysicalgap isnot automatically aninstanceboundary.',
        scope='Exposeddevelopment necessaryconditionprobe,not fullobjectformation,benchmarkAP,newindependenttest,robotvalidation or noveltyproof. Raycarving itself is established; do not relabel it as novel.'))
    print('GAP PROBE FROZEN',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    return p


def planes(tree,cloud,query):
    distance,index=tree.query(query,k=32,distance_upper_bound=.4,workers=1);valid=np.isfinite(distance)
    x=cloud[np.minimum(index,len(cloud)-1)];count=valid.sum(1);mean=(x*valid[:,:,None]).sum(1)/np.maximum(count[:,None],1)
    centered=(x-mean[:,None])*valid[:,:,None];cov=np.einsum('bni,bnj->bij',centered,centered)/np.maximum(count[:,None,None],1)
    values,vectors=np.linalg.eigh(cov);values=np.maximum(values,0)
    good=(count>=6)&(np.sqrt(values[:,0])<=.05)&(np.sqrt(values[:,1])>=.05)
    return vectors[:,:,0],good


def prepare():
    p=verify();assert not (OUT/'prepared.json').exists();files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];full=np.load(INPUT/key/'geometry.npz');xyz=full['xyz'];oldmap=np.load(INPUT/key/'old_to_full.npy')
        anchor=np.zeros(len(xyz),np.int64);anchor[oldmap]=source_anchors(key,len(oldmap));core=np.flatnonzero(anchor>0)
        _,representatives=np.unique(np.floor(xyz/.2).astype(np.int64),axis=0,return_index=True);representatives.sort()
        representatives=representatives[anchor[representatives]==0]
        distance,near=cKDTree(xyz[core]).query(xyz[representatives],workers=1)
        valid=(distance>.05)&(distance<=.4);query=representatives[valid];owners=core[near[valid]]
        front=full['old_source_id'][query]>=0;chosen=[]
        for group in [front,~front]:
            indices=np.flatnonzero(group)
            if len(indices):chosen.extend(indices[np.linspace(0,len(indices)-1,min(4096,len(indices)),dtype=int)].tolist())
        chosen=np.sort(chosen);a=owners[chosen];b=query[chosen];tree=cKDTree(xyz)
        normal,good=planes(tree,xyz,np.r_[xyz[a],xyz[b]]);n=len(a);delta=xyz[b]-xyz[a]
        tangent=(np.abs(np.einsum('ij,ij->i',normal[:n],normal[n:]))>=np.cos(np.deg2rad(15)))
        tangent&=(np.abs(np.einsum('ij,ij->i',delta,normal[:n]))<=.05)&(np.abs(np.einsum('ij,ij->i',delta,normal[n:]))<=.05)
        keep=good[:n]&good[n:]&tangent;a=a[keep];b=b[keep]
        folder=OUT/key;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'links.npz',anchor_source=a,query_source=b,anchor_owner=anchor[a],
            starts=xyz[a],ends=xyz[b],birth=np.maximum(full['source_frame'][a],full['source_frame'][b]),
            additional_source=full['old_source_id'][b]<0)
        row=dict(full_sources=len(xyz),unowned_representatives=len(representatives),nearby_candidates=len(query),sampled_before_geometry=len(chosen),
                 geometric_links=len(a),old_camera_links=int((full['old_source_id'][b]>=0).sum()),additional_links=int((full['old_source_id'][b]<0).sum()),no_gt_read=True)
        stats[key]=row;files[str(folder/'links.npz')]=sha(folder/'links.npz');print('GAP LINKS',key,row,flush=True)
        del tree,full,xyz,anchor
    dump(OUT/'prepared.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json')))


def observe():
    p=verify();prepared=read(OUT/'prepared.json');assert prepared['complete'] and not (OUT/'observation_execution.json').exists()
    for f,h in prepared['files'].items():assert sha(f)==h
    dump(OUT/'observation_execution.json',dict(started_epoch=time.time(),no_gt_read=True));files={};stats={};calib=load_left_calibration(CALIB)
    for scene in p['scenes']:
        key=scene['key'];d=np.load(OUT/key/'links.npz');n=len(d['birth']);job=read(scene['job']);counts=np.zeros(n,np.int32)
        naive=np.zeros(n,bool);first=np.full(n,-1,np.int64);firstgap=np.zeros(n,bool);gapframe=np.full(n,-1,np.int64);known=np.zeros(n,np.int32)
        folder=OUT/key/'frames';folder.mkdir(exist_ok=False)
        for frame in job['frames']:
            fid=frame['frame_id'];ix=np.flatnonzero(d['birth']<=fid);raw=np.fromfile(frame['raw_scan'],dtype='<f4').reshape(-1,4)[:,:3].astype(float)
            raw=raw[np.isfinite(raw).all(1)&(np.linalg.norm(raw,axis=1)>0)]
            pose=np.array(frame['pose']);t=pose@calib['rectified_from_velodyne'];world=raw@t[:3,:3].T+t[:3,3]
            rays=MeasuredRayTubes(t[:3,3],world);gap,state=rays.gaps(d['starts'][ix],d['ends'][ix])
            common=(state!=UNKNOWN).all(0);novel=(first[ix]<0)&common;first[ix[novel]]=fid;firstgap[ix[novel]]=gap[novel]
            newgap=(gapframe[ix]<0)&gap;gapframe[ix[newgap]]=fid
            counts[ix]+=gap;known[ix]+=common;naive[ix]|=state[1]==FREE
            file=folder/f'{fid:06d}.npz';np.savez_compressed(file,link_indices=ix,status=state);files[str(file)]=sha(file)
        file=OUT/key/'evidence.npz';np.savez_compressed(file,gap_count=counts,naive_free=naive,first_common_frame=first,first_common_gap=firstgap,first_gap_frame=gapframe,common_observations=known)
        files[str(file)]=sha(file);stats[key]=dict(links=n,strict_gap_links=int((counts>0).sum()),naive_free_links=int(naive.sum()),first_common_gap_links=int(firstgap.sum()),no_common_observation=int((known==0).sum()),scans=len(job['frames']))
        print('GAP EVIDENCE',key,stats[key],flush=True)
    dump(OUT/'evidence_complete.json',dict(complete=True,files=files,stats=stats,prepared_sha256=sha(OUT/'prepared.json'),no_gt_read=True))


def score():
    from evaluate_kitti360_development import read_static_ply,transfer
    p=verify();seal=read(OUT/'evidence_complete.json');assert seal['complete'] and not (OUT/'summary.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label'];results={};gate=True
    for scene in p['evaluation_scenes']:
        key=scene['key'];ref=np.load(scene['reference']);raw=read_static_ply(p['references'][key]['raw'])
        semantic=np.array([taxonomy[int(s)].trainId for s in raw['semantic']]);known=(semantic!=255)&(ref['confidence']>0);truth=ref['train_instance']
        mass=np.bincount(truth,weights=ref['confidence']);eligible=np.flatnonzero((mass>=100)&np.isin(np.arange(len(mass))//1000,[2,13]))
        d=np.load(OUT/key/'links.npz');n=len(d['birth']);q=np.r_[d['starts'],d['ends']]
        nn=transfer(ref['xyz'],np.arange(1,len(truth)+1),q,.2);ix=np.maximum(nn-1,0);ok=(nn>0)&known[ix];gt=truth[ix]
        use=ok[:n]&ok[n:]&np.isin(gt[:n],eligible);same=use&(gt[:n]==gt[n:]);different=use&~same
        e=np.load(OUT/key/'evidence.npz');methods=dict(naive_midpoint=e['naive_free'],first_common=e['first_common_gap'],ours_gap=e['gap_count']>0);rows={}
        for method,cut in methods.items():
            correct=int((cut&different).sum());wrong=int((cut&same).sum())
            rows[method]=dict(cut_total=int(cut.sum()),known_different_cut=correct,same_object_cut=wrong,unknown_or_nonforeground_cut=int((cut&~use).sum()),
                precision=correct/(correct+wrong) if correct+wrong else None,known_boundary_recall=correct/int(different.sum()) if different.any() else None)
        ours=rows['ours_gap'];naive=rows['naive_midpoint'];passed=ours['known_different_cut']>=10 and ours['precision'] is not None and ours['precision']>=.9 and naive['precision'] is not None and ours['precision']>naive['precision']
        gate&=passed;results[key]=dict(eligible_links=int(use.sum()),same_object_links=int(same.sum()),different_object_links=int(different.sum()),methods=rows,necessary_gate=bool(passed))
        np.savez_compressed(OUT/key/'diagnostic_labels.npz',eligible=use,same_object=same,different_object=different,anchor_gt=gt[:n],query_gt=gt[n:])
        print('GAP SCREEN',key,results[key],flush=True)
    dump(OUT/'summary.json',dict(complete=True,results=results,advance_gate=bool(gate),evidence_seal_sha256=sha(OUT/'evidence_complete.json'),
        scope='Necessary link-discrimination screen only, not full-map AP, complete object formation, independent evaluation or robot validation.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','observe','score']);globals()[ap.parse_args().action]()
