"""Replay raw provenance and independently verify every first gap witness."""
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from prepare_full_lidar_surface_inputs import OUT as INPUT, OLD, CALIB, verify
from omni_semantic_map.kitti360_input import load_left_calibration


def direct_state(origin, returns, point):
    delta=returns-origin;distance=np.linalg.norm(delta,axis=1);delta=delta[distance>0];distance=distance[distance>0]
    direction=delta/distance[:,None];q=point-origin;r=np.linalg.norm(q)
    if r<=.1:return 0
    along=direction@q;perp=np.maximum(0,r*r-along*along);aligned=(along>0)&(perp<=.05**2+1e-10)
    residual=along-distance
    if np.any(aligned&(np.abs(residual)<=.05)):return 1
    if np.any(aligned&(residual>.05)):return 2
    if np.any(aligned&(residual<-.05)):return 3
    return 0


def run():
    p=verify();seal=read(INPUT/'prepared.json');assert seal['complete'];calib=load_left_calibration(CALIB);results={}
    for f,h in seal['files'].items():assert sha(f)==h
    probe=ROOT/'build/benchmarks/surface_gap_witness_v1';evidence=read(probe/'evidence_complete.json')
    for f,h in evidence['files'].items():assert sha(f)==h
    for scene in p['scenes']:
        key=scene['key'];full=np.load(INPUT/key/'geometry.npz');xyz=full['xyz'];frame_ids=full['source_frame'];raw_ids=full['source_raw_index'];mapping=np.load(INPUT/key/'old_to_full.npy')
        old=np.load(read(OLD/key/'manifest.json')['geometry']);np.testing.assert_array_equal(xyz[mapping],old['xyz'])
        np.testing.assert_array_equal(frame_ids[mapping],old['source_frame']);np.testing.assert_array_equal(raw_ids[mapping],old['source_raw_index'])
        token=(frame_ids.astype(np.uint64)<<np.uint64(32))|raw_ids.astype(np.uint64);assert np.all(token[1:]>token[:-1])
        links=np.load(probe/key/'links.npz');e=np.load(probe/key/'evidence.npz');witnesses=np.flatnonzero(e['first_gap_frame']>=0);seen=0;checked=0
        for frame in read(scene['job'])['frames']:
            fid=frame['frame_id'];raw=np.fromfile(frame['raw_scan'],dtype='<f4').reshape(-1,4)[:,:3].astype(float)
            pose=np.array(frame['pose']);t=pose@calib['rectified_from_velodyne'];world=raw@t[:3,:3].T+t[:3,3]
            lo,hi=np.searchsorted(frame_ids,[fid,fid],side='left')[0],np.searchsorted(frame_ids,fid,side='right')
            ids=raw_ids[lo:hi];assert np.all(ids<len(raw));np.testing.assert_allclose(xyz[lo:hi],world[ids],rtol=0,atol=2e-12);seen+=hi-lo
            valid=np.isfinite(raw).all(1)&(np.linalg.norm(raw,axis=1)>0);returns=world[valid]
            for i in witnesses[e['first_gap_frame'][witnesses]==fid]:
                a,b=links['starts'][i],links['ends'][i]
                states=[direct_state(t[:3,3],returns,x) for x in [a,(a+b)/2,b]]
                assert states==[1,3,1],(key,int(i),fid,states);checked+=1
        assert seen==len(xyz) and checked==len(witnesses)
        results[key]=dict(all_sources_replayed=int(seen),old_sources_exact=len(mapping),unique_chronological_returns=True,
            first_gap_witnesses_brute_verified=checked,source_world_tolerance_m=2e-12)
        print('RAW SOURCE AUDIT',key,results[key],flush=True)
    dump(INPUT/'raw_replay_audit.json',dict(complete=True,results=results,source_sha256=sha(__file__),prepared_sha256=sha(INPUT/'prepared.json'),gap_evidence_sha256=sha(probe/'evidence_complete.json')))


if __name__=='__main__':run()
