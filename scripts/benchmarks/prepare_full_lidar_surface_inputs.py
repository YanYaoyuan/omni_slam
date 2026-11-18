"""Full raw-scan geometry with exact preservation of the old camera-source set.

This is input preparation and a necessary-coverage diagnosis, not a new method
result. No perception predictions or evaluation labels select source points.
"""
import argparse, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.kitti360_input import load_left_calibration, project_lidar
from omni_semantic_map.sparse_lidar_support import MeasuredVoxelMap

OUT = ROOT/'build/benchmarks/full_lidar_surface_inputs_v1'
OLD = ROOT/'build/benchmarks/discovery_measured_inputs_v1'
CALIB = ROOT/'build/datasets/kitti360_prepare/extracted/calibration'


def freeze():
    assert not OUT.exists(); old = read(OLD/'protocol.json'); scenes=[s for s in old['scenes'] if s['kind']=='outdoor']; inputs={}
    for scene in scenes:
        m=read(OLD/scene['key']/'manifest.json'); job=read(scene['job'])
        for f in [OLD/scene['key']/'manifest.json',m['geometry'],scene['job']]:inputs[str(f)]=sha(f)
        for frame in job['frames']:
            for f in [frame['raw_scan'],frame['measured']]:inputs[f]=sha(f)
    for f in CALIB.glob('*.txt'):inputs[str(f)]=sha(f)
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/kitti360_input.py',ROOT/'omni_semantic_map/omni_semantic_map/sparse_lidar_support.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=scenes,input_hashes=inputs,source_hashes={str(f):sha(f) for f in sources},
        hypothesis='Check whether withholding measured non-camera-facing geometry materially limits object support, before designing surface-continuation inference.',
        source='Exactly the same60/48paired scans and calibrated poses. Keep every finite nonzero-range raw return before camera/FOV/zbuffer selection. First actual return per5cm world voxel; additionally retain every exact old camera-source return, deduplicate by(frame_id,raw_index), then sort by this chronological key. At most two representatives per5cm voxel; do not average coordinates or synthesize surfaces.',
        coordinates='Same calibrated rectified-camera transform and published pose as old adapter. Independently replay all old measured frame points against raw indices before export. No new deskew/interpolation claim.',
        image='Pixel values present only for original front-camera zbuffer-selected returns; -1 means no image evidence at acquisition. Zero RGB is a placeholder, not a colored map. Later visibility/semantic inference is not run here.',
        diagnosis='Only after full geometry is sealed, compare confidence-weighted reference surface coverage within strict20cm for ALL94eligible building/carGT. Raw geometry contains no labels: coverage assumes perfect labeling and is only a necessary condition, not IoU/AP, recall or achievable performance. Do not remove uncoveredGT from official/local scoring.',
        scope='Existing exposed development windows. No new algorithm, map predictions, training, robot candidate comparison or independent test is completed by this cache. Later controls must share full geometry to isolate algorithmic effects.'))
    print('FULL LIDAR INPUT FROZEN',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    return p


def prepare():
    p=verify();assert not (OUT/'execution.json').exists();dump(OUT/'execution.json',dict(started_epoch=time.time(),no_gt_read=True))
    files={};stats={};calib=load_left_calibration(CALIB)
    for scene in p['scenes']:
        key=scene['key'];job=read(scene['job']);m=read(OLD/key/'manifest.json');previous=dict(np.load(m['geometry']))
        ledger=MeasuredVoxelMap(.05);frame_rows=[];start=time.monotonic();last=-1
        for frame in job['frames']:
            fid=frame['frame_id'];assert fid>last;last=fid
            raw=np.fromfile(frame['raw_scan'],dtype='<f4').reshape(-1,4)
            t=calib['rectified_from_velodyne'];camera=raw[:,:3].astype(float)@t[:3,:3].T+t[:3,3]
            pose=np.array(frame['pose']);world=camera@pose[:3,:3].T+pose[:3,3]
            measured=np.load(frame['measured']);np.testing.assert_array_equal(world[measured['raw_indices']],measured['xyz'])
            valid=np.isfinite(raw[:,:3]).all(1)&(np.linalg.norm(raw[:,:3],axis=1)>0)
            ids=np.flatnonzero(valid);pixels=np.full((len(raw),2),-1,np.int32);pixels[measured['raw_indices']]=measured['pixel_xy']
            chosen=ledger.append(world[ids],np.zeros((len(ids),3),np.uint8),fid,ids,pixels[ids])
            frame_rows.append(dict(frame_id=fid,raw_returns=len(raw),finite_nonzero_returns=len(ids),new_voxel_returns=len(chosen),old_measured_points=len(measured['xyz']),old_frame_geometry_exact=True))
            if len(frame_rows)%20==0:print('FULL LIDAR RAW',key,len(frame_rows),'frames',flush=True)
        full=ledger.arrays();del ledger
        # Preserve exact original front-source returns, not merely their voxels.
        frames=np.r_[full['source_frame'],previous['source_frame']]
        rawids=np.r_[full['source_raw_index'],previous['source_raw_index']]
        assert np.all((frames>=0)&(frames<2**32)) and np.all((rawids>=0)&(rawids<2**32))
        token=(frames.astype(np.uint64)<<np.uint64(32))|rawids.astype(np.uint64)
        unique,first=np.unique(token,return_index=True)
        result={name:np.concatenate([full[name],previous[name]])[first] for name in full}
        oldtoken=(previous['source_frame'].astype(np.uint64)<<np.uint64(32))|previous['source_raw_index'].astype(np.uint64)
        mapping=np.searchsorted(unique,oldtoken);np.testing.assert_array_equal(unique[mapping],oldtoken)
        np.testing.assert_array_equal(result['xyz'][mapping],previous['xyz'])
        assert len(np.unique(mapping))==len(mapping)
        result['old_source_id']=np.full(len(unique),-1,np.int64);result['old_source_id'][mapping]=np.arange(len(mapping))
        result['image_evidence_at_acquisition']=(result['source_pixel_xy']>=0).all(1)
        folder=OUT/key;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'geometry.npz',**result)
        np.save(folder/'old_to_full.npy',mapping)
        records=[]
        for frame in job['frames']:
            records.append(dict(frame_id=frame['frame_id'],raw_scan=frame['raw_scan'],camera_pose=frame['pose'],map_points=int(np.searchsorted(result['source_frame'],frame['frame_id'],side='right'))))
        stat=dict(frames=len(records),raw_returns=sum(r['raw_returns'] for r in frame_rows),old_sources=len(mapping),full_sources=len(unique),
                  added_measured_sources=len(unique)-len(mapping),image_evidence_at_acquisition=int(result['image_evidence_at_acquisition'].sum()),
                  old_sources_preserved_exactly=True,all_old_frame_geometry_replayed=True,seconds=time.monotonic()-start)
        dump(folder/'manifest.json',dict(complete=True,key=key,geometry=str(folder/'geometry.npz'),geometry_sha256=sha(folder/'geometry.npz'),records=records,frame_audit=frame_rows,stats=stat))
        for f in folder.iterdir():files[str(f)]=sha(f)
        stats[key]=stat;print('FULL LIDAR PREPARED',key,stat,flush=True)
        del result,full,previous
    dump(OUT/'prepared.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json'),no_gt_read=True))


def coverage():
    p=verify();seal=read(OUT/'prepared.json');assert seal['complete'] and not (OUT/'coverage.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h
    scenes=read(ROOT/'build/benchmarks/provenance_object_memory_v1/protocol.json')['scenes'];results={}
    for scene in scenes:
        key=scene['key'];ref=np.load(scene['reference']);truth=ref['train_instance'];weight=ref['confidence'].astype(float)
        previous=np.load(read(OLD/key/'manifest.json')['geometry'])['xyz'];full=np.load(OUT/key/'geometry.npz')['xyz']
        olddistance=cKDTree(previous).query(ref['xyz'],workers=1)[0];fulldistance=cKDTree(full).query(ref['xyz'],workers=1)[0]
        assert np.all(fulldistance<=olddistance+1e-10)
        mass=np.bincount(truth,weights=weight);gids=np.flatnonzero((mass>=100)&np.isin(np.arange(len(mass))//1000,[2,13]))
        oldmass=np.bincount(truth,weights=weight*(olddistance<.2),minlength=len(mass));newmass=np.bincount(truth,weights=weight*(fulldistance<.2),minlength=len(mass))
        rows=[dict(gt=int(g),category='building' if g//1000==2 else 'car',reference_mass=float(mass[g]),old_coverage=float(oldmass[g]/mass[g]),full_coverage=float(newmass[g]/mass[g])) for g in gids]
        category={}
        for cls in ['building','car']:
            selected=[r for r in rows if r['category']==cls]
            category[cls]=dict(gt=len(selected),old_over_half=sum(r['old_coverage']>.5 for r in selected),full_over_half=sum(r['full_coverage']>.5 for r in selected),new_over_half=[r['gt'] for r in selected if r['old_coverage']<=.5<r['full_coverage']])
        np.savez_compressed(OUT/key/'coverage_diagnostic.npz',old_distance=olddistance,full_distance=fulldistance)
        results[key]=dict(objects=rows,category=category,reference_sha256=sha(scene['reference']))
        print('FULL LIDAR COVERAGE',key,category,flush=True)
    dump(OUT/'coverage.json',dict(complete=True,results=results,prepared_sha256=sha(OUT/'prepared.json'),
        scope='GT-assisted necessary geometry coverage under perfect labeling; not a predicted method, instanceIoU/AP, independent test or official leaderboard. All eligibleGT retained.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','coverage']);globals()[ap.parse_args().action]()
