"""Prepare reserved measured inputs, without semantic inference or GT decoding."""
import argparse,copy,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from freeze_batch_transport_assessment import OUT,DATA
from omni_semantic_map.kitti360_input import load_left_calibration,load_poses,load_times,project_lidar
from omni_semantic_map.sparse_lidar_support import MeasuredVoxelMap,project_measured_visibility
from omni_semantic_map.measured_query_cells import query_cell_ids

META=ROOT/'build/datasets/kitti360_prepare'
INPUT=OUT/'measured'
FULL=OUT/'full_measured'


class ArrayVoxelLedger(MeasuredVoxelMap):
    """Same first-return rule, compact sorted keys instead of Python tuples."""
    def __init__(self,voxel_size_m=.05):
        super().__init__(voxel_size_m);self._keys=np.empty(0,dtype='V24')

    def append(self,xyz,colors,frame_id,raw_indices,pixel_xy):
        xyz=np.asarray(xyz,dtype=np.float64);colors,raw_indices,pixel_xy=map(np.asarray,[colors,raw_indices,pixel_xy]);n=len(xyz)
        if xyz.shape!=(n,3) or colors.shape!=(n,3) or raw_indices.shape!=(n,) or pixel_xy.shape!=(n,2) or not np.isfinite(xyz).all():raise ValueError('Invalid measured source arrays')
        keys=np.ascontiguousarray(np.floor(xyz/self.voxel_size_m).astype('<i8')).view('V24').reshape(-1)
        unique,first=np.unique(keys,return_index=True)
        if len(self._keys):
            pos=np.searchsorted(self._keys,unique);present=pos<len(self._keys);present[present]&=self._keys[pos[present]]==unique[present]
        else:present=np.zeros(len(unique),bool)
        selected=np.sort(first[~present]);self._keys=np.sort(np.r_[self._keys,unique[~present]])
        self._xyz.append(xyz[selected].copy());self._colors.append(colors[selected].copy());self._frames.append(np.full(len(selected),frame_id,np.int64))
        self._raw.append(raw_indices[selected].copy());self._pixels.append(pixel_xy[selected].copy())
        return selected


def verify():
    p=read(OUT/'protocol.json')
    for path,digest in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(path)==digest,path
    return p


def project(scan,pose,calib):
    raw=np.fromfile(scan,dtype='<f4').reshape(-1,4);projection=project_lidar(raw[:,:3],calib);ids=projection['selected_indices']
    yy,xx=np.nonzero(projection['source_point_index']>=0);np.testing.assert_array_equal(projection['source_point_index'][yy,xx],ids)
    xyz=projection['camera_xyz'][ids]@pose[:3,:3].T+pose[:3,3]
    return raw,projection,ids,np.column_stack([xx,yy]).astype(np.int32),xyz


def golden():
    verify();assert not (OUT/'input_golden.json').exists();calib=load_left_calibration(META/'extracted/calibration');results={}
    # Compare first-return behavior, including duplicate/negative/boundary keys.
    rng=np.random.RandomState(5);native=MeasuredVoxelMap(.05);compact=ArrayVoxelLedger(.05)
    for fid in range(12):
        xyz=rng.randint(-8,9,(3000,3))*.05+rng.choice([0.,1e-12,-1e-12],(3000,3));colors=rng.randint(0,256,(3000,3)).astype(np.uint8);ids=np.arange(len(xyz));xy=np.tile([3,4],(len(xyz),1))
        np.testing.assert_array_equal(native.append(xyz,colors,fid,ids,xy),compact.append(xyz,colors,fid,ids,xy))
    for name,value in native.arrays().items():np.testing.assert_array_equal(value,compact.arrays()[name])
    for scene in read(ROOT/'build/benchmarks/closed_vocabulary_semantics_v1/protocol.json')['scenes']:
        job=read(scene['job']);poses=load_poses(META/'extracted'/job['scene']/'poses.txt');count=0;ledger=ArrayVoxelLedger(.05)
        for row in job['frames']:
            pose=poses[row['frame_id']]@calib['pose_from_cam']@np.linalg.inv(calib['rectification']);pose[:3,3]-=np.asarray(job['origin_world'])
            np.testing.assert_array_equal(pose,np.asarray(row['pose']));raw,proj,ids,xy,xyz=project(row['raw_scan'],pose,calib);old=np.load(row['measured'])
            for actual,expected in [(xyz,old['xyz']),(ids,old['raw_indices']),(xy,old['pixel_xy']),(proj['depth'],old['depth'])]:np.testing.assert_array_equal(actual,expected)
            count+=len(ids)
            t=calib['rectified_from_velodyne'];camera=raw[:,:3].astype(float)@t[:3,:3].T+t[:3,3];world=camera@pose[:3,:3].T+pose[:3,3]
            valid=np.isfinite(raw[:,:3]).all(1)&(np.linalg.norm(raw[:,:3],axis=1)>0);ri=np.flatnonzero(valid);px=np.full((len(raw),2),-1,np.int32);px[ids]=xy
            ledger.append(world[ri],np.zeros((len(ri),3),np.uint8),row['frame_id'],ri,px[ri])
        actual=ledger.arrays();del ledger
        old=np.load(read(ROOT/'build/benchmarks/discovery_measured_inputs_v1'/scene['key']/'manifest.json')['geometry'])
        frames=np.r_[actual['source_frame'],old['source_frame']];rawids=np.r_[actual['source_raw_index'],old['source_raw_index']]
        token=(frames.astype(np.uint64)<<np.uint64(32))|rawids.astype(np.uint64);unique,first=np.unique(token,return_index=True)
        expected=np.load(ROOT/'build/benchmarks/full_lidar_surface_inputs_v1'/scene['key']/'geometry.npz')
        for name,value in actual.items():np.testing.assert_array_equal(np.r_[value,old[name]][first],expected[name])
        results[scene['key']]=dict(frames=len(job['frames']),projected_points=count,all_xyz_depth_pixels_and_poses_exact=True,compact_full_sources_exact=len(unique))
        print('COMPACT LEDGER GOLDEN',scene['key'],len(unique),flush=True)
        del actual,old,expected
    dump(OUT/'input_golden.json',dict(complete=True,results=results,source_sha256=sha(__file__),compact_ledger_random_boundary_frames=12,scope='All108oldmeasuredinputframes andall7468489fullsourcefields,notbackbonepredictiongolden. No newRGBinference orreferenceaccess.'))
    print('ASSESSMENT INPUT GOLDEN',results,flush=True)


def prepare():
    p=verify();gold=read(OUT/'input_golden.json');assert gold['complete'] and gold['source_sha256']==sha(__file__)
    assert not (OUT/'input_execution.json').exists()
    reference=ROOT/'build/benchmarks/hypersim_shared_v1/jobs/ai_001_001.json'
    dump(OUT/'input_execution.json',dict(started_epoch=time.time(),source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),config_template=str(reference),config_template_sha256=sha(reference),gt_read=False,semantic_inference=False))
    calib=load_left_calibration(META/'extracted/calibration');files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];w=scene['window'];seq=w['scene'];poses=load_poses(META/'extracted'/seq/'poses.txt');rect=np.linalg.inv(calib['rectification'])
        camera_poses={f:poses[f]@calib['pose_from_cam']@rect for f in w['input_frames']};origin=camera_poses[w['input_frames'][0]][:3,3].copy()
        image_times=load_times(META/'extracted'/seq/'image_00/timestamps.txt');scan_times=load_times(META/'extracted'/seq/'velodyne_points/timestamps.txt')
        cfg=copy.deepcopy(read(reference)['config']);cfg['dataset_name']='kitti360_measured_lidar';cfg['semantic']['depth_filter']=False
        cfg['cam']=dict(H=calib['height'],W=calib['width'],fx=float(calib['K'][0,0]),fy=float(calib['K'][1,1]),cx=float(calib['K'][0,2]),cy=float(calib['K'][1,2]),depth_scale=1.)
        folder=INPUT/key;(folder/'frames').mkdir(parents=True,exist_ok=False);measure=OUT/'prepared'/key/'inputs'/seq;measure.mkdir(parents=True,exist_ok=False)
        old_ledger=ArrayVoxelLedger(.05);full_ledger=ArrayVoxelLedger(.05);records=[];job_frames=[];hashes={};raw_count=0
        for index,fid in enumerate(w['input_frames']):
            scan=DATA/'data/data_3d_raw'/seq/'velodyne_points/data'/f'{fid:010d}.bin';rgbpath=DATA/'data/data_2d_raw'/seq/'image_00/data_rect'/f'{fid:010d}.png'
            pose=camera_poses[fid].copy();pose[:3,3]-=origin;raw,proj,ids,xy,xyz=project(scan,pose,calib)
            measured=measure/f'{fid:06d}_measured.npz';np.savez_compressed(measured,depth=proj['depth'],xyz=xyz,raw_indices=ids,pixel_xy=xy,origin_world=origin)
            rgb=np.asarray(Image.open(rgbpath).convert('RGB'));assert rgb.shape[:2]==(calib['height'],calib['width'])
            old_ledger.append(xyz,rgb[xy[:,1],xy[:,0]],fid,ids,xy);old=old_ledger.arrays();n=len(old['xyz'])
            x,y,_,visible=project_measured_visibility(old['xyz'],pose,cfg['cam'],proj['depth'],SimpleNamespace(depth_absolute_m=.05,depth_relative=0.));vis=np.flatnonzero(visible)
            dest=folder/'frames'/f'{fid:06d}.npz';np.savez_compressed(dest,visible_source_ids=vis,pixel_xy=np.column_stack([x[vis],y[vis]]));files[str(dest)]=sha(dest)
            records.append(dict(frame_id=fid,rgb=str(rgbpath),file=str(dest),sha256=sha(dest),map_points=n,visible_points=len(vis),camera_pose=pose.tolist()))
            # Match full-LiDAR preparation exactly: transform all raw returns,
            # with pixels only on front-camera z-buffer winners at acquisition.
            transform=calib['rectified_from_velodyne'];camera=raw[:,:3].astype(float)@transform[:3,:3].T+transform[:3,3];world=camera@pose[:3,:3].T+pose[:3,3]
            np.testing.assert_array_equal(world[ids],xyz);valid=np.isfinite(raw[:,:3]).all(1)&(np.linalg.norm(raw[:,:3],axis=1)>0);rawids=np.flatnonzero(valid)
            px=np.full((len(raw),2),-1,np.int32);px[ids]=xy;full_ledger.append(world[rawids],np.zeros((len(rawids),3),np.uint8),fid,rawids,px[rawids]);raw_count+=len(raw)
            job_frames.append(dict(frame_id=fid,rgb=str(rgbpath),raw_scan=str(scan),measured=str(measured),pose=pose.tolist(),semantic=True,image_timestamp_ns=int(image_times[fid]),scan_timestamp_ns=int(scan_times[fid]),scan_minus_image_ms=float((scan_times[fid]-image_times[fid])/1e6)))
            for path in [rgbpath,scan,measured]:hashes[str(path)]=sha(path)
            if (index+1)%20==0:print('ASSESSMENT PREPARE',key,index+1,len(w['input_frames']),flush=True)
        old=old_ledger.arrays();geometry=folder/'geometry.npz';np.savez_compressed(geometry,**old);files[str(geometry)]=sha(geometry)
        job=dict(kind='kitti360_reserved_validation',scene=seq,frames=job_frames,config=cfg,output=str(OUT/'prepared'/key/'frontend'/seq),origin_world=origin.tolist(),no_gt_in_inference=True,input_sha256=hashes,
            geometry='Exact measured sources; published paired framepose without newdeskew/interpolation.',scope='batch_transport_assessment_v1 registered independent quality assessment; opaque references until all predictions sealed.')
        dump(scene['job'],job);files[scene['job']]=sha(scene['job']);hashes[scene['job']]=sha(scene['job'])
        dump(folder/'manifest.json',dict(complete=True,key=key,kind='outdoor',records=records,geometry=str(geometry),geometry_sha256=sha(geometry),input_hashes=hashes,source_points=len(old['xyz']),frames=len(records),no_masks_or_gt=True));files[str(folder/'manifest.json')]=sha(folder/'manifest.json')
        cells=query_cell_ids(old['xyz'],.2);dest=INPUT/'query_cells'/key;dest.mkdir(parents=True,exist_ok=False);np.save(dest/'source_to_cell.npy',cells);files[str(dest/'source_to_cell.npy')]=sha(dest/'source_to_cell.npy')
        for row in records[::max(1,len(records)//4)]:np.testing.assert_array_equal(query_cell_ids(old['xyz'][:row['map_points']],.2),cells[:row['map_points']])
        allsource=full_ledger.arrays();frames=np.r_[allsource['source_frame'],old['source_frame']];rawids=np.r_[allsource['source_raw_index'],old['source_raw_index']]
        token=(frames.astype(np.uint64)<<np.uint64(32))|rawids.astype(np.uint64);unique,first=np.unique(token,return_index=True)
        full={name:np.concatenate([allsource[name],old[name]])[first] for name in allsource}
        oldtoken=(old['source_frame'].astype(np.uint64)<<np.uint64(32))|old['source_raw_index'].astype(np.uint64);mapping=np.searchsorted(unique,oldtoken)
        np.testing.assert_array_equal(unique[mapping],oldtoken);np.testing.assert_array_equal(full['xyz'][mapping],old['xyz'])
        full['old_source_id']=np.full(len(unique),-1,np.int64);full['old_source_id'][mapping]=np.arange(len(mapping));full['image_evidence_at_acquisition']=(full['source_pixel_xy']>=0).all(1)
        dest=FULL/key;dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'geometry.npz',**full);np.save(dest/'old_to_full.npy',mapping)
        stat=dict(frames=len(records),raw_returns=raw_count,old_camera_sources=len(mapping),full_sources=len(unique),added_full_sources=len(unique)-len(mapping),old_sources_exact=True)
        dump(dest/'manifest.json',dict(complete=True,key=key,geometry=str(dest/'geometry.npz'),geometry_sha256=sha(dest/'geometry.npz'),stats=stat))
        for path in dest.iterdir():files[str(path)]=sha(path)
        stats[key]=stat;print('ASSESSMENT INPUT READY',key,stat,flush=True)
        del old_ledger,full_ledger,old,full,allsource
    dump(OUT/'prepared_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json'),source_sha256=sha(__file__),semantic_inference=False,reference_labels_decoded=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['golden','prepare']);globals()[a.parse_args().action]()
