"""Convert released HDR/radial depth to the frozen shared frontend interface.

No semantic/instance annotation is decoded. Test jobs can be prepared before
inference; creating this file does not run a test prediction or evaluate it.
"""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import cv2
from paper_final_common import ROOT,read,dump,sha
from prepare_strong_jobs import sources
from audit_evidence_retention import CACHE

META=ROOT/'build/datasets/hypersim_prepare'
sys.path.insert(0,str(META/'python_deps'))
import h5py


def array(path):
    with h5py.File(path,'r') as h:return h['dataset'][:]


def tonemap(rgb,entity):
    rgb=np.asarray(rgb,np.float32)
    assert rgb.shape[:2]==entity.shape and rgb.shape[2]==3
    assert np.isfinite(rgb).all() and np.all(entity!=0)
    valid=entity!=-1
    brightness=.3*rgb[:,:,0]+.59*rgb[:,:,1]+.11*rgb[:,:,2]
    percentile=float(np.percentile(brightness[valid],90)) if valid.any() else None
    scale=1. if percentile is None else 0. if percentile<.0001 else .8**2.2/percentile
    result=np.clip(np.maximum(scale*rgb,0)**(1/2.2),0,1)
    return np.rint(result*255).astype(np.uint8),scale


def main():
    parser=argparse.ArgumentParser();parser.add_argument('scene');parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'build/benchmarks/hypersim_shared_v1')
    args=parser.parse_args();scene=args.scene
    download_path=args.input/(scene+'_download.json');download=read(download_path)
    assert download['extracted'] and download['hdr']
    for f in download['files']:assert sha(ROOT/f['path'])==f['sha256']
    is_train=scene=='ai_001_001'
    if not is_train:
        reserved=read(META/'reserved_test_scenes.json')
        assert download['frames']==reserved['scenes'][scene] and download['reserved']
    target=args.output/'inputs'/scene;target.mkdir(parents=True,exist_ok=False)
    (target/'rgb').mkdir();(target/'depth').mkdir()
    calibration=next(r for r in csv.DictReader((META/'metadata_camera_parameters.csv').open()) if r['scene_name']==scene)
    w,h=int(float(calibration['settings_output_img_width'])),int(float(calibration['settings_output_img_height']))
    m=np.array([[float(calibration[f'M_cam_from_uv_{i}{j}']) for j in range(3)] for i in range(3)])
    scale=float(calibration['settings_units_info_meters_scale'])
    convention=np.diag([1.,-1.,-1.]);pixels_to_uv=np.array([[2/w,0,1/w-1],[0,-2/h,1-1/h],[0,0,1.]])
    k=np.linalg.inv(convention@m@pixels_to_uv);k/=k[2,2]
    expected=np.array([[k[0,0],0,k[0,2]],[0,k[1,1],k[1,2]],[0,0,1.]])
    np.testing.assert_allclose(k,expected,atol=1e-8,err_msg='Native frontend cannot represent this camera; do not silently discard skew')
    root=args.input/'data'/scene;detail=root/'_detail/cam_00'
    pose_paths=[detail/f'camera_keyframe_{name}.hdf5' for name in ['frame_indices','positions','orientations']]
    frame_ids,positions,rotations=[array(p) for p in pose_paths]
    positions=positions.astype(np.float64)*scale;rotations=rotations.astype(np.float64)
    x,y=np.meshgrid(np.arange(w),np.arange(h));uv=np.stack([2*(x+.5)/w-1,1-2*(y+.5)/h,np.ones((h,w))],-1)
    rays=uv@m.T;rays/=np.linalg.norm(rays,axis=2,keepdims=True)
    reference=CACHE/'office_0';cfg=read(reference/'manifest.json')['config'];cfg['dataset_name']='hypersim'
    cfg['cam']=dict(H=h,W=w,fx=float(k[0,0]),fy=float(k[1,1]),cx=float(k[0,2]),cy=float(k[1,2]),depth_scale=1000.)
    inputs={str(p.resolve()):sha(p) for p in [download_path,META/'metadata_camera_parameters.csv',META/'official_tonemap.py',*pose_paths]}
    records=[];verification=[]
    for frame in download['frames']:
        prefix=f'frame.{frame:04d}';geom=root/'images/scene_cam_00_geometry_hdf5'
        rgb_path=root/'images/scene_cam_00_final_hdf5'/(prefix+'.color.hdf5')
        radial_path=geom/(prefix+'.depth_meters.hdf5');entity_path=geom/(prefix+'.render_entity_id.hdf5')
        for p in [rgb_path,radial_path,entity_path]:inputs[str(p.resolve())]=sha(p)
        rgb,exposure=tonemap(array(rgb_path),array(entity_path))
        distance=array(radial_path).astype(np.float64);axial=-rays[:,:,2]*distance
        valid=np.isfinite(axial)&(axial>0)&(axial<=65.535)
        depth=np.zeros((h,w),np.uint16);depth[valid]=np.rint(axial[valid]*1000).astype(np.uint16)
        rgb_out=target/'rgb'/f'{frame:06d}.png';dep_out=target/'depth'/f'{frame:06d}.png'
        assert cv2.imwrite(str(rgb_out),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)) and cv2.imwrite(str(dep_out),depth)
        np.testing.assert_array_equal(cv2.imread(str(dep_out),cv2.IMREAD_UNCHANGED),depth)
        i=int(np.flatnonzero(frame_ids==frame)[0]);pose=np.eye(4);pose[:3,:3]=rotations[i]@convention;pose[:3,3]=positions[i]
        # Verify native integer-pixel pinhole unprojection against official rays.
        axial_camera=np.stack([(x-k[0,2])*axial/k[0,0],(y-k[1,2])*axial/k[1,1],axial],-1)
        ray_camera=(rays*distance[:,:,None])@convention
        error=np.linalg.norm(axial_camera-ray_camera,axis=2)[valid]
        assert not len(error) or error.max()<1e-7
        for p in [rgb_out,dep_out]:inputs[str(p.resolve())]=sha(p)
        records.append(dict(frame_id=frame,rgb=str(rgb_out.resolve()),depth=str(dep_out.resolve()),pose=pose.tolist(),semantic=True))
        verification.append(dict(frame_id=frame,exposure_scale=exposure,valid_depth_pixels=int(valid.sum()),
            out_of_uint16_range=int((np.isfinite(axial)&(axial>65.535)).sum()),
            pinhole_ray_max_error_m=float(error.max()) if len(error) else None,
            millimeter_quantization_max_error_m=float(np.max(np.abs(depth[valid]/1000-axial[valid]))) if valid.any() else None))
    src=sources();src[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    job=args.output/'jobs'/(scene+'.json');assert not job.exists()
    for name in ['text_features.npy','vocabulary.json']:inputs[str(reference/name)]=sha(reference/name)
    dump(target/'conversion.json',dict(complete=True,scene=scene,partition='train' if is_train else 'reserved_test',
        frames=verification,source_metadata_sha256=sha(META/'metadata_camera_parameters.csv'),
        no_semantic_or_instance_gt_decoded=True,convention='Official radial rays -> axial camera depth; OpenGL right/up/back to CV right/down/forward; translations scaled to meters.',
        rgb='Official 90th-percentile luminance normalization to 0.8 with gamma 1/2.2; lossless 8-bit PNG. Render entity IDs are used solely as a valid-region mask for official exposure normalization.',
        reference_geometry_check=str(META/'geometry_verification.json')))
    inputs[str((target/'conversion.json').resolve())]=sha(target/'conversion.json')
    dump(job,dict(kind='hypersim_adapter_smoke' if is_train else 'hypersim_reserved_test',scene=scene,
        output=str((args.output/'frontend'/scene).resolve()),config=cfg,reference_cache=str(reference),
        frames=records,input_sha256=inputs,source_sha256=src,no_gt_in_inference=True,
        semantic_frontend='Unchanged frozen SAM2 + SigLIP + learned merger, staged batch8 BF16.',
        execution_gate='Training adapter may run now. Reserved-test inference must wait for final method/evaluation protocol freeze.',
        peak_cuda_allocator_limit_gib=6))
    print(job,flush=True)


if __name__=='__main__':main()
