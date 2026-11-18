"""Check dataset coordinate conversion against released world positions.

Only the training adapter fixture is read. This is not a mapping score and does
not use semantic labels to construct or modify predictions.
"""
import csv
from pathlib import Path
import sys
import numpy as np
from paper_final_common import ROOT,read,dump,sha
BASE=ROOT/'build/datasets/hypersim_prepare'
sys.path.insert(0,str(BASE/'python_deps'))
import h5py


def array(path):
    with h5py.File(path,'r') as h:return h['dataset'][:]


def main():
    scene='ai_001_001';download=read(BASE/(scene+'_download.json'))
    for record in download['files']:assert sha(ROOT/record['path'])==record['sha256']
    metadata=next(r for r in csv.DictReader((BASE/'metadata_camera_parameters.csv').open()) if r['scene_name']==scene)
    w,h=int(float(metadata['settings_output_img_width'])),int(float(metadata['settings_output_img_height']))
    scale=float(metadata['settings_units_info_meters_scale'])
    m=np.array([[float(metadata[f'M_cam_from_uv_{i}{j}']) for j in range(3)] for i in range(3)])
    root=BASE/'data'/scene;detail=root/'_detail/cam_00'
    scene_scale=float(next(r for r in csv.DictReader((root/'_detail/metadata_scene.csv').open()) if r['parameter_name']=='meters_per_asset_unit')['parameter_value'])
    assert np.isclose(scale,scene_scale)
    frames=array(detail/'camera_keyframe_frame_indices.hdf5')
    positions=array(detail/'camera_keyframe_positions.hdf5').astype(np.float64)*scale
    rotations=array(detail/'camera_keyframe_orientations.hdf5').astype(np.float64)
    x,y=np.meshgrid(np.arange(w),np.arange(h))
    uv=np.stack([2*(x+.5)/w-1,1-2*(y+.5)/h,np.ones((h,w))],-1)
    rays=uv@m.T;rays/=np.linalg.norm(rays,axis=2,keepdims=True)
    assert (rays[:,:,2]<0).all()
    record=[]
    for frame in download['frames']:
        index=int(np.flatnonzero(frames==frame)[0]);r=rotations[index];t=positions[index]
        # Released matrices are rounded and not exactly orthonormal. Preserve
        # the published transform and use its actual inverse for round trips.
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-4)
        geom=root/'images/scene_cam_00_geometry_hdf5'
        raw_distance=array(geom/f'frame.{frame:04d}.depth_meters.hdf5')
        distance=raw_distance.astype(np.float64)
        assert distance.shape==(h,w)
        camera=rays*distance[:,:,None];world=camera@r.T+t
        valid=np.isfinite(world).all(2)&(distance>0)
        # Actual axial depth, rather than treating radial distances as z.
        axial=-camera[:,:,2]
        recovered=(world-t)@np.linalg.inv(r).T
        ndc=recovered@np.linalg.inv(m).T;ndc=ndc[:,:,:2]/ndc[:,:,2:]
        pixels=np.stack([(ndc[:,:,0]+1)*w/2-.5,(1-ndc[:,:,1])*h/2-.5],-1)
        residual=np.linalg.norm(pixels-np.stack([x,y],-1),axis=2)[valid]
        assert residual.max()<1e-5
        item=dict(frame_id=frame,valid_pixels=int(valid.sum()),pixel_roundtrip_max=float(residual.max()),
                  median_radial_minus_axial_m=float(np.median((distance-axial)[valid])))
        ground=geom/f'frame.{frame:04d}.position.hdf5'
        if ground.exists():
            raw_reference=array(ground)
            reference=raw_reference.astype(np.float64)*scale
            v=valid&np.isfinite(reference).all(2)
            error=np.linalg.norm(world-reference,axis=2)[v]
            item.update(world_position_median_error_m=float(np.median(error)),
                        world_position_p99_error_m=float(np.quantile(error,.99)),
                        world_position_max_error_m=float(error.max()),reference_pixels=int(v.sum()))
            # Both released images use float16. Bound their independent
            # round-to-nearest quantization errors rather than demand sub-ULP
            # agreement. Also bound the rounded rotation's length distortion.
            position_bound=np.linalg.norm(np.abs(np.spacing(raw_reference)).astype(float)*.5*scale,axis=2)
            distance_bound=np.abs(np.spacing(raw_distance)).astype(float)*.5
            rotation_bound=np.linalg.norm(r.T@r-np.eye(3),ord=2)*distance
            bound=(position_bound+distance_bound+rotation_bound+1e-5)[v]
            item.update(depth_storage_dtype=str(raw_distance.dtype),position_storage_dtype=str(raw_reference.dtype),
                        quantization_bound_pass_fraction=float((error<=bound).mean()),
                        maximum_error_over_bound=float((error/bound).max()))
            assert (error<=bound).all(),'Coordinate error exceeds storage-precision bound'
        record.append(item)
    dump(BASE/'geometry_verification.json',dict(complete=True,scene=scene,partition='train',frames=record,
        camera_metadata_sha256=sha(BASE/'metadata_camera_parameters.csv'),download_sha256=sha(BASE/(scene+'_download.json')),
        script_sha256=sha(__file__),test_predictions_run=False,
        scope='Training adapter fixture only; ray depth and world pose verified against independently stored position image. No semantic accuracy or new benchmark result.'))
    print(record)


if __name__=='__main__':main()
