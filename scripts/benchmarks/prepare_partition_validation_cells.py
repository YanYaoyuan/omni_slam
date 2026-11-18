"""Observe physical query cells from NEW actual LiDAR returns.

The previous strict old-point reprojection adapter is retained unchanged.
This new representation is explicit: current measured 20cm cell support,
not a claim of passing the old 5cm point-depth gate. All final arms must share
it. New returns are validation evidence only, never new output-map geometry.
"""
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from prepare_partition_revision import OUT,META,INPUT,PARENT
from run_signed_query_association import pieces
from omni_semantic_map.kitti360_input import load_left_calibration,project_lidar


def main():
    assert not (OUT/'validation_cells_complete.json').exists()
    inputs=read(OUT/'verification_inputs.json');scope=read(OUT/'common_view_execution_scope.json')
    grid=read(INPUT/'query_cells/protocol.json');assert grid['cell_m']==.2
    for f,h in inputs['files'].items():assert sha(f)==h,f
    calib=load_left_calibration(META/'calibration');files={};stats={};results={};comparisons=[]
    dump(OUT/'validation_cells_protocol.json',dict(input_seal_sha256=sha(OUT/'verification_inputs.json'),
        candidate_seal_sha256=sha(OUT/'candidates_complete.json'),common_scope_sha256=sha(OUT/'common_view_execution_scope.json'),
        grid_protocol_sha256=sha(INPUT/'query_cells/protocol.json'),source_sha256=sha(__file__),
        cell_m=.2,selection_changes=False,output_map_geometry_changes=False,
        reason='Old map has no returns originating at these new frames. Exact sparse-pixel reprojection loses physical surface observations. Read current real returns in the existing physical query grid instead of inventing depth or relaxing the old depth threshold.',
        definition='Current scan nearest-pixel z-buffer and rectified calibration; actual selected camera xyz transformed with exact published camera pose and fixed origin. Floor world/.2 keys. Existing cell IDs retained, new keys appended in stable frame/pixel order. No interpolation, dilation, nearest-distance search or GT.',
        limitation='Measured CELL visibility, not measured visibility of every old POINT in that cell. Boundary aliasing at20cm remains. Not equivalent to5cm old-point depth gating. All candidate methods/controls must share this input, and none may append these new returns to the scored map.'))
    for key in scope['scenes']:
        manifest=read(INPUT/key/'manifest.json');source=np.load(manifest['geometry'])['xyz'];oldcell=np.load(INPUT/'query_cells'/key/'source_to_cell.npy')
        ids,first=np.unique(oldcell,return_index=True);np.testing.assert_array_equal(ids,np.arange(len(ids)))
        oldkeys=np.floor(source[first]/.2).astype(np.int64);lookup={tuple(k):int(i) for k,i in zip(oldkeys,ids)}
        assert len(lookup)==len(ids)
        # Confirm every old source still maps to its frozen physical cell ID.
        remapped=np.fromiter((lookup[tuple(k)] for k in np.floor(source/.2).astype(np.int64)),dtype=np.int64,count=len(source))
        np.testing.assert_array_equal(remapped,oldcell)
        scene_records=[];currentkeys=[k.tolist() for k in oldkeys]
        for rec in sorted((r for r in inputs['records'] if r['key']==key),key=lambda r:r['frame_id']):
            raw=np.fromfile(rec['raw_scan'],dtype='<f4').reshape(-1,4)[:,:3];p=project_lidar(raw,calib);raw_ids=p['selected_indices']
            pose=np.load(rec['file'])['camera_pose'];world=p['camera_xyz'][raw_ids]@pose[:3,:3].T+pose[:3,3]
            keys=np.floor(world/.2).astype(np.int64);assigned=[]
            for k in keys:
                token=tuple(k)
                if token not in lookup:lookup[token]=len(lookup);currentkeys.append(k.tolist())
                assigned.append(lookup[token])
            assigned=np.asarray(assigned,np.int64);y,x=np.nonzero(p['source_point_index']>=0)
            np.testing.assert_array_equal(p['source_point_index'][y,x],raw_ids)
            np.testing.assert_array_equal(np.asarray(currentkeys,np.int64)[assigned],keys)
            path=OUT/'validation_cells'/key/f'{rec["frame_id"]:06d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,cell_ids=assigned,pixel_xy=np.column_stack([x,y]),raw_indices=raw_ids)
            files[str(path)]=sha(path);scene_records.append(dict(frame_id=rec['frame_id'],rgb=rec['rgb'],file=str(path),
                current_returns=len(raw_ids),visible_cells=len(np.unique(assigned)),visible_old_cells=len(np.unique(assigned[assigned<len(ids)])),
                cell_count_at_frame=len(lookup)))
        keypath=OUT/'validation_cells'/key/'cell_keys.npy';np.save(keypath,np.asarray(currentkeys,np.int64));files[str(keypath)]=sha(keypath)
        # Geometry-only opportunity counts; NEVER replace selected pairs/frames.
        d=dict(np.load(OUT/key/'groups.npz'));sets=pieces(d['source_ids'],d['source_offsets']);views={r['frame_id']:np.unique(np.load(r['file'])['cell_ids']) for r in scene_records}
        pair_rows=[]
        for candidate in read(OUT/key/'selected.json'):
            parts=[np.unique(oldcell[sets[g]]) for g in candidate['groups']];counts=[]
            for frame in candidate['policies']['max_visible']['verification_frames']:
                values=[len(np.intersect1d(part,views[frame])) for part in parts]
                counts.append(dict(frame=frame,visible_cells=values,both16=min(values)>=16))
            pair_rows.append(dict(key=key,groups=candidate['groups'],views=counts))
        comparisons.extend(pair_rows)
        stats[key]=dict(old_cells=len(ids),new_cells=len(lookup)-len(ids),frames=len(scene_records),
                       pairs_with_validation=sum(any(v['both16'] for v in r['views']) for r in pair_rows),
                       pairs_with_two_views=sum(all(v['both16'] for v in r['views']) for r in pair_rows))
        results[key]=dict(records=scene_records,cell_keys=str(keypath),cell_count=len(lookup))
        print('CURRENT MEASUREMENT CELLS',key,stats[key],flush=True)
    dump(OUT/'validation_cells_complete.json',dict(complete=True,protocol_sha256=sha(OUT/'validation_cells_protocol.json'),
         files=files,stats=stats,results=results,opportunities=comparisons,old_point_adapter_unchanged=True,
         gt_read=False,selection_changed=False,output_map_geometry_changed=False))


if __name__=='__main__':main()
