"""Verify complete real-input preparation and physical source provenance."""
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha

OUT=ROOT/'build/benchmarks/discovery_measured_inputs_v1'
p=read(OUT/'protocol.json');seal=read(OUT/'complete.json');assert seal['complete'] and seal['protocol_sha256']==sha(OUT/'protocol.json')
for f,h in {**seal['files'],**p['source_hashes']}.items():assert sha(f)==h,f
scenes=[]
for spec in p['scenes']:
    m=read(OUT/spec['key']/'manifest.json');assert m['complete'] and sha(m['geometry'])==m['geometry_sha256']
    for f,h in m['input_hashes'].items():assert sha(f)==h,f
    geometry=np.load(m['geometry']);xyz=geometry['xyz'];born=geometry['source_frame'];last=-1;last_size=0;frequency=np.zeros(len(xyz),np.int32)
    for r in m['records']:
        assert r['frame_id']>last and last_size<=r['map_points']<=len(xyz)
        z=np.load(r['file']);ids=z['visible_source_ids'];pixels=z['pixel_xy'];assert pixels.shape==(len(ids),2)
        assert len(np.unique(ids))==len(ids) and np.all(ids>=0) and np.all(ids<r['map_points'])
        assert np.all(born[ids]<=r['frame_id']);frequency[ids]+=1;last=r['frame_id'];last_size=r['map_points']
    assert last_size==len(xyz)
    provenance_checks=0
    if spec['kind']=='robot':
        raw_indices=geometry['source_raw_index'];job=read(spec['job']);frames={r['frame_id']:r for r in job['frames']}
        selected=np.linspace(0,len(xyz)-1,min(256,len(xyz)),dtype=int)
        for frame_id in np.unique(born[selected]):
            local=selected[born[selected]==frame_id];r=frames[int(frame_id)];points=np.load(r['cloud'])['points'];t=np.asarray(r['pose'])
            actual=(points@t[:3,:3].T+t[:3,3])[raw_indices[local]]
            np.testing.assert_array_equal(xyz[local],actual);provenance_checks+=len(local)
    scenes.append(dict(key=spec['key'],frames=len(m['records']),source_points=len(xyz),queriable_points=int((frequency>0).sum()),
        points_visible_in_multiple_frames=int((frequency>1).sum()),total_point_frame_pairs=int(frequency.sum()),
        sampled_robot_provenance_checks=provenance_checks,all_frame_ids_causal=True,all_frame_source_ids_unique=True))
dump(OUT/'verification.json',dict(complete=True,files_verified=len(seal['files']),scenes=scenes,source_sha256=sha(__file__),input_seal_sha256=sha(OUT/'complete.json')))
print(scenes)
