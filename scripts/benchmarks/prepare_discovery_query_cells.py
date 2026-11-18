"""Prepare common query-memory resolution for all future selection controls."""
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.measured_query_cells import query_cell_ids

BASE=ROOT/'build/benchmarks/discovery_measured_inputs_v1'
OUT=BASE/'query_cells'
assert not OUT.exists();old=read(BASE/'complete.json')
for f,h in old['files'].items():assert sha(f)==h,f
parent=ROOT/'build/benchmarks/kitti360_connected_development_v1/mapping_protocol.json';size=read(parent)['sensor_overlap_m'];assert size==.2
source=ROOT/'omni_semantic_map/omni_semantic_map/measured_query_cells.py'
dump(OUT/'protocol.json',dict(cell_m=size,scale_source=str(parent),scale_source_sha256=sha(parent),input_seal_sha256=sha(BASE/'complete.json'),
    source_hashes={str(source):sha(source),str(Path(__file__)):sha(__file__)},
    scope='Common query-memory grid for all selection controls, not a new map or semantic labeling. Uses existing outdoor association scale without score-based search. Grid and nearest-neighbor gates are different operations.',
    geometry_unchanged=True,cell_ids='First measured source occurrence order; future spatial cells cannot renumber existing IDs.'))
stats=[];files={}
for s in old['scenes']:
    key=s['key'];manifest=read(BASE/key/'manifest.json');assert sha(manifest['geometry'])==manifest['geometry_sha256']
    xyz=np.load(manifest['geometry'])['xyz'];ids=query_cell_ids(xyz,size);folder=OUT/key;folder.mkdir();path=folder/'source_to_cell.npy';np.save(path,ids);files[str(path)]=sha(path)
    frequency=np.zeros(int(ids.max())+1,np.int32)
    for r in manifest['records']:
        v=np.load(r['file'])['visible_source_ids'];cells=np.unique(ids[v]);frequency[cells]+=1
    # Verify causality at multiple real prefix boundaries, not just a synthetic test.
    for r in manifest['records'][::max(1,len(manifest['records'])//4)]:
        n=r['map_points'];np.testing.assert_array_equal(query_cell_ids(xyz[:n],size),ids[:n])
    stats.append(dict(key=key,total_cells=len(frequency),visible_cells=int((frequency>0).sum()),
        cells_visible_in_multiple_frames=int((frequency>1).sum()),cell_frame_pairs=int(frequency.sum()),prefix_causality_verified=True))
dump(OUT/'complete.json',dict(complete=True,files=files,scenes=stats,protocol_sha256=sha(OUT/'protocol.json')))
print(stats)
