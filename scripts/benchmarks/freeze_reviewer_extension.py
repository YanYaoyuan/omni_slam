"""Bound four spatial holdout windows using official metadata only; no RGB/GT."""
from pathlib import Path
import re,subprocess,time
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.kitti360_input import load_poses
OUT=ROOT/'build/benchmarks/reviewer_extension_20261117'
META=ROOT/'build/datasets/kitti360_prepare'
DATA=ROOT/'build/datasets/reviewer_extension_20261117'

def main():
    assert not OUT.exists() and not DATA.exists()
    source=META/'splits/2013_05_28_drive_val.txt';index=ROOT/'build/datasets/kitti360_reserved_v4/semantic_archive_index.json';seenfile=ROOT/'build/datasets/kitti360_reserved_v4/geometry_separation_audit.json'
    entries={v['name']:v for v in read(index)['entries']};seen=read(seenfile)['boxes'];boxes=[np.asarray(v['aabb']) for v in seen.values()];inputs={str(p):sha(p) for p in [source,index,seenfile]};chosen=[];considered=[];poses={}
    def gap(x,y):return float(np.linalg.norm(np.maximum(np.maximum(x[0,:2]-y[1,:2],y[0,:2]-x[1,:2]),0)))
    for member in sorted(n for n in entries if '/static/' in n and n.endswith('.ply')):
        scene=re.search(r'2013_05_28_drive_\d+_sync',member)[0];sid=int(scene.split('_')[-2])
        if sid in [0,2] or scene in {v['scene'] for v in chosen}:continue
        if scene not in poses:
            p=META/'extracted'/scene/'cam0_to_world.txt'
            if not p.exists():continue
            poses[scene]=load_poses(p);inputs[str(p)]=sha(p)
        first,last=map(int,Path(member).stem.split('_'));frames=sorted(f for f in poses[scene] if first<=f<=last)
        if len(frames)<100:continue
        xyz=np.array([poses[scene][f][:3,3] for f in frames]);box=np.array([xyz.min(0),xyz.max(0)]);dist=[gap(box,b) for b in boxes];reason='selected' if min(dist)>300 else 'near_exposed_or_selected_extent';considered.append(dict(member=member,frames=len(frames),minimum_trajectory_to_exposed_raw_aabb_m=min(dist),decision=reason))
        if reason!='selected':continue
        row=dict(scene=scene,member=member,start=first,end=last,posed_frames=frames,input_frames=[frames[i] for i in np.unique(np.linspace(0,len(frames)-1,min(61,len(frames)),dtype=int))],trajectory_aabb=box.tolist(),minimum_separation_m=min(dist),gt_bytes=entries[member]['bytes'],role='spatial_holdout_within_previously_exposed_dataset_sequences');chosen.append(row);boxes.append(box)
        if len(chosen)==4:break
    print('Metadata selection',[(v['scene'],v['start'],v['end']) for v in chosen], 'eligible considered',len(considered),flush=True)
    if len(chosen)!=4:
        dump(ROOT/'build/benchmarks/reviewer_extension_selection_attempt3_20261117.json',dict(requested=4,selected=chosen,considered=considered,reason='No feasible four-window reservation under first metadata-only rule; no RGB or GT read.'))
    assert len(chosen)==4,'Do not silently relax selection; report reservation failure.'
    selection=dict(windows=chosen,source_sha256={str(Path(k).relative_to(ROOT)):v for k,v in inputs.items()},selection_code_sha256=sha(__file__),metadata_only_amendment='Validation-only candidate pool yielded no window at >300m from all prior measured extents, even after widening length. Before viewing RGB/GT, expand candidate pool to official annotated static windows; continue excluding the training sequences0000/0002, retain >300m gate and max61frames. Both failed metadata selections remain archived.',selection='First lexicographic official annotated static windows (train/validation pool; local spatial holdout, not official validation-only),>=100posedframes,exclude training sequences0000/0002,one window per sequence,trajectory AABB >300m from all nine previously measured window AABBs and chosen trajectories. No image/GT contents read.',sampling='61 uniformly spaced posed frames including endpoints across the entire selected window; same samples for all methods.',independence='New spatial windows within an already-exposed dataset/sequence family. NOT new sequences or a second dataset. Actual measured-surface separation remains a mandatory check before independent spatial claims.')
    dump(DATA/'window_selection.json',selection);dump(DATA/'semantic_archive_index.json',read(index))
    current=read(ROOT/'build/benchmarks/revision_assessment_v1/protocol.json')
    dump(OUT/'protocol.json',dict(date='2026-11-17',created_epoch=time.time(),selection=str(DATA/'window_selection.json'),selection_sha256=sha(DATA/'window_selection.json'),scientific_parent=str(ROOT/'build/benchmarks/revision_assessment_v1/protocol.json'),scientific_parent_sha256=sha(ROOT/'build/benchmarks/revision_assessment_v1/protocol.json'),sources=inputs,methods=['strong_baseline','unconditional','history_only','witness_history','nearest_witness_history','full_authorization','MaskClustering_shared_core'],primary='full_authorization unchanged; nearest/witness are controls, not post-hoc replacement Ours',fixed_models={k:current[k] for k in ['selected_model','sam3_checkpoint','sam3_sha256'] if k in current},budget=dict(windows=4,max_input_pairs=244,max_complete_maps=28,max_score_records=84,parameter_search=False),protocol='Reuse all parent geometric/semantic/model thresholds; same parent candidates for internal controls. Export all frozen maps before annotation decoding. Primary/canonical/uniform and ALL eligible object outcomes retained. No replacement if a window has no eligible target; report NA. Include unchanged/negative results; do not promote based on best windows.',required_before_inference=['Download SHA/CRC checks','Raw measured-space separation versus all earlier windows','Isolated-root execution adapter plus exposed golden-output checks','Working GPU and pinned model/runtime preflight'],status='reserved_not_executed',new_rgb_read=False,new_gt_decoded=False,model_calls=0,considered=considered))
    # Reusable official range downloader, isolated destination. No automatic model launch.
    dump(OUT/'preflight.json',dict(cuda_device_nodes=sorted(str(p) for p in Path('/dev').glob('nvidia*')),gpu_ready=bool(list(Path('/dev').glob('nvidia[0-9]*'))),old_python_runtime_resolves=(ROOT/'build/benchmarks/ovo_full_20261110/venv311/bin/python').exists(),status='GPU execution blocked in current environment; selected windows remain unprocessed.'))
    print([(x['scene'],x['start'],x['end'],len(x['input_frames'])) for x in chosen],flush=True)

if __name__=='__main__':main()
