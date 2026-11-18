"""Freeze all output-partition disputes and genuinely unused image observations.

Development capability probe only. Does not read references, diagnostic GT,
or change any map. Old failed signed grouping is only a proposal source.
"""
from pathlib import Path
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from run_signed_query_association import OUT as PARENT, INPUT, pieces, sealed
from omni_semantic_map.kitti360_input import load_poses, load_times

OUT = ROOT/'build/benchmarks/partition_query_probe_v1'
META = ROOT/'build/datasets/kitti360_prepare/extracted'


def anchor(pixels, source_ids):
    """Choose a measured pixel, never the possibly empty image centroid."""
    center = np.median(pixels, axis=0)
    order = np.lexsort((source_ids, np.sum((pixels-center)**2, axis=1)))
    first = order[0]
    # The second prompt stays central but uses a different measured location.
    distinct = order[np.any(pixels[order] != pixels[first], axis=1)]
    second = distinct[0] if len(distinct) else first
    return pixels[[first, second]].astype(float).tolist(), source_ids[[first, second]].tolist()


def main():
    assert not OUT.exists(), 'Do not overwrite frozen probe selection'
    sealed('nodes_complete'); sealed('predictions_complete')
    protocol = read(PARENT/'protocol.json')
    pairs_out, sources, requests = [], {}, {}
    for scene in protocol['scenes']:
        key = scene['key']
        folders = {m: PARENT/'maps'/key/m for m in ['geometry_graph', 'ours_signed']}
        geometry_groups = read(folders['geometry_graph']/'groups.json')['groups']
        signed_groups = read(folders['ours_signed']/'groups.json')['groups']
        owner = {j: g['object_id'] for g in geometry_groups for j in g['node_ids']}
        disputes = set()
        for group in signed_groups:
            ids = sorted(set(owner[j] for j in group['node_ids'] if j in owner))
            disputes.update((a, b) for i, a in enumerate(ids) for b in ids[i+1:])
        manifest = read(INPUT/key/'manifest.json')
        job = read(scene['job']); sequence = job['scene']
        poses_path = META/sequence/'poses.txt'
        poses = load_poses(poses_path)
        image_time = META/sequence/'image_00/timestamps.txt'
        scan_time = META/sequence/'velodyne_points/timestamps.txt'
        images, scans = load_times(image_time), load_times(scan_time)
        old_frames = set(r['frame_id'] for r in manifest['records'])
        unused = sorted(f for f in poses if min(old_frames) <= f <= max(old_frames)
                        and f not in old_frames and f < len(images) and f < len(scans))
        for f in [INPUT/key/'manifest.json', scene['job'], poses_path, image_time, scan_time]:
            sources[str(f)] = sha(f)
        nodes = dict(np.load(PARENT/'nodes'/key/'nodes.npz'))
        meta = read(PARENT/'nodes'/key/'nodes.json')
        sets = pieces(nodes['source_ids'], nodes['source_offsets'])
        old_sets = {m['old_object_id']: sets[j] for j, m in enumerate(meta) if m['old_object_id'] >= 0}
        baseline = np.load(PARENT/'maps'/key/'baseline/geometry.npz')
        mapped = dict(np.load(folders['geometry_graph']/'geometry.npz'))
        n = len(baseline['xyz'])
        groups = {g['object_id']: g for g in geometry_groups}
        measured = {}
        for oid, g in groups.items():
            new = mapped['added_source_ids'][mapped['instance'][n:] == oid]
            measured[oid] = np.unique(np.concatenate([new]+[old_sets[i] for i in g['old_ids']]))
        views = [(r, dict(np.load(r['file']))) for r in manifest['records']]
        for a, b in sorted(disputes):
            assert groups[a]['semantic_id'] == groups[b]['semantic_id']
            sa, sb = np.setdiff1d(measured[a], measured[b]), np.setdiff1d(measured[b], measured[a])
            eligible = []
            for rec, view in views:
                va = np.isin(view['visible_source_ids'], sa); vb = np.isin(view['visible_source_ids'], sb)
                if min(va.sum(), vb.sum()) < 16: continue
                pa, pb = view['pixel_xy'][va], view['pixel_xy'][vb]
                ca, cb = np.median(pa, axis=0), np.median(pb, axis=0)
                distance = np.linalg.norm(ca-cb)
                spread = np.sqrt(np.mean(np.sum((pa-ca)**2, axis=1)))+np.sqrt(np.mean(np.sum((pb-cb)**2, axis=1)))+1
                fa, fb = va.sum()/max(1, len(sa)), vb.sum()/max(1, len(sb))
                utility = 2*fa*fb/max(1e-12, fa+fb)*distance/spread
                eligible.append((float(utility), -rec['frame_id'], rec, va, vb, view))
            row = dict(key=key, sequence=sequence, object_ids=[a,b], semantic_id=groups[a]['semantic_id'],
                       exclusive_source_counts=[len(sa),len(sb)], eligible_views=len(eligible))
            if not eligible:
                row.update(status='no_common_measured_view'); pairs_out.append(row); continue
            utility, _, rec, va, vb, view = max(eligible, key=lambda r:r[:2])
            points_a, ids_a = anchor(view['pixel_xy'][va], view['visible_source_ids'][va])
            points_b, ids_b = anchor(view['pixel_xy'][vb], view['visible_source_ids'][vb])
            if points_a[0] == points_b[0]:
                row.update(status='coincident_prompt_pixels'); pairs_out.append(row); continue
            # Fixed temporal offsets; no RGB, segmentation or visibility inspection
            # of verification frames contributes to this selection.
            heldout = []
            for offset in [1, 6]:
                candidates = [f for f in unused if rec['frame_id']+offset <= f <= rec['frame_id']+offset+4]
                if candidates and candidates[0] not in heldout: heldout.append(candidates[0])
            if len(heldout) != 2:
                row.update(status='insufficient_unused_pose_frames', acquisition_frame=rec['frame_id'])
                pairs_out.append(row); continue
            row.update(status='ready', acquisition_frame=rec['frame_id'], acquisition_rgb=rec['rgb'],
                       acquisition_view=rec['file'], utility=utility, point_a=points_a, point_b=points_b,
                       prompt_source_ids=[ids_a,ids_b], verification_frames=heldout)
            name=f'{key}_{a}_{b}'
            dest=OUT/'sources'/(name+'.npz')
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, a=sa,b=sb)
            row['source_sets']=str(dest)
            sources[str(dest)] = sha(dest)
            sources[rec['rgb']] = sha(rec['rgb']); sources[rec['file']] = sha(rec['file'])
            requests.setdefault(sequence,set()).update(heldout)
            pairs_out.append(row)
    inputs = dict(created_epoch=time.time(), parent_prediction_sha256=sha(PARENT/'predictions_complete.json'),
        selection='All unordered pairs of exported geometry-graph objects whose member nodes share an exported frozen signed group. No GT, audit results or metric ranking read. Only pairs with >=16 exclusive measured sources visible on each side and two unused pose frames proceed; every skipped pair recorded.',
        action='Maximize harmonic mean of exclusive measured-source visible fractions times median-pixel separation/(sum RMS pixel radii+1). Stable earliest-frame tie. This is a fixed geometric dispute utility, not learned information gain.',
        prompts='Four instance decodes per acquisition: split(A positive,B negative), split(B positive,A negative), joint(A1 positive,B1 positive), joint(A2 positive,B2 positive). A1/B1 are measured pixels nearest their coordinate median; A2/B2 next distinct measured pixel. Prompt choices cannot access verification images. Select native maximum predicted-IoU mask among multimask outputs; retain all outputs for audit.',
        verification='Two new image/LiDAR frames: first unused exact-pose frame within [action+1,action+5] and [action+6,action+10]. No visual/visibility search. They were absent from old map image queries, but are correlated nearby observations of the SAME exposed development window, not independent benchmark scenes. Same class-only native query, no candidate-specific prompts. Missing measured support abstains.',
        budget='All six currently detected disputes, no resampling or threshold scan. Four instance decodes per ready dispute; one image encode per acquisition image; one native class decode per distinct verification image/class. No new fits or maps in this capability probe.',
        gate='Capability evidence only: record whether both prompted hypotheses yield masks and whether distinct unprompted verification observations discriminate their real measured support. No AP or paper-advantage claim. Freeze full-map repair and equal-budget controls separately only if evidence warrants them. Failed signed variant remains closed.',
        checkpoint='/home/user/vbot/sam3/checkpoints/sam3.pt', checkpoint_sha256=sha('/home/user/vbot/sam3/checkpoints/sam3.pt'),
        model='Local SAM3 with enable_inst_interactivity=True, resolution1008, BF16, seed0; GPU1 only, >=18GiB free, half-device allocator cap.',
        source_hashes=sources, generator_sha256=sha(__file__), pairs=pairs_out,
        downloads={k:sorted(v) for k,v in requests.items()}, reference_labels_read=False)
    dump(OUT/'protocol.json', inputs)
    print('PREPARED',[(r['key'],r['object_ids'],r['status'],r.get('verification_frames')) for r in pairs_out],flush=True)


if __name__=='__main__':main()
