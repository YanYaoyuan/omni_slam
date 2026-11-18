"""Reserve bounded outdoor windows from official split/pose metadata only."""
from pathlib import Path
import re
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.kitti360_input import load_poses

BASE = ROOT/'build/datasets/kitti360_prepare'


def main():
    output = BASE/'window_selection.json'; assert not output.exists()
    sources = [BASE/'splits/2013_05_28_drive_train.txt', BASE/'splits/2013_05_28_drive_val.txt', BASE/'semantic_archive_index.json']
    entries = {r['name']:r for r in read(sources[2])['entries']}; poses = {}
    def describe(member):
        scene = re.search(r'(2013_05_28_drive_\d+_sync)', member)[1]
        first, last = map(int, Path(member).stem.split('_'))
        if scene not in poses:
            path = BASE/'extracted'/scene/'cam0_to_world.txt'; poses[scene] = load_poses(path); sources.append(path)
        frames = [f for f in poses[scene] if first <= f <= last]
        xyz = np.array([poses[scene][f][:3,3] for f in frames])
        return dict(member=member, scene=scene, start=first, end=last, posed_frames=frames,
                    trajectory_aabb=np.array([xyz.min(0),xyz.max(0)]).tolist(), gt_bytes=entries[member]['bytes'])
    def separation(a, b):
        x, y = np.asarray(a['trajectory_aabb']), np.asarray(b['trajectory_aabb'])
        return float(np.linalg.norm(np.maximum(np.maximum(x[0,:2]-y[1,:2], y[0,:2]-x[1,:2]), 0)))
    development = describe(sources[0].read_text().splitlines()[0]); development['role'] = 'development'
    chosen = [development]
    for member in sorted(sources[1].read_text().splitlines()):
        candidate = describe(member)
        if candidate['scene'] in {r['scene'] for r in chosen} or len(candidate['posed_frames']) < 100: continue
        distances = [separation(candidate, previous) for previous in chosen]
        if min(distances) <= 200: continue
        candidate.update(role='reserved_validation', trajectory_aabb_separation_m=distances); chosen.append(candidate)
        if len(chosen) == 3: break
    assert len(chosen) == 3
    for row in chosen:
        row['input_frames'] = sorted(set(row['posed_frames'][::5]+row['posed_frames'][-1:]))
    dump(output, dict(windows=chosen,
        selection='First official train window is exposed development. Select first lexicographic official validation windows from two other sequences, >=100 available camera poses, with trajectory XY AABB distance >200m from all previously selected windows. Only metadata used.',
        sampling='Every fifth available rectified left-camera pose in the entire selected window, plus final pose. Image and raw Velodyne same frame IDs. This is sampled-window mapping, not processing every scan.',
        independence='Distinct sequences and separated recorded camera trajectories. Actual input/reference point extents still need auditing; not yet a guarantee of disjoint observed geometry.',
        gt_policy='Development reference may be inspected for adapter validation. Reserved references may be downloaded as opaque files; labels must not be decoded for method selection. Prediction/scoring protocol will be frozen separately before reserved inference.',
        protocol_status='Dataset reservation only. No algorithm hyperparameters or official-score claim implied.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources}, selection_code_sha256=sha(__file__)))
    print([(r['scene'],r['role'],len(r['input_frames'])) for r in chosen])


if __name__ == '__main__': main()
