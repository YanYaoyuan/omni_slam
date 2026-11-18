"""Bounded read-only observation audit; no new predictions or map variants."""
import time
import runpy
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha

B = ROOT / 'build/benchmarks'
PARENT = B / 'joint_semantic_revision_v1'
OUT = B / 'object_observation_audit_v1'
DEST = ROOT / 'output/object_observation_audit_v1'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    assert not OUT.exists()
    parent = read(PARENT / 'protocol.json'); predictions = read(PARENT / 'predictions_complete.json')
    contexts = parent['contexts']; cases = {}
    for c in contexts:
        cases[c['key']] = [r['object_id'] for r in predictions['results'][c['key']]['methods']['ours_joint']['class_changes']]
    inputs = {str(PARENT / name): sha(PARENT / name) for name in ['protocol.json', 'predictions_complete.json', 'summary.json']}
    for c in contexts:
        for f in [c['sem'], c['field'], c['job'], c['observations'], Path(c['full']) / 'old_to_full.npy', Path(c['full']) / 'geometry.npz']:
            inputs[str(f)] = sha(f)
        obs = read(c['observations'])
        for r in obs['results'][c['key']]:
            assert sha(r['file']) == obs['files'][r['file']]
            inputs[r['file']] = obs['files'][r['file']]
            inputs[r['rgb']] = sha(r['rgb'])
    dump(OUT / 'protocol.json', dict(created_epoch=time.time(), cases=cases, input_hashes=inputs,
        source_sha256=sha(__file__), scope='All seven objects whose class changed in the frozen joint candidate. Diagnosis, not a new candidate, independent evaluation or GT-selected visualization. Inspect every cached observation for these objects; display each maximum-visible-source frame (earliest frame breaks ties). No new model/GT/map/AP.',
        checks='Compare source probabilities against stored dense class map. Public data: independently project full measured coordinates with job camera pose/intrinsics and compare cached integer pixels; measure fractional pixel residual and camera depth. Preserve all frame counts/probabilities. Robot: original camera model separately handled; do not apply public camera convention.',
        next='Determine whether observations sample the target surface or suffer coordinate/occlusion/front-end ambiguity. No threshold or posterior weighting search.'))
    names = {}
    for label in runpy.run_path(str(ROOT / 'build/datasets/kitti360_prepare/labels.py'))['labels']:
        if 0 <= label.trainId < 19: names.setdefault(label.trainId, label.name)
    colors = np.array([[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
                       [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
                       [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],
                       [0,80,100],[0,0,230],[119,11,32]], dtype=np.uint8)
    DEST.mkdir(parents=True, exist_ok=False); all_results = {}; figures = []
    for c in contexts:
        key = c['key']; owners = np.load(c['sem'])['owners']; full = np.load(Path(c['full']) / 'geometry.npz')
        mapping = np.load(Path(c['full']) / 'old_to_full.npy'); job = read(c['job']); frames = {r['frame_id']: r for r in job['frames']}
        records = {oid: [] for oid in cases[key]}; selected = {}
        for record in read(c['observations'])['results'][key]:
            d = np.load(record['file']); ids = d['source_ids']; xy = d['source_pixel_xy']; q = d['source_probabilities']
            for oid in cases[key]:
                take = owners[ids] == oid
                if not take.any(): continue
                pixels = xy[take]; prob = q[take]; src = ids[take]
                np.testing.assert_array_equal(d['class_map'][pixels[:, 1], pixels[:, 0]], prob.argmax(1))
                row = dict(frame_id=record['frame_id'], source_count=len(src), unique_pixels=len(np.unique(pixels, axis=0)),
                    argmax_counts=np.bincount(prob.argmax(1), minlength=19).tolist(), mean_probability=prob.mean(0).tolist(),
                    pixel_bounds=[pixels.min(0).tolist(), pixels.max(0).tolist()])
                if c['kind'] == 'outdoor':
                    inv = np.linalg.inv(frames[record['frame_id']]['pose']); xyz = full['xyz'][mapping[src]]
                    camera = xyz @ inv[:3, :3].T + inv[:3, 3]; cam = job['config']['cam']
                    uv = camera[:, :2] / camera[:, 2:3] * [cam['fx'], cam['fy']] + [cam['cx'], cam['cy']]
                    expected = np.floor(uv + .5).astype(int)
                    np.testing.assert_array_equal(expected, pixels)
                    row.update(max_pixel_residual=float(np.abs(uv - pixels).max()),
                               camera_depth_range=[float(camera[:, 2].min()), float(camera[:, 2].max())])
                records[oid].append(row)
                if oid not in selected or len(src) > selected[oid]['count']:
                    selected[oid] = dict(record=record, count=len(src), pixels=pixels.copy(), source_ids=src.copy(),
                                         class_map=d['class_map'], probability=prob.copy())
        all_results[key] = {}
        for oid in cases[key]:
            rows = records[oid]; choice = selected[oid]; rec = choice['record']; pixels = choice['pixels']
            data = OUT / key / f'object_{oid}_selected.npz'; data.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(data, source_ids=choice['source_ids'], pixel_xy=pixels, probabilities=choice['probability'])
            with Image.open(rec['rgb']) as im: rgb = np.asarray(im.convert('RGB'))
            height, width = rgb.shape[:2]
            low = np.maximum(pixels.min(0) - 40, 0); high = np.minimum(pixels.max(0) + 41, [width, height]); x0,y0 = low; x1,y1 = high
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            axes[0].imshow(rgb); axes[0].scatter(pixels[:, 0], pixels[:, 1], c='#ffed4b', s=2, alpha=.5)
            axes[0].set_title('Full RGB + measured object samples')
            axes[1].imshow(rgb[y0:y1, x0:x1]); axes[1].set_title('Same RGB region (no overlay)')
            blend = (.55 * rgb + .45 * colors[choice['class_map']]).astype(np.uint8)
            axes[2].imshow(blend[y0:y1, x0:x1]); axes[2].scatter(pixels[:,0]-x0,pixels[:,1]-y0,c='#ffed4b',s=3,alpha=.7)
            top = np.argsort(choice['probability'].mean(0))[::-1][:3]
            axes[2].set_title('Cached SegFormer + samples\n' + ', '.join(names[int(i)] for i in top))
            for ax in axes: ax.axis('off')
            fig.suptitle(f'{key} / predicted object {oid} / frame {rec["frame_id"]} / {len(pixels)} samples')
            path = DEST / f'{key}_object_{oid}.png'; fig.savefig(path,dpi=150); plt.close(fig); figures.append(str(path))
            all_results[key][str(oid)] = dict(frames=rows, selected_frame=rec['frame_id'], selection='Maximum source count, earliest tie',
                selected_probability_mean=choice['probability'].mean(0).tolist(), selected_rgb=rec['rgb'], figure=str(path), data=str(data))
            print('OBSERVATION AUDIT',key,oid,'frames',len(rows),'best frame',rec['frame_id'],'top',[names[int(i)] for i in top],flush=True)
    for f,h in inputs.items(): assert sha(f)==h,f
    dump(OUT / 'summary.json',dict(complete=True,results=all_results,figures=figures,gt_read=False,new_maps=0,new_ap=0,model_calls=0,
         files={str(f):sha(f) for f in [*map(Path,figures),*OUT.glob('*/object_*_selected.npz')]},protocol_sha256=sha(OUT/'protocol.json')))


if __name__ == '__main__': main()
