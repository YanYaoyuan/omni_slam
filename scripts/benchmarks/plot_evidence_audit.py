"""Deterministic post-hoc original-image evidence for the retention audit."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_evidence_retention import ROOT, BASE, CACHE, REPORT, dump

SOURCE = CACHE.parents[1] / 'OVO_Replica_full'


def ground_truth_names(scene):
    for root in ['replica_independent_20261109', 'ownership_20261109/heldout', 'expanded_semantic_20261109/replica']:
        p = ROOT / 'build/benchmarks' / root / 'data' / scene / 'info_semantic.json'
        if p.exists():
            return {int(x['id'])+1: x['class_name'] for x in json.loads(p.read_text())['objects']}
    raise FileNotFoundError(scene)


def show(ax, scene, row, title, cases, names):
    ax.axis('off')
    if row is None:
        ax.text(.5, .5, title+'\nNo case in this scene', ha='center', va='center')
        return
    oid = row['selected_observation_ids'][0]  # largest raw-mask area, fixed tie ID
    manifest = json.loads((CACHE/scene/'manifest.json').read_text())
    record = next(r for r in manifest['records'] if r['observation_offset'] <= oid < r['observation_offset']+r['masks'])
    with np.load(CACHE/scene/'frames'/record['file']) as f:
        mask = np.unpackbits(f['masks_packed'][oid-record['observation_offset']], axis=1)[:, :manifest['config']['cam']['W']].astype(bool)
    frame = record['frame_id']
    rgb = np.asarray(Image.open(SOURCE/scene/'raw'/f'rgb_{frame}.png')).astype(float)/255
    rgb[mask] = .55*rgb[mask]+.45*np.array([1., .22, .15])
    ax.imshow(rgb)
    b = row['before']
    gt = names.get(b['dominant_gt'], '?') if b else '?'
    ax.set_title(f'{scene} / {title} / ID {row["object_id"]}\npred: {row["prediction"]["name"]}; dominant GT: {gt}\nframe {frame}; {row["unique_frames"]} views', fontsize=9)
    cases.append(dict(scene=scene, role=title, object_id=row['object_id'], observation_id=oid,
        frame_id=frame, dominant_gt_name=gt, prediction=row['prediction'],
        rgb=str(SOURCE/scene/'raw'/f'rgb_{frame}.png'), mask=str(CACHE/scene/'frames'/record['file']),
        view_selection='largest raw-mask area among unique-frame selected observations, lowest ID tie'))


def main():
    summary = json.loads((REPORT/'stage1/summary.json').read_text())
    scenes = summary['scenes']; cases = []
    fig, axes = plt.subplots(len(scenes), 3, figsize=(16, 3.2*len(scenes)), layout='constrained')
    gt_losses = {}
    for si, scene in enumerate(scenes):
        result = json.loads((REPORT/'stage1'/f'{scene}.json').read_text())
        rows = result['objects']; by_id = {r['object_id']:r for r in rows}; names = ground_truth_names(scene)
        lost = [l for l in result['filter_losses'] if l['cause']=='direct_background_filter']
        chosen_loss = by_id[min(lost, key=lambda l:l['object_id'])['object_id']] if lost else None
        partial = [r for r in rows if r['after'] and r['after']['category'] in ('fragmentation_consistent', 'partial_coverage')]
        chosen_part = max(partial, key=lambda r:(r['after']['voxels'], -r['object_id'])) if partial else None
        matched = [r for r in rows if r['after'] and r['after']['category']=='matched']
        control = max(matched, key=lambda r:(r['after']['voxels'], -r['object_id'])) if matched else None
        for ci, (row, title) in enumerate([(chosen_loss,'direct filter loss'), (chosen_part,'partial / fragment'), (control,'matched control')]):
            show(axes[si, ci], scene, row, title, cases, names)
        gt_losses[scene] = [dict(**l, gt_class_name=names.get(l['gt_id'])) for l in lost]
    fig.suptitle('Frozen guarded + mean baseline | red overlay = cached predicted mask\nCases selected by a fixed rule; GT names are post-hoc annotations only', fontsize=13)
    fig.savefig(REPORT/'stage1/image_cases.png', dpi=135); plt.close(fig)
    dump(REPORT/'stage1/case_manifest.json', dict(cases=cases, direct_losses_with_gt=gt_losses))

    categories = ['matched','partial_coverage','fragmentation_consistent','background_dominated','mixed_foreground']
    labels = ['Matched','Partial','Fragments','Background','Mixed']
    fig, axes = plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    for ax, key, title in zip(axes, ['unique_frames','median_area_px','max_view_angle_deg'], ['Unique observed frames','Median raw-mask area (pixels)','Maximum view separation (degrees)']):
        stats=[]
        for c, label in zip(categories,labels):
            q=summary['statistics'][c][key]
            stats.append(dict(label=label,med=q['median'],q1=q['q25'],q3=q['q75'],whislo=q['min'],whishi=q['max'],fliers=[]))
        ax.bxp(stats,showfliers=False); ax.set_title(title,fontsize=10);ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
        if key=='median_area_px':ax.set_yscale('log')
    fig.suptitle('Object-level associations with errors; box = IQR, whiskers = range; not causal evidence',fontsize=11)
    fig.savefig(REPORT/'stage1/evidence_statistics.png',dpi=150);plt.close(fig)
    print('PLOTTED',len(cases),'cases',flush=True)


if __name__=='__main__':main()
