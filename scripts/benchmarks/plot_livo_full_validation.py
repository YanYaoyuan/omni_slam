"""Full spatial extent and temporal coverage of the completed LIVO experiment."""
import json
from itertools import combinations, product
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_evidence_retention import OUT, REPORT


def main():
    root = OUT / 'livo_full_strong_v2'
    job = json.loads((OUT / 'jobs/livo_full_v2.json').read_text())
    capture = json.loads((REPORT / 'stage3/livo_full_v2_capture_audit.json').read_text())
    xyz = np.load(root / 'cache/geometry.npz')['xyz']
    assert np.isfinite(xyz).all()
    sample = xyz[::max(1, int(np.ceil(len(xyz) / 100000)))]
    trajectory = np.array([np.asarray(f['pose'])[:3, 3] for f in job['frames']])
    low = np.minimum(xyz.min(0), trajectory.min(0)) - .5
    high = np.maximum(xyz.max(0), trajectory.max(0)) + .5
    fig = plt.figure(figsize=(14, 9), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    for col, (run, report, title) in enumerate([
        ('cg', 'livo_full_v2', 'Original sparse supports'),
        ('cg_supportcc', 'livo_full_v2_supportcc', 'Largest connected supports')]):
        data = json.loads((REPORT / 'stage3' / report / 'summary.json').read_text())
        g = np.load(root / run / 'geometry.npz')
        ax = fig.add_subplot(grid[0, col], projection='3d')
        ax.scatter(*sample.T, s=.2, c='#b6c0cc', alpha=.15, rasterized=True)
        ax.plot(*trajectory.T, c='#202a35', lw=1.2)
        for i, obj in enumerate(data['retained_objects']):
            color = plt.get_cmap('tab20')(i % 20)
            points = g['xyz'][g['instance'] == obj['evaluation_id']]
            ax.scatter(*points.T, s=1.5, color=color, rasterized=True)
            corners = np.array(list(product(*zip(points.min(0), points.max(0)))))
            for a, b in combinations(corners, 2):
                if np.count_nonzero(a != b) == 1:
                    ax.plot(*np.stack([a, b]).T, color=color, lw=.8)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]),
               xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
               title=f'{title}\n{len(data["retained_objects"])} retained predictions')
        ax.set_box_aspect(high-low); ax.view_init(38, -55)
    ax = fig.add_subplot(grid[1, :])
    bins = capture['timeline']; edges = np.array(bins['bin_edges_sensor_epoch_s'])
    centers = (edges[:-1]+edges[1:])/2-edges[0]; width = np.diff(edges)*.38
    ax.bar(centers-width/2, bins['published_visual_frames'], width, label='Native visual keyframes', color='#8ba2bb')
    ax.bar(centers+width/2, bins['captured_frames'], width, label='Retained semantic packets', color='#187aaa')
    ax.set(xlabel='Sensor time since first update (s)', ylabel='Frames per interval',
           title='Every temporal interval represented; missing semantic packet IDs: 0, 2')
    ax.legend(ncols=2, frameon=False)
    fig.suptitle('Full recording: LIVO + SAM2 / SigLIP / CG\nShared axes, no spatial crop; boxes describe observed surfaces, not GT accuracy', fontsize=14)
    path = REPORT / 'stage3/livo_full_comparison.png'
    fig.savefig(path, dpi=170); plt.close(fig); print(path)


if __name__ == '__main__':
    main()
