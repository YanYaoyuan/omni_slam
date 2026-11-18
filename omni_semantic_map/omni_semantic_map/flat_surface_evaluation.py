"""Observed-surface voxel instance diagnostic, NOT official ScanNet mesh AP.

Reference labels are derived from rendered GT masks and provided RGB-D poses.
All methods vote over the same 5 cm world voxels, including missed predictions.
This evaluates association on observed supports, not TSDF reconstruction.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from .flat_benchmark import frame_index, camera_points, read_ids, read_pose, read_labels, instance_metrics
from .objects import transform_points


def majority_labels(voxel_indices, labels, n_voxels):
    pairs, counts = np.unique(np.column_stack([voxel_indices, labels]), axis=0, return_counts=True)
    # Most votes wins; ties break towards the smaller ID (background is zero).
    order = np.lexsort((pairs[:, 1], -counts, pairs[:, 0]))
    pairs, counts = pairs[order], counts[order]
    first = np.r_[True, np.diff(pairs[:, 0]) != 0]
    result = np.zeros(n_voxels, np.int32)
    result[pairs[first, 0]] = pairs[first, 1]
    return result


def evaluate_surfaces(sequence, runs, output, voxel_size=.05):
    if voxel_size <= 0:
        raise ValueError('voxel size must be positive')
    output.mkdir(parents=True, exist_ok=False)
    ids, _ = frame_index(sequence)
    labels = read_labels(sequence.parent / 'groundtruth_labels.csv')
    coordinates, truths, validity, lengths = [], [], [], []
    stride = 4
    for frame in ids:
        depth = np.asarray(Image.open(sequence / (frame + '_depth.tiff')))
        points, valid, _, _ = camera_points(depth, stride)
        world = transform_points(points[valid], read_pose(sequence / (frame + '_pose.txt')))
        coordinates.append(np.floor(world / voxel_size).astype(np.int32))
        raw = read_ids(sequence / (frame + '_segmentation.png'))[::stride, ::stride][valid]
        truth = np.zeros(len(raw), np.int32)
        for value in np.unique(raw):
            if labels[int(value)]['thing']:
                truth[raw == value] = int(value) + 1
        truths.append(truth)
        validity.append(valid)
        lengths.append(len(raw))
    voxels, inverse = np.unique(np.concatenate(coordinates), axis=0, return_inverse=True)
    truth = majority_labels(inverse, np.concatenate(truths), len(voxels))
    centers = (voxels.astype(float) + .5) * voxel_size
    np.savez_compressed(output / 'reference.npz', xyz=centers, instance=truth)
    results = dict(scope=__doc__, voxel_size_m=voxel_size, depth_stride=stride,
                   evaluation_voxels=len(voxels), sampled_pixels=sum(lengths),
                   tie_policy='most votes, smallest ID on tie; zero is background/missing', methods={})
    for path in runs:
        protocol = json.loads((path / 'run.json').read_text())
        if (not protocol['complete'] or protocol['depth_stride'] != stride
                or Path(protocol['sequence']).resolve() != sequence.resolve()
                or protocol['frames_processed'] != len(ids)):
            raise ValueError(f'incompatible run: {path}')
        pieces = []
        for frame, valid in zip(ids, validity):
            with np.load(path / 'frames' / (frame + '.npz')) as data:
                if not np.array_equal(valid, data['valid']):
                    raise ValueError('different support domain')
                pieces.append(data['instance'][valid])
        predicted = majority_labels(inverse, np.concatenate(pieces), len(voxels))
        pairs, counts = np.unique(np.column_stack([truth, predicted]), axis=0, return_counts=True)
        contingency = Counter({tuple(map(int, p)): int(c) for p, c in zip(pairs, counts)})
        metrics = instance_metrics(contingency)
        metrics['observed_voxel_instance_pq'] = metrics.pop('sequence_pixel_instance_pq')
        metrics['gt_thing_voxel_coverage'] = metrics.pop('gt_thing_sample_coverage')
        results['methods'][path.name] = metrics
        np.savez_compressed(output / (path.name + '.npz'), xyz=centers, instance=predicted)
    (output / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    plot_maps(centers, truth, runs, output)
    return results


def plot_maps(centers, truth, runs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Static top-down scientific plot. Color indices are arbitrary instance IDs,
    # not semantic classes; corresponding colors across methods are not identity.
    n = len(runs) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), constrained_layout=True, squeeze=False)
    datasets = [('GT observed supports', truth)]
    for path in runs:
        with np.load(output / (path.name + '.npz')) as data:
            datasets.append((path.name.replace('run1_', '').replace('run2_', ''), data['instance']))
    for axis, (name, instance) in zip(axes[0], datasets):
        bg = instance == 0
        axis.scatter(centers[bg, 0], centers[bg, 1], c='#dddddd', s=.2, rasterized=True)
        selected = ~bg
        axis.scatter(centers[selected, 0], centers[selected, 1],
                     c=(instance[selected] * .61803398875) % 1, cmap='turbo', vmin=0, vmax=1,
                     s=.4, rasterized=True)
        axis.set(title=name, xlabel='World X [m]', ylabel='World Y [m]')
        axis.set_aspect('equal')
    fig.suptitle('Observed surface instances | shared 5 cm voxel domain | colors are arbitrary IDs')
    fig.savefig(output / 'instance_maps.png', dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sequence', type=Path, required=True)
    ap.add_argument('--runs', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = evaluate_surfaces(args.sequence, args.runs, args.output)
    print(json.dumps({n: r['observed_voxel_instance_pq'] for n, r in result['methods'].items()}, indent=2))


if __name__ == '__main__':
    main()
