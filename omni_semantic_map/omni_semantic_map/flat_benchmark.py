"""Flat development adapter: frozen poses/detections, anchored object supports.

Run never reads GT segmentation in predicted mode. Evaluation is a separate
command. Metrics operate on sampled, valid-depth pixels across the sequence,
NOT on a reconstructed 3D mesh and NOT the official ScanNet AP protocol.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .objects import associate, summarize_object, transform_points


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def frame_index(sequence):
    with (sequence / 'timestamps.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row['TimeStamp']))
    ids = [row['ImageID'] for row in rows]
    stamps = np.array([int(row['TimeStamp']) for row in rows], dtype=np.int64)
    if not len(rows) or len(set(ids)) != len(ids) or np.any(np.diff(stamps) <= 0):
        raise ValueError('invalid frame index')
    if any(not frame.isdigit() for frame in ids):
        raise ValueError('unsafe frame ID')
    return ids, (stamps - stamps[0]).astype(float) / 1e9


def read_ids(path):
    data = np.asarray(Image.open(path))
    if data.ndim == 3:
        if not np.all(data == data[:, :, :1]):
            raise ValueError('expected grayscale segment IDs, not RGB colors')
        data = data[:, :, 0]
    if data.ndim != 2 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError('invalid segmentation image')
    return data


def read_labels(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    labels = {int(r['InstanceID']): dict(class_id=int(r['ClassID']),
              thing=bool(int(r['PanopticID'])), name=r['Name']) for r in rows}
    if len(labels) != len(rows):
        raise ValueError('duplicate label ID')
    return labels


def camera_points(depth, stride=4):
    # Published Flat intrinsics, depth is camera Z in metres.
    if depth.shape != (480, 640) or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError('expected 640x480 floating depth in metres')
    if stride < 1:
        raise ValueError('stride must be positive')
    v, u = np.mgrid[0:480:stride, 0:640:stride]
    z = depth[v, u]
    valid = np.isfinite(z) & (z >= .1) & (z <= 5.)
    points = np.stack([(u - 320) * z / 320, (v - 240) * z / 320, z], axis=-1)
    return points, valid, u, v


def read_pose(path):
    pose = np.loadtxt(path)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
            or abs(np.linalg.det(pose[:3, :3]) - 1) > 1e-4):
        raise ValueError(f'invalid camera-to-world pose: {path}')
    return pose


def geometry_associate(objects, observations, poses, radius=1.):
    """Class-gated spatial Hungarian baseline, with no appearance/ambiguity test."""
    for obj in objects:
        summarize_object(obj, poses)
    costs = np.full((len(observations), len(objects)), 1e6)
    for i, obs in enumerate(observations):
        p = transform_points([obs['local_position']], poses[obs['keyframe_id']])[0]
        for j, obj in enumerate(objects):
            if obj['label'] == obs['label']:
                distance = np.linalg.norm(p - obj['position'])
                if distance <= radius:
                    costs[i, j] = distance
    matched = set()
    if costs.size:
        rows, cols = linear_sum_assignment(costs)
        for i, j in zip(rows, cols):
            if costs[i, j] < 1e5:
                observations[i]['object_id'] = objects[j]['id']
                objects[j]['observations'].append(observations[i])
                matched.add(i)
    for i, obs in enumerate(observations):
        if i not in matched:
            obs['object_id'] = f"object_{len(objects) + 1:05d}"
            objects.append(dict(id=obs['object_id'], label=obs['label'], observations=[obs]))
    for obj in objects:
        summarize_object(obj, poses)


def run(sequence, output, mode='predicted', method='omni_slam', stride=4):
    ids, stamps = frame_index(sequence)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'frames').mkdir()
    (output / 'supports').mkdir()
    gt_labels = read_labels(sequence.parent / 'groundtruth_labels.csv') if mode == 'oracle_masks' else None
    poses = np.array([read_pose(sequence / (frame + '_pose.txt')) for frame in ids])
    np.save(output / 'camera_poses.npy', poses)
    protocol = dict(sequence=str(sequence.resolve()), mode=mode, method=method,
                    pose_source='provided camera-to-world poses; no SLAM estimated here',
                    depth_stride=stride, depth_range_m=[.1, 5.], min_sampled_support=8,
                    association_radius_m=1., appearance_limit=.35,
                    ambiguity_margin=0. if method == 'omni_no_ambiguity' else .25,
                    geometry='sampled measured RGB-D surface; no amodal completion',
                    gt_segmentation_used_by_mapper=mode == 'oracle_masks',
                    source_sha256={str(p): sha256(p) for p in
                        [Path(__file__), Path(__file__).with_name('objects.py')]},
                    complete=False, frames_expected=len(ids))
    (output / 'run.json').write_text(json.dumps(protocol, indent=2))
    objects, numbers, frames = [], {}, []
    hashes = {str((sequence / 'timestamps.csv').resolve()): sha256(sequence / 'timestamps.csv')}
    if mode == 'oracle_masks':
        path = sequence.parent / 'groundtruth_labels.csv'
        hashes[str(path.resolve())] = sha256(path)
    started = time.monotonic()
    for k, frame in enumerate(ids):
        prefix = sequence / frame
        paths = [Path(str(prefix) + suffix) for suffix in ['_color.png', '_depth.tiff', '_pose.txt']]
        if mode == 'predicted':
            paths += [Path(str(prefix) + '_predicted.png'), Path(str(prefix) + '_labels.json')]
            segments = json.loads(paths[-1].read_text())
            segmentation = read_ids(paths[-2])
        else:
            paths += [Path(str(prefix) + '_segmentation.png')]
            segmentation = read_ids(paths[-1])
            unknown = set(map(int, np.unique(segmentation))) - set(gt_labels)
            if unknown:
                raise ValueError(f'unknown GT IDs: {unknown}')
            segments = [dict(id=int(i), isthing=gt_labels[int(i)]['thing'],
                        category_id=gt_labels[int(i)]['class_id'], score=1.) for i in np.unique(segmentation)]
        if len({int(s['id']) for s in segments}) != len(segments):
            raise ValueError('duplicate segment metadata')
        for path in paths:
            hashes[str(path.resolve())] = sha256(path)
        color = cv2.imread(str(paths[0]))
        if color is None or color.shape != (480, 640, 3) or segmentation.shape != (480, 640):
            raise ValueError('invalid frame dimensions')
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        depth = np.asarray(Image.open(paths[1]))
        points, valid, u, v = camera_points(depth, stride)
        observations, selected_masks = [], []
        rejected = 0
        for seg in segments:
            if not seg['isthing']:
                continue
            mask = segmentation == int(seg['id'])
            selected = mask[v, u] & valid
            if selected.sum() < 8:
                rejected += 1
                continue
            hist = cv2.calcHist([hsv], [0, 1, 2], mask.astype(np.uint8),
                               [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
            hist /= max(hist.sum(), 1)
            local = points[selected].astype(np.float32)
            support = f'supports/{frame}_{int(seg["id"])}.npz'
            np.savez_compressed(output / support, local_points=local,
                                pixels=np.column_stack([u[selected], v[selected]]).astype(np.int16))
            observations.append(dict(keyframe_id=k, stamp=float(stamps[k]),
                label=f"category_{int(seg['category_id'])}", score=float(seg.get('score', 1.)),
                local_position=np.median(local, axis=0).tolist(), appearance=hist.tolist(),
                evidence=str(paths[0].resolve()), support_file=support,
                support_points=len(local)))
            selected_masks.append(selected)
        if method in ('omni_slam', 'omni_no_ambiguity'):
            associate(objects, observations, poses,
                      ambiguity_margin=0. if method == 'omni_no_ambiguity' else .25)
        else:
            geometry_associate(objects, observations, poses)
        predicted = np.zeros(valid.shape, np.int32)
        for obs, selected in zip(observations, selected_masks):
            number = numbers.setdefault(obs['object_id'], len(numbers) + 1)
            predicted[selected] = number
            obs['numeric_object_id'] = number
        np.savez_compressed(output / 'frames' / (frame + '.npz'), instance=predicted, valid=valid)
        frames.append(dict(frame=frame, observations=len(observations), rejected_small_support=rejected,
                           valid_depth_samples=int(valid.sum())))
        if k % 50 == 0:
            print(method, mode, sequence.name, k, 'objects', len(objects), flush=True)
    document = dict(frame='flat_world', map_revision='provided_poses', objects=objects,
                    numeric_object_ids=numbers, frames=frames)
    (output / 'objects.json').write_text(json.dumps(document, indent=2))
    (output / 'input_sha256.json').write_text(json.dumps(hashes, indent=2))
    protocol.update(complete=True, frames_processed=len(frames), wall_s=time.monotonic() - started,
                    object_records=len(objects), confirmed_records=sum(o['confirmed'] for o in objects))
    (output / 'run.json').write_text(json.dumps(protocol, indent=2))
    return protocol


def instance_metrics(contingency):
    """Class-agnostic sequence-pixel instance matching; 0 denotes background/miss."""
    gt_counts, pred_counts = Counter(), Counter()
    for (gt, pred), count in contingency.items():
        if gt > 0:
            gt_counts[gt] += count
        if pred > 0:
            pred_counts[pred] += count
    matches = []
    for (gt, pred), intersection in contingency.items():
        if gt > 0 and pred > 0:
            iou = intersection / (gt_counts[gt] + pred_counts[pred] - intersection)
            if iou > .5:
                matches.append((gt, pred, iou))
    # Disjoint label partitions imply at most one IoU>0.5 match per instance.
    assert len({x[0] for x in matches}) == len(matches) == len({x[1] for x in matches})
    tp, ng, npred = len(matches), len(gt_counts), len(pred_counts)
    denominator = .5 * (ng + npred)
    overlap = sum(c for (g, p), c in contingency.items() if g > 0 and p > 0)
    return dict(gt_visible_instances=ng, predicted_instances=npred, tp=tp, fp=npred-tp, fn=ng-tp,
                sequence_pixel_instance_pq=sum(m[2] for m in matches)/denominator if denominator else None,
                sq=sum(m[2] for m in matches)/tp if tp else None,
                rq=tp/denominator if denominator else None,
                gt_thing_sample_coverage=overlap/sum(gt_counts.values()) if gt_counts else None,
                matches=[dict(gt_id=g, predicted_id=p, iou=i) for g, p, i in matches])


def evaluate(sequence, run_path, output):
    protocol = json.loads((run_path / 'run.json').read_text())
    if not protocol['complete'] or Path(protocol['sequence']).resolve() != sequence.resolve():
        raise ValueError('run incomplete or wrong sequence')
    input_hashes = json.loads((run_path / 'input_sha256.json').read_text())
    for path, expected in input_hashes.items():
        if sha256(path) != expected:
            raise ValueError(f'input changed since mapping: {path}')
    labels = read_labels(sequence.parent / 'groundtruth_labels.csv')
    ids, _ = frame_index(sequence)
    if len(ids) != protocol['frames_processed']:
        raise ValueError('frame coverage mismatch')
    counts = Counter()
    gt_hashes = {}
    for frame in ids:
        path = sequence / (frame + '_segmentation.png')
        gt_hashes[str(path.resolve())] = sha256(path)
        gt = read_ids(path)[::protocol['depth_stride'], ::protocol['depth_stride']]
        unknown = set(map(int, np.unique(gt))) - set(labels)
        if unknown:
            raise ValueError(f'unmapped GT IDs: {unknown}')
        truth = np.zeros(gt.shape, np.int32)
        for number in np.unique(gt):
            if labels[int(number)]['thing']:
                truth[gt == number] = int(number) + 1
        with np.load(run_path / 'frames' / (frame + '.npz')) as data:
            pred, valid = data['instance'], data['valid']
        depth = np.asarray(Image.open(sequence / (frame + '_depth.tiff')))
        _, expected_valid, _, _ = camera_points(depth, protocol['depth_stride'])
        if pred.shape != truth.shape or not np.array_equal(valid, expected_valid):
            raise ValueError('evaluation sampling domain differs from locked depth domain')
        pairs, n = np.unique(np.column_stack([truth[valid], pred[valid]]), axis=0, return_counts=True)
        counts.update({tuple(map(int, pair)): int(count) for pair, count in zip(pairs, n)})
    result = dict(protocol=protocol, scope='class-agnostic sequence-pixel instance PQ; sampled valid-depth domain',
                  not_official_scannet_ap=True, not_3d_map_pq=True,
                  gt_label_table_sha256=sha256(sequence.parent / 'groundtruth_labels.csv'),
                  gt_segmentation_sha256=gt_hashes, metrics=instance_metrics(counts))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['run', 'evaluate'])
    ap.add_argument('--sequence', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--run', type=Path)
    ap.add_argument('--mode', choices=['predicted', 'oracle_masks'], default='predicted')
    ap.add_argument('--method', choices=['omni_slam', 'geometry_hungarian', 'omni_no_ambiguity'], default='omni_slam')
    ap.add_argument('--stride', type=int, default=4)
    args = ap.parse_args()
    if args.action == 'run':
        result = run(args.sequence, args.output, args.mode, args.method, args.stride)
    else:
        if not args.run:
            ap.error('evaluate requires --run')
        result = evaluate(args.sequence, args.run, args.output)['metrics']
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
