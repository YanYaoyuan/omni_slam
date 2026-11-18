#!/usr/bin/env python3
"""Convert selected original NTU ROS1 sensors to the deterministic native transport.

Ground truth is deliberately excluded. Timestamps and point fields are preserved;
the published NTU -0.1s LiDAR header offset remains an estimator parameter.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct
import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
import yaml


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bag', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = Path(__file__).resolve().parents[1] / 'config/upstream_ros1'
    camera = yaml.safe_load((config / 'camera_NTU_VIRAL.yaml').read_text())
    k = np.array([[camera['cam_fx'], 0, camera['cam_cx']],
                  [0, camera['cam_fy'], camera['cam_cy']], [0, 0, 1.]])
    distortion = [camera[f'cam_d{i}'] for i in range(4)] + [0.]
    width, height = camera['cam_width'], camera['cam_height']
    maps = cv2.initUndistortRectifyMap(k, np.array(distortion), None, k,
        (width, height), cv2.CV_32FC1)
    cv2.setNumThreads(1)
    store = get_typestore(Stores.ROS2_HUMBLE)
    topics = {'/imu/imu': 1, '/os1_cloud_node1/points': 2, '/left/image_raw': 3}
    counts = Counter(); stamps = defaultdict(list); first = {}; bag_times = []; initial_acceleration = []
    target = args.output / 'sensors.bin'
    with Reader(args.bag) as bag, target.open('wb') as stream:
        connections = [c for c in bag.connections if c.topic in topics]
        if {c.topic for c in connections} != set(topics):
            raise ValueError('required NTU topics absent')
        stream.write(b'OMNILIVO1')
        for connection, recorded_time, raw in bag.messages(connections=connections):
            kind = topics[connection.topic]
            payload = store.ros1_to_cdr(raw, connection.msgtype)
            message = store.deserialize_cdr(payload, connection.msgtype)
            stamp = message.header.stamp.sec*10**9 + message.header.stamp.nanosec
            stamps[connection.topic].append(stamp); bag_times.append(recorded_time)
            if kind == 1 and counts[kind] < 1000:
                a = message.linear_acceleration
                initial_acceleration.append(np.linalg.norm([a.x,a.y,a.z]))
            if kind == 3:
                if message.encoding not in ('mono8', '8UC1') or (message.width, message.height) != (width, height):
                    raise ValueError('unexpected original NTU image format')
                raw_image = np.frombuffer(message.data, np.uint8).reshape(height, message.step)[:, :width]
                image = cv2.cvtColor(cv2.remap(raw_image, *maps, cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR)
                payload = struct.pack('<qII', stamp, width, height) + image.tobytes()
                if not counts[kind]:
                    cv2.imwrite(str(args.output / 'first_rectified.jpg'), image)
            elif kind == 2 and not counts[kind]:
                fields = {f.name: dict(offset=f.offset, datatype=f.datatype, count=f.count) for f in message.fields}
                if not {'x','y','z','intensity','t','ring'}.issubset(fields):
                    raise ValueError('Ouster fields missing')
                if message.is_bigendian or fields['t']['datatype'] != 6:
                    raise ValueError('expected little-endian uint32 Ouster relative time')
                offsets = np.ndarray((message.height, message.width), dtype='<u4', buffer=message.data,
                    offset=fields['t']['offset'], strides=(message.row_step, message.point_step))
                first['lidar'] = dict(fields=fields, points=message.width*message.height,
                    relative_time_min_s=float(offsets.min()*1e-9), relative_time_max_s=float(offsets.max()*1e-9))
            if connection.topic not in first:
                first[connection.topic] = dict(frame_id=message.header.frame_id, stamp_ns=stamp)
            stream.write(struct.pack('<BI', kind, len(payload))); stream.write(payload)
            counts[kind] += 1
            if kind == 2 and counts[kind] % 200 == 0:
                print('converted lidar frames', counts[kind], flush=True)
        stream.write(struct.pack('<BIQQQ', 0, 24, counts[1], counts[2], counts[3]))
    audit = dict(source=str(args.bag.resolve()), source_sha256=digest(args.bag),
        transport_sha256=digest(target), first=first, input_counts=dict(counts),
        record_duration_s=(max(bag_times)-min(bag_times))*1e-9,
        preprocessing='original timestamps; no restamping; mono8 rectified with pinned upstream K,D then replicated to BGR',
        ground_truth_passed_to_estimator=False, topics={})
    audit['first_1000_imu_acceleration_norm_median'] = float(np.median(initial_acceleration))
    for topic, values in stamps.items():
        times = np.array(values, dtype=np.int64); gaps = np.diff(times)*1e-9
        audit['topics'][topic] = dict(count=len(times), first_ns=int(times[0]), last_ns=int(times[-1]),
            nonpositive_gaps=int((gaps<=0).sum()), interval_p50_s=float(np.median(gaps)),
            max_gap_s=float(gaps.max()))
    (args.output / 'input_audit.json').write_text(json.dumps(audit, indent=2))
    if any(v['nonpositive_gaps'] for v in audit['topics'].values()):
        raise ValueError('nonmonotonic original sensor stamps; inspect before estimation')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
