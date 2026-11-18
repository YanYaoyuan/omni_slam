#!/usr/bin/env python3
"""Build a car-object prototype from complete LIVO archives and actual camera frames."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import av
import cv2
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
import open3d as o3d
import torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    maskrcnn_resnet50_fpn_v2,
)
import yaml
from map_quality import trajectory_poses
from revisit_geometry import poses_from_rows
from prepare_vbot_bag import time_ns
from omni_semantic_map.objects import associate, project


def images_at(entry, targets, maps):
    """One sequential decode; select nearest frames without changing acquisition time."""
    codec = av.CodecContext.create("hevc", "r")
    codec.thread_count = 1
    decoders = {}
    factory = DecoderFactory()
    pending = 0
    previous = None
    with Path(entry["path"]).open("rb") as stream:
        records = make_reader(stream).iter_messages(topics=["/image_left_raw/h265"])

        def frames():
            for schema, channel, record in records:
                if channel.id not in decoders:
                    decoders[channel.id] = factory.decoder_for(
                        channel.message_encoding, schema
                    )
                msg = decoders[channel.id](record.data)
                packet = av.Packet(bytes(msg.data))
                packet.pts = packet.dts = time_ns(msg.timestamp)
                packet.time_base = Fraction(1, 1_000_000_000)
                try:
                    yield from codec.decode(packet)
                except av.error.FFmpegError:
                    continue
            yield from codec.decode(None)

        for frame in frames():
            stamp = float(Fraction(frame.pts) * frame.time_base)
            while pending < len(targets) and stamp >= targets[pending]:
                chosen = (stamp, frame)
                if previous is not None and abs(previous[0] - targets[pending]) < abs(
                    stamp - targets[pending]
                ):
                    chosen = previous
                image = cv2.remap(
                    chosen[1].to_ndarray(format="bgr24"), *maps, cv2.INTER_LINEAR
                )
                yield pending, chosen[0], image
                pending += 1
            previous = (stamp, frame)
            if pending == len(targets):
                return


def depth_support(points, pixels, depth, valid, box, mask=None):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    # Interior upper/middle body reduces floor and neighbouring-vehicle contamination.
    keep = (
        valid
        & (depth > 1)
        & (depth < 18)
        & (pixels[:, 0] > x1 + 0.12 * w)
        & (pixels[:, 0] < x2 - 0.12 * w)
        & (pixels[:, 1] > y1 + 0.12 * h)
        & (pixels[:, 1] < y2 - 0.23 * h)
    )
    if mask is not None:
        uv = np.nan_to_num(pixels, nan=-1).astype(int)
        inside = (
            valid
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < mask.shape[1])
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < mask.shape[0])
        )
        masked = np.zeros(len(points), dtype=bool)
        masked[inside] = mask[uv[inside, 1], uv[inside, 0]]
        keep &= masked
    ids = np.flatnonzero(keep)
    if len(ids) < 8:
        return None
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[ids]))
    labels = np.asarray(
        cloud.cluster_dbscan(eps=0.65, min_points=4, print_progress=False)
    )
    candidates = []
    for label in set(labels) - {-1}:
        selected = ids[labels == label]
        if len(selected) < 8:
            continue
        extent = np.ptp(points[selected], axis=0)
        if np.max(extent) > 6 or np.max(extent) < 0.25:
            continue
        center = np.median(pixels[selected], axis=0)
        score = np.linalg.norm(
            (center - [(x1 + x2) / 2, (y1 + y2) / 2]) / [w, h]
        ) + 0.01 * np.median(depth[selected])
        candidates.append((score, selected))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.05:
        return None
    return candidates[0][1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ["inventory", "source", "maps", "weights", "output"]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--bag", default="rosbag2245")
    ap.add_argument("--period", type=float, default=2.0)
    ap.add_argument("--detector", choices=["ssdlite", "maskrcnn"], default="maskrcnn")
    ap.add_argument("--score", type=float, default=0.6)
    ap.add_argument("--limit-frames", type=int, default=0)
    args = ap.parse_args()
    if args.period <= 0 or not 0 < args.score <= 1 or args.limit_frames < 0:
        ap.error("period must be positive, score in (0,1], limit-frames nonnegative")
    if args.output.exists():
        ap.error("output exists; retain previous experiment")
    args.output.mkdir(parents=True)
    (args.output / "evidence").mkdir()
    (args.output / "frames").mkdir()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    av.logging.set_level(av.logging.FATAL)
    weights_sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    expected = "73cbd019" if args.detector == "maskrcnn" else "a79551df"
    if not weights_sha.startswith(expected):
        raise ValueError("unexpected official checkpoint")
    model = (
        maskrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=None, min_size=360, max_size=640
        )
        if args.detector == "maskrcnn"
        else ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
    )
    model.load_state_dict(
        torch.load(args.weights, map_location="cpu", weights_only=True)
    )
    model.eval()
    entry = next(
        e for e in json.loads(args.inventory.read_text()) if e["name"] == args.bag
    )
    cfg = yaml.safe_load((args.source.parent / "livo.yaml").read_text())["/**"][
        "ros__parameters"
    ]
    rows = np.atleast_1d(
        np.genfromtxt(args.source / "keyframes.csv", delimiter=",", names=True)
    )
    native = poses_from_rows(rows)
    trajectory = np.loadtxt(args.source / "trajectory_final.tum")
    mapped = np.load(args.maps / "keyframe_poses.npz")
    poses = mapped["after"]
    assert np.array_equal(mapped["stamps"], rows["stamp"])
    assert np.array_equal(rows["id"], np.arange(len(rows)))
    map_result = json.loads((args.maps / "result.json").read_text())
    if (
        map_result["source_index_sha256"]
        != hashlib.sha256((args.source / "keyframes.csv").read_bytes()).hexdigest()
    ):
        raise ValueError(
            "map poses and source keyframes do not belong to the same archive"
        )
    ids = []
    last = -np.inf
    for i, row in enumerate(rows):
        if row["stamp"] - last >= args.period:
            ids.append(i)
            last = row["stamp"]
    if args.limit_frames:
        ids = ids[: args.limit_frames]
    c = entry["calibration"]
    kraw = np.array(c["K"]).reshape(3, 3)
    k = kraw.copy()
    k[0] *= 640 / c["width"]
    k[1] *= 360 / c["height"]
    rect = cv2.initUndistortRectifyMap(
        kraw, np.array(c["D"]), np.eye(3), k, (640, 360), cv2.CV_32FC1
    )
    ext = cfg["extrin_calib"]
    il = np.eye(4)
    il[:3, :3] = np.array(ext["extrinsic_R"]).reshape(3, 3)
    il[:3, 3] = ext["extrinsic_T"]
    cl = np.eye(4)
    cl[:3, :3] = np.array(ext["Rcl"]).reshape(3, 3)
    cl[:3, 3] = ext["Pcl"]
    ci = cl @ np.linalg.inv(il)
    objects = []
    frames = []
    start = time.monotonic()
    for sequence, image_stamp, image in images_at(entry, rows["stamp"][ids], rect):
        i = ids[sequence]
        row = rows[i]
        offset = image_stamp - row["stamp"]
        if (
            abs(offset) > 0.04
            or not trajectory[0, 0] <= image_stamp <= trajectory[-1, 0]
        ):
            frames.append({"keyframe_id": int(row["id"]), "skipped": "image_time"})
            continue
        image_pose = trajectory_poses(trajectory, [image_stamp])[0]
        cv2.imwrite(
            str(args.output / "frames" / f"keyframe_{int(row['id']):05d}.jpg"), image
        )
        camera_from_keyframe = ci @ np.linalg.inv(image_pose) @ native[i]
        cloud = o3d.io.read_point_cloud(
            str(args.source / "keyframes" / f"{int(row['id'])}.pcd")
        )
        points = np.asarray(cloud.points)
        pixels, depth, valid = project(points, camera_from_keyframe, k)
        tensor = (
            torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy())
            .permute(2, 0, 1)
            .float()
            / 255
        )
        with torch.inference_mode():
            prediction = model([tensor])[0]
        observations = []
        annotated = image.copy()
        detections = 0
        for detection_index, (box, label, score) in enumerate(
            zip(
                prediction["boxes"].numpy(),
                prediction["labels"].numpy(),
                prediction["scores"].numpy(),
            )
        ):
            if label != 3 or score < args.score:
                continue
            mask = None
            if "masks" in prediction:
                mask = cv2.erode(
                    (prediction["masks"][detection_index, 0].numpy() > 0.7).astype(
                        np.uint8
                    ),
                    np.ones((3, 3), np.uint8),
                ).astype(bool)
            detections += 1
            selected = depth_support(points, pixels, depth, valid, box, mask)
            if selected is None:
                continue
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(640, x2)
            y2 = min(360, y2)
            crop = image[y1:y2, x1:x2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist(
                [hsv],
                [0, 1, 2],
                (mask[y1:y2, x1:x2].astype(np.uint8) if mask is not None else None),
                [8, 4, 4],
                [0, 180, 0, 256, 0, 256],
            ).ravel()
            hist /= max(hist.sum(), 1)
            evidence = f"evidence/keyframe_{int(row['id']):05d}.jpg"
            observations.append(
                {
                    "keyframe_id": int(row["id"]),
                    "stamp": float(row["stamp"]),
                    "image_stamp": image_stamp,
                    "image_offset_ms": offset * 1000,
                    "label": "car",
                    "score": float(score),
                    "box": box.tolist(),
                    "local_position": np.median(points[selected], axis=0).tolist(),
                    "support_points": len(selected),
                    "appearance": hist.tolist(),
                    "evidence": evidence,
                }
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                annotated,
                f"car {score:.2f} / {len(selected)} points",
                (x1, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 220, 0),
                1,
            )
            for u, v in pixels[selected].astype(int):
                cv2.circle(annotated, (u, v), 2, (0, 0, 255), -1)
        associate(objects, observations, poses)
        for obs in observations:
            cv2.putText(
                annotated,
                obs["object_id"],
                (int(obs["box"][0]), min(350, int(obs["box"][3]) + 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )
        if observations:
            cv2.imwrite(str(args.output / observations[0]["evidence"]), annotated)
        frames.append(
            {
                "keyframe_id": int(row["id"]),
                "stamp": float(row["stamp"]),
                "image_stamp": image_stamp,
                "detections": detections,
                "supported_observations": len(observations),
            }
        )
        if sequence % 20 == 0:
            print(
                sequence + 1,
                "/",
                len(ids),
                "objects",
                len(objects),
                "confirmed",
                sum(o["confirmed"] for o in objects),
                flush=True,
            )
    doc = {
        "schema_version": 1,
        "bag": args.bag,
        "frame": "offline_livo_map",
        "map_revision": hashlib.sha256(
            (args.maps / "keyframe_poses.npz").read_bytes()
        ).hexdigest(),
        "keyframe_source": str(args.source.resolve()),
        "keyframe_index_sha256": hashlib.sha256(
            (args.source / "keyframes.csv").read_bytes()
        ).hexdigest(),
        "source_map": str((args.maps / "map_loop_corrected.pcd").resolve()),
        "detector": args.detector + " COCO_V1",
        "weights_sha256": weights_sha,
        "model_source": "https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection."
        + (
            "maskrcnn_resnet50_fpn_v2"
            if args.detector == "maskrcnn"
            else "ssdlite320_mobilenet_v3_large"
        )
        + ".html",
        "parameters": {
            "period_s": args.period,
            "score": args.score,
            "association_radius_m": 1.0,
            "appearance_distance_limit": 0.35,
            "minimum_observations": 3,
            "maximum_position_spread_p90_m": 0.8,
        },
        "objects": objects,
        "frames": frames,
        "requested_frames": len(ids),
        "decoded_target_frames": len(frames),
        "wall_s": time.monotonic() - start,
        "limitations": [
            "Car class prototype; confirmed means repeated geometric observations, not ground-truth identity.",
            "Surface positions are partial-view estimates, not full object centers.",
            "Potentially moving cars require further dynamic filtering.",
            "Offline corrected map frame is not the live TF map frame.",
        ],
    }
    (args.output / "objects.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2)
    )
    print(
        "complete",
        len(frames),
        "frames",
        len(objects),
        "objects",
        sum(o["confirmed"] for o in objects),
        "confirmed",
        flush=True,
    )


if __name__ == "__main__":
    main()
