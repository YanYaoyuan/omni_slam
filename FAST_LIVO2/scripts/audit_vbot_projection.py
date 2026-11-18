#!/usr/bin/env python3
"""Plot acquisition-near LiDAR points on rectified images; no calibration fitting."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

import av
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np

from prepare_vbot_bag import time_ns


def audit(entry, output):
    c = entry["calibration"]
    K = np.array(c["K"]).reshape(3, 3)
    Ko = K.copy()
    Ko[:2] /= 3
    maps = cv2.initUndistortRectifyMap(
        K, np.array(c["D"]), np.eye(3), Ko, (640, 360), cv2.CV_32FC1
    )
    T = np.linalg.inv(c["transforms"]["stereo_left"]) @ np.array(
        c["transforms"]["lidar"]
    )
    targets = [entry["first_record_ns"] + int(t * 1e9) for t in (5, 20)]
    samples = [None, None]
    clouds = []
    codec = av.CodecContext.create("hevc", "r")
    codec.thread_count = 1
    cv2.setNumThreads(1)

    def frame(image):
        stamp = int(Fraction(image.pts) * image.time_base * 1e9)
        for i, target in enumerate(targets):
            if samples[i] is None or abs(stamp - target) < abs(samples[i][0] - target):
                samples[i] = (
                    stamp,
                    cv2.remap(
                        image.to_ndarray(format="bgr24"), *maps, cv2.INTER_LINEAR
                    ),
                )

    with Path(entry["path"]).open("rb") as f:
        for _, ch, rec, msg in make_reader(
            f, decoder_factories=[DecoderFactory()]
        ).iter_decoded_messages(topics=["/lidar_points", "/image_left_raw/h265"]):
            if rec.log_time > targets[-1] + 1_000_000_000:
                break
            if ch.topic == "/lidar_points":
                fields = {v.name: v for v in msg.fields}
                dt = np.dtype(
                    {
                        "names": ["x", "y", "z", "t"],
                        "formats": ["<f4"] * 3 + ["<f8"],
                        "offsets": [
                            fields[k].offset for k in ("x", "y", "z", "timestamp")
                        ],
                        "itemsize": msg.point_step,
                    }
                )
                arr = np.ndarray(
                    (msg.height, msg.width),
                    dtype=dt,
                    buffer=bytes(msg.data),
                    strides=(msg.row_step, msg.point_step),
                ).ravel()
                clouds.append(np.column_stack([arr[k] for k in ("x", "y", "z", "t")]))
            else:
                packet = av.Packet(bytes(msg.data))
                packet.pts = time_ns(msg.timestamp)
                packet.dts = packet.pts
                packet.time_base = Fraction(1, 1_000_000_000)
                for f in codec.decode(packet):
                    frame(f)
        for f in codec.decode(None):
            frame(f)
    points = np.concatenate(clouds)
    rows = []
    for index, (stamp, img) in enumerate(samples):
        pts = points[
            np.isfinite(points).all(axis=1) & (abs(points[:, 3] - stamp * 1e-9) < 0.05),
            :3,
        ]
        pc = pts @ T[:3, :3].T + T[:3, 3]
        front = pc[:, 2] > 0.3
        pc = pc[front]
        uv = pc @ Ko.T
        uv = uv[:, :2] / uv[:, 2:]
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 360)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
        for ax in axes:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_xlim(0, 640)
            ax.set_ylim(360, 0)
            ax.axis("off")
        dots = axes[1].scatter(
            uv[valid, 0],
            uv[valid, 1],
            c=pc[valid, 2],
            s=4,
            cmap="turbo",
            vmin=0.3,
            vmax=15,
        )
        fig.colorbar(dots, ax=axes[1], label="Camera depth (m)")
        axes[0].set_title("Rectified left image")
        axes[1].set_title(f"Recorded TF projection: {sum(valid)} visible points")
        fig.suptitle(
            f"{entry['name']} · {5 if index == 0 else 20} s · point times within ±50 ms\nNo motion compensation; visual plausibility check, not calibrated accuracy"
        )
        name = f"{entry['name']}_{index}.png"
        fig.savefig(output / name, dpi=140)
        plt.close(fig)
        rows.append(
            {
                "image_stamp_ns": stamp,
                "near_time_points": len(pts),
                "front_points": len(pc),
                "visible_points": int(sum(valid)),
                "figure": name,
            }
        )
    return {
        "bag": entry["name"],
        "T_camera_lidar": T.tolist(),
        "samples": rows,
        "note": "Recorded TF/K/D; no fitted calibration, no motion compensation or quantitative calibration accuracy claim.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = []
    for e in json.loads(args.inventory.read_text()):
        result.append(audit(e, args.output))
        print(e["name"], "projection plotted", flush=True)
        (args.output / "audit.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
