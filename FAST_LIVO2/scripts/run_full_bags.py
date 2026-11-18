#!/usr/bin/env python3
"""Stream complete VBot bags to native estimator workers with explicit visual selection.

The workers use the regular ROS 2 callbacks/estimator, with synchronous draining
and verified input counts instead of DDS delivery. Images are decoded once and
never written as large uncompressed intermediate bags. Source bags are read-only.
"""

import argparse
from collections import Counter, defaultdict
import copy
from fractions import Fraction
import hashlib
import heapq
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import time

import av
import cv2
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
import yaml

from prepare_vbot_bag import time_ns

MODES = ("lio", "livo", "lidar_scheduled")


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    temporary.replace(path)


def worker_limit():
    resource.setrlimit(resource.RLIMIT_AS, (24 << 30, 24 << 30))


def summarize_mode(folder, mode, counts, wall):
    out = folder / mode
    transport = json.loads((out / "offline_transport.json").read_text())
    if not transport["completed"] or transport["exit_code"]:
        raise RuntimeError("native worker did not complete")
    trajectory = np.loadtxt(out / "trajectory.txt", ndmin=2)
    timing = np.atleast_1d(np.genfromtxt(out / "timing.csv", delimiter=",", names=True))
    if len(trajectory) != len(timing) or not np.isfinite(trajectory).all():
        raise RuntimeError("invalid or nonfinite trajectory")
    params = yaml.safe_load((folder / (mode + ".yaml")).read_text())["/**"][
        "ros__parameters"
    ]
    lidar_timed = params.get("vio", {}).get("lidar_timed_fusion", False)
    if mode != "lio" and lidar_timed:
        final = trajectory[np.r_[np.diff(trajectory[:, 0]) != 0, True], :8]
    else:
        final = trajectory[trajectory[:, -1] == (2 if mode == "lio" else 1), :8]
    if len(final) < 2:
        raise RuntimeError("expected estimator updates absent")
    np.savetxt(out / "trajectory_final.tum", final, fmt="%.12f")
    metrics = {
        "mode": mode,
        "transport": "synchronous CDR/image stream; no DDS sensor delivery",
        "exit_code": 0,
        "wall_s": wall,
        "input_counts": dict(counts),
        "final_poses": len(final),
        "trajectory_start_s": float(final[0, 0]),
        "trajectory_end_s": float(final[-1, 0]),
        "path_length_m": float(
            np.linalg.norm(np.diff(final[:, 1:4], axis=0), axis=1).sum()
        ),
        "endpoint_displacement_m": float(
            np.linalg.norm(final[-1, 1:4] - final[0, 1:4])
        ),
        "z_range_m": float(np.ptp(final[:, 3])),
        "peak_rss_mib": transport["peak_rss_mib"],
        "pending_at_eof": {
            k: transport[k] for k in ("pending_imu", "pending_lidar", "pending_image")
        },
        "accuracy_note": "No independent ground truth; trajectory differences are not accuracy errors.",
        "timing_note": "Concurrent workers share CPU; update timings exclude video decode/input and are not isolated runtime benchmarks.",
    }
    atomic_json(out / "metrics.json", metrics)
    return metrics


def configure(entry, folder, width, height, local_map_half_size, overrides=None):
    calib = entry["calibration"]
    source = Path(__file__).resolve().parent.parent / "config/vbot_bag_example.yaml"
    if not source.exists():
        from ament_index_python.packages import get_package_share_directory

        source = (
            Path(get_package_share_directory("fast_livo"))
            / "config/vbot_bag_example.yaml"
        )
    params = yaml.safe_load(source.read_text())["/**"]["ros__parameters"]

    def merge(target, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    merge(params, overrides or {})
    K = np.asarray(calib["K"]).reshape(3, 3)
    Kout = K.copy()
    Kout[0, :] *= width / calib["width"]
    Kout[1, :] *= height / calib["height"]
    if abs(width / height - calib["width"] / calib["height"]) > 1e-6:
        raise ValueError("output aspect ratio differs from source calibration")
    tf = {k: np.asarray(v) for k, v in calib["transforms"].items()}
    T_i_l = np.linalg.inv(tf["lidar_imu"]) @ tf["lidar"]
    T_c_l = np.linalg.inv(tf["stereo_left"]) @ tf["lidar"]
    params["use_sim_time"] = False
    params["local_map"] = {
        "map_sliding_en": local_map_half_size > 0,
        "half_map_size": max(1, local_map_half_size),
        "sliding_thresh": 5.0,
    }
    params["camera"].update(
        width=width,
        height=height,
        scale=1.0,
        fx=float(Kout[0, 0]),
        fy=float(Kout[1, 1]),
        cx=float(Kout[0, 2]),
        cy=float(Kout[1, 2]),
        distortion=[0.0] * 5,
    )
    params["extrin_calib"] = {
        "extrinsic_T": T_i_l[:3, 3].tolist(),
        "extrinsic_R": T_i_l[:3, :3].ravel().tolist(),
        "Rcl": T_c_l[:3, :3].ravel().tolist(),
        "Pcl": T_c_l[:3, 3].tolist(),
    }
    for mode in MODES:
        config = copy.deepcopy(params)
        config["common"]["img_en"] = 0 if mode == "lio" else 1
        config["vio"]["update_enabled"] = mode != "lidar_scheduled"
        config["runtime"]["output_directory"] = str(folder / mode)
        (folder / (mode + ".yaml")).write_text(
            yaml.safe_dump({"/**": {"ros__parameters": config}}, sort_keys=False)
        )
    maps = cv2.initUndistortRectifyMap(
        K, np.asarray(calib["D"]), np.eye(3), Kout, (width, height), cv2.CV_32FC1
    )
    return maps


def run_bag(
    entry,
    root,
    executable,
    width,
    height,
    limit_seconds=None,
    image_stride=3,
    local_map_half_size=60,
    overrides=None,
):
    folder = root / entry["name"]
    folder.mkdir(parents=True, exist_ok=True)
    if entry.get("calibration_status") != "ok":
        raise RuntimeError(entry.get("calibration_error", "calibration absent"))
    completed = {}
    for mode in MODES:
        path = folder / mode / "metrics.json"
        if path.exists():
            native = folder / mode / "offline_transport.json"
            if native.exists() and json.loads(native.read_text()).get("completed"):
                completed[mode] = json.loads(path.read_text())
    if len(completed) == len(MODES):
        print("RESUME COMPLETE", entry["name"], flush=True)
        return json.loads((folder / "batch_result.json").read_text())
    maps = configure(entry, folder, width, height, local_map_half_size, overrides)
    workers = {}
    failures = {}
    started = time.monotonic()
    counts = Counter()
    stamps = defaultdict(list)
    offsets = defaultdict(list)
    video_errors = []
    gnss = Counter()
    reference = []
    pending_events = []
    excluded = Counter()
    event_sequence = 0
    delivered_stamp = -1
    reorder_ns = 1_000_000_000
    selected = [
        "/lidar_imu",
        "/lidar_points",
        "/image_left_raw/h265",
        "/odometry",
        "/gnss/fix",
    ]
    expected = {
        "imu": entry["topics"]["/lidar_imu"]["count"],
        "lidar": entry["topics"]["/lidar_points"]["count"],
        "video_packets": entry["topics"]["/image_left_raw/h265"]["count"],
    }
    first_record = entry["first_record_ns"]
    last_record = first_record
    decoder_factory = DecoderFactory()
    decoders = {}
    codec = av.CodecContext.create("hevc", "r")
    codec.thread_count = 1
    cv2.setNumThreads(1)
    try:
        for index, mode in enumerate(MODES):
            if mode in completed:
                continue
            out = folder / mode
            if out.exists():
                suffix = int(time.time())
                out.rename(folder / f"{mode}_previous_{suffix}")
            out.mkdir()
            env = os.environ.copy()
            env.update(
                ROS_DOMAIN_ID=str(
                    int(os.environ.get("FAST_LIVO_DOMAIN_BASE", "121")) + index
                ),
                ROS_LOCALHOST_ONLY="1",
                ROS_LOG_DIR=str(out / "ros_logs"),
                OMP_NUM_THREADS="3",
                OPENBLAS_NUM_THREADS="1",
                RCUTILS_COLORIZED_OUTPUT="0",
            )
            log = (out / "node.log").open("wb")
            proc = subprocess.Popen(
                [
                    str(executable),
                    "--ros-args",
                    "--params-file",
                    str(folder / (mode + ".yaml")),
                ],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=1 << 20,
                preexec_fn=worker_limit,
            )
            proc.stdin.write(b"OMNILIVO1")
            workers[mode] = {
                "process": proc,
                "log": log,
                "counts": Counter(),
                "failed": False,
            }

        def deliver(kind, payload):
            for mode, worker in workers.items():
                if worker["failed"] or (kind == 3 and mode == "lio"):
                    continue
                proc = worker["process"]
                try:
                    proc.stdin.write(struct.pack("<BI", kind, len(payload)))
                    proc.stdin.write(payload)
                    worker["counts"][kind] += 1
                except (BrokenPipeError, OSError) as error:
                    worker["failed"] = True
                    failures[mode] = (
                        f"worker input failed at record {last_record}: {error}"
                    )

        def flush_events(watermark):
            nonlocal delivered_stamp
            while pending_events and pending_events[0][0] <= watermark:
                stamp, _, kind, payload = heapq.heappop(pending_events)
                if stamp < delivered_stamp:
                    raise RuntimeError("sensor arrival exceeded 1 s reorder window")
                deliver(kind, payload)
                delivered_stamp = stamp

        def emit(kind, payload):
            nonlocal event_sequence
            if kind == 3:
                stamp = struct.unpack_from("<q", payload)[0]
            else:
                sec, nsec = struct.unpack_from("<iI", payload, 4)
                stamp = sec * 1_000_000_000 + nsec
            # Match the validated pilot: exclude an incomplete leading scan and
            # any acquisition outside the recording window, while auditing all
            # original records. Never change the sensor timestamps themselves.
            if stamp < first_record or stamp > entry["last_record_ns"]:
                excluded[kind] += 1
                return
            heapq.heappush(pending_events, (stamp, event_sequence, kind, payload))
            event_sequence += 1

        decoded_frames = 0

        def process_frame(frame):
            nonlocal decoded_frames
            if frame.pts is None:
                raise RuntimeError("video decoder lost acquisition timestamp")
            pts = int(Fraction(frame.pts) * frame.time_base * 1_000_000_000)
            if (frame.width, frame.height) != (
                entry["calibration"]["width"],
                entry["calibration"]["height"],
            ):
                raise RuntimeError("decoded dimensions mismatch calibration")
            decoded_frames += 1
            counts["decoded_frames"] += 1
            if (decoded_frames - 1) % image_stride:
                return
            image = cv2.remap(
                frame.to_ndarray(format="bgr24"), maps[0], maps[1], cv2.INTER_LINEAR
            )
            emit(3, struct.pack("<qII", pts, width, height) + image.tobytes())
            counts["image"] += 1
            stamps["image"].append(pts)
            if decoded_frames == 1:
                cv2.imwrite(str(folder / "sample_undistorted.jpg"), image)

        with Path(entry["path"]).open("rb") as stream:
            reader = make_reader(stream)
            last_report = time.monotonic()
            for schema, channel, record in reader.iter_messages(topics=selected):
                if limit_seconds is not None and record.log_time > first_record + int(
                    limit_seconds * 1e9
                ):
                    break
                last_record = record.log_time
                topic = channel.topic
                if topic in ("/lidar_imu", "/lidar_points"):
                    key, kind = ("imu", 1) if topic == "/lidar_imu" else ("lidar", 2)
                    emit(kind, record.data)
                    counts[key] += 1
                    # Standard ROS2 CDR starts with encapsulation then Header stamp.
                    sec, nsec = struct.unpack_from("<iI", record.data, 4)
                    stamp = sec * 1_000_000_000 + nsec
                    stamps[key].append(stamp)
                    offsets[key].append((record.log_time - stamp) * 1e-6)
                else:
                    if channel.id not in decoders:
                        decoders[channel.id] = decoder_factory.decoder_for(
                            channel.message_encoding, schema
                        )
                    msg = decoders[channel.id](record.data)
                    if topic == "/image_left_raw/h265":
                        counts["video_packets"] += 1
                        if msg.format not in ("h265", "hevc"):
                            raise RuntimeError("unexpected compressed video codec")
                        stamp = time_ns(msg.timestamp)
                        offsets["image"].append((record.log_time - stamp) * 1e-6)
                        packet = av.Packet(bytes(msg.data))
                        packet.pts = stamp
                        packet.dts = stamp
                        packet.time_base = Fraction(1, 1_000_000_000)
                        try:
                            for frame in codec.decode(packet):
                                process_frame(frame)
                        except av.error.FFmpegError as error:
                            counts["video_decode_errors"] += 1
                            if len(video_errors) < 10:
                                video_errors.append(str(error))
                    elif topic == "/gnss/fix":
                        gnss[int(msg.status.status)] += 1
                    else:
                        p, q = msg.pose.pose.position, msg.pose.pose.orientation
                        reference.append(
                            [
                                time_ns(msg.header.stamp) * 1e-9,
                                p.x,
                                p.y,
                                p.z,
                                q.x,
                                q.y,
                                q.z,
                                q.w,
                            ]
                        )
                flush_events(last_record - reorder_ns)
                if time.monotonic() - last_report > 10:
                    state = {
                        "source": entry["name"],
                        "source_elapsed_s": (last_record - first_record) * 1e-9,
                        "source_duration_s": entry["duration_s"],
                        "wall_s": time.monotonic() - started,
                        "counts": dict(counts),
                        "failures": failures,
                    }
                    atomic_json(root / "progress.json", state)
                    print(json.dumps(state, ensure_ascii=False), flush=True)
                    last_report = time.monotonic()
                    # Keep auditing the complete source even if all workers
                    # fail; transport completion is reported separately.
            try:
                for frame in codec.decode(None):
                    process_frame(frame)
            except av.error.FFmpegError as error:
                video_errors.append(str(error))
                counts["video_decode_errors"] += 1
        flush_events(float("inf"))
        if limit_seconds is None:
            for key, value in expected.items():
                if counts[key] != value:
                    raise RuntimeError(
                        f"incomplete source {key}: {counts[key]} != {value}"
                    )
        for mode, worker in workers.items():
            if worker["failed"]:
                continue
            count = worker["counts"]
            payload = struct.pack("<QQQ", count[1], count[2], count[3])
            worker["process"].stdin.write(struct.pack("<BI", 0, len(payload)) + payload)
            worker["process"].stdin.close()
        for mode, worker in workers.items():
            proc = worker["process"]
            try:
                code = proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                code = -9
            worker["log"].close()
            if code:
                failures[mode] = failures.get(mode, f"worker exited {code}")
            else:
                try:
                    completed[mode] = summarize_mode(
                        folder, mode, worker["counts"], time.monotonic() - started
                    )
                except Exception as error:
                    failures[mode] = str(error)
    finally:
        for worker in workers.values():
            proc = worker["process"]
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if not worker["log"].closed:
                worker["log"].close()
    np.savetxt(
        folder / "vendor_odometry.txt",
        np.asarray(reference).reshape(-1, 8),
        fmt="%.12f",
    )

    def stat(values):
        values = np.asarray(values, dtype=np.int64)
        delta = np.diff(values) * 1e-9
        return {
            "count": len(values),
            "first_ns": int(values[0]),
            "last_ns": int(values[-1]),
            "nonmonotonic": int(np.sum(delta <= 0)),
            "max_gap_s": float(delta.max()) if len(delta) else None,
        }

    audit = {
        "source": entry,
        "source_complete": limit_seconds is None,
        "limit_seconds": limit_seconds,
        "image_stride": image_stride,
        "input_counts": dict(counts),
        "expected_source_counts": expected,
        "streams": {k: stat(v) for k, v in stamps.items() if v},
        "video_errors": video_errors,
        "gnss_status_counts": dict(gnss),
        "record_minus_sensor_ms": {
            k: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
            for k, v in offsets.items()
        },
        "timestamp_policy": "unchanged acquisition timestamps; 1 s bounded reorder; acquisitions outside recording window excluded",
        "acquisition_window_exclusions_by_type": dict(excluded),
        "reorder_window_s": 1,
        "schedule": "all modes receive identical LiDAR/IMU events; visual modes receive the same decoded image events",
    }
    atomic_json(folder / "input_audit.json", audit)
    result = {
        "bag": entry["name"],
        "duration_s": entry["duration_s"],
        "source_complete": limit_seconds is None,
        "counts": dict(counts),
        "modes_completed": list(completed),
        "failures": failures,
        "wall_s": time.monotonic() - started,
    }
    atomic_json(folder / "batch_result.json", result)
    return result


def main():
    global MODES
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--executable", required=True, type=Path)
    ap.add_argument(
        "--image-stride",
        default=3,
        type=int,
        help="visual keyframe stride; every video packet is decoded",
    )
    ap.add_argument("--width", default=640, type=int)
    ap.add_argument("--height", default=360, type=int)
    ap.add_argument(
        "--local-map-half-size",
        default=60,
        type=int,
        help="half window in 0.5 m voxels; 0 disables sliding",
    )
    ap.add_argument(
        "--limit-seconds", type=float, help="smoke test only; omit for complete bags"
    )
    ap.add_argument("--only", help="one inventory bag name")
    ap.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    ap.add_argument(
        "--params-override",
        type=Path,
        help="nested ROS parameter overrides (YAML mapping)",
    )
    args = ap.parse_args()
    if len(set(args.modes)) != len(args.modes):
        ap.error("modes must be unique")
    MODES = tuple(args.modes)
    overrides = (
        yaml.safe_load(args.params_override.read_text()) if args.params_override else {}
    )
    if not isinstance(overrides, dict):
        ap.error("parameter overrides must be a mapping")
    if (
        args.width <= 0
        or args.image_stride < 1
        or args.height <= 0
        or args.local_map_half_size < 0
        or (args.limit_seconds is not None and args.limit_seconds <= 0)
    ):
        ap.error("dimensions and optional duration must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries = json.loads(args.inventory.read_text())
    if args.only:
        entries = [e for e in entries if e["name"] == args.only]
        if not entries:
            ap.error("bag name absent from inventory")
    # Smallest first surfaces input/algorithm errors before the long sessions.
    entries.sort(key=lambda e: e["duration_s"])
    executable = args.executable.resolve()
    run = {
        "inventory": str(args.inventory.resolve()),
        "executable": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "width": args.width,
        "height": args.height,
        "image_stride": args.image_stride,
        "local_map_half_size": args.local_map_half_size,
        "parameter_overrides": overrides,
        "modes": list(MODES),
        "limit_seconds": args.limit_seconds,
        "inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        "timestamp_policy": "acquisition-order-1s-recording-window-v1",
        "transport": "synchronous stream",
        "worker_execution": f"{len(MODES)} separate estimator process(es); identical acquisition-time ordering with 1 s bounded reorder",
        "results": [],
    }
    existing = root / "status.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if any(
            old.get(key) != run.get(key)
            for key in (
                "executable_sha256",
                "image_stride",
                "width",
                "height",
                "limit_seconds",
                "inventory_sha256",
                "timestamp_policy",
                "local_map_half_size",
                "parameter_overrides",
                "modes",
            )
        ):
            raise RuntimeError(
                "resume configuration/executable differs; choose a new output directory"
            )
    atomic_json(root / "status.json", run)
    for entry in entries:
        print("START", entry["name"], entry["duration_s"], flush=True)
        try:
            result = run_bag(
                entry,
                root,
                executable,
                args.width,
                args.height,
                args.limit_seconds,
                args.image_stride,
                args.local_map_half_size,
                overrides,
            )
        except Exception as error:
            result = {
                "bag": entry["name"],
                "source_complete": False,
                "error": str(error),
            }
        run["results"].append(result)
        atomic_json(root / "status.json", run)
        print("FINISH", json.dumps(result, ensure_ascii=False), flush=True)
    if any(r.get("error") or r.get("failures") for r in run["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
