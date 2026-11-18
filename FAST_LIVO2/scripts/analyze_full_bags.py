#!/usr/bin/env python3
"""Summarize full-bag execution and trajectory diagnostics without claiming GT accuracy."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_experiment import load_mode

MODES = ("lio", "livo", "lidar_scheduled")
COLORS = {"lio": "#64748b", "livo": "#167d8d", "lidar_scheduled": "#d58a20"}


def inspect_map(path, root):
    header = {}
    with path.open("rb") as stream:
        for _ in range(100):
            line = stream.readline(4096).decode("ascii").strip()
            if line and not line.startswith("#"):
                key, *values = line.split()
                header[key] = values
                if key == "DATA":
                    break
        header_bytes = stream.tell()
    points = int(header["POINTS"][0])
    point_bytes = sum(int(s) * int(c) for s, c in zip(header["SIZE"], header["COUNT"]))
    return {
        "path": str(path.relative_to(root)),
        "points": points,
        "size_bytes": path.stat().st_size,
        "binary_size_valid": bool(
            header["DATA"] == ["binary"]
            and header_bytes + points * point_bytes == path.stat().st_size
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.experiment.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = json.loads((root / "status.json").read_text())
    summary = {
        "run": status,
        "bags": [],
        "interpretation": [
            "Source completeness, transport completion and physical trajectory quality are separate checks.",
            "All source video packets are decoded; the recorded stride selects visual input frames.",
            "Acquisition-time reordering and incomplete leading scan exclusion are part of this offline protocol.",
            "No independent ground truth. Neither vendor odometry nor mode separation establishes accuracy.",
            "Workers share CPU. Estimator update timings exclude video decoding and are not isolated runtime benchmarks.",
        ],
    }
    loaded_bags = []
    for result in status["results"]:
        folder = root / result["bag"]
        if not (folder / "input_audit.json").exists():
            summary["bags"].append(
                {"result": result, "analysis_error": "input audit absent"}
            )
            continue
        audit = json.loads((folder / "input_audit.json").read_text())
        row = {"result": result, "audit": audit, "modes": {}, "errors": {}}
        loaded = {}
        for mode in MODES:
            try:
                trajectory, local, timing, metrics = load_mode(folder, mode)
                dt = np.diff(trajectory[:, 0])
                if np.any(dt <= 0):
                    raise ValueError("non-increasing final pose timestamps")
                step = np.linalg.norm(np.diff(trajectory[:, 1:4], axis=0), axis=1)
                speed = step / dt
                metrics.update(
                    speed_p99_m_s=float(np.percentile(speed, 99)),
                    speed_max_m_s=float(np.max(speed)),
                    max_step_m=float(np.max(step)),
                    local_z_range_m=float(np.ptp(local[:, 2])),
                    max_quaternion_norm_error=float(
                        np.max(np.abs(np.linalg.norm(trajectory[:, 4:8], axis=1) - 1.0))
                    ),
                    start_after_first_record_s=float(
                        trajectory[0, 0] - audit["source"]["first_record_ns"] * 1e-9
                    ),
                    end_before_last_record_s=float(
                        audit["source"]["last_record_ns"] * 1e-9 - trajectory[-1, 0]
                    ),
                )
                excessive = np.flatnonzero(speed > 10.0)
                metrics["first_speed_over_10m_s_after_first_pose_s"] = (
                    float(trajectory[excessive[0] + 1, 0] - trajectory[0, 0])
                    if len(excessive)
                    else None
                )
                metrics["catastrophic_trajectory_flag"] = bool(
                    metrics["speed_p99_m_s"] > 10.0
                    or metrics["path_length_m"] > 100_000
                )
                metrics["quality_note"] = (
                    "Localization failed: catastrophic trajectory scale/speed despite finite states."
                    if metrics["catastrophic_trajectory_flag"]
                    else "No catastrophic scale/speed flag; accuracy and accumulated drift remain unverified."
                )
                metrics["transport_audit"] = json.loads(
                    (folder / mode / "offline_transport.json").read_text()
                )
                metrics["map_artifacts"] = [
                    inspect_map(p, root)
                    for p in sorted((folder / mode / "Log/pcd").glob("*.pcd"))
                ]
                row["modes"][mode] = metrics
                loaded[mode] = (trajectory, local)
            except Exception as error:
                row["errors"][mode] = str(error)
        if "livo" in loaded and "lidar_scheduled" in loaded:
            a, al = loaded["livo"]
            b, bl = loaded["lidar_scheduled"]
            common = (a[:, 0] >= b[0, 0]) & (a[:, 0] <= b[-1, 0])
            t = a[common, 0]
            b_interp = np.stack(
                [np.interp(t, b[:, 0], bl[:, i]) for i in range(3)], axis=1
            )
            separation = np.linalg.norm(al[common] - b_interp, axis=1)
            row["matched_schedule_comparison"] = {
                "common_samples": len(t),
                "identical_final_timestamps": bool(np.array_equal(a[:, 0], b[:, 0])),
                "median_relative_path_separation_m": float(np.median(separation)),
                "endpoint_relative_path_separation_m": float(separation[-1]),
                "note": "First-pose-frame trajectory separation; not error versus ground truth.",
            }
        summary["bags"].append(row)
        loaded_bags.append((result["bag"], loaded))
    summary["totals"] = {
        "bags": len(status["results"]),
        "source_duration_s": sum(
            b["result"].get("duration_s", 0) for b in summary["bags"]
        ),
        "completed_modes": sum(len(b.get("modes", {})) for b in summary["bags"]),
        "catastrophically_flagged_modes": sum(
            m.get("catastrophic_trajectory_flag", False)
            for b in summary["bags"]
            for m in b.get("modes", {}).values()
        ),
        "source_counts": {
            key: sum(
                b.get("audit", {}).get("input_counts", {}).get(key, 0)
                for b in summary["bags"]
            )
            for key in (
                "imu",
                "lidar",
                "video_packets",
                "decoded_frames",
                "image",
                "video_decode_errors",
            )
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    if loaded_bags:
        plt.rcParams.update(
            {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
        )
        rows = (len(loaded_bags) + 1) // 2
        for plot in ("xy", "z"):
            fig, axes = plt.subplots(
                rows, 2, figsize=(13, rows * 4), layout="constrained", squeeze=False
            )
            for ax, (name, loaded) in zip(axes.flat, loaded_bags):
                for mode, (traj, local) in loaded.items():
                    x, y = (
                        (local[:, 0], local[:, 1])
                        if plot == "xy"
                        else (traj[:, 0] - traj[0, 0], local[:, 2])
                    )
                    ax.plot(x, y, color=COLORS[mode], label=mode, linewidth=1.1)
                ax.set_title(name)
                flagged = [
                    mode
                    for mode, m in next(
                        b for b in summary["bags"] if b["result"]["bag"] == name
                    )["modes"].items()
                    if m["catastrophic_trajectory_flag"]
                ]
                if flagged:
                    ax.set_title(
                        name + "\nFLAGGED: " + ", ".join(flagged), color="#a4262c"
                    )
                ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 4))
                ax.set_xlabel(
                    "X in initial IMU frame (m)"
                    if plot == "xy"
                    else "Time from first valid pose (s)"
                )
                ax.set_ylabel("Y (m)" if plot == "xy" else "Z in initial IMU frame (m)")
                if plot == "xy":
                    ax.set_aspect("equal", adjustable="datalim")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
            for ax in list(axes.flat)[len(loaded_bags) :]:
                ax.set_visible(False)
            fig.suptitle(
                f"VBot complete-bag experiment · ROS 2 Humble · image stride {status['image_stride']}\n"
                "No independent ground truth; trajectory differences are not accuracy errors"
            )
            fig.savefig(output / f"trajectories_{plot}.png", dpi=150)
            plt.close(fig)
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    main()
