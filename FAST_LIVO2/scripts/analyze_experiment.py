#!/usr/bin/env python3
"""Summarize execution and matched-schedule visual ablation, without claiming GT accuracy."""

import argparse
import json
from pathlib import Path
import numpy as np
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def load_mode(folder, mode):
    out = folder / mode
    data = np.loadtxt(out / "trajectory.txt", ndmin=2)
    timing = np.atleast_1d(np.genfromtxt(out / "timing.csv", delimiter=",", names=True))
    if len(data) != len(timing):
        raise ValueError("trajectory/timing rows differ")
    valid = (timing["points"] > 0) & np.isfinite(data).all(axis=1)
    final_type = 2 if mode == "lio" else 1
    selected = valid & (data[:, -1] == final_type)
    config = folder / (mode + ".yaml")
    if mode != "lio" and config.exists():
        params = yaml.safe_load(config.read_text())["/**"]["ros__parameters"]
        if params.get("vio", {}).get("lidar_timed_fusion", False):
            selected = valid & np.r_[np.diff(data[:, 0]) != 0, True]
    trajectory = data[selected, :8]
    if len(trajectory) < 2:
        raise ValueError("no valid final trajectory")
    np.savetxt(out / "trajectory_final.tum", trajectory, fmt="%.12f")
    q = Rotation.from_quat(trajectory[0, 4:8]).as_matrix()
    local = (trajectory[:, 1:4] - trajectory[0, 1:4]) @ q
    metrics = json.loads((out / "metrics.json").read_text())
    metrics.update(
        valid_final_poses=len(trajectory),
        valid_duration_s=float(trajectory[-1, 0] - trajectory[0, 0]),
        path_length_m=float(
            np.linalg.norm(np.diff(trajectory[:, 1:4], axis=0), axis=1).sum()
        ),
        finite_state=bool(np.isfinite(data).all()),
        valid_timing={},
    )
    for kind, name in [(1, "visual"), (2, "lidar")]:
        values = timing[(timing["update_type"] == kind) & valid]["processing_ms"]
        if len(values):
            metrics["valid_timing"][name] = {
                "count": len(values),
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "p99_ms": float(np.percentile(values, 99)),
            }
    visual = timing[(timing["update_type"] == 1) & valid]
    if len(visual):
        metrics["visual_evidence"] = {
            "frames": len(visual),
            "frames_with_support": int(np.sum(visual["visual_points"] > 0)),
            "median_support_points": float(np.median(visual["visual_points"])),
            "frames_with_position_correction_over_1um": int(
                np.sum(visual["position_correction_m"] > 1e-6)
            ),
            "median_position_correction_mm": float(
                np.median(visual["position_correction_m"]) * 1e3
            ),
            "max_position_correction_mm": float(
                np.max(visual["position_correction_m"]) * 1e3
            ),
            "median_rotation_correction_deg": float(
                np.median(visual["rotation_correction_rad"]) * 180 / np.pi
            ),
        }
    return trajectory, local, timing, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    folder = args.experiment.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    modes = [
        m
        for m in ["lio", "livo", "lidar_scheduled"]
        if (folder / m / "metrics.json").exists()
    ]
    loaded = {m: load_mode(folder, m) for m in modes}
    summary = {
        "experiment": str(folder),
        "modes": {m: v[3] for m, v in loaded.items()},
        "interpretation": [
            "Execution validation on one segment; not an accuracy benchmark.",
            "LIO-only and LIVO use different scan-cut/update schedules.",
            "lidar_scheduled preserves the image-timed scan cuts while disabling the photometric EKF update.",
            "Processing times exclude input video decode, ROS delivery and IMU propagation.",
            "No independent ground truth is available. Vendor odometry is not used as ground truth.",
        ],
    }
    if "livo" in loaded and "lidar_scheduled" in loaded:
        a, al, _, _ = loaded["livo"]
        b, bl, _, _ = loaded["lidar_scheduled"]
        keep = (a[:, 0] >= b[0, 0]) & (a[:, 0] <= b[-1, 0])
        t = a[keep, 0]
        b_interp = np.stack([np.interp(t, b[:, 0], bl[:, i]) for i in range(3)], axis=1)
        d = np.linalg.norm(al[keep] - b_interp, axis=1)
        summary["matched_schedule_comparison"] = {
            "common_samples": len(t),
            "median_relative_path_separation_m": float(np.median(d)),
            "endpoint_relative_path_separation_m": float(d[-1]),
            "note": "Each trace is expressed in its first-pose frame; separation is not error versus truth.",
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {"lio": "#64748b", "livo": "#167d8d", "lidar_scheduled": "#e09539"}
    labels = {
        "lio": "LIO only (full scans)",
        "livo": "LIVO (visual update on)",
        "lidar_scheduled": "Image-timed LiDAR (visual update off)",
    }
    for mode, (traj, local, timing, _) in loaded.items():
        axes[0, 0].plot(
            local[:, 0],
            local[:, 1],
            color=colors[mode],
            label=labels[mode],
            linewidth=1.7,
        )
        axes[0, 1].plot(
            traj[:, 0] - traj[0, 0],
            local[:, 2],
            color=colors[mode],
            label=labels[mode],
            linewidth=1.4,
        )
        visual = timing[(timing["update_type"] == 1) & (timing["points"] > 0)]
        if len(visual):
            t = visual["stamp"] - traj[0, 0]
            axes[1, 0].plot(
                t,
                visual["position_correction_m"] * 1e3,
                color=colors[mode],
                label=labels[mode],
                linewidth=1,
            )
            axes[1, 1].plot(
                t,
                visual["visual_points"],
                color=colors[mode],
                label=labels[mode],
                linewidth=1,
            )
    axes[0, 0].set(
        title="Trajectory shape (each starts in its first-pose frame)",
        xlabel="X (m)",
        ylabel="Y (m)",
    )
    axes[0, 0].axis("equal")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(
        title="Z in each initial IMU frame (not altitude truth)",
        xlabel="Time from first valid pose (s)",
        ylabel="Z (m)",
    )
    axes[1, 0].set(
        title="Position correction during the visual update",
        xlabel="Time (s)",
        ylabel="Correction norm (mm)",
    )
    axes[1, 1].set(
        title="Visual map points supporting each frame",
        xlabel="Time (s)",
        ylabel="Points",
    )
    for ax in axes.ravel():
        ax.grid(alpha=0.18)
    fig.suptitle(
        "VBot garage segment · ROS 2 Humble execution and visual ablation\nNo independent ground truth — these curves do not establish accuracy improvement",
        fontsize=13,
    )
    fig.savefig(output / "experiment_overview.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                "modes": {
                    m: {
                        k: v[3].get(k)
                        for k in [
                            "valid_final_poses",
                            "valid_duration_s",
                            "valid_timing",
                            "visual_evidence",
                        ]
                    }
                    for m, v in loaded.items()
                },
                "comparison": summary.get("matched_schedule_comparison"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
