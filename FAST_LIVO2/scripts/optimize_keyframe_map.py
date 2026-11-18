#!/usr/bin/env python3
"""Offline geometry-verified loop correction of native IMU-frame RGB keyframes."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from revisit_geometry import (
    gravity_frame,
    poses_from_rows,
    registration_cloud,
    scan_descriptor,
    validate_pair,
    optimize_graph,
    loop_fit_mask,
)


def edge_errors(edges, poses):
    values = []
    for e in edges:
        i, j = e["i"], e["j"]
        measurement = np.array(e["matrix"])
        prediction = np.linalg.inv(poses[j]) @ poses[i]
        vertical = float(
            (poses[j, :3, :3] @ (prediction[:3, 3] - measurement[:3, 3]))[2]
        )
        values.append(
            {
                "i": i,
                "j": j,
                "vertical_error_m": vertical,
                "translation_error_m": float(
                    np.linalg.norm(prediction[:3, 3] - measurement[:3, 3])
                ),
            }
        )
    return {
        "count": len(values),
        "vertical_rmse_m": float(
            np.sqrt(np.mean([v["vertical_error_m"] ** 2 for v in values]))
        )
        if values
        else None,
        "pairs": values,
    }


def corrected_trajectory(trajectory, key_times, base, optimized, gauge):
    correction = optimized @ np.linalg.inv(base)
    t = np.clip(trajectory[:, 0], key_times[0], key_times[-1])
    rotation = Slerp(key_times, Rotation.from_matrix(correction[:, :3, :3]))(
        t
    ).as_matrix()
    translation = np.column_stack(
        [np.interp(t, key_times, correction[:, i, 3]) for i in range(3)]
    )
    original = np.repeat(np.eye(4)[None], len(trajectory), axis=0)
    original[:, :3, :3] = Rotation.from_quat(trajectory[:, 4:8]).as_matrix()
    original[:, :3, 3] = trajectory[:, 1:4]
    aligned = gauge @ original
    result = aligned.copy()
    result[:, :3, :3] = rotation @ aligned[:, :3, :3]
    result[:, :3, 3] = (
        np.einsum("nij,nj->ni", rotation, aligned[:, :3, 3]) + translation
    )

    def tum(poses):
        return np.column_stack(
            [
                trajectory[:, 0],
                poses[:, :3, 3],
                Rotation.from_matrix(poses[:, :3, :3]).as_quat(),
            ]
        )

    return tum(aligned), tum(result)


def export_map(source, rows, poses, path):
    points = []
    colors = []
    for row, pose in zip(rows, poses):
        cloud = o3d.io.read_point_cloud(
            str(source / "keyframes" / f"{int(row['id'])}.pcd")
        )
        if len(cloud.points) != int(row["points"]):
            raise RuntimeError("keyframe PCD count mismatch")
        points.append(np.asarray(cloud.points) @ pose[:3, :3].T + pose[:3, 3])
        colors.append(np.asarray(cloud.colors))
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(np.vstack(points))
    result.colors = o3d.utility.Vector3dVector(np.vstack(colors))
    result = result.voxel_down_sample(0.15)
    if not o3d.io.write_point_cloud(
        str(path), result, write_ascii=False, compressed=False
    ):
        raise RuntimeError("map export failed")
    return len(result.points)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        type=Path,
        required=True,
        help="completed livo folder with keyframes.csv",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--minimum-seconds", type=float, default=90.0)
    ap.add_argument("--max-candidates", type=int, default=2)
    ap.add_argument("--submap-neighbors", type=int, default=2)
    ap.add_argument("--descriptor-distance", type=float, default=0.6)
    ap.add_argument(
        "--loop-fit-half-period-s",
        type=float,
        default=0.0,
        help="optional whole-scan validation split; even time blocks fit loops, odd blocks withheld",
    )
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--allow-prefix", action="store_true")
    args = ap.parse_args()
    if args.submap_neighbors < 0 or args.max_candidates < 1:
        ap.error("submap-neighbors must be nonnegative and max-candidates positive")
    if args.output.exists():
        ap.error("output exists; retain prior experiments and choose another directory")
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True)
    completed = json.loads((source.parent / "batch_result.json").read_text())
    if not completed["source_complete"] and not args.allow_prefix:
        ap.error("source is a prefix, not a complete recording")
    rows = np.atleast_1d(
        np.genfromtxt(source / "keyframes.csv", delimiter=",", names=True)
    )
    if len(rows) < 3 or not np.all(np.diff(rows["stamp"]) > 0):
        raise RuntimeError("invalid keyframe timeline")
    gauge, gravity = gravity_frame(rows)
    poses = gauge @ poses_from_rows(rows)
    times = rows["stamp"]
    trajectory = np.loadtxt(source / "trajectory_final.tum", ndmin=2)
    fit_mask = loop_fit_mask(times, trajectory[0, 0], args.loop_fit_half_period_s)
    tree = cKDTree(poses[:, :2, 3])

    @lru_cache(maxsize=128)
    def raw_cloud(index):
        c = o3d.io.read_point_cloud(
            str(source / "keyframes" / f"{int(rows[index]['id'])}.pcd")
        )
        if len(c.points) != int(rows[index]["points"]):
            raise RuntimeError("keyframe PCD count mismatch")
        return c

    @lru_cache(maxsize=128)
    def cloud(index):
        local = o3d.geometry.PointCloud()
        for neighbor in range(
            max(0, index - args.submap_neighbors),
            min(len(rows), index + args.submap_neighbors + 1),
        ):
            if not fit_mask[neighbor]:
                continue
            relative = np.linalg.inv(poses[index]) @ poses[neighbor]
            if np.linalg.norm(relative[:3, 3]) > 3.0:
                continue
            # Copy before transforming: cached native clouds remain immutable.
            c = o3d.geometry.PointCloud(raw_cloud(neighbor))
            local += c.transform(relative)
        return registration_cloud(local)

    descriptors = []
    edges = []
    attempts = []
    started = time.monotonic()
    last_report = started
    for i in range(len(rows)):
        descriptor = scan_descriptor(cloud(i).points, poses[i, :3, :3])
        descriptors.append(descriptor)
        nearby = [
            j
            for j in tree.query_ball_point(poses[i, :2, 3], args.radius)
            if j < i and times[i] - times[j] >= args.minimum_seconds
        ]
        candidates = sorted(
            [(float(1 - descriptor @ descriptors[j]), j) for j in nearby]
        )
        chosen = []
        for distance, j in candidates:
            if distance > args.descriptor_distance:
                continue
            # Nearby reference keyframes share most submap points; try a
            # different local view when the first view fails geometry checks.
            if any(abs(times[j] - times[k]) < 4 for k in chosen):
                continue
            chosen.append(j)
            initial = np.linalg.inv(poses[j]) @ poses[i]
            fit, audit = validate_pair(
                cloud(i), cloud(j), initial, poses[j, :3, :3].T @ np.array([0, 0, 1.0])
            )
            audit.update(
                i=i,
                j=j,
                descriptor_distance=distance,
                time_separation_s=float(times[i] - times[j]),
            )
            attempts.append(audit)
            if fit is not None:
                matrix, info = fit
                edges.append(
                    dict(
                        i=i,
                        j=j,
                        matrix=matrix.tolist(),
                        information=info.tolist(),
                        audit=audit,
                    )
                )
                break
            if len(chosen) >= args.max_candidates:
                break
        if time.monotonic() - last_report > 10:
            state = {
                "keyframe": i + 1,
                "total": len(rows),
                "geometric_matches": len(edges),
                "attempts": len(attempts),
                "wall_s": time.monotonic() - started,
            }
            print(json.dumps(state), flush=True)
            (output / "progress.json").write_text(json.dumps(state, indent=2))
            last_report = time.monotonic()
    # Reserve measured revisit constraints from the optimizer, not just points
    # from an ICP fit, to assess improvement beyond fitted graph edges.
    heldout = [e for k, e in enumerate(edges) if k % 4 == 3]
    training = [e for k, e in enumerate(edges) if k % 4 != 3]
    optimized, retained = (
        optimize_graph(poses, training) if training else (poses.copy(), [])
    )
    for e in edges:
        e["heldout_from_graph"] = e in heldout
    result = {
        "source": str(source),
        "source_complete": completed["source_complete"],
        "keyframes": len(rows),
        "source_index_sha256": hashlib.sha256(
            (source / "keyframes.csv").read_bytes()
        ).hexdigest(),
        "backend_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("optimize_keyframe_map.py", "revisit_geometry.py")
        },
        "gravity_alignment": gravity,
        "gauge_transform": gauge.tolist(),
        "parameters": {
            "radius_m": args.radius,
            "minimum_seconds": args.minimum_seconds,
            "max_candidates": args.max_candidates,
            "submap_neighbors": args.submap_neighbors,
            "submap_max_translation_m": 3.0,
            "descriptor_distance": args.descriptor_distance,
            "reference_candidate_separation_s": 4.0,
            "loop_fit_half_period_s": args.loop_fit_half_period_s,
            "loop_fit_origin_s": float(trajectory[0, 0]),
        },
        "loop_candidate_fit_scan_stamps": times[fit_mask].tolist(),
        "geometric_loops": len(edges),
        "training_loops": len(training),
        "heldout_loops": len(heldout),
        "retained_loop_edges": retained,
        "heldout_before": edge_errors(heldout, poses),
        "heldout_after": edge_errors(heldout, optimized),
        "training_before": edge_errors(training, poses),
        "training_after": edge_errors(training, optimized),
        "attempts": attempts,
        "edges": edges,
        "wall_s": time.monotonic() - started,
        "note": "Offline map pose graph only; odometry is unchanged. 3D geometry supplies height differences; no zero-height or flat-ground constraint. Heldout consistency is not independent ground truth.",
    }
    np.savez(output / "keyframe_poses.npz", before=poses, after=optimized, stamps=times)
    aligned, corrected = corrected_trajectory(
        trajectory, times, poses, optimized, gauge
    )
    np.savetxt(output / "trajectory_gravity.tum", aligned, fmt="%.12f")
    np.savetxt(output / "trajectory_corrected.tum", corrected, fmt="%.12f")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for values, label in [
        (aligned, "Gravity-aligned odometry"),
        (corrected, "Geometry-verified loop map"),
    ]:
        axes[0].plot(values[:, 1], values[:, 2], label=label, lw=1)
        axes[1].plot(
            values[:, 0] - values[0, 0], values[:, 3] - values[0, 3], label=label, lw=1
        )
    axes[0].set(xlabel="X (m)", ylabel="Y (m)")
    axes[0].axis("equal")
    axes[1].set(xlabel="Time (s)", ylabel="Gravity-direction height change (m)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"{source.parent.name}: {len(training)} fitted / {len(heldout)} withheld revisit constraints\nOffline map correction; no independent trajectory ground truth"
    )
    fig.savefig(output / "trajectory_comparison.png", dpi=150)
    plt.close(fig)
    if not args.no_export:
        result["gravity_map_points"] = export_map(
            source, rows, poses, output / "map_gravity_only.pcd"
        )
        result["corrected_map_points"] = export_map(
            source, rows, optimized, output / "map_loop_corrected.pcd"
        )
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "keyframes",
                    "geometric_loops",
                    "heldout_before",
                    "heldout_after",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
