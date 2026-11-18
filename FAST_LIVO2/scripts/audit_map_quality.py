#!/usr/bin/env python3
"""Compare fixed LIO/LIVO runs using common scan geometry withheld from loop fits."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import yaml

from map_quality import (
    event_measurement_scatter,
    planar_patches,
    surface_spread,
    temporal_events,
    trajectory_poses,
)
from revisit_geometry import (
    gravity_rotation,
    poses_from_rows,
    registration_cloud,
    validate_pair,
    loop_fit_mask,
)


def load_mode(source, maps):
    result = json.loads((maps / "result.json").read_text())
    rows = np.atleast_1d(
        np.genfromtxt(source / "keyframes.csv", delimiter=",", names=True)
    )
    if (
        result["source_index_sha256"]
        != hashlib.sha256((source / "keyframes.csv").read_bytes()).hexdigest()
    ):
        raise RuntimeError("source keyframes differ from mapped data")
    complete = json.loads((source.parent / "batch_result.json").read_text())
    if not complete["source_complete"] or complete["failures"]:
        raise RuntimeError("incomplete or failed input run")
    return dict(
        source=source,
        maps=maps,
        result=result,
        rows=rows,
        poses=poses_from_rows(rows),
        times=rows["stamp"],
        native=np.loadtxt(source / "trajectory_final.tum", ndmin=2),
        before=np.loadtxt(maps / "trajectory_gravity.tum", ndmin=2),
        after=np.loadtxt(maps / "trajectory_corrected.tum", ndmin=2),
    )


def excluded_training_times(mode):
    result, rows = mode["result"], mode["rows"]
    neighbors = result["parameters"]["submap_neighbors"]
    excluded = set()
    fit_mask = loop_fit_mask(
        mode["times"],
        result["parameters"].get("loop_fit_origin_s", mode["times"][0]),
        result["parameters"].get("loop_fit_half_period_s", 0),
    )
    for e in result["edges"]:
        if e["heldout_from_graph"]:
            continue
        for center in (e["i"], e["j"]):
            for k in range(
                max(0, center - neighbors), min(len(rows), center + neighbors + 1)
            ):
                if not fit_mask[k]:
                    continue
                if (
                    np.linalg.norm(
                        mode["poses"][k, :3, 3] - mode["poses"][center, :3, 3]
                    )
                    <= 3
                ):
                    excluded.add(int(round(rows["stamp"][k] * 1e6)))
    return excluded


def mode_measurement(mode, edge, stamps):
    # Move the loop measurement to the SAME physical IMU epochs used by all
    # four evaluated trajectories. No per-variant geometry refitting follows.
    i, j = edge["native_i"], edge["native_j"]
    sampled = trajectory_poses(mode["native"], stamps)
    return (
        np.linalg.inv(sampled[1])
        @ mode["poses"][j]
        @ np.asarray(edge["matrix"])
        @ np.linalg.inv(mode["poses"][i])
        @ sampled[0]
    )


def plane_evaluation(source, target, fitted, predictions, up):
    src, tgt = np.asarray(source.points), np.asarray(target.points)
    moved = src @ fitted[:3, :3].T + fitted[:3, 3]
    distances, nearest = cKDTree(tgt).query(moved)
    agreement = np.abs(
        np.sum(
            (np.asarray(source.normals) @ fitted[:3, :3].T)
            * np.asarray(target.normals)[nearest],
            axis=1,
        )
    )
    patches, labels = planar_patches(tgt, target.normals, up)
    rows = []
    for patch_id, patch in enumerate(patches):
        ids = np.flatnonzero(
            (distances < 0.25) & (agreement > 0.95) & (labels[nearest] == patch_id)
        )
        if len(ids) < 15:
            continue
        ids = ids[:: max(1, len(ids) // 500)]
        ref_ids = patch["target_ids"]
        ref = (tgt[ref_ids] - patch["center"]) @ patch["normal"]
        values = {}
        samples = {}
        for name, matrix in predictions.items():
            points = src[ids] @ matrix[:3, :3].T + matrix[:3, 3]
            signed = (points - patch["center"]) @ patch["normal"]
            values[name] = surface_spread(ref, signed)
            samples[name] = signed
        rows.append(
            dict(
                kind=patch["kind"],
                center=patch["center"].tolist(),
                normal=patch["normal"].tolist(),
                source_points=len(ids),
                target_points=len(ref_ids),
                values=values,
                samples=samples,
                reference_samples=ref,
            )
        )
    return rows


def summarize_patches(patches):
    out = {}
    for kind in ("ground", "vertical_plane"):
        selected = [p for p in patches if p["kind"] == kind]
        out[kind] = {"patches": len(selected), "variants": {}}
        if not selected:
            continue
        for name in selected[0]["values"]:
            out[kind]["variants"][name] = {
                "median_abs_layer_gap_m": float(
                    np.median([p["values"][name]["abs_layer_gap_m"] for p in selected])
                ),
                "p90_abs_layer_gap_m": float(
                    np.quantile(
                        [p["values"][name]["abs_layer_gap_m"] for p in selected], 0.9
                    )
                ),
                "median_balanced_width90_m": float(
                    np.median(
                        [p["values"][name]["balanced_width90_m"] for p in selected]
                    )
                ),
            }
    return out


def save_plane_plot(patches, output, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), layout="constrained")
    for ax, kind in zip(axes, ("ground", "vertical_plane")):
        selected = [p for p in patches if p["kind"] == kind]
        if not selected:
            ax.text(
                0.5,
                0.5,
                "Insufficient verified planar support",
                ha="center",
                transform=ax.transAxes,
            )
            ax.set_title(kind)
            continue
        patch = max(
            selected, key=lambda p: p["values"]["LIVO_before"]["abs_layer_gap_m"]
        )
        samples = [patch["reference_samples"]] + list(patch["samples"].values())
        lower = min(float(np.min(x)) for x in samples)
        upper = max(float(np.max(x)) for x in samples)
        bins = np.linspace(lower - 0.03, upper + 0.03, 65)
        ax.hist(
            samples[0],
            bins=bins,
            histtype="step",
            density=True,
            label="Reference visit",
            color="black",
            lw=1.5,
        )
        for key, sample in patch["samples"].items():
            ax.hist(sample, bins=bins, histtype="step", density=True, label=key, lw=1)
        ax.set(
            title=f"{kind}: fixed points / target-only plane",
            xlabel="Signed plane distance (m)",
            ylabel="Density",
        )
        ax.legend(fontsize=7)
    fig.suptitle(
        name
        + "\nScans excluded from BOTH graphs' fitted loop submaps; internal consistency only"
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)


def overlay_plot(source, target, predictions, up, output, name):
    s, t = np.asarray(source.points), np.asarray(target.points)
    rotate = gravity_rotation(up)
    t = t @ rotate.T
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, (key, matrix) in zip(axes.ravel(), predictions.items()):
        moved = (s @ matrix[:3, :3].T + matrix[:3, 3]) @ rotate.T
        for a, label, color in [
            (t, "Reference visit", "0.55"),
            (moved, key, "#2563eb"),
        ]:
            a = a[:: max(1, len(a) // 15000)]
            keep = (
                (np.abs(a[:, 0]) < 18)
                & (np.abs(a[:, 1]) < 18)
                & (a[:, 2] > -10)
                & (a[:, 2] < 10)
            )
            a = a[keep]
            ax.scatter(a[:, 0], a[:, 2], s=0.5, alpha=0.4, label=label, c=color)
        ax.set(
            title=key,
            xlabel="Local gravity-frame X (m)",
            ylabel="Z (m)",
            xlim=(-18, 18),
            ylim=(-10, 10),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle(
        name
        + "\nSAME reserved LIO scans, shown with four relative map poses; no flat-ground fit"
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lio-source", type=Path, required=True)
    ap.add_argument("--lio-maps", type=Path, required=True)
    ap.add_argument("--livo-source", type=Path, required=True)
    ap.add_argument("--livo-maps", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("retain existing audit; choose another output directory")
    args.output.mkdir(parents=True)
    modes = {
        "LIO": load_mode(args.lio_source, args.lio_maps),
        "LIVO": load_mode(args.livo_source, args.livo_maps),
    }
    lio, livo = modes["LIO"], modes["LIVO"]
    configurations = []
    for key, mode in modes.items():
        config = yaml.safe_load(
            (mode["source"].parent / (key.lower() + ".yaml")).read_text()
        )["/**"]["ros__parameters"]
        del config["runtime"]["output_directory"]
        del config["common"]["img_en"]
        configurations.append(config)
    if configurations[0] != configurations[1]:
        raise RuntimeError(
            "LIO/LIVO configurations differ beyond camera enable/output path"
        )
    if not np.array_equal(lio["native"][:, 0], livo["native"][:, 0]):
        raise RuntimeError("LIO/LIVO final state epochs differ")
    if lio["result"]["backend_sha256"] != livo["result"]["backend_sha256"]:
        raise RuntimeError("different map backends")
    fit_times = excluded_training_times(lio) | excluded_training_times(livo)
    eligible = np.array([int(round(t * 1e6)) not in fit_times for t in lio["times"]])
    split_period = lio["result"]["parameters"].get("loop_fit_half_period_s", 0)
    if split_period:
        if split_period != livo["result"]["parameters"].get("loop_fit_half_period_s"):
            raise RuntimeError("different whole-scan validation splits")
        eligible &= ~loop_fit_mask(lio["times"], lio["native"][0, 0], split_period)
    times = np.r_[lio["times"], livo["times"]]
    all_edges = []
    for key, mode in modes.items():
        offset = 0 if key == "LIO" else len(lio["times"])
        for edge in mode["result"]["edges"]:
            all_edges.append(
                dict(
                    edge,
                    mode=key,
                    native_i=edge["i"],
                    native_j=edge["j"],
                    i=edge["i"] + offset,
                    j=edge["j"] + offset,
                )
            )
    groups = temporal_events(all_edges, times)

    @lru_cache(maxsize=128)
    def raw(index):
        path = lio["source"] / "keyframes" / f"{int(lio['rows'][index]['id'])}.pcd"
        cloud = o3d.io.read_point_cloud(str(path))
        if len(cloud.points) != int(lio["rows"][index]["points"]):
            raise RuntimeError("PCD count differs from archive index")
        return cloud

    def reserved_ids(index):
        # All selected observations are outside every fitted loop submap in
        # BOTH modes, not just outside this edge's two central scans.
        distance = np.linalg.norm(
            lio["poses"][:, :3, 3] - lio["poses"][index, :3, 3], axis=1
        )
        delta = np.abs(lio["times"] - lio["times"][index])
        ids = np.flatnonzero(eligible & (distance < 8) & (delta < 35))
        return ids[np.argsort(delta[ids], kind="stable")][:5]

    def reserved(index):
        ids = reserved_ids(index)
        cloud = o3d.geometry.PointCloud()
        for k in ids:
            relative = np.linalg.inv(lio["poses"][index]) @ lio["poses"][k]
            cloud += o3d.geometry.PointCloud(raw(int(k))).transform(relative)
        return registration_cloud(cloud) if len(ids) else cloud, ids

    variants = {
        f"{key}_{which}": mode[which]
        for key, mode in modes.items()
        for which in ("before", "after")
    }
    events = []
    all_patches = []
    for number, group in enumerate(groups, 1):
        representatives = {}
        own_groups = {}
        scatters = {}
        for key, mode in modes.items():
            own = [
                dict(e, i=e["native_i"], j=e["native_j"])
                for e in group
                if e["mode"] == key
            ]
            if own:
                own.sort(key=lambda e: (mode["times"][e["i"]], mode["times"][e["j"]]))
                own_groups[key] = own
                representatives[key] = own[len(own) // 2]
                scatters[key] = event_measurement_scatter(own, mode["poses"])
        reference_key = "LIVO" if "LIVO" in representatives else "LIO"
        tm = modes[reference_key]["times"]
        candidates = []
        for e in own_groups[reference_key]:
            ii = int(np.argmin(abs(lio["times"] - tm[e["native_i"]])))
            jj = int(np.argmin(abs(lio["times"] - tm[e["native_j"]])))
            a, b = reserved_ids(ii), reserved_ids(jj)
            distance = sum(
                float(np.max(abs(lio["times"][ids] - lio["times"][center])))
                if len(ids)
                else 1000
                for ids, center in ((a, ii), (b, jj))
            )
            candidates.append((min(len(a), len(b)), -distance, ii, jj))
        # Choose anchors by reserved scan availability BEFORE registration,
        # without selecting on the resulting residual or map improvement.
        _, _, i, j = max(candidates, key=lambda x: x[:2])
        stamps = np.array([lio["times"][i], lio["times"][j]])
        for key, own in own_groups.items():
            mt = modes[key]["times"]
            representatives[key] = min(
                own,
                key=lambda e: (
                    abs(mt[e["native_i"]] - stamps[0])
                    + abs(mt[e["native_j"]] - stamps[1])
                ),
            )
        event = {
            "event": number,
            "anchor_i": i,
            "anchor_j": j,
            "stamps": stamps.tolist(),
            "source_elapsed_s": float(stamps[0] - lio["times"][0]),
            "reference_elapsed_s": float(stamps[1] - lio["times"][0]),
            "edge_scatter": scatters,
            "mode_checks": {},
            "source_interval_s": [
                float(min(times[x["i"]] for x in group) - lio["times"][0]),
                float(max(times[x["i"]] for x in group) - lio["times"][0]),
            ],
            "reference_interval_s": [
                float(min(times[x["j"]] for x in group) - lio["times"][0]),
                float(max(times[x["j"]] for x in group) - lio["times"][0]),
            ],
        }
        source, source_ids = reserved(i)
        target, target_ids = reserved(j)
        event["reserved_source_ids"] = source_ids.tolist()
        event["reserved_target_ids"] = target_ids.tolist()
        event["reserved_stamps"] = {
            key: lio["times"][ids].tolist()
            for key, ids in [("source", source_ids), ("target", target_ids)]
        }
        assert not any(
            int(round(t * 1e6)) in fit_times
            for ids in (source_ids, target_ids)
            for t in lio["times"][ids]
        )
        up = lio["before"][np.argmin(abs(lio["before"][:, 0] - stamps[1]))]
        up = Rotation.from_quat(up[4:8]).as_matrix().T @ np.array([0.0, 0.0, 1.0])
        fitted = None
        for key, edge in representatives.items():
            initial = mode_measurement(modes[key], edge, stamps)
            if min(len(source_ids), len(target_ids)) < 2:
                event["mode_checks"][key] = {"status": "insufficient_reserved_scans"}
                continue
            fit, audit = validate_pair(source, target, initial, up)
            check = {"registration": audit, "original_measurement": initial.tolist()}
            if fit is None:
                check["status"] = "reserved_geometry_not_verified"
            else:
                matrix, _ = fit
                delta = np.linalg.inv(initial) @ matrix
                shift = float(np.linalg.norm(delta[:3, 3]))
                angle = float(
                    np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude())
                )
                check.update(
                    measurement_change_m=shift,
                    measurement_change_deg=angle,
                    verification_transform=matrix.tolist(),
                    status="supported_by_reserved_scans"
                    if shift <= 0.3 and angle <= 3
                    else "inconsistent_reserved_geometry",
                )
                if fitted is None or key == "LIVO":
                    fitted = matrix
            event["mode_checks"][key] = check
        predictions = {}
        for key, trajectory in variants.items():
            poses = trajectory_poses(trajectory, stamps)
            predictions[key] = np.linalg.inv(poses[1]) @ poses[0]
        if fitted is not None:
            patches = plane_evaluation(source, target, fitted, predictions, up)
            event["surface_summary"] = summarize_patches(patches)
            event["patches"] = [
                {
                    k: v
                    for k, v in p.items()
                    if k not in ("samples", "reference_samples")
                }
                for p in patches
            ]
            all_patches.extend(patches)
            save_plane_plot(
                patches,
                args.output / f"event_{number:02d}_surfaces.png",
                args.lio_source.parent.name + f" event {number}",
            )
            np.savez_compressed(
                args.output / f"event_{number:02d}_geometry.npz",
                source=np.asarray(source.points),
                target=np.asarray(target.points),
                fitted=fitted,
                up=up,
                **{k: v for k, v in predictions.items()},
            )
        if min(len(source_ids), len(target_ids)) >= 2:
            overlay_plot(
                source,
                target,
                predictions,
                up,
                args.output / f"event_{number:02d}_xz.png",
                args.lio_source.parent.name + f" event {number}",
            )
        events.append(event)
        print(
            json.dumps(
                {
                    "event": number,
                    "checks": {k: v["status"] for k, v in event["mode_checks"].items()},
                    "patches": len(event.get("patches", [])),
                }
            ),
            flush=True,
        )

    # Shared initial SE3 alignment for trajectory plots, with no scale fitting.
    q = trajectory_poses(livo["before"], [livo["before"][0, 0]])[0] @ np.linalg.inv(
        trajectory_poses(lio["before"], [lio["before"][0, 0]])[0]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    trajectory_metrics = {}
    for key, trajectory in variants.items():
        poses = trajectory_poses(trajectory, trajectory[:, 0])
        if key.startswith("LIO_"):
            poses = q @ poses
        xyz = poses[:, :3, 3]
        axes[0].plot(xyz[:, 0], xyz[:, 1], label=key, lw=1)
        axes[1].plot(
            trajectory[:, 0] - trajectory[0, 0], xyz[:, 2] - xyz[0, 2], label=key, lw=1
        )
        speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.diff(trajectory[:, 0])
        trajectory_metrics[key] = {
            "path_m": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
            "speed_p99_m_s": float(np.quantile(speed, 0.99)),
            "end_height_change_m": float(xyz[-1, 2] - xyz[0, 2]),
        }
    axes[0].set(xlabel="X (m)", ylabel="Y (m)")
    axes[0].axis("equal")
    axes[1].set(
        xlabel="Time (s)", ylabel="Height change in common initial gravity frame (m)"
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle(
        args.lio_source.parent.name
        + "\nIdentical SI configuration / native binary; no trajectory ground truth"
    )
    fig.savefig(args.output / "trajectories.png", dpi=150)
    plt.close(fig)
    output = {
        "bag": args.lio_source.parent.name,
        "configuration_equal_except_camera_enable_and_output": True,
        "identical_state_epochs": True,
        "native_poses_per_mode": len(lio["native"]),
        "backend_sha256": lio["result"]["backend_sha256"],
        "sources": {k: str(v["source"].resolve()) for k, v in modes.items()},
        "map_results": {
            k: str((v["maps"] / "result.json").resolve()) for k, v in modes.items()
        },
        "accepted_edges": {k: v["result"]["geometric_loops"] for k, v in modes.items()},
        "excluded_training_scan_epochs": len(fit_times),
        "eligible_reserved_lio_scans": int(eligible.sum()),
        "visit_events_union": len(events),
        "events": events,
        "surfaces": summarize_patches(all_patches),
        "trajectory_metrics": trajectory_metrics,
        "protocol": {
            "event_window_s": 15,
            "reserved_window_s": 35,
            "reserved_max_distance_m": 8,
            "reserved_max_scans_per_side": 5,
            "loop_measurement_stability_m": 0.3,
            "loop_measurement_stability_deg": 3,
            "plane_cell_m": 3,
            "ground_reference_plane": "target-only RANSAC; threshold .06 m, at least 40 points, at most 3 planes; gravity normal is not forced",
            "fixed_pair_radius_m": 0.25,
            "anchor_selection": "maximize minimum reserved scan count, then minimize time extent; no residual-based selection",
            "loop_fit_half_period_s": split_period,
        },
        "note": "Same reserved LIO scan points for all four map variants, excluded from both graphs' fitted loop submaps. Plane fitting/correspondence selection occurs once. No external truth; low overlap or inconsistent local odometry can leave an event unverified. Vertical planes are wall candidates, not semantic labels.",
    }
    (args.output / "result.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
