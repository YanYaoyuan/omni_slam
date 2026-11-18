"""Revisit-event and fixed-surface diagnostics, without a flat-ground prior."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def temporal_events(edges, stamps, window=15.0):
    """Connected components in BOTH source and reference time coordinates."""
    times = np.array([[stamps[e["i"]], stamps[e["j"]]] for e in edges])
    remaining = set(range(len(edges)))
    groups = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        group, queue = [seed], [seed]
        while queue:
            current = queue.pop()
            near = [
                k
                for k in sorted(remaining)
                if np.max(np.abs(times[k] - times[current])) <= window
            ]
            remaining.difference_update(near)
            group.extend(near)
            queue.extend(near)
        groups.append([edges[k] for k in sorted(group)])
    return groups


def trajectory_poses(trajectory, stamps):
    stamps = np.asarray(stamps)
    if not np.all(np.diff(trajectory[:, 0]) > 0):
        raise ValueError("trajectory times must increase")
    if (
        stamps.min() < trajectory[0, 0] - 1e-5
        or stamps.max() > trajectory[-1, 0] + 1e-5
    ):
        raise ValueError("requested pose outside measured timeline")
    t = np.clip(stamps, trajectory[0, 0], trajectory[-1, 0])
    result = np.repeat(np.eye(4)[None], len(t), axis=0)
    result[:, :3, :3] = Slerp(trajectory[:, 0], Rotation.from_quat(trajectory[:, 4:8]))(
        t
    ).as_matrix()
    result[:, :3, 3] = np.column_stack(
        [np.interp(t, trajectory[:, 0], trajectory[:, k]) for k in (1, 2, 3)]
    )
    return result


def surface_spread(reference, observed):
    """Keep identical signed-distance samples for all evaluated map variants."""
    reference, observed = np.asarray(reference), np.asarray(observed)
    if min(len(reference), len(observed)) < 10:
        raise ValueError("insufficient surface samples")
    n = min(500, len(reference), len(observed))
    a = reference[np.linspace(0, len(reference) - 1, n).astype(int)]
    b = observed[np.linspace(0, len(observed) - 1, n).astype(int)]
    mixed = np.r_[a, b]
    return {
        "signed_layer_gap_m": float(np.median(observed) - np.median(reference)),
        "abs_layer_gap_m": float(abs(np.median(observed) - np.median(reference))),
        "balanced_width90_m": float(
            np.quantile(mixed, 0.95) - np.quantile(mixed, 0.05)
        ),
        "reference_width90_m": float(
            np.quantile(reference, 0.95) - np.quantile(reference, 0.05)
        ),
        "source_width90_m": float(
            np.quantile(observed, 0.95) - np.quantile(observed, 0.05)
        ),
        "balanced_points_per_visit": n,
    }


def planar_patches(points, normals, up, cell_size=3.0):
    """Fit target-only local planes; classify lower horizontal / vertical planes."""
    points, normals = np.asarray(points), np.asarray(normals)
    up = np.asarray(up) / np.linalg.norm(up)
    alignment = np.abs(normals @ up)
    height = points @ up
    category = np.full(len(points), -1, int)
    category[(alignment > 0.85) & (height < -0.15) & (height > -3)] = 0
    category[alignment < 0.2] = 1
    cells = np.floor(points / cell_size).astype(int)
    labels = np.full(len(points), -1, int)
    patches = []
    # Sparse horizontal laser rings need a plane supported across multiple
    # cells. Fit ONLY the independent reference visit, with no normal forced
    # to the gravity axis and no refitting for any evaluated map variant.
    lower_ids = np.flatnonzero(category == 0)
    rng = np.random.default_rng(24)
    for _ in range(3):
        if len(lower_ids) < 40:
            break
        best = np.array([], dtype=int)
        for _ in range(300):
            three = points[rng.choice(lower_ids, 3, replace=False)]
            normal = np.cross(three[1] - three[0], three[2] - three[0])
            length = np.linalg.norm(normal)
            if length < 1e-6:
                continue
            normal /= length
            if abs(normal @ up) < 0.85:
                continue
            mask = (np.abs((points[lower_ids] - three[0]) @ normal) < 0.06) & (
                np.abs(normals[lower_ids] @ normal) > 0.95
            )
            ids = lower_ids[mask]
            if len(ids) > len(best):
                best = ids
        if len(best) < 40:
            break
        center = points[best].mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov((points[best] - center).T))
        normal = vectors[:, 0]
        if values[1] < 0.03 or abs(normal @ up) < 0.85:
            break
        labels[best] = len(patches)
        patches.append(
            {"kind": "ground", "normal": normal, "center": center, "target_ids": best}
        )
        lower_ids = lower_ids[~np.isin(lower_ids, best)]
    keys = np.column_stack([category, cells])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    for k, key in enumerate(unique):
        if key[0] != 1:
            continue
        ids = np.flatnonzero(inverse == k)
        if len(ids) < 30:
            continue
        kept = ids.copy()
        for _ in range(3):
            center = points[kept].mean(axis=0)
            values, vectors = np.linalg.eigh(np.cov((points[kept] - center).T))
            normal = vectors[:, 0]
            kept = ids[np.abs((points[ids] - center) @ normal) < 0.08]
            if len(kept) < 30:
                break
        if len(kept) < 30 or values[0] / max(values[1], 1e-12) > 0.04:
            continue
        if (key[0] == 0 and abs(normal @ up) < 0.85) or (
            key[0] == 1 and abs(normal @ up) > 0.2
        ):
            continue
        if np.mean(np.abs(normals[kept] @ normal) > 0.95) < 0.75:
            continue
        labels[kept] = len(patches)
        patches.append(
            {
                "kind": "ground" if key[0] == 0 else "vertical_plane",
                "normal": normal,
                "center": center,
                "target_ids": kept,
            }
        )
    return patches, labels


def event_measurement_scatter(group, poses):
    representative = group[len(group) // 2]
    i0, j0 = representative["i"], representative["j"]
    transforms = np.array(
        [
            np.linalg.inv(poses[j0])
            @ poses[e["j"]]
            @ np.asarray(e["matrix"])
            @ np.linalg.inv(poses[e["i"]])
            @ poses[i0]
            for e in group
        ]
    )
    delta = np.linalg.inv(transforms[len(group) // 2]) @ transforms
    shift = np.linalg.norm(delta[:, :3, 3], axis=1)
    angle = np.rad2deg(Rotation.from_matrix(delta[:, :3, :3]).magnitude())
    return {
        "edges": len(group),
        "translation_p90_m": float(np.quantile(shift, 0.9)),
        "rotation_p90_deg": float(np.quantile(angle, 0.9)),
    }
