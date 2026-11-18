"""Geometry checks for offline keyframe loop correction; no flat-ground prior."""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REG = o3d.pipelines.registration


def loop_fit_mask(stamps, origin, half_period_s=0.0):
    """Optional whole-scan split: even time blocks fit loops; odd blocks audit."""
    if not np.isfinite(half_period_s) or half_period_s < 0:
        raise ValueError("loop fit half period must be finite and nonnegative")
    stamps = np.asarray(stamps)
    if half_period_s == 0:
        return np.ones(stamps.shape, dtype=bool)
    blocks = np.floor((stamps - origin + 1e-6) / half_period_s).astype(np.int64)
    return blocks % 2 == 0


def gravity_rotation(up):
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    target = np.array([0.0, 0.0, 1.0])
    cross = np.cross(up, target)
    cosine = float(up @ target)
    if cosine < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    x, y, z = cross
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + skew + skew @ skew / (1 + cosine)


def poses_from_rows(rows):
    poses = np.repeat(np.eye(4)[None], len(rows), axis=0)
    poses[:, :3, :3] = Rotation.from_quat(
        np.column_stack([rows[k] for k in ("qx", "qy", "qz", "qw")])
    ).as_matrix()
    poses[:, :3, 3] = np.column_stack([rows[k] for k in ("x", "y", "z")])
    return poses


def gravity_frame(rows):
    force = np.column_stack([rows[k] for k in ("ax_world", "ay_world", "az_world")])
    speed = np.linalg.norm(
        np.column_stack([rows[k] for k in ("vx", "vy", "vz")]), axis=1
    )
    norm = np.linalg.norm(force, axis=1)
    stationary = (speed < 0.04) & (rows["gyro_norm"] < 0.03) & (norm > 8) & (norm < 11)
    if stationary.sum() >= 5:
        vectors = force[stationary] / norm[stationary, None]
        method = "robust stationary specific-force direction"
    else:
        vectors = -np.column_stack([rows[k] for k in ("gx", "gy", "gz")])
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
        method = "estimated gravity fallback; insufficient stationary frames"
    up = np.median(vectors, axis=0)
    up /= np.linalg.norm(up)
    keep = vectors @ up > np.cos(np.deg2rad(3))
    if not keep.any():
        raise RuntimeError("no consistent gravity direction; cannot define height")
    up = vectors[keep].mean(axis=0)
    up /= np.linalg.norm(up)
    rotation = gravity_rotation(up)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ np.array([rows[k][0] for k in ("x", "y", "z")])
    angles = np.rad2deg(np.arccos(np.clip(vectors @ up, -1, 1)))
    return transform, {
        "method": method,
        "stationary_frames": int(stationary.sum()),
        "up_in_odom": up.tolist(),
        "direction_p95_deg": float(np.percentile(angles, 95)),
        "frame_rotation_deg": float(
            np.rad2deg(Rotation.from_matrix(rotation).magnitude())
        ),
    }


def scan_descriptor(points, rotation):
    p = np.asarray(points) @ rotation.T
    radius = np.linalg.norm(p[:, :2], axis=1)
    mask = (radius > 1) & (radius < 35) & np.isfinite(p).all(axis=1)
    p = p[mask]
    radius = radius[mask]
    ring = np.minimum((radius / 35 * 20).astype(int), 19)
    sector = np.minimum(
        ((np.arctan2(p[:, 1], p[:, 0]) + np.pi) / (2 * np.pi) * 60).astype(int), 59
    )
    out = np.zeros((20, 60))
    np.maximum.at(out, (ring, sector), np.maximum(0.1, p[:, 2] + 2))
    v = out.ravel()
    return v / max(np.linalg.norm(v), 1e-12)


def registration_cloud(cloud, voxel=0.3):
    xyz = np.asarray(cloud.points)
    radius = np.linalg.norm(xyz, axis=1)
    valid = np.flatnonzero(np.isfinite(xyz).all(axis=1) & (radius > 1) & (radius < 35))
    result = cloud.select_by_index(valid).voxel_down_sample(voxel)
    result.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.9, max_nn=30))
    return result


def normal_information(points, normals):
    jacobian = np.column_stack([np.cross(points, normals), normals])
    hessian = jacobian.T @ jacobian / max(1, len(points))
    eigen = np.linalg.eigvalsh(hessian)
    ratio = float(eigen[0] / max(eigen[-1], 1e-12))
    return hessian, ratio


def validate_pair(source, target, initial, up_in_target):
    if min(len(source.points), len(target.points)) < 150:
        return None, {"reason": "too_few_points"}
    source_ids = np.arange(len(source.points))
    target_ids = np.arange(len(target.points))
    train_s = source.select_by_index(source_ids[source_ids % 4 != 0])
    train_t = target.select_by_index(target_ids[target_ids % 4 != 0])
    coarse_s = train_s.voxel_down_sample(0.7)
    coarse_t = train_t.voxel_down_sample(0.7)
    trials = []
    for height in (-6.0, -3.0, 0.0, 3.0, 6.0):
        seed = initial.copy()
        seed[:3, 3] += up_in_target * height
        fit = REG.registration_icp(
            coarse_s,
            coarse_t,
            1.8,
            seed,
            REG.TransformationEstimationPointToPoint(),
            REG.ICPConvergenceCriteria(max_iteration=35),
        )
        trials.append((fit.fitness - 0.3 * fit.inlier_rmse, fit))
    trials.sort(key=lambda x: x[0], reverse=True)
    best = trials[0][1]
    if best.fitness < 0.4:
        return None, {"reason": "coarse_overlap", "fitness": best.fitness}
    for score, other in trials[1:]:
        if score < trials[0][0] - 0.025:
            continue
        if np.linalg.norm(other.transformation[:3, 3] - best.transformation[:3, 3]) > 1:
            return None, {"reason": "ambiguous_height_or_structure"}
    fine = REG.registration_icp(
        train_s,
        train_t,
        0.65,
        best.transformation,
        REG.TransformationEstimationPointToPlane(REG.HuberLoss(0.15)),
        REG.ICPConvergenceCriteria(max_iteration=50),
    )
    matrix = fine.transformation
    if not np.isfinite(matrix).all():
        return None, {"reason": "nonfinite_transform"}
    delta = np.linalg.inv(initial) @ matrix
    if np.linalg.norm(delta[:3, 3]) > 12 or Rotation.from_matrix(
        delta[:3, :3]
    ).magnitude() > np.deg2rad(20):
        return None, {"reason": "excessive_correction"}
    s = np.asarray(source.points)[source_ids % 4 == 0]
    t = np.asarray(target.points)
    moved = s @ matrix[:3, :3].T + matrix[:3, 3]
    distances, indices = cKDTree(t).query(moved)
    inlier = distances < 0.35
    reverse = np.asarray(target.points)[target_ids % 4 == 0]
    reverse = (reverse - matrix[:3, 3]) @ matrix[:3, :3]
    reverse_distance = cKDTree(np.asarray(source.points)).query(reverse)[0]
    overlap = float(inlier.mean())
    reverse_overlap = float((reverse_distance < 0.35).mean())
    rmse = (
        float(np.sqrt(np.mean(distances[inlier] ** 2)))
        if inlier.any()
        else float("inf")
    )
    normals = np.asarray(target.normals)[indices[inlier]]
    residual = np.sum(normals * (moved[inlier] - t[indices[inlier]]), axis=1)
    plane_rmse = float(np.sqrt(np.mean(residual**2))) if len(residual) else float("inf")
    info, ratio = normal_information(moved[inlier], normals)
    audit = {
        "fitness": overlap,
        "reverse_fitness": reverse_overlap,
        "rmse_m": rmse,
        "plane_rmse_m": plane_rmse,
        "observability_ratio": ratio,
        "vertical_seed_offsets_m": [-6, -3, 0, 3, 6],
    }
    if (
        min(overlap, reverse_overlap) < 0.4
        or (overlap + reverse_overlap) / 2 < 0.5
        or rmse > 0.2
        or plane_rmse > 0.12
    ):
        return None, dict(audit, reason="heldout_geometry")
    if ratio < 1e-4:
        return None, dict(audit, reason="degenerate_geometry")
    info = info / max(0.05, plane_rmse) ** 2
    return (matrix, info), dict(audit, reason="accepted")


def optimize_graph(poses, edges):
    graph = REG.PoseGraph()
    for pose in poses:
        graph.nodes.append(REG.PoseGraphNode(pose.copy()))
    odom_info = np.diag([1 / 0.02**2] * 3 + [1 / 0.1**2] * 3)
    for i in range(1, len(poses)):
        graph.edges.append(
            REG.PoseGraphEdge(
                i - 1, i, np.linalg.inv(poses[i]) @ poses[i - 1], odom_info, False
            )
        )
    for edge in edges:
        graph.edges.append(
            REG.PoseGraphEdge(
                edge["i"],
                edge["j"],
                np.asarray(edge["matrix"]),
                np.asarray(edge["information"]),
                True,
            )
        )
    REG.global_optimization(
        graph,
        REG.GlobalOptimizationLevenbergMarquardt(),
        REG.GlobalOptimizationConvergenceCriteria(),
        REG.GlobalOptimizationOption(
            max_correspondence_distance=2.0,
            edge_prune_threshold=0.1,
            preference_loop_closure=2.0,
            reference_node=0,
        ),
    )
    return np.array([n.pose for n in graph.nodes]), [
        {
            "i": int(e.source_node_id),
            "j": int(e.target_node_id),
            "confidence": float(e.confidence),
        }
        for e in graph.edges
        if e.uncertain
    ]
