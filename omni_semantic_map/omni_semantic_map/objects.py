"""Keyframe-anchored object observations, conservative association and queries."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def transform_points(points, pose):
    return np.asarray(points) @ np.asarray(pose)[:3, :3].T + np.asarray(pose)[:3, 3]


def project(points, camera_from_tracking, intrinsic):
    camera = transform_points(points, camera_from_tracking)
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.2)
    pixels = np.full((len(camera), 2), np.nan)
    xy = camera[valid] @ np.asarray(intrinsic).T
    pixels[valid] = xy[:, :2] / xy[:, 2:3]
    return pixels, camera[:, 2], valid


def summarize_object(obj, poses):
    positions = [
        transform_points([o["local_position"]], poses[int(o["keyframe_id"])])[0]
        for o in obj["observations"]
    ]
    center = np.median(positions, axis=0)
    residual = np.linalg.norm(np.array(positions) - center, axis=1)
    unique = len({o["keyframe_id"] for o in obj["observations"]})
    duration = max(o["stamp"] for o in obj["observations"]) - min(
        o["stamp"] for o in obj["observations"]
    )
    obj["position"] = center.tolist()
    if any('local_support_bounds' in ob for ob in obj['observations']):
        from .geometry import object_box
        obj['box3d']=object_box(obj['observations'],poses,center)
    obj["position_spread_p90_m"] = float(np.quantile(residual, 0.9))
    obj["observation_count"] = len(positions)
    obj["keyframe_count"] = unique
    obj["confirmed"] = (
        unique >= 3 and duration >= 4 and obj["position_spread_p90_m"] <= 0.8
    )
    # A compact observed-surface position, not an amodal vehicle center.
    obj["position_definition"] = (
        "median of observed surface positions in the selected map"
    )
    best = max(obj["observations"], key=lambda o: o["score"])
    obj["evidence"] = best["evidence"]
    return obj


def associate(objects, observations, poses, radius=1.0, appearance_limit=0.35,
              ambiguity_margin=0.25):
    """One observation per object per keyframe; ambiguous matches create proposals."""
    if not observations:
        return
    for obj in objects:
        summarize_object(obj, poses)
    costs = np.full((len(observations), len(objects)), 1e6)
    for i, obs in enumerate(observations):
        pos = transform_points([obs["local_position"]], poses[int(obs["keyframe_id"])])[
            0
        ]
        for j, obj in enumerate(objects):
            if obj["label"] != obs["label"] or any(
                o["keyframe_id"] == obs["keyframe_id"] for o in obj["observations"]
            ):
                continue
            distance = np.linalg.norm(pos - obj["position"])
            hist = np.mean([o["appearance"] for o in obj["observations"]], axis=0)
            hist /= max(hist.sum(), 1e-12)
            appearance = np.sqrt(max(0.0, 1 - np.sqrt(hist * obs["appearance"]).sum()))
            if distance <= radius and appearance <= appearance_limit:
                costs[i, j] = distance + 0.6 * appearance
        ranked = np.sort(costs[i])
        if len(ranked) > 1 and ranked[1] < 1e5 and ranked[1] - ranked[0] < ambiguity_margin:
            costs[i] = 1e6
    assigned = set()
    if costs.size:
        rows, cols = linear_sum_assignment(costs)
        for i, j in zip(rows, cols):
            if costs[i, j] < 1e5:
                observations[i]["object_id"] = objects[j]["id"]
                objects[j]["observations"].append(observations[i])
                assigned.add(int(i))
    for i, obs in enumerate(observations):
        if i not in assigned:
            obs["object_id"] = f"{obs['label']}_{len(objects) + 1:04d}"
            objects.append(
                {"id": obs["object_id"], "label": obs["label"], "observations": [obs]}
            )
    for obj in objects:
        summarize_object(obj, poses)


def reproject_map(document, poses, revision):
    import copy

    output = copy.deepcopy(document)
    for obj in output["objects"]:
        summarize_object(obj, poses)
    output["map_revision"] = revision
    return output


def query(document, text, include_tentative=False):
    aliases = {
        "车": "car", "汽车": "car", "车辆": "car", "找车": "car", "轿车": "car",
        "门": "door", "入口": "door", "建筑入口": "door",
        "长椅": "bench", "座椅": "bench", "花盆": "potted_plant", "盆栽": "potted_plant",
        "消火栓": "fire_hydrant", "消防栓": "fire_hydrant", "灭火器": "fire_extinguisher",
    }
    target = aliases.get(text.strip(), text.strip().lower())
    return [
        dict(
            id=o["id"],
            label=o["label"],
            position=o["position"],
            evidence=o["evidence"],
            observation_count=o["observation_count"],
            position_spread_p90_m=o["position_spread_p90_m"],
            confirmed=o["confirmed"],
            **({"box3d":o["box3d"]} if o.get("box3d") else {}),
        )
        for o in document["objects"]
        if (include_tentative or o["confirmed"])
        and (target == o["label"] or target == o["id"])
    ]
