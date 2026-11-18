"""ROS-independent geometry for the bounded live trial."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def rigid_matrix(position, quaternion):
    values = np.asarray([*position, *quaternion], dtype=float)
    if not np.isfinite(values).all() or abs(np.linalg.norm(quaternion) - 1) > .01:
        raise ValueError("invalid rigid transform")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = position
    return matrix


def mask_points(pixels, valid, mask):
    uv = np.nan_to_num(pixels, nan=-1, posinf=-1, neginf=-1).astype(int)
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1])
    inside &= (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0])
    selected = np.zeros(len(pixels), bool)
    selected[inside] = mask[uv[inside, 1], uv[inside, 0]]
    return selected


def support_indices(points, pixels, depth, valid, mask):
    """Reject mixed-depth/isolated supports; never infer unseen dimensions."""
    ids = np.flatnonzero(mask_points(pixels, valid, mask) & (depth > 1) & (depth < 18))
    if len(ids) < 8:
        return None
    # Connected components in measured XYZ; bounded input from a single keyframe.
    tree = cKDTree(points[ids]); unseen = set(range(len(ids))); groups = []
    while unseen:
        seed = unseen.pop(); stack = [seed]; group = [seed]
        while stack:
            neighbours = set(tree.query_ball_point(points[ids[stack.pop()]], .45)) & unseen
            unseen.difference_update(neighbours); stack.extend(neighbours); group.extend(neighbours)
        if len(group) >= 8:
            groups.append(ids[group])
    groups.sort(key=lambda g: len(g), reverse=True)
    if not groups or (len(groups) > 1 and len(groups[1]) > .7 * len(groups[0])):
        return None
    selected = groups[0]
    if np.ptp(points[selected], axis=0).max() > 6:
        return None
    return selected


class LatestSlot:
    """One pending frame; callbacks and pop run in the same ROS executor."""
    def __init__(self):
        self.value = None
        self.dropped = 0

    def put(self, value):
        if self.value is not None:
            self.dropped += 1
        self.value = value

    def pop(self):
        value, self.value = self.value, None
        return value
