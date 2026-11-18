"""Training-free, fixed-membership semantic readout controls.

The proposed control balances angular observation density using measured camera
positions. It changes neither instance memberships nor geometry. It cannot
recover a foreground category dominated by the same background class in every
input descriptor; high feature consistency is not evidence of correctness.
"""
import numpy as np

METHODS = ('mean', 'mean_min3', 'area', 'view_fps10', 'density', 'area_density')
ANGULAR_BANDWIDTH_DEG = 30.0


def aggregate(features, areas, camera_positions, center, *, method='mean', observation_ids=None):
    features = np.asarray(features, np.float64)
    areas = np.asarray(areas, np.float64)
    positions = np.asarray(camera_positions, np.float64)
    center = np.asarray(center, np.float64)
    if features.ndim != 2 or not len(features) or areas.shape != (len(features),) or positions.shape != (len(features), 3) or center.shape != (3,):
        raise ValueError('incompatible observation arrays')
    if method not in METHODS or any(not np.isfinite(x).all() for x in (features, areas, positions, center)) or (areas <= 0).any():
        raise ValueError('invalid method or observations')
    n = len(features)
    ids = np.arange(n) if observation_ids is None else np.asarray(observation_ids)
    if ids.shape != (n,) or len(np.unique(ids)) != n:
        raise ValueError('observation IDs must be distinct')
    direction = positions-center
    norms = np.linalg.norm(direction, axis=1)
    if (norms < 1e-9).any():
        raise ValueError('camera at object center; viewing direction undefined')
    direction /= norms[:, None]
    dissimilarity = np.maximum(0, 1-np.clip(direction @ direction.T, -1, 1))
    # At 30 degrees the angular correlation is exp(-1). Chosen before scores;
    # no per-scene threshold or search. K_ii = 1 and K_ij in (0,1].
    kernel = np.exp(-dissimilarity / (1-np.cos(np.deg2rad(ANGULAR_BANDWIDTH_DEG))))
    weights = np.ones(n)
    if method == 'area':
        weights = areas.copy()
    elif method == 'density':
        weights = 1 / kernel.sum(axis=1)
    elif method == 'area_density':
        quality = areas / areas.max()
        weights = quality / (kernel @ quality)
    elif method == 'view_fps10':
        # Area seeds the set; farthest angular views afterwards, stable ID ties.
        first = min(range(n), key=lambda i: (-areas[i], ids[i]))
        selected = [first]
        while len(selected) < min(10, n):
            candidates = [i for i in range(n) if i not in selected]
            selected.append(min(candidates, key=lambda i: (-dissimilarity[i, selected].min(), ids[i])))
        weights = np.zeros(n); weights[selected] = 1
    weights /= weights.sum()
    descriptor = weights @ features
    norm = np.linalg.norm(descriptor)
    if norm < 1e-12:
        raise ValueError('degenerate weighted descriptor')
    return dict(descriptor=descriptor/norm, weights=weights,
                effective_views=float(1/(weights @ kernel @ weights)),
                retained_by_count=method != 'mean_min3' or n >= 3)
