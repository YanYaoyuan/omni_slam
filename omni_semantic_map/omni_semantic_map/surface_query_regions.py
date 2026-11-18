"""Causal measured-surface coverage memory for selecting additional image crops.

Unexplained means absent from the supplied predicted masks, not true foreground.
The smoothed miss frequency is a heuristic, not a calibrated object probability.
"""
import numpy as np
from scipy.spatial import cKDTree


def crop_grid(width, height):
    w, h = (width+1)//2, int(np.ceil(2*height/3))
    if min(width, height) <= 0: raise ValueError('Positive image size required')
    return np.asarray([(x, y, x+w, y+h) for y in [0, height-h] for x in [0, (width-w)//2, width-w]], np.int64)


def vertical_surface_weight(xyz, gravity):
    """Continuous planarity x verticality from the current visible cloud only."""
    points = np.asarray(xyz, np.float64); up = np.asarray(gravity, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or up.shape != (3,) or np.linalg.norm(up) == 0: raise ValueError('Invalid geometry/gravity')
    if not np.isfinite(points).all() or not np.isfinite(up).all(): raise ValueError('Finite geometry required')
    if len(points) < 3: return np.zeros(len(points))
    up = up/np.linalg.norm(up); tree = cKDTree(points)
    distance, ids = tree.query(points, k=min(17, len(points)), workers=1)
    local = points[ids[:, :16]]; local = local-local.mean(1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', local, local)/local.shape[1]
    # Include every tie at the neighbor boundary. Arbitrary tree tie-breaking
    # must not make a rigidly transformed planar surface change its utility.
    if len(points) > 16:
        tolerance = 1e-9*np.maximum(1., distance[:, 15])
        tied = np.flatnonzero(distance[:, 16]-distance[:, 15] <= tolerance)
        for i in tied:
            neighbor = tree.query_ball_point(points[i], distance[i, 15]+tolerance[i])
            centered = points[neighbor]-points[neighbor].mean(0)
            covariance[i] = centered.T@centered/len(neighbor)
    values, vectors = np.linalg.eigh(covariance); values = np.maximum(values, 0.)
    planarity = (values[:, 1]-values[:, 0])/np.maximum(values[:, 2], 1e-12)
    vertical = 1.-np.square(vectors[:, :, 0]@up)
    return np.clip(planarity*vertical, 0., 1.)


class SurfaceQueryMemory:
    def __init__(self):
        self.seen = np.zeros(0, np.int64); self.covered = np.zeros(0, np.int64); self.last_frame = -1

    def observe(self, frame, cell_ids, positive):
        ids = np.asarray(cell_ids); positive = np.asarray(positive, bool)
        if ids.ndim != 1 or ids.shape != positive.shape or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0): raise ValueError('Invalid measured cells')
        if frame <= self.last_frame: raise ValueError('Frames must be new and increasing')
        size = int(ids.max(initial=-1))+1
        if size > len(self.seen):
            self.seen = np.pad(self.seen, (0, size-len(self.seen))); self.covered = np.pad(self.covered, (0, size-len(self.covered)))
        self.seen[np.unique(ids)] += 1
        self.covered[np.unique(ids[positive])] += 1
        self.last_frame = int(frame)

    def rank(self, frame, cell_ids, pixel_xy, weights, positive, boxes, uniform_rank):
        if frame != self.last_frame: raise ValueError('Rank only the latest observed frame')
        ids = np.asarray(cell_ids, np.int64); xy = np.asarray(pixel_xy, float); weights = np.asarray(weights, float); positive = np.asarray(positive, bool)
        if xy.shape != (len(ids), 2) or weights.shape != ids.shape or positive.shape != ids.shape: raise ValueError('Aligned supports required')
        if np.any(ids < 0) or np.any(ids >= len(self.seen)) or np.any(self.seen[ids] == 0): raise ValueError('Only observed measured cells may be ranked')
        if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isfinite(xy).all(): raise ValueError('Finite nonnegative weights required')
        unique, inv = np.unique(ids, return_inverse=True); count = np.bincount(inv)
        centers = np.column_stack([np.bincount(inv, weights=xy[:, k])/count for k in range(2)])
        geom = np.bincount(inv, weights=weights)/count
        hit = np.bincount(inv, weights=positive) > 0
        miss_frequency = (self.seen[unique]-self.covered[unique]+1)/(self.seen[unique]+2)
        components = dict(geometry=geom, current=geom*(~hit), memory=geom*miss_frequency)
        scores = {key: [] for key in components}; cell_counts = []
        for x0, y0, x1, y1 in boxes:
            inside = (centers[:, 0]>=x0)&(centers[:, 0]<x1)&(centers[:, 1]>=y0)&(centers[:, 1]<y1)
            cell_counts.append(int(inside.sum()))
            for key, value in components.items(): scores[key].append(float(value[inside].sum()))
        chosen = {key: int(np.argmax(value)) for key, value in scores.items()}
        chosen['uniform'] = int(uniform_rank % len(boxes))
        return dict(chosen=chosen, scores=scores, cell_counts=cell_counts, visible_cells=len(unique),
                    repeat_visible_cells=int((self.seen[unique]>1).sum()), explained_cells=int(hit.sum()))
