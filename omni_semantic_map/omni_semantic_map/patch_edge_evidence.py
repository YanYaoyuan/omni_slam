"""Exact bilinear patch cosine through a small Gram matrix.

This is a readout utility, not a trained boundary detector. Zero-norm features
are unavailable evidence, not negative evidence.
"""
import numpy as np


def patch_weights(pixel_xy, image_hw, grid_hw):
    xy = np.asarray(pixel_xy, np.float64)
    h, w = image_hw
    gh, gw = grid_hw
    if min(h, w, gh, gw) <= 0 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('Invalid image/grid/pixel shape')
    if not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy >= [w, h]):
        raise ValueError('Pixel outside image')
    uv = (xy + .5) * [gw/w, gh/h] - .5
    lo = np.floor(uv).astype(np.int64)
    frac = uv-lo
    indices, weights = [], []
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        indices.append(np.clip(lo[:, 1]+dy, 0, gh-1)*gw + np.clip(lo[:, 0]+dx, 0, gw-1))
        weights.append((frac[:, 0] if dx else 1-frac[:, 0]) *
                       (frac[:, 1] if dy else 1-frac[:, 1]))
    return np.stack(indices, 1), np.stack(weights, 1)


def bilinear_edge_cosine(grid, pixel_xy, image_hw, left, right):
    grid = np.asarray(grid, np.float64)
    if grid.ndim != 3 or not np.isfinite(grid).all():
        raise ValueError('Invalid dense feature grid')
    indices, weights = patch_weights(pixel_xy, image_hw, grid.shape[:2])
    left, right = np.asarray(left, np.int64), np.asarray(right, np.int64)
    if left.shape != right.shape or left.ndim != 1 or np.any(left < 0) or np.any(right < 0) or np.any(left >= len(indices)) or np.any(right >= len(indices)):
        raise ValueError('Invalid edge endpoints')
    flat = grid.reshape(-1, grid.shape[-1])
    gram = flat @ flat.T
    norms = np.zeros(len(indices), np.float64)
    dot = np.zeros(len(left), np.float64)
    for i in range(4):
        for j in range(4):
            norms += weights[:, i]*weights[:, j]*gram[indices[:, i], indices[:, j]]
            dot += weights[left, i]*weights[right, j]*gram[indices[left, i], indices[right, j]]
    denominator = np.sqrt(np.maximum(norms[left], 0)*np.maximum(norms[right], 0))
    valid = denominator > 1e-12
    cosine = np.full(len(left), np.nan)
    cosine[valid] = np.clip(dot[valid]/denominator[valid], -1, 1)
    return cosine, valid
