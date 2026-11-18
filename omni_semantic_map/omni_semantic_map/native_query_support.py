"""Native-score arbitration on measured pixels, without materializing images."""
import numpy as np


def exclusive_measured_queries(packed_masks, scores, pixel_xy, width, threshold):
    packed = np.asarray(packed_masks); scores = np.asarray(scores); px = np.asarray(pixel_xy)
    if packed.ndim != 3 or scores.shape != (len(packed),) or px.ndim != 2 or px.shape[1] != 2:
        raise ValueError('Invalid masks, scores or measured coordinates')
    if packed.dtype != np.uint8 or not np.issubdtype(px.dtype, np.integer): raise ValueError('Packed bytes and integer pixels required')
    if not np.isfinite(scores).all() or np.any(px < 0) or np.any(px[:, 0] >= width) or np.any(px[:, 1] >= packed.shape[1]): raise ValueError('Invalid score or out-of-image pixel')
    if packed.shape[2] != (width+7)//8: raise ValueError('Packed width mismatch')
    order = np.argsort(-scores, kind='stable'); order = order[scores[order] > threshold]
    winners = np.full(len(px), -1, np.int64)
    for q in order:
        inside = ((packed[q, px[:, 1], px[:, 0]//8] >> (7-px[:, 0]%8)) & 1).astype(bool)
        winners[inside & (winners < 0)] = q
    return winners
