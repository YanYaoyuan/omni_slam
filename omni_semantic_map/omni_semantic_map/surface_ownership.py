"""Learned allocation of previously unowned measurements to existing objects.

An explicit null alternative prevents a candidate mask from forcing ownership.
This window-end research module preserves core geometry and object identities;
it does not discover objects or infer missing depth.
"""
import numpy as np
import torch
from torch import nn


FEATURES = ('log_nearest_core', 'log_nearest_other_core', 'core_distance_ratio',
            'log_center_distance_scaled', 'outside_x_scaled', 'outside_y_scaled',
            'outside_z_scaled', 'log_extent_x', 'log_extent_y', 'log_extent_z',
            'is_car', 'is_building', 'log_exact_own_frames', 'log_cell_own_frames',
            'log_cell_other_frames', 'log_visible_frames', 'own_visible_fraction',
            'cell_own_fraction', 'mean_mask_score', 'log_object_frames')
GEOMETRY_FEATURES = 12


class OwnershipScorer(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimensions, 32), nn.ReLU(),
                                 nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def grouped_loss(logits, valid, positive):
    """Set likelihood: multiple candidate IDs may denote the same GT object.

Null has fixed logit zero and is the target iff none of the candidate owners is
correct. Padding is never an alternative. Caller removes unknown supervision.
"""
    if logits.shape != valid.shape or valid.shape != positive.shape:
        raise ValueError('Mismatched candidate arrays')
    if torch.any(positive & ~valid) or torch.any(~valid.any(1)):
        raise ValueError('Invalid candidate supervision')
    scores = torch.cat((torch.zeros_like(logits[:, :1]), logits.masked_fill(~valid, -torch.inf)), 1)
    target = torch.cat((~positive.any(1, keepdim=True), positive), 1)
    return torch.logsumexp(scores, 1) - torch.logsumexp(scores.masked_fill(~target, -torch.inf), 1)


def allocate(logits, valid, object_ids, *, independent=False):
    """Return at most one owner per source; ambiguous/low confidence stays null.

Joint posterior must exceed 1/2. In the independent ablation each pair has its
own sigmoid, then only the highest pair may win. Exact ties abstain in both.
"""
    z = np.asarray(logits, float); valid = np.asarray(valid, bool)
    ids = np.asarray(object_ids, np.int64)
    if z.shape != valid.shape or ids.shape != z.shape or z.ndim != 2:
        raise ValueError('Mismatched candidate arrays')
    if not np.isfinite(z[valid]).all() or np.any(ids[valid] <= 0):
        raise ValueError('Finite scores and positive object IDs required')
    scores = np.where(valid, z, -np.inf)
    maximum = np.maximum(0., scores.max(1))
    if independent:
        probability = 1 / (1 + np.exp(-np.clip(scores, -80, 80)))
        probability[~valid] = 0
    else:
        exp = np.exp(scores - maximum[:, None])
        probability = exp / (np.exp(-maximum) + exp.sum(1))[:, None]
    best = probability.argmax(1)
    confidence = probability[np.arange(len(best)), best]
    ties = (probability == confidence[:, None]).sum(1) > 1
    chosen = ids[np.arange(len(best)), best].copy()
    chosen[(confidence <= .5) | ties] = 0
    return chosen, confidence


def append_owned(core_xyz, core_instance, source_xyz, source_ids, owners):
    """Immutable old map prefix, exact measured additions, exclusive source IDs."""
    sid = np.asarray(source_ids, np.int64); owners = np.asarray(owners, np.int64)
    if sid.ndim != 1 or sid.shape != owners.shape or len(np.unique(sid)) != len(sid):
        raise ValueError('Each measured source must have exactly one decision')
    if np.any(sid < 0) or np.any(sid >= len(source_xyz)) or np.any(owners < 0):
        raise ValueError('Invalid measured-source assignment')
    if not np.isin(owners[owners > 0], np.unique(core_instance)).all():
        raise ValueError('Cannot invent an object identity')
    use = owners > 0
    return dict(xyz=np.concatenate((core_xyz, source_xyz[sid[use]])),
                instance=np.r_[core_instance, owners[use]], added_source_ids=sid[use])
