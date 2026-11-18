"""Local instance re-partitioning with separate proposal/validation views.

No classes, reference labels, model training, point synthesis or pose fitting.
The caller supplies depth-verified point/mask incidence. A proposal never reads
the validation block, and each object receives at most one proposal.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SplitConfig:
    minimum_part_points: int = 12
    minimum_part_fraction: float = .03
    minimum_pair_coverage: float = .70
    maximum_mask_overlap: float = .10
    proposal_fraction: float = .50
    validation_fraction: float = .65
    minimum_visible_part_points: int = 3
    minimum_separating_views: int = 2
    minimum_view_angle_deg: float = 12.
    minimum_camera_translation_m: float = .15
    minimum_view_gain: float = .10
    complexity_penalty: float = .05


@dataclass
class EvidenceView:
    frame_id: int
    camera_center: np.ndarray
    indices: np.ndarray  # global source indices with valid measured depth
    membership: np.ndarray  # raw eroded masks x visible source indices
    observation_ids: np.ndarray

    def __post_init__(self):
        if self.membership.shape != (len(self.observation_ids), len(self.indices)):
            raise ValueError('incompatible incidence shape')
        if len(np.unique(self.indices)) != len(self.indices):
            raise ValueError('a point may occur only once per view')


def object_view(view, labels, object_id):
    selected = np.flatnonzero(labels[view.indices] == object_id)
    return view.indices[selected], view.membership[:, selected]


def _distinct_view(a, b, center, cfg):
    if np.linalg.norm(a - b) < cfg.minimum_camera_translation_m:
        return False
    u, v = a - center, b - center
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm < 1e-12:
        return False
    angle = math.degrees(math.acos(float(np.clip(u @ v / norm, -1., 1.))))
    return angle >= cfg.minimum_view_angle_deg


def propose_splits(xyz, labels, views, cfg=SplitConfig()):
    """Build one deterministic candidate per object from the initial block only."""
    xyz, labels = np.asarray(xyz), np.asarray(labels)
    if xyz.shape != (len(labels), 3) or not np.isfinite(xyz).all():
        raise ValueError('finite point array with one label per point required')
    best = {}
    sizes = {int(o): int((labels == o).sum()) for o in np.unique(labels) if o > 0}
    for vi, view in enumerate(views[:int(len(views) * cfg.proposal_fraction)]):
        for oid in np.unique(labels[view.indices]):
            if oid <= 0:
                continue
            ids, membership = object_view(view, labels, oid)
            required = max(cfg.minimum_part_points, int(math.ceil(sizes[int(oid)] * cfg.minimum_part_fraction)))
            eligible = np.flatnonzero(membership.sum(1) >= required)
            for pos, a in enumerate(eligible):
                for b in eligible[pos + 1:]:
                    ma, mb = membership[a], membership[b]
                    na, nb = int(ma.sum()), int(mb.sum())
                    overlap = int((ma & mb).sum()) / min(na, nb)
                    union = int((ma | mb).sum()) / max(len(ids), 1)
                    if overlap > cfg.maximum_mask_overlap or union < cfg.minimum_pair_coverage:
                        continue
                    ia, ib = ids[ma & ~mb], ids[mb & ~ma]
                    if min(len(ia), len(ib)) < required:
                        continue
                    score = union * math.sqrt(2 * min(len(ia), len(ib)) / (len(ia) + len(ib)))
                    # Stable first-view/mask tie break; validation never ranks proposals.
                    if oid not in best or score > best[oid]['proposal_score']:
                        best[int(oid)] = dict(object_id=int(oid), view_index=vi,
                            frame_id=int(view.frame_id), observation_ids=[int(view.observation_ids[a]), int(view.observation_ids[b])],
                            proposal_score=float(score), seed_a=ia, seed_b=ib,
                            camera_center=view.camera_center.copy())
    proposals = []
    for oid, candidate in sorted(best.items()):
        ids = np.flatnonzero(labels == oid)
        da = cKDTree(xyz[candidate['seed_a']]).query(xyz[ids], workers=1)[0]
        db = cKDTree(xyz[candidate['seed_b']]).query(xyz[ids], workers=1)[0]
        child = (db < da).astype(np.uint8)
        if min(int(child.sum()), int((child == 0).sum())) < cfg.minimum_part_points:
            continue
        proposals.append(dict(candidate, source_indices=ids, child=child,
                              center=xyz[ids].mean(0)))
    return proposals


def view_gain(membership, child, cfg):
    """One-mask vs exclusive two-mask explanation on the same visible points.

    Membership is recomputed from geometry and raw masks, never inherited object
    associations. Overlapping masks cannot provide a separating explanation.
    """
    if len(child) == 0 or len(membership) == 0:
        return None
    parts = [child == k for k in (0, 1)]
    if min(int(x.sum()) for x in parts) < cfg.minimum_visible_part_points:
        return None
    keep = float(membership.sum(1).max() / len(child))
    counts = np.column_stack([membership[:, part].sum(1) for part in parts])
    best = None
    for a in range(len(membership)):
        for b in range(len(membership)):
            if a == b or min(counts[a, 0], counts[b, 1]) < cfg.minimum_visible_part_points:
                continue
            size = min(int(membership[a].sum()), int(membership[b].sum()))
            if not size or (membership[a] & membership[b]).sum() / size > cfg.maximum_mask_overlap:
                continue
            # Wrong-child incidence is a penalty, not supporting evidence.
            score = float((counts[a, 0] + counts[b, 1] - counts[a, 1] - counts[b, 0]) / len(child))
            if best is None or score > best:
                best = score
    # Seeing a single mask covering both parts actively supports keeping them.
    if best is None:
        return -keep
    return best - keep


def validate_split(proposal, labels, views, cfg=SplitConfig()):
    ids = proposal['source_indices']
    assignment = np.full(len(labels), -1, np.int8)
    assignment[ids] = proposal['child']
    selected, gains, events = [], [], []
    for view in views[int(len(views) * cfg.validation_fraction):]:
        if not _distinct_view(view.camera_center, proposal['camera_center'], proposal['center'], cfg):
            continue
        if any(not _distinct_view(view.camera_center, center, proposal['center'], cfg) for center in selected):
            continue
        visible, membership = object_view(view, labels, proposal['object_id'])
        gain = view_gain(membership, assignment[visible], cfg)
        if gain is None:
            continue
        selected.append(view.camera_center)
        gains.append(gain)
        events.append(dict(frame_id=int(view.frame_id), visible_points=len(visible), gain=float(gain)))
    separating = sum(g >= cfg.minimum_view_gain for g in gains)
    mean = float(np.mean(gains)) if gains else None
    accepted = separating >= cfg.minimum_separating_views and mean > cfg.complexity_penalty
    return dict(accepted=bool(accepted), distinct_views=len(gains), separating_views=separating,
                mean_gain=mean, events=events,
                reason='accepted' if accepted else 'insufficient_independent_evidence')


def apply_splits(labels, proposals, accepted):
    """Lossless source-index partition and ID lineage; rollback needs no remapping."""
    labels = np.asarray(labels)
    result = labels.copy()
    next_id = int(labels.max(initial=0)) + 1
    lineage = []
    for p, decision in zip(proposals, accepted):
        if not decision:
            continue
        ids, child = p['source_indices'], p['child']
        if not np.all(labels[ids] == p['object_id']):
            raise ValueError('proposal ownership differs from source')
        small = 1 if int(child.sum()) <= int((child == 0).sum()) else 0
        changed = ids[child == small]
        result[changed] = next_id
        lineage.append(dict(parent=p['object_id'], child=next_id,
                            changed_source_indices=changed, proposal_frame=p['frame_id']))
        next_id += 1
    return result, lineage


def rollback(labels, lineage):
    result = np.asarray(labels).copy()
    for event in reversed(lineage):
        result[event['changed_source_indices']] = event['parent']
    return result
