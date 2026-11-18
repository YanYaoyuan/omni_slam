"""Reversible appearance partitions using disjoint observation blocks.

Frozen patch descriptors propose a source-point partition. Held-out descriptor
agreement and independently matched masks decide whether it is an object edit.
This module has no reference-label input or semantic-class-dependent rules.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from .reversible_split import _distinct_view, SplitConfig, object_view


@dataclass(frozen=True)
class AppearanceConfig:
    minimum_part_points: int = 12
    minimum_part_fraction: float = .05
    minimum_feature_coverage: float = .50
    minimum_prototype_gap: float = .10
    minimum_validation_margin: float = .03
    minimum_validation_agreement: float = .80
    mask_part_coverage: float = .50
    mask_part_purity: float = .80
    minimum_support_views: int = 2
    kmeans_iterations: int = 20


def unit(array):
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)


def partition(xyz, indices, train, train_count, cfg=AppearanceConfig()):
    observed = train_count[indices] > 0
    if len(indices) < 2*cfg.minimum_part_points or observed.mean() < cfg.minimum_feature_coverage:
        return None
    measured = indices[observed]
    f = unit(train[measured])
    mean = unit(f.mean(0))
    a = int(np.argmin(f @ mean))
    b = int(np.argmin(f @ f[a]))
    centers = f[[a,b]].copy()
    previous = None
    for _ in range(cfg.kmeans_iterations):
        assignment = np.argmax(f @ centers.T, axis=1)
        if min(np.bincount(assignment, minlength=2)) == 0:
            return None
        if previous is not None and np.array_equal(assignment, previous):
            break
        centers = unit(np.stack([f[assignment == k].mean(0) for k in (0,1)]))
        previous = assignment.copy()
    gap = 1-float(centers[0] @ centers[1])
    if gap < cfg.minimum_prototype_gap:
        return None
    child = np.empty(len(indices), np.uint8)
    child[observed] = assignment
    if not observed.all():
        _, near = cKDTree(xyz[measured]).query(xyz[indices[~observed]], workers=1)
        child[~observed] = assignment[near]
    counts = np.bincount(child, minlength=2)
    if min(counts) < max(cfg.minimum_part_points, cfg.minimum_part_fraction*len(indices)):
        return None
    return dict(source_indices=indices, child=child, centers=centers,
                prototype_gap=gap, feature_coverage=float(observed.mean()))


def validate(proposal, labels, validation, validation_count, views,
             validation_indices, cfg=AppearanceConfig()):
    ids, child = proposal['source_indices'], proposal['child']
    observed = validation_count[ids] > 0
    selected = ids[observed]
    if min(np.bincount(child[observed], minlength=2)) < cfg.minimum_part_points:
        return dict(accepted=False, reason='insufficient_heldout_features')
    scores = unit(validation[selected]) @ proposal['centers'].T
    agreement = float((scores.argmax(1) == child[observed]).mean())
    margin = scores[np.arange(len(selected)), child[observed]]-scores[np.arange(len(selected)), 1-child[observed]]
    part_margins = [float(margin[child[observed] == k].mean()) for k in (0,1)]
    mapping = np.full(len(labels), -1, np.int8);mapping[ids] = child
    accepted_views = [[], []]
    events = []
    diversity = SplitConfig()
    for vi in validation_indices:
        view = views[vi]
        visible, masks = object_view(view, labels, proposal['object_id'])
        parts = [mapping[visible] == k for k in (0,1)]
        if min(int(p.sum()) for p in parts) < 3:
            continue
        for k in (0,1):
            if not _distinct_view(view.camera_center, proposal['camera_center'], proposal['center'], diversity):
                continue
            if any(not _distinct_view(view.camera_center, c, proposal['center'], diversity) for c in accepted_views[k]):
                continue
            for mi, mask in enumerate(masks):
                overlap = int((mask & parts[k]).sum())
                if overlap < 3 or overlap/max(int(parts[k].sum()),1) < cfg.mask_part_coverage:
                    continue
                if overlap/max(int(mask.sum()),1) < cfg.mask_part_purity:
                    continue
                accepted_views[k].append(view.camera_center)
                events.append(dict(part=k,frame_id=view.frame_id,observation_id=int(view.observation_ids[mi]),
                                   coverage=overlap/int(parts[k].sum()),purity=overlap/int(mask.sum())))
                break
    support = [len(v) for v in accepted_views]
    accepted = (agreement >= cfg.minimum_validation_agreement and
                min(part_margins) >= cfg.minimum_validation_margin and
                min(support) >= cfg.minimum_support_views)
    return dict(accepted=bool(accepted), reason='accepted' if accepted else 'insufficient_cross_view_object_evidence',
                heldout_points=len(selected), feature_agreement=agreement,
                part_margins=part_margins, independent_mask_support=support, events=events)
