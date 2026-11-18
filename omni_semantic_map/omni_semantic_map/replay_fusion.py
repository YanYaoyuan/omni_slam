"""Research v0: immutable observations and transactional replay of object fusion.

The native ConceptGraphs adapter below supplies the association/fusion algorithm.
Replay is a correctness baseline, not a new incremental optimizer or a claim of
novelty. Optional upstream dependencies are imported only by that adapter.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np


def _array(value, shape, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"invalid {name}")
    result.setflags(write=False)
    return result


def _pose(value):
    result = _array(value, (4, 4), "pose")
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(result[:3, :3]), 1., atol=1e-5)):
        raise ValueError("pose must be a rigid camera-to-world transform")
    return result


@dataclass(frozen=True)
class FusionObservation:
    observation_id: int
    frame_id: int
    class_id: int
    points: np.ndarray
    colors: np.ndarray
    image_feature: np.ndarray
    text_feature: np.ndarray

    def __post_init__(self):
        for key in ("observation_id", "frame_id", "class_id"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"invalid {key}")
        n = len(self.points)
        for key in ("points", "colors"):
            object.__setattr__(self, key, _array(getattr(self, key), (n, 3), key))
        if np.any(self.colors < 0) or np.any(self.colors > 1):
            raise ValueError("colors must be in [0, 1]")
        for key in ("image_feature", "text_feature"):
            value = np.array(getattr(self, key), dtype=np.float32, copy=True)
            if value.ndim != 1 or not len(value) or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
                raise ValueError(f"invalid {key}")
            value.setflags(write=False)
            object.__setattr__(self, key, value)


class ReplayableObjectMap:
    """Backend factory returns a fresh engine with integrate/labels/export methods.

    Causal IDs are immutable historical emissions. Current IDs may change, and
    callers must consume a revision plus observation IDs rather than assume an
    object number is permanent. Unassigned observations have label zero.
    """
    def __init__(self, backend_factory):
        self._factory = backend_factory
        self._backend = backend_factory()
        self._frames, self._poses, self._causal = [], {}, {}
        self._ids = set()
        self._feature_dims = None
        self.revision = 0
        self.events = []

    def add_frame(self, frame_id, observations, pose):
        observations = tuple(observations)
        if isinstance(frame_id, bool) or not isinstance(frame_id, (int, np.integer)) or frame_id < 0:
            raise ValueError("invalid frame ID")
        if self._frames and frame_id <= self._frames[-1][0]:
            raise ValueError("frame IDs must increase")
        pose = _pose(pose)
        ids = [o.observation_id for o in observations]
        if len(set(ids)) != len(ids) or self._ids.intersection(ids):
            raise ValueError("duplicate observation ID")
        dims = self._feature_dims
        for o in observations:
            if o.frame_id != frame_id:
                raise ValueError("observation belongs to another frame")
            current = (len(o.image_feature), len(o.text_feature))
            if dims is not None and dims != current:
                raise ValueError("feature dimensions changed")
            dims = current
        try:
            labels = self._backend.integrate(observations, pose)
            if set(labels) != set(ids):
                raise ValueError("backend must label every current observation, including rejected ones")
        except Exception:
            # Discard a partially modified backend; retained evidence is intact.
            self._backend = self._replay(self._poses)
            raise
        self._frames.append((int(frame_id), observations))
        self._poses[frame_id] = pose
        self._ids.update(ids)
        self._feature_dims = dims
        self._causal.update(labels)
        return dict(labels)

    def _replay(self, poses):
        backend = self._factory()
        for frame_id, observations in self._frames:
            backend.integrate(observations, poses[frame_id])
        return backend

    def revise_poses(self, poses):
        """Validate a complete pose table and publish only a successful rebuild."""
        if set(poses) != set(self._poses):
            raise ValueError("pose revision must cover exactly the observed frames")
        validated = {frame: _pose(pose) for frame, pose in poses.items()}
        old = self.labels()
        candidate = self._replay(validated)
        new = candidate.labels(finalize=True)
        self._backend, self._poses = candidate, validated
        self.revision += 1
        self.events.append(dict(revision=self.revision, reason="pose_revision",
            changed=[dict(observation_id=i, old=old[i], new=new[i])
                     for i in sorted(old) if old[i] != new[i]]))
        return dict(new)

    def labels(self, finalize=True):
        return dict(self._backend.labels(finalize=finalize))

    def causal_labels(self):
        return dict(self._causal)

    def export_objects(self, finalize=True):
        """World clouds, boxes, features and evidence IDs at the current revision."""
        return deepcopy(self._backend.export_objects(finalize=finalize))


class ConceptGraphsBackend:
    """Official native association with optional final denoise/filter/merge.

    Input is unperturbed measured camera-frame support. A private RNG reproduces
    native 4 mm jitter in ingestion order, without modifying global NumPy state.
    No GT, categories, co-visibility constraints or pose-error oracle are added.
    """
    def __init__(self, config, seed=0):
        import open3d as o3d
        import torch
        from conceptgraph.slam import mapping, utils
        from conceptgraph.slam.slam_classes import DetectionList, MapObjectList
        self.o3d, self.torch, self.mapping, self.utils = o3d, torch, mapping, utils
        self.DetectionList, self.MapObjectList = DetectionList, MapObjectList
        self.config = deepcopy(config)
        self._objects = MapObjectList()
        self._ids, self.rejected = [], []
        self._rng = np.random.RandomState(seed)
        self._frame_count = 0

    def integrate(self, observations, pose):
        cfg = self.config
        detections = self.DetectionList()
        for obs in observations:
            self._ids.append(obs.observation_id)
            points = obs.points + self._rng.normal(0, 4e-3, obs.points.shape)
            if len(points) < max(cfg.min_points_threshold, 5):
                self.rejected.append(obs.observation_id)
                continue
            pcd = self.o3d.geometry.PointCloud()
            pcd.points = self.o3d.utility.Vector3dVector(points)
            pcd.colors = self.o3d.utility.Vector3dVector(np.array(obs.colors, order='C', copy=True))
            pcd = self.utils.process_pcd(pcd.transform(np.array(pose)), cfg)
            box = self.utils.get_bounding_box(cfg, pcd)
            if box.volume() < 1e-6:
                self.rejected.append(obs.observation_id)
                continue
            detections.append(dict(pcd=pcd, bbox=box, num_detections=1,
                observation_ids=[obs.observation_id], class_id=[obs.class_id],
                clip_ft=self.torch.tensor(obs.image_feature), text_ft=self.torch.tensor(obs.text_feature)))
        self._frame_count += 1
        if detections:
            if not self._objects:
                self._objects.extend(detections)
            else:
                spatial = self.mapping.compute_spatial_similarities(cfg, detections, self._objects)
                visual = self.mapping.compute_visual_similarities(cfg, detections, self._objects)
                similarity = self.mapping.aggregate_similarities(cfg, spatial, visual)
                similarity[similarity < cfg.sim_threshold] = float('-inf')
                self._objects = self.mapping.merge_detections_to_objects(cfg, detections, self._objects, similarity)
                for interval, function in ((cfg.denoise_interval, self.utils.denoise_objects),
                        (cfg.filter_interval, self.utils.filter_objects), (cfg.merge_interval, self.utils.merge_objects)):
                    if interval > 0 and self._frame_count % interval == 0:
                        self._objects = function(cfg, self._objects)
        current = self.labels(finalize=False)
        return {o.observation_id: current[o.observation_id] for o in observations}

    def _snapshot(self, finalize):
        if not finalize:
            return self._objects
        objects = deepcopy(self._objects)
        objects = self.utils.denoise_objects(self.config, objects)
        objects = self.utils.filter_objects(self.config, objects)
        return self.utils.merge_objects(self.config, objects)

    def labels(self, finalize=True):
        result = dict.fromkeys(self._ids, 0)
        for obj in self._snapshot(finalize):
            result.update(dict.fromkeys(obj['observation_ids'], min(obj['observation_ids']) + 1))
        return result

    def export_objects(self, finalize=True):
        return [dict(object_id=min(o['observation_ids']) + 1,
                     observation_ids=list(o['observation_ids']),
                     points=np.asarray(o['pcd'].points).copy(),
                     colors=np.asarray(o['pcd'].colors).copy(),
                     box=np.asarray(o['bbox'].get_box_points()).copy(),
                     image_feature=o['clip_ft'].cpu().numpy().copy(),
                     text_feature=o['text_ft'].cpu().numpy().copy())
                for o in self._snapshot(finalize)]
