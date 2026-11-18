"""Conservative spatial rollback of the cached native object-fusion history.

Unchanged objects are reusable only outside an interaction-envelope closure.
Bounds cover ALL jittered raw supports, including points later discarded by
DBSCAN. This is intentionally conservative. Large corrections can select the
whole map; neither constant update cost nor universal speedup is promised.
"""
from copy import deepcopy
import time
import numpy as np
from .replay_fusion import ConceptGraphsBackend, ReplayableObjectMap, _pose


def envelope(bounds):
    bounds = [b for b in bounds if b is not None]
    if not bounds:
        return None
    return np.min([b[0] for b in bounds], axis=0), np.max([b[1] for b in bounds], axis=0)


def interaction_closure(groups, old_bounds, new_bounds, changed, margin):
    """Close old member groups under old/new raw-support AABB interactions.

    A group can split or combine during replay. Its resulting voxel centroids
    remain inside the union envelope of its raw points. Distinct envelopes
    separated by more than the native nearest-point radius cannot associate.
    Recompute the selected union after every addition, including empty gaps.
    """
    regions = [envelope([b for i in group for b in (old_bounds[i], new_bounds[i])])
               for group in groups]
    selected = {j for j, group in enumerate(groups) if set(group).intersection(changed)}
    iterations = 0
    while selected:
        iterations += 1
        area = envelope([regions[j] for j in selected])
        if area is None:
            break
        added = {j for j, box in enumerate(regions) if j not in selected and box is not None
                 and np.all(box[0] <= area[1] + margin)
                 and np.all(area[0] <= box[1] + margin)}
        if not added:
            break
        selected.update(added)
    return set().union(*(set(groups[j]) for j in selected)) if selected else set(), iterations


class PreparedConceptGraphsBackend(ConceptGraphsBackend):
    """Native core with owned pre-integration records for deterministic reuse."""
    def __init__(self, config, seed=0):
        super().__init__(config, seed)
        if config.filter_interval > 0 or config.merge_interval > 0:
            raise ValueError('cached replay currently requires online filter/merge disabled; final processing is supported')
        self.prepared, self.raw_bounds, self.jittered = {}, {}, {}
        self.frame_ids = []
        self._global_started = False
        self._final_objects = None

    def _snapshot(self, finalize):
        # Both cached-full and local controls reuse already requested final
        # snapshots, so final-query caching is not credited to local replay.
        if not finalize:
            return self._objects
        if self._final_objects is None:
            self._final_objects = super()._snapshot(True)
        return self._final_objects

    def prepare(self, obs, pose, jittered):
        cfg = self.config
        world = jittered @ pose[:3, :3].T + pose[:3, 3]
        bounds = (world.min(axis=0), world.max(axis=0)) if len(world) else None
        if len(jittered) < max(cfg.min_points_threshold, 5):
            return None, bounds
        pcd = self.o3d.geometry.PointCloud()
        pcd.points = self.o3d.utility.Vector3dVector(np.array(jittered, order='C', copy=True))
        pcd.colors = self.o3d.utility.Vector3dVector(np.array(obs.colors, order='C', copy=True))
        pcd = self.utils.process_pcd(pcd.transform(np.array(pose)), cfg)
        box = self.utils.get_bounding_box(cfg, pcd)
        if box.volume() < 1e-6:
            return None, bounds
        return dict(pcd=pcd, bbox=box, num_detections=1,
            observation_ids=[obs.observation_id], class_id=[obs.class_id],
            clip_ft=self.torch.tensor(obs.image_feature), text_ft=self.torch.tensor(obs.text_feature)), bounds

    def apply_frame(self, ids, selected=None):
        """Keep the ORIGINAL frame clock and global denoise eligibility.

        Even a frame with no selected detections can trigger denoising of the
        selected objects when the complete native stream had detections.
        """
        self._final_objects = None
        global_nonempty = any(self.prepared[i] is not None for i in ids)
        already_started = self._global_started
        self._global_started |= global_nonempty
        detections = self.DetectionList([deepcopy(self.prepared[i]) for i in ids
            if self.prepared[i] is not None and (selected is None or i in selected)])
        cfg = self.config
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
        if global_nonempty and already_started:
            for interval, function in ((cfg.denoise_interval, self.utils.denoise_objects),
                    (cfg.filter_interval, self.utils.filter_objects), (cfg.merge_interval, self.utils.merge_objects)):
                if interval > 0 and self._frame_count % interval == 0:
                    self._objects = function(cfg, self._objects)

    def integrate(self, observations, pose):
        ids = []
        for obs in observations:
            oid = obs.observation_id
            self._ids.append(oid)
            ids.append(oid)
            points = obs.points + self._rng.normal(0, 4e-3, obs.points.shape)
            points.setflags(write=False)
            self.jittered[oid] = points
            self.prepared[oid], self.raw_bounds[oid] = self.prepare(obs, pose, points)
            if self.prepared[oid] is None:
                self.rejected.append(oid)
        self.frame_ids.append(ids)
        self.apply_frame(ids)
        current = self.labels(finalize=False)
        return {i: current[i] for i in ids}

    def corrected(self, observations, frame_poses, changed, mode):
        start = time.perf_counter()
        candidate = PreparedConceptGraphsBackend(self.config)
        candidate._rng.set_state(self._rng.get_state())
        candidate._ids = list(self._ids)
        candidate.frame_ids = [list(ids) for ids in self.frame_ids]
        candidate.jittered = dict(self.jittered)
        candidate.prepared = dict(self.prepared)
        candidate.raw_bounds = dict(self.raw_bounds)
        for oid in sorted(changed):
            obs = observations[oid]
            candidate.prepared[oid], candidate.raw_bounds[oid] = candidate.prepare(
                obs, frame_poses[obs.frame_id], self.jittered[oid])
        preparation_s = time.perf_counter()-start
        cfg = self.config
        # The bound argument relies on association requiring positive spatial
        # overlap, point averaging, and monotone online membership histories.
        supported = (cfg.spatial_sim_type == 'overlap' and cfg.match_method == 'sim_sum'
            and cfg.phys_bias == 0 and cfg.sim_threshold > 1
            and cfg.filter_interval <= 0 and cfg.merge_interval <= 0)
        start = time.perf_counter()
        groups = [list(o['observation_ids']) for o in self._objects]
        covered = {i for group in groups for i in group}
        groups.extend([[i] for i in self._ids if i not in covered])
        iterations = 0
        if mode == 'full' or not supported:
            selected = set(self._ids)
            reason = 'cached_full' if mode == 'full' else 'unsupported_configuration_full_fallback'
        else:
            selected, iterations = interaction_closure(groups, self.raw_bounds,
                candidate.raw_bounds, changed, float(cfg.downsample_voxel_size)+1e-9)
            reason = 'envelope_closure'
        closure_s = time.perf_counter()-start
        start = time.perf_counter()
        for ids in candidate.frame_ids:
            candidate.apply_frame(ids, selected)
        # Detached copies keep publication transactional and finalization safe.
        unchanged = [deepcopy(o) for o in self._objects if not selected.intersection(o['observation_ids'])]
        candidate._objects = self.MapObjectList(sorted(list(candidate._objects)+unchanged,
            key=lambda o: min(o['observation_ids'])))
        candidate.rejected = [i for i in candidate._ids if candidate.prepared[i] is None]
        replay_s = time.perf_counter()-start
        stats = dict(mode=mode, reason=reason, changed_observations=len(changed),
            replayed_observations=len(selected), total_observations=len(self._ids),
            reused_online_objects=len(unchanged), closure_iterations=iterations,
            preparation_s=preparation_s, closure_s=closure_s, replay_s=replay_s)
        return candidate, stats


class LocalReplayObjectMap(ReplayableObjectMap):
    """Exactness candidate with a cached-full control, both publicly selectable."""
    def __init__(self, config, mode='local', seed=0):
        if mode not in ['local', 'full']:
            raise ValueError(mode)
        super().__init__(lambda: PreparedConceptGraphsBackend(config, seed))
        self.mode = mode

    def revise_poses(self, poses):
        if set(poses) != set(self._poses):
            raise ValueError('pose revision must cover exactly the observed frames')
        start = time.perf_counter()
        validated = {frame: _pose(pose) for frame, pose in poses.items()}
        old = self.labels()
        changed_frames = {frame for frame in self._poses if not np.array_equal(validated[frame], self._poses[frame])}
        observations = {o.observation_id: o for _, group in self._frames for o in group}
        changed = {i for i, o in observations.items() if o.frame_id in changed_frames}
        candidate, stats = self._backend.corrected(observations, validated, changed, self.mode)
        new = candidate.labels()
        self._backend, self._poses = candidate, validated
        self.revision += 1
        self.events.append(dict(revision=self.revision, reason='pose_revision',
            changed=[dict(observation_id=i, old=old[i], new=new[i]) for i in sorted(old) if old[i] != new[i]],
            update_stats=dict(stats, total_s=time.perf_counter()-start, changed_frames=len(changed_frames))))
        return dict(new)
