"""Stateful dispatch among dependency reuse, valid-prefix replay and full replay.

No scene names, GT, measured alternative runtimes or post-hoc best-of selection
enter dispatch. Full replay drops the DAG rather than rebuilding it invisibly;
later local changes can still use prefix snapshots in the cold-cache state.
"""
from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np
from .dependency_fusion import DependencyConceptGraphsBackend
from .local_fusion import PreparedConceptGraphsBackend, LocalReplayObjectMap
from .replay_fusion import ReplayableObjectMap


@dataclass(frozen=True)
class DispatchConfig:
    checkpoint_interval: int = 20
    min_prefix_fraction: float = .5
    max_tainted_pair_fraction: float = .5

    def __post_init__(self):
        if self.checkpoint_interval < 1 or not 0 <= self.min_prefix_fraction <= 1 or not 0 <= self.max_tainted_pair_fraction <= 1:
            raise ValueError('invalid dispatch configuration')


def affected_pair_fraction(engine, changed):
    """Conservative taint on the OLD DAG; a cost estimate, not an accuracy gate.

    New associations can introduce extra dependencies. Every route still runs
    real association on the corrected observations, so a bad estimate affects
    runtime rather than retaining stale object memberships.
    """
    tainted = {engine.input_tokens[i] for i in changed}
    for recipe, token in sorted(engine.recipes.items(), key=lambda kv: kv[1]):
        if recipe[0] != 'input' and any(i in tainted for i in recipe[1:]):
            tainted.add(token)
    pairs = engine.pair_scores
    affected = sum(a in tainted or b in tainted for a,b in pairs)
    return affected / len(pairs) if pairs else 1.


class AdaptiveFusionBackend:
    def __init__(self, config, dispatch=DispatchConfig(), policy='auto', seed=0):
        if policy not in ['auto', 'checkpoint']:
            raise ValueError(policy)
        self.config, self.dispatch, self.policy = deepcopy(config), dispatch, policy
        self.engine = (DependencyConceptGraphsBackend(config, seed) if policy == 'auto'
                       else PreparedConceptGraphsBackend(config, seed))
        self.checkpoints = {0: self.engine.MapObjectList()}

    def _capture(self, engine, checkpoints):
        if engine._frame_count % self.dispatch.checkpoint_interval == 0:
            checkpoints[engine._frame_count] = deepcopy(engine._objects)

    def integrate(self, observations, pose):
        labels = self.engine.integrate(observations, pose)
        self._capture(self.engine, self.checkpoints)
        return labels

    def labels(self, finalize=True):
        return self.engine.labels(finalize)

    def export_objects(self, finalize=True):
        return self.engine.export_objects(finalize)

    def corrected(self, observations, frame_poses, changed, mode):
        started = time.perf_counter()
        engine = self.engine
        count = len(engine.frame_ids)
        first = next((j for j, ids in enumerate(engine.frame_ids) if changed.intersection(ids)), count)
        prefix = max(k for k in self.checkpoints if k <= first)
        warm = isinstance(engine, DependencyConceptGraphsBackend)
        estimate = None
        if not changed:
            return self, dict(route='noop', reason='no_observation_pose_changed', planner_s=time.perf_counter()-started,
                              changed_observations=0, replayed_frames=0, warm_dag=warm)
        if prefix > 0 and (self.policy == 'checkpoint' or prefix/count >= self.dispatch.min_prefix_fraction):
            route = 'checkpoint'
        elif self.policy == 'auto' and warm:
            estimate = affected_pair_fraction(engine, changed)
            route = 'dependency' if estimate < self.dispatch.max_tainted_pair_fraction else 'full'
        elif prefix > 0:
            route = 'checkpoint'
        else:
            route = 'full'
        planner_s = time.perf_counter()-started
        if route == 'dependency':
            candidate, checkpoints, work = self._dependency(observations, frame_poses, changed)
            replayed = count
        else:
            start_frame = prefix if route == 'checkpoint' else 0
            candidate, checkpoints, work = self._prefix(observations, frame_poses, changed, start_frame)
            replayed = count-start_frame
        result = object.__new__(AdaptiveFusionBackend)
        result.config, result.dispatch, result.policy = self.config, self.dispatch, self.policy
        result.engine, result.checkpoints = candidate, checkpoints
        return result, dict(route=route, reason='fixed_input_only_dispatch', planner_s=planner_s,
            first_changed_frame_index=first, valid_prefix_frames=prefix, total_frames=count,
            replayed_frames=replayed, warm_dag=warm, dag_retained=isinstance(candidate, DependencyConceptGraphsBackend),
            predicted_tainted_pair_fraction=estimate, changed_observations=len(changed),
            checkpoint_count=len(checkpoints), **work)

    def _base_copy(self, candidate):
        old = self.engine
        candidate._rng.set_state(old._rng.get_state())
        candidate._ids = list(old._ids)
        candidate.frame_ids = [list(ids) for ids in old.frame_ids]
        for name in ['jittered','prepared','raw_bounds']:
            setattr(candidate, name, dict(getattr(old, name)))

    def _prefix(self, observations, poses, changed, start_frame):
        started = time.perf_counter()
        candidate = PreparedConceptGraphsBackend(self.config)
        self._base_copy(candidate)
        for oid in sorted(changed):
            o = observations[oid]
            candidate.prepared[oid], candidate.raw_bounds[oid] = candidate.prepare(o, poses[o.frame_id], candidate.jittered[oid])
        preparation_s = time.perf_counter()-started
        checkpoints = {k:v for k,v in self.checkpoints.items() if k <= start_frame}
        candidate._objects = deepcopy(self.checkpoints[start_frame])
        candidate._frame_count = start_frame
        candidate._global_started = bool(candidate._objects)
        started = time.perf_counter()
        for ids in candidate.frame_ids[start_frame:]:
            candidate.apply_frame(ids)
            self._capture(candidate, checkpoints)
        candidate.rejected = [i for i in candidate._ids if candidate.prepared[i] is None]
        return candidate, checkpoints, dict(preparation_s=preparation_s, replay_s=time.perf_counter()-started)

    def _dependency(self, observations, poses, changed):
        started = time.perf_counter()
        old = self.engine
        candidate = DependencyConceptGraphsBackend(self.config)
        self._base_copy(candidate)
        for name in ['input_tokens','recipes','states','pair_scores']:
            setattr(candidate, name, dict(getattr(old, name)))
        candidate.next_token = old.next_token
        for oid in sorted(changed):
            o = observations[oid]
            candidate.prepared[oid], candidate.raw_bounds[oid] = candidate.prepare(o, poses[o.frame_id], candidate.jittered[oid])
        preparation_s = time.perf_counter()-started
        checkpoints = {0: candidate.MapObjectList()}
        started = time.perf_counter()
        for ids in candidate.frame_ids:
            candidate.apply_frame(ids)
            self._capture(candidate, checkpoints)
        candidate.states = {k:v for k,v in candidate.states.items() if k in candidate.used_tokens}
        candidate.recipes = {k:v for k,v in candidate.recipes.items() if v in candidate.used_tokens}
        candidate.pair_scores = {k:v for k,v in candidate.pair_scores.items() if k in candidate.used_pairs}
        candidate.rejected = [i for i in candidate._ids if candidate.prepared[i] is None]
        return candidate, checkpoints, dict(preparation_s=preparation_s, replay_s=time.perf_counter()-started, **candidate.work)


class AdaptiveReplayObjectMap(LocalReplayObjectMap):
    def __init__(self, config, dispatch=DispatchConfig(), policy='auto', seed=0):
        frozen_config = deepcopy(config)
        ReplayableObjectMap.__init__(self, lambda: AdaptiveFusionBackend(frozen_config, dispatch, policy, seed))
        self.mode = policy
