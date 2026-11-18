"""Dependency-local re-evaluation of the native fusion computation history.

Exact-input memoization, not approximate geometric gating: raw measurement pose
versions name input nodes; ordered fusion/denoise inputs name derived nodes.
Only changed pair scores and derived objects are recomputed. All frame indices
are visited, retaining native causal order and denoising schedule.
"""
from copy import deepcopy
import hashlib
import time
import numpy as np
from .local_fusion import PreparedConceptGraphsBackend, LocalReplayObjectMap
from .replay_fusion import ReplayableObjectMap


class DependencyConceptGraphsBackend(PreparedConceptGraphsBackend):
    def __init__(self, config, seed=0):
        super().__init__(config, seed)
        if config.spatial_sim_type != 'overlap':
            raise ValueError('dependency cache currently supports the native overlap configuration')
        self.input_tokens, self.recipes, self.states, self.pair_scores = {}, {}, {}, {}
        self.object_tokens = []
        self.next_token = 0
        self.used_tokens, self.used_pairs = set(), set()
        self.work = dict(pair_hits=0, pair_misses=0, fusion_hits=0, fusion_misses=0,
                         denoise_hits=0, denoise_misses=0, native_fallback_frames=0)

    def token(self, recipe):
        if recipe not in self.recipes:
            self.recipes[recipe] = self.next_token
            self.next_token += 1
        result = self.recipes[recipe]
        self.used_tokens.add(result)
        return result

    def prepare(self, obs, pose, jittered):
        record, bounds = super().prepare(obs, pose, jittered)
        version = hashlib.sha256(np.asarray(pose, dtype=np.float64).tobytes()).digest()
        token = self.token(('input', obs.observation_id, version))
        self.input_tokens[obs.observation_id] = token
        self.states[token] = record
        return record, bounds

    def derived(self, operation, inputs):
        token = self.token((operation, *inputs))
        if token in self.states:
            self.work[operation+'_hits'] += 1
        else:
            self.work[operation+'_misses'] += 1
            if operation == 'fusion':
                self.states[token] = self.utils.merge_obj2_into_obj1(self.config,
                    deepcopy(self.states[inputs[0]]), deepcopy(self.states[inputs[1]]), run_dbscan=False)
            elif operation == 'denoise':
                self.states[token] = self.utils.denoise_objects(self.config,
                    self.MapObjectList([deepcopy(self.states[inputs[0]])]))[0]
            else:
                raise ValueError(operation)
        return token

    def apply_frame(self, ids, selected=None):
        if selected is not None:
            raise ValueError('dependency replay visits the full frame sequence')
        self._final_objects = None
        det_tokens = [self.input_tokens[i] for i in ids if self.prepared[i] is not None]
        self.used_tokens.update(self.input_tokens[i] for i in ids)
        previous = list(self.object_tokens)
        self._frame_count += 1
        if det_tokens and not previous:
            self.object_tokens.extend(det_tokens)
        elif det_tokens:
            # Native scores are computed against the pre-frame map, then all
            # same-frame detections merge sequentially using those scores.
            # Native PyTorch3D failure falls back for the WHOLE frame matrix.
            # A per-pair cache must not hide that nonlocal numerical policy.
            from conceptgraph.utils.ious import compute_3d_iou_accuracte_batch
            full_detections = self.DetectionList([self.states[t] for t in det_tokens])
            full_objects = self.MapObjectList([self.states[t] for t in previous])
            fallback_values = None
            try:
                compute_3d_iou_accuracte_batch(full_objects.get_stacked_values_torch('bbox'),
                    full_detections.get_stacked_values_torch('bbox'))
            except ValueError:
                self.work['native_fallback_frames'] += 1
                spatial = self.mapping.compute_spatial_similarities(self.config, full_detections, full_objects)
                visual = self.mapping.compute_visual_similarities(self.config, full_detections, full_objects)
                fallback_values = self.mapping.aggregate_similarities(self.config, spatial, visual).cpu().numpy()
            decisions = []
            for row, det in enumerate(det_tokens):
                if fallback_values is not None:
                    self.work['pair_misses'] += len(previous)
                    scores = fallback_values[row]
                    best = int(np.argmax(scores))
                    decisions.append(best if scores[best] >= self.config.sim_threshold else None)
                    continue
                missing = [obj for obj in previous if (det, obj) not in self.pair_scores]
                self.work['pair_hits'] += len(previous)-len(missing)
                self.work['pair_misses'] += len(missing)
                if missing:
                    detections = self.DetectionList([self.states[det]])
                    objects = self.MapObjectList([self.states[obj] for obj in missing])
                    spatial = self.mapping.compute_spatial_similarities(self.config, detections, objects)
                    visual = self.mapping.compute_visual_similarities(self.config, detections, objects)
                    values = self.mapping.aggregate_similarities(self.config, spatial, visual).cpu().numpy()[0]
                    self.pair_scores.update({(det, obj): float(value) for obj, value in zip(missing, values)})
                pairs = [(det, obj) for obj in previous]
                self.used_pairs.update(pairs)
                scores = np.array([self.pair_scores[pair] for pair in pairs])
                best = int(np.argmax(scores))
                decisions.append(best if scores[best] >= self.config.sim_threshold else None)
            for det, best in zip(det_tokens, decisions):
                if best is None:
                    self.object_tokens.append(det)
                else:
                    self.object_tokens[best] = self.derived('fusion', (self.object_tokens[best], det))
        # As in native code, initial empty-map insertion does not denoise.
        if det_tokens and previous and self.config.denoise_interval > 0 and self._frame_count % self.config.denoise_interval == 0:
            self.object_tokens = [self.derived('denoise', (obj,)) for obj in self.object_tokens]
        self._objects = self.MapObjectList([self.states[token] for token in self.object_tokens])
        self._global_started |= bool(det_tokens)

    def corrected(self, observations, frame_poses, changed, mode):
        start = time.perf_counter()
        candidate = DependencyConceptGraphsBackend(self.config)
        candidate._rng.set_state(self._rng.get_state())
        candidate._ids = list(self._ids)
        candidate.frame_ids = [list(ids) for ids in self.frame_ids]
        for name in ['jittered','prepared','raw_bounds','input_tokens','recipes','states','pair_scores']:
            setattr(candidate, name, dict(getattr(self, name)))
        candidate.next_token = self.next_token
        for oid in sorted(changed):
            obs = observations[oid]
            candidate.prepared[oid], candidate.raw_bounds[oid] = candidate.prepare(
                obs, frame_poses[obs.frame_id], self.jittered[oid])
        preparation_s = time.perf_counter()-start
        start = time.perf_counter()
        for ids in candidate.frame_ids:
            candidate.apply_frame(ids)
        replay_s = time.perf_counter()-start
        candidate.rejected = [i for i in candidate._ids if candidate.prepared[i] is None]
        # Retain only the current computation DAG, rather than accumulating all
        # obsolete pose versions after an unbounded sequence of corrections.
        old_states, old_pairs = len(candidate.states), len(candidate.pair_scores)
        candidate.states = {k:v for k,v in candidate.states.items() if k in candidate.used_tokens}
        candidate.recipes = {k:v for k,v in candidate.recipes.items() if v in candidate.used_tokens}
        candidate.pair_scores = {k:v for k,v in candidate.pair_scores.items() if k in candidate.used_pairs}
        stats = dict(mode=mode, reason='exact_input_dependency_reuse', changed_observations=len(changed),
            visited_observations=len(self._ids), preparation_s=preparation_s, replay_s=replay_s,
            state_cache_entries=len(candidate.states), pair_cache_entries=len(candidate.pair_scores),
            obsolete_states_evicted=old_states-len(candidate.states), obsolete_pairs_evicted=old_pairs-len(candidate.pair_scores),
            **candidate.work)
        return candidate, stats


class DependencyReplayObjectMap(LocalReplayObjectMap):
    def __init__(self, config, seed=0):
        ReplayableObjectMap.__init__(self, lambda: DependencyConceptGraphsBackend(config, seed))
        self.mode = 'dependency'
