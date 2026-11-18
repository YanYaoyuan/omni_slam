"""Separate few accepted detections from few measured opportunities.

Existing predictions only. References choose and describe diagnostic targets;
no output of this audit is a predictor input or a method benchmark score.
"""
import time
import runpy
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from capture_prethreshold_candidates import OUT as CAPTURE, FRONT, INPUT, verify
from audit_prethreshold_candidates import AUDIT, CLASSES
from run_surface_ownership import OUT as OWNERSHIP
from evaluate_kitti360_development import read_static_ply, transfer
from omni_semantic_map.native_query_support import exclusive_measured_queries

OUT = CAPTURE/'observation_opportunities'


def main():
    assert not OUT.exists(); p = verify(); predictions = read(CAPTURE/'predictions_complete.json'); earlier = read(AUDIT/'summary.json')
    assert earlier['prediction_seal_sha256'] == sha(CAPTURE/'predictions_complete.json')
    for f, h in predictions['files'].items(): assert sha(f) == h
    targets = {key: [o for o in r['objects'] if not o['already_matched'] and o['coverage']['combined_connected']>.5] for key, r in earlier['results'].items()}
    dump(OUT/'protocol.json', dict(created_epoch=time.time(), source_sha256=sha(__file__), capture_sha256=sha(CAPTURE/'predictions_complete.json'), earlier_audit_sha256=sha(AUDIT/'summary.json'),
        target_rule='All previously unmatched GTs with combined connected coverage>0.5 in the previous diagnostic, including those with zero deferred pure frames. GT-selected mechanism audit only.',
        visibility='Count distinct depth-verified visible measured source IDs with a known nearest same-GT reference within0.2m; report frames with>=16 such points separately. This is measured visibility, not mere projected GT bounding box presence.',
        matching='For each selected GT and frame, compute confidence-weighted IoU over all currently visible known-labelled measured sources for each same-class raw query mask. Raw masks can overlap; no exclusive arbitration or largest-component truncation in this diagnostic. Best IoU uses GT and is NOT a deployable selection.',
        score_groups='high>0.5; deferred0.1<score<=0.5; any_positive score>0 including below0.1. Native zero/suppressed queries excluded. Also report actual0.1-exclusive same-class coverage on the target surface. No cutoffs changed in prediction.',
        scope='No inference, fitting, map building, new evaluation data or changed earlier gates. Per-view measured-point IoU is not dense image IoU or full-map instance AP.'))
    parent = read(OWNERSHIP/'protocol.json'); taxonomy = runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']; results = {}
    for scene in p['scenes']:
        key = scene['key']; manifest = read(INPUT/key/'manifest.json'); xyz = np.load(manifest['geometry'])['xyz']; measured = {r['frame_id']: r for r in manifest['records']}
        ref = np.load(scene['reference']); info = parent['references'][key]; assert sha(info['raw']) == info['sha256']; raw = read_static_ply(info['raw']); train = np.array([taxonomy[int(v)].trainId for v in raw['semantic']], np.int64)
        nearest = transfer(ref['xyz'], np.arange(1, len(raw)+1), xyz, .2); ix = np.maximum(nearest-1, 0); known = (nearest>0)&(train[ix]!=255)&(ref['confidence'][ix]>0); truth = ref['train_instance'][ix]; weight = ref['confidence'][ix]
        rows = {o['gt_id']: [] for o in targets[key]}
        for r in predictions['results'][key]:
            d = np.load(r['file']); m = np.load(measured[r['frame_id']]['file']); ids = m['visible_source_ids']; px = m['pixel_xy']; valid = known[ids]; weights = np.where(valid, weight[ids], 0.)
            query_ids = np.flatnonzero(d['scores']>0); packed = d['masks_packed'][query_ids]
            inside = ((packed[:, px[:, 1], px[:, 0]//8] >> (7-px[:, 0]%8)[None, :]) & 1).astype(bool)
            masses = inside@weights; scores = d['scores'][query_ids]; semantics = d['semantic_ids'][query_ids]
            exclusive = exclusive_measured_queries(d['masks_packed'], d['scores'], px, int(d['original_size'][0]), .1)
            predicted_class = np.where(exclusive>=0, d['semantic_ids'][np.maximum(exclusive, 0)], -1)
            for target in targets[key]:
                gid = target['gt_id']; own = valid & (truth[ids] == gid); count = int(own.sum()); own_mass = float(weights[own].sum()); sem = next(s for s,c in CLASSES.items() if c==gid//1000)
                if not count:
                    rows[gid].append(dict(frame_id=r['frame_id'], visible_sources=0, visible_mass=0., enough_measured_support=False)); continue
                intersection = inside[:, own]@weights[own]; iou = intersection/np.maximum(own_mass+masses-intersection, 1e-12)
                best = {}
                for name, group in [('high', scores>.5), ('deferred', (scores>.1)&(scores<=.5)), ('any_positive', scores>0)]:
                    index = np.flatnonzero(group & (semantics==sem))
                    if not len(index): best[name] = None; continue
                    q = index[np.lexsort((query_ids[index], -scores[index], -iou[index]))[0]]
                    best[name] = dict(query_id=int(query_ids[q]), score=float(scores[q]), measured_iou=float(iou[q]), measured_recall=float(intersection[q]/own_mass), known_precision=float(intersection[q]/masses[q]) if masses[q]>0 else 0.)
                rows[gid].append(dict(frame_id=r['frame_id'], visible_sources=count, visible_mass=own_mass, enough_measured_support=count>=16,
                    exclusive_same_class_recall=float(weights[own & (predicted_class==sem)].sum()/own_mass), best_raw_queries=best))
        result = []
        for target in targets[key]:
            gid = target['gt_id']; frames = rows[gid]; eligible = [r for r in frames if r['enough_measured_support']]
            counts = {name: sum(r['best_raw_queries'][name] is not None and r['best_raw_queries'][name]['measured_iou']>.5 for r in eligible) for name in ['high', 'deferred', 'any_positive']}
            record = dict(gt_id=gid, category=target['category'], measured_visible_frames=len(eligible), any_measured_visible_frames=sum(r['visible_sources']>0 for r in frames), old_pure_deferred_frames=len(target['diagnostic_deferred_frames']), best_raw_iou_over_half_frames=counts, frames=frames)
            result.append(record); print('OPPORTUNITIES', key, gid, {k:v for k,v in record.items() if k!='frames'}, flush=True)
        results[key] = result
    dump(OUT/'summary.json', dict(complete=True, diagnostic_only=True, results=results, protocol_sha256=sha(OUT/'protocol.json')))


if __name__ == '__main__': main()
