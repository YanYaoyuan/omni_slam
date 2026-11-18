"""Matched-query diagnosis of information lost by per-frame mean prototypes.

Reuses cached unit features. No new encoder calls or changed map decisions.
The alternative is an explanatory score, not a validated mapping candidate.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from audit_anchor_retraction_failures import OUT as AUDIT, verify

PARENT = ROOT / 'build/benchmarks/anchor_counterevidence_v1'
OUT = AUDIT / 'compression'


def inputs(c):
    units, features, prototypes, labels, frames = [], [], [], [], []
    for frame in read(c['job'])['frames']:
        d = np.load(PARENT / 'features' / c['key'] / f'{frame["frame_id"]:06d}.npz')
        units.append(d['units']); features.append(d['features']); prototypes.append(d['prototype_features'])
        labels.append(d['prototype_label']); frames.append(np.full(len(d['prototype_label']), frame['frame_id']))
    u, q, p, label, pf = [np.concatenate(v) for v in [units, features, prototypes, labels, frames]]
    q /= np.linalg.norm(q, axis=1)[:, None]; p /= np.linalg.norm(p, axis=1)[:, None]
    return u, q, p, label, pf


def calculate():
    import torch
    p = verify(); assert not (OUT / 'scores_complete.json').exists()
    assert not OUT.exists()
    dump(OUT / 'protocol.json', dict(created_epoch=time.time(), parent_audit_sha256=sha(AUDIT / 'protocol.json'),
        source_sha256=sha(__file__),
        scope='Posthoc matched-query information diagnosis after a failed map candidate. Not a new method, decision gate or map trial.',
        fixed='Same actual foreground anchor unit queries, identities, query-frame exclusion, background prototypes and comparable frames as the closed candidate. No GT-selected samples.',
        change='Compare saved normalized per-frame object mean prototype with the maximum cosine over its already-cached foreground unit features in that SAME other frame. Background evidence is held fixed. This is not a fair alternative full classifier because only foreground retains local units; it isolates loss from compressing foreground support.',
        outputs='Continuous mean margins only. Do not generate alternative rejection masks or maps, fit thresholds, or promote from AUROC. Report both correct and wrong semantic anchors. No claim of independent label provenance.',
        budget='Both exposed windows, all queries,0encoder forwards,0training,0maps,0APscores. CUDA matrix products on cached features only.'))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.5)
    files, stats = {}, {}
    for c in p['contexts']:
        started = time.monotonic(); u, q, proto, label, pf = inputs(c)
        query_gpu, proto_gpu = torch.as_tensor(q, device='cuda'), torch.as_tensor(proto, device='cuda')
        mean_margin = np.zeros(len(u), np.float64); local_margin = mean_margin.copy()
        eligible = np.zeros(len(u), np.int32)
        with torch.inference_mode():
            for oid in np.unique(u[:, 1]):
                own_p = np.flatnonzero(label == oid)
                frames = np.intersect1d(pf[own_p], np.unique(pf[label < 0]))
                if not len(frames): continue
                rows = np.flatnonzero(u[:, 1] == oid)
                own_unit = np.flatnonzero((u[:, 1] == oid) & np.isin(u[:, 2], frames))
                bg = np.flatnonzero((label < 0) & np.isin(pf, frames))
                own_p = np.array([np.flatnonzero((label == oid) & (pf == f))[0] for f in frames])
                unit_group = torch.as_tensor(np.searchsorted(frames, u[own_unit, 2]), device='cuda')
                bg_group = torch.as_tensor(np.searchsorted(frames, pf[bg]), device='cuda')
                frame_gpu = torch.as_tensor(frames, device='cuda')
                for start in range(0, len(rows), 256):
                    ix = rows[start:start + 256]; qg = query_gpu[ix]
                    scores = qg @ query_gpu[own_unit].T
                    maximum = torch.full((len(ix), len(frames)), -torch.inf, device='cuda')
                    maximum.scatter_reduce_(1, unit_group.expand(len(ix), -1), scores, reduce='amax')
                    bg_scores = qg @ proto_gpu[bg].T
                    background = torch.full_like(maximum, -torch.inf)
                    background.scatter_reduce_(1, bg_group.expand(len(ix), -1), bg_scores, reduce='amax')
                    own_mean = qg @ proto_gpu[own_p].T
                    usable = torch.as_tensor(u[ix, 2], device='cuda')[:, None] != frame_gpu[None, :]
                    count = usable.sum(1)
                    original = torch.where(usable, background - own_mean, 0).sum(1) / count.clamp(min=1)
                    alternative = torch.where(usable, background - maximum, 0).sum(1) / count.clamp(min=1)
                    mean_margin[ix] = original.cpu().numpy(); local_margin[ix] = alternative.cpu().numpy(); eligible[ix] = count.cpu().numpy()
        closed = np.load(PARENT / 'decisions' / c['key'] / 'anchors.npz')
        np.testing.assert_array_equal(closed['units'], u)
        np.testing.assert_array_equal(closed['eligible'], eligible)
        np.testing.assert_allclose(closed['mean_margin'], mean_margin, rtol=0, atol=3e-6)
        # Independent scalar/NumPy reference on fixed indices, all other frames.
        checked = np.unique(np.linspace(0, len(u) - 1, 24, dtype=int))
        for i in checked:
            values = []
            for frame in np.unique(pf[(label == u[i, 1])]):
                if frame == u[i, 2]: continue
                bg = np.flatnonzero((pf == frame) & (label < 0))
                own = np.flatnonzero((u[:, 2] == frame) & (u[:, 1] == u[i, 1]))
                if not len(bg): continue
                assert len(own)
                values.append(float((proto[bg] @ q[i]).max() - (q[own] @ q[i]).max()))
            expected = float(np.mean(values)) if values else 0.
            assert abs(expected - local_margin[i]) < 3e-6
        path = OUT / c['key'] / 'unit_scores.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, units=u, eligible=eligible, replay_mean_margin=mean_margin, local_support_margin=local_margin)
        files[str(path)] = sha(path)
        stats[c['key']] = dict(queries=len(u), independent_checks=len(checked), seconds=time.monotonic() - started,
                               original_margin_max_error=float(np.abs(closed['mean_margin'] - mean_margin).max()))
        print('COMPRESSION SCORES', c['key'], stats[c['key']], flush=True)
        del query_gpu, proto_gpu
    dump(OUT / 'scores_complete.json', dict(complete=True, files=files, stats=stats,
                                          protocol_sha256=sha(OUT / 'protocol.json'), alternative_decisions_generated=False))


def measure():
    from sklearn.metrics import roc_auc_score
    p = verify(); s = read(OUT / 'scores_complete.json'); assert s['complete']
    assert not (OUT / 'summary.json').exists()
    assert sha(__file__) == read(OUT / 'protocol.json')['source_sha256']
    for file, expected in s['files'].items(): assert sha(file) == expected
    results = {}
    for c in p['contexts']:
        a = np.load(AUDIT / c['key'] / 'anchor_readout.npz')
        scores = np.load(OUT / c['key'] / 'unit_scores.npz')
        local = scores['local_support_margin'][a['unit_index']]
        y, w, known = a['wrong_semantic_class'], a['confidence'], a['known']
        rows = {}
        for name, t in [('all', known), ('building', known & (a['declared_class'] == 2)), ('car', known & (a['declared_class'] == 13))]:
            rows[name] = dict(anchors=int(t.sum()), wrong_mass=float(w[t & y].sum()), correct_mass=float(w[t & ~y].sum()),
                auroc={k: float(roc_auc_score(y[t], v[t], sample_weight=w[t])) if len(np.unique(y[t])) == 2 else None
                       for k, v in [('closed_mean_margin', a['mean_margin']), ('local_support_margin', local)]})
        results[c['key']] = rows
        print('COMPRESSION DIAGNOSIS', c['key'], rows, flush=True)
    dump(OUT / 'summary.json', dict(complete=True, results=results,
        no_new_decisions_maps_or_ap_scores=True, score_seal_sha256=sha(OUT / 'scores_complete.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['calculate', 'measure'])
    globals()[parser.parse_args().action]()
