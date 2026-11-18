"""Post-seal diagnosis of all physical hypotheses, never predictor input.

No map admission or new detector threshold is fitted here. Query choices must
already be sealed before references are loaded. GT chooses seed descriptions
and evaluation targets only; diagnostic rows are dependent, not map instances.
"""
import argparse
import runpy
import time
from pathlib import Path

import numpy as np
from paper_final_common import ROOT, read, dump, sha
from extract_query_hypothesis_evidence import OUT as REPRESENTATION, verify, sealed
from capture_prethreshold_candidates import OUT as RAW, INPUT
from run_surface_ownership import OUT as OWNERSHIP
from evaluate_kitti360_development import read_static_ply, transfer

OUT = REPRESENTATION/'readout_diagnostic'
EXAMPLES = ROOT/'output/prethreshold_candidates_v1/images.json'
CLASSES = {11: 2, 26: 13}


def freeze():
    assert not OUT.exists()
    verify()
    dump(OUT/'protocol.json', dict(created_epoch=time.time(), source_sha256=sha(__file__),
        representation_protocol_sha256=sha(REPRESENTATION/'protocol.json'),
        examples_sha256=sha(EXAMPLES),
        scope='Post-seal development diagnostic of all hypotheses; not map AP, admission, training, or independent test. Four examples fixed by the previous report, never reselected by retrieval quality.',
        seed_labels='Nearest known reference within0.2m; ignore raw trainId255 and zero-confidence/unmatched sources. Confidence-weighted majority over ALL known seed sources. Same predicted class and purity>0.5 defines a diagnostic anchored seed. It is not an algorithm filter.',
        view_metric='For each already-selected raw query, confidence-weighted measured-source IoU against seed anchor GT in that frame. Unknown labels ignored. Other-frame target visibility>=16 is reported separately from hypothesis-cell visibility. Oracle best same-class query is diagnostic headroom, not a control algorithm.',
        risks='Report all seed score groups, including unknown and wrong-class/mixed seeds; count how often each has at least3 other frames with cell agreement>0.5 and>=16 visible cells. This fixed descriptive confirmation statistic is not an admission rule or tuned threshold. Repeated seeds/views are dependent.',
        exclusions='Seed frame excluded from all retrieval summaries. No new masks/maps, robot data,0004/0005 references or parameter sweep. Old failed gate unchanged.'))
    print('AUDIT FROZEN', flush=True)


def mean_or_none(values):
    values = np.asarray(values)
    return float(values.mean()) if values.size else None


def run():
    protocol = read(OUT/'protocol.json')
    expected = protocol['source_sha256']
    amendment = REPRESENTATION/'technical_amendment_npz_io.json'
    if amendment.exists():
        a = read(amendment)
        assert sha(a['original_audit']) == expected
        assert a['original_audit_protocol_sha256'] == sha(OUT/'protocol.json')
        expected = a['revised_audit_sha256']
    assert sha(__file__) == expected
    assert sha(EXAMPLES) == protocol['examples_sha256']
    assert sha(REPRESENTATION/'protocol.json') == protocol['representation_protocol_sha256']
    p = verify(); index = sealed('index_complete'); evidence = sealed('evidence_complete')
    assert evidence['index_sha256'] == sha(REPRESENTATION/'index_complete.json')
    assert not (OUT/'summary.json').exists()
    capture = read(RAW/'predictions_complete.json')
    for f, h in capture['files'].items(): assert sha(f) == h
    parent = read(OWNERSHIP/'protocol.json')
    taxonomy = runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    examples = read(EXAMPLES); results = {}; files = {}
    for scene in p['scenes']:
        key = scene['key']; start = time.monotonic()
        manifest = read(INPUT/key/'manifest.json'); xyz = np.load(manifest['geometry'])['xyz']
        by_frame = {r['frame_id']: r for r in manifest['records']}
        ref = np.load(scene['reference']); info = parent['references'][key]
        assert sha(info['raw']) == info['sha256']
        raw = read_static_ply(info['raw'])
        train = np.array([taxonomy[int(v)].trainId for v in raw['semantic']], np.int64)
        nearest = transfer(ref['xyz'], np.arange(1, len(raw)+1), xyz, .2)
        ix = np.maximum(nearest-1, 0)
        known = (nearest>0)&(train[ix]!=255)&(ref['confidence'][ix]>0)
        truth = ref['train_instance'][ix]; weight = np.where(known, ref['confidence'][ix], 0.)
        folder = REPRESENTATION/'index'/key
        meta = read(folder/'hypotheses.json'); supports = np.load(folder/'hypotheses.npz')
        source_values = supports['source_ids']; source_offsets = supports['source_offsets']
        readout = np.load(REPRESENTATION/'evidence'/key/'readout.npz')
        n = len(meta); t = len(readout['witness_frames'])
        anchor = np.full(n, -1, np.int64); purity = np.zeros(n, np.float32)
        known_fraction = np.zeros(n, np.float32); label_status = np.zeros(n, np.int8)
        # status0 unknown,1 same-class majority,2 wrong-class or mixed.
        for j, m in enumerate(meta):
            a,b = source_offsets[j:j+2]; ids = source_values[a:b]
            known_fraction[j] = known[ids].mean()
            ids = ids[known[ids]]
            if not len(ids): continue
            histogram = np.bincount(truth[ids], weights=weight[ids]); gid = int(histogram.argmax())
            purity[j] = histogram[gid]/histogram.sum(); label_status[j] = 2
            if purity[j]>.5 and gid//1000 == CLASSES[m['semantic_id']]:
                anchor[j] = gid; label_status[j] = 1
        iou = np.zeros((n,t), np.float32); precision = np.zeros_like(iou)
        oracle = np.zeros_like(iou); target_visible = np.zeros((n,t), np.int32)
        case_rows = {}
        for name, case in examples.items():
            if not name.startswith(key+'/'): continue
            node = case['node']
            rows = [j for j,m in enumerate(meta) if m['frame_id']==node['frame_id'] and node['query_id'] in m['aliases']]
            assert len(rows)==1, (name, rows)
            j = rows[0]
            assert anchor[j] == node['diagnostic_gt'], (name, int(anchor[j]))
            case_rows[name] = dict(hypothesis_index=j, fixed_seed=node, frames=[])
        for k,r in enumerate(capture['results'][key]):
            assert r['frame_id'] == readout['witness_frames'][k]
            data = np.load(r['file']); measured = np.load(by_frame[r['frame_id']]['file'])
            ids = measured['visible_source_ids']; px = measured['pixel_xy']
            active = np.flatnonzero(data['scores']>0)
            packed = data['masks_packed'][active]
            inside = ((packed[:,px[:,1],px[:,0]//8] >> (7-px[:,0]%8)[None,:]) & 1).astype(bool)
            weights = weight[ids]; masses = inside@weights
            lookup = np.full(len(data['scores']), -1, np.int64); lookup[active] = np.arange(len(active))
            for gid in np.unique(anchor[anchor>=0]):
                rows = np.flatnonzero(anchor==gid); own = known[ids] & (truth[ids]==gid)
                count = int(own.sum()); target_visible[rows,k] = count
                if not count: continue
                mass = weights[own].sum(); intersection = inside[:,own]@weights[own]
                scores = intersection/np.maximum(mass+masses-intersection, 1e-12)
                same = data['semantic_ids'][active] == next(s for s,c in CLASSES.items() if c==gid//1000)
                oracle[rows,k] = scores[same].max() if same.any() else 0.
                selected = readout['query_ids'][rows,k]; present = selected>=0
                valid_rows = rows[present]; q = lookup[selected[present]]
                assert np.all(q>=0)
                iou[valid_rows,k] = scores[q]
                precision[valid_rows,k] = intersection[q]/np.maximum(masses[q],1e-12)
            for case in case_rows.values():
                j = case['hypothesis_index']
                case['frames'].append(dict(frame_id=r['frame_id'], seed_frame=bool(meta[j]['frame_id']==r['frame_id']),
                    target_visible_sources=int(target_visible[j,k]), hypothesis_visible_cells=int(readout['visible_cells'][j,k]),
                    query_id=int(readout['query_ids'][j,k]), native_score=float(readout['scores'][j,k]),
                    cell_agreement=float(readout['agreement'][j,k]), measured_iou=float(iou[j,k]),
                    known_precision=float(precision[j,k]), oracle_measured_iou=float(oracle[j,k])))
            if (k+1)%15==0: print('AUDIT',key,k+1,t,flush=True)
        other = readout['seed_frames'][:,None] != readout['witness_frames'][None,:]
        eligible = (target_visible>=16)&other&(label_status[:,None]==1)
        retrieved = readout['query_ids']>=0
        confirmation = ((readout['agreement']>.5)&(readout['visible_cells']>=16)&other).sum(1)
        score = np.array([m['score'] for m in meta]); groups = {}
        for name, mask in [('all', score>0), ('high', score>.5), ('deferred', (score>.1)&(score<=.5)), ('weak', (score>0)&(score<=.1))]:
            selected = eligible&mask[:,None]
            groups[name] = dict(hypotheses=int(mask.sum()), same_class_majority=int((mask&(label_status==1)).sum()),
                unknown=int((mask&(label_status==0)).sum()), wrong_class_or_mixed=int((mask&(label_status==2)).sum()),
                same_class_three_agreements=int((mask&(label_status==1)&(confirmation>=3)).sum()),
                wrong_or_mixed_three_agreements=int((mask&(label_status==2)&(confirmation>=3)).sum()),
                unknown_three_agreements=int((mask&(label_status==0)&(confirmation>=3)).sum()),
                eligible_seed_view_pairs=int(selected.sum()), retrieved_pairs=int((selected&retrieved).sum()),
                iou_over_half_pairs=int((selected&(iou>.5)).sum()), oracle_iou_over_half_pairs=int((selected&(oracle>.5)).sum()),
                mean_iou=mean_or_none(iou[selected]), mean_oracle_iou=mean_or_none(oracle[selected]))
        for case in case_rows.values():
            j = case['hypothesis_index']; rows = eligible[j]
            case['summary'] = dict(other_measured_views=int(rows.sum()), retrieved=int((rows&retrieved[j]).sum()),
                iou_over_half=int((rows&(iou[j]>.5)).sum()), oracle_over_half=int((rows&(oracle[j]>.5)).sum()),
                weak_good=int((rows&(iou[j]>.5)&(readout['scores'][j]<=.1)&retrieved[j]).sum()))
        dest = OUT/key; dest.mkdir(parents=True,exist_ok=False)
        path = dest/'GT_DIAGNOSTIC_ONLY.npz'
        np.savez_compressed(path, anchor_gt=anchor, seed_known_purity=purity, seed_known_fraction=known_fraction,
            seed_status=label_status, selected_measured_iou=iou, selected_known_precision=precision,
            oracle_measured_iou=oracle, target_visible_sources=target_visible, eligible=eligible,
            descriptive_confirmation_count=confirmation)
        files[str(path)] = sha(path)
        results[key] = dict(groups=groups, fixed_examples=case_rows, seconds=time.monotonic()-start)
        print('AUDIT RESULT', key, groups, {k:v['summary'] for k,v in case_rows.items()}, flush=True)
    dump(OUT/'summary.json', dict(complete=True, diagnostic_only=True, results=results, files=files,
        evidence_seal_sha256=sha(REPRESENTATION/'evidence_complete.json'), protocol_sha256=sha(OUT/'protocol.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze','run'])
    globals()[parser.parse_args().action]()
