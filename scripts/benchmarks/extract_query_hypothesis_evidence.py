"""GT-free persistent hypothesis/evidence representation from sealed raw queries.

This creates no accepted object map. Original0.1/three-frame audit remains
failed; this distinct representation preserves native score evidence rather
than changing that audit's thresholds or relabelling its outcome.
"""
import argparse
import time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from capture_prethreshold_candidates import OUT as RAW, INPUT, verify as verify_raw
from omni_semantic_map.sparse_support_components import largest_measured_component
from omni_semantic_map.query_hypothesis_evidence import membership, retrieve

OUT = ROOT/'build/benchmarks/query_hypothesis_evidence_v1'


def freeze():
    assert not OUT.exists(); p = verify_raw(); capture = read(RAW/'predictions_complete.json')
    for f,h in capture['files'].items(): assert sha(f) == h
    sources = [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/query_hypothesis_evidence.py', ROOT/'omni_semantic_map/test/test_query_hypothesis_evidence.py']
    dump(OUT/'protocol.json', dict(created_epoch=time.time(), scenes=p['scenes'], source_hashes={str(f): sha(f) for f in sources}, raw_seal_sha256=sha(RAW/'predictions_complete.json'),
        scope='Evidence representation only, not admission/association/new map. Old exposed0000/0002 only. No GT features, training, new model inference, benchmark scoring, robot or0004/0005 inputs.',
        motivation='Previous0.1/three-pure-frame gate failed and remains unchanged. Subsequent observation audit showed missed buildings visible7/8frames and usable raw masks below0.1. This separate representation asks whether physical hypotheses can retrieve those raw queries without GT; it does not declare a lower detector cutoff to be a method.',
        hypotheses='Each native positive-score query with>=16 measured points in its largest0.4m component seeds an uncommitted physical hypothesis. No custom confidence threshold, per-frame exclusivity or global object ID. Native zero/suppressed queries excluded. Exact same-class/same-frame component-source duplicates collapse to highest native score then query index; all aliases retained.',
        witnesses='Each native positive-score query keeps ALL its nonexclusive measured source cells, even when too small to seed. Use existing20cm cells. At each witness frame restrict hypothesis to actually measured-visible cells; choose same-class query maximizing cell IoU, native-score tie then query row. Return raw native score/logits as evidence, not as an acceptance decision. Large all-image masks incur union penalty.',
        timing='Window-end representation with explicit seed/witness timestamps. Seed frame cannot confirm itself; one selected query per different frame. A historical-view retrieval becomes available no earlier than max(seed_frame,witness_frame). Repeated nearby views are not claimed independent.',
        unknown='No measured-visible hypothesis cells is unknown, never negative. Visible cells with no overlapping same-class query is explicitly distinguishable from unknown; neither is automatically a deletion rule.',
        budget='One CPU index/evidence pass over108 existing frames. No parameter sweep. Full maps/admission/training require a separate concrete hypothesis and frozen comparison. Cannot claim model advantage from oracle diagnostics or representation coverage.'))
    print('FROZEN', flush=True)


def verify():
    p = read(OUT/'protocol.json'); verify_raw(); assert sha(RAW/'predictions_complete.json') == p['raw_seal_sha256']
    sources = dict(p['source_hashes'])
    amendment = OUT/'technical_amendment_npz_io.json'
    if amendment.exists():
        a = read(amendment)
        assert a['original_protocol_sha256'] == sha(OUT/'protocol.json')
        assert a['index_seal_sha256'] == sha(OUT/'index_complete.json')
        assert sha(a['original_extractor']) == sources[str(Path(__file__).resolve())]
        sources[str(Path(__file__).resolve())] = a['revised_extractor_sha256']
    for f,h in sources.items(): assert sha(f) == h, f
    return p


def sealed(name):
    d = read(OUT/(name+'.json'))
    for f,h in d['files'].items(): assert sha(f) == h, f
    return d


def ragged(sets):
    return (np.concatenate(sets) if sets else np.empty(0, np.int64)), np.r_[0, np.cumsum([len(s) for s in sets])]


def index():
    p = verify(); raw = read(RAW/'predictions_complete.json'); files = {}; results = {}; assert not (OUT/'index_complete.json').exists()
    for scene in p['scenes']:
        key = scene['key']; manifest = read(INPUT/key/'manifest.json'); by_frame = {r['frame_id']:r for r in manifest['records']}; xyz = np.load(manifest['geometry'])['xyz']; cells = np.load(INPUT/'query_cells'/key/'source_to_cell.npy'); hypotheses = []; seed_sources = []; seed_cells = []; frames = []; start = time.monotonic()
        folder = OUT/'index'/key; folder.mkdir(parents=True, exist_ok=False)
        for rank, r in enumerate(raw['results'][key]):
            assert sha(r['file']) == raw['files'][r['file']]
            data = np.load(r['file']); measured = np.load(by_frame[r['frame_id']]['file']); ids = measured['visible_source_ids']; px = measured['pixel_xy']; active = np.flatnonzero(data['scores']>0)
            query_cells = []; chosen = {}
            for q in active:
                inside = ((data['masks_packed'][q, px[:, 1], px[:, 0]//8] >> (7-px[:, 0]%8)) & 1).astype(bool); sources = ids[inside]
                query_cells.append(np.unique(cells[sources]))
                if len(sources)<16: continue
                component = sources[largest_measured_component(xyz[sources], .4, 16)]
                if len(component)<16: continue
                component = np.sort(component); identity = (int(data['semantic_ids'][q]), component.tobytes()); score = float(data['scores'][q])
                if identity not in chosen:
                    chosen[identity] = dict(query_id=int(q), score=score, sources=component, aliases=[int(q)])
                else:
                    existing = chosen[identity]; existing['aliases'].append(int(q))
                    if score > existing['score']: existing.update(query_id=int(q), score=score)
            for (sem, _), hypothesis in sorted(chosen.items(), key=lambda kv: kv[1]['query_id']):
                q = hypothesis['query_id']; sources = hypothesis.pop('sources'); hc = np.unique(cells[sources]); seed_sources.append(sources); seed_cells.append(hc)
                hypotheses.append(dict(index=len(hypotheses), frame_id=r['frame_id'], semantic_id=sem, source_points=len(sources), cells=len(hc), class_logit=float(data['class_logits'][q]), presence_logit=float(data['presence_logits'][q]), **hypothesis))
            values, offsets = ragged(query_cells); path = folder/f'{r["frame_id"]:06d}.npz'
            np.savez_compressed(path, query_ids=active, semantic_ids=data['semantic_ids'][active], scores=data['scores'][active], class_logits=data['class_logits'][active], presence_logits=data['presence_logits'][active], cell_ids=values, offsets=offsets, visible_cells=np.unique(cells[ids]))
            files[str(path)] = sha(path); frames.append(dict(frame_id=r['frame_id'], file=str(path), queries=len(active)))
            if (rank+1)%15==0: print('INDEX',key,rank+1,len(raw['results'][key]),'hypotheses',len(hypotheses),flush=True)
        sv, so = ragged(seed_sources); cv, co = ragged(seed_cells); path = folder/'hypotheses.npz'; np.savez_compressed(path, source_ids=sv, source_offsets=so, cell_ids=cv, cell_offsets=co); files[str(path)] = sha(path)
        path = folder/'hypotheses.json'; dump(path,hypotheses); files[str(path)] = sha(path)
        results[key] = dict(frames=frames, hypotheses=len(hypotheses), original_seed_queries=sum(len(h['aliases']) for h in hypotheses), cell_count=int(cells.max())+1, seconds=time.monotonic()-start)
    dump(OUT/'index_complete.json',dict(complete=True,files=files,results=results,protocol_sha256=sha(OUT/'protocol.json')))
    print('INDEX COMPLETE',{k:{n:v for n,v in x.items() if n!='frames'} for k,x in results.items()},flush=True)


def evidence():
    p = verify(); index_data = sealed('index_complete'); files = {}; stats = {}; assert not (OUT/'evidence_complete.json').exists()
    for scene in p['scenes']:
        key = scene['key']; info = index_data['results'][key]; folder = OUT/'index'/key; meta = read(folder/'hypotheses.json'); d = np.load(folder/'hypotheses.npz'); n = len(meta); t = len(info['frames']); start = time.monotonic()
        # NpzFile does not cache decoded arrays. Decode once before making views;
        # otherwise every view retains a separate full decompressed array.
        cell_values = d['cell_ids']; cell_offsets = d['cell_offsets']
        sets = [cell_values[a:b] for a,b in zip(cell_offsets[:-1],cell_offsets[1:])]; h = membership(sets,info['cell_count']); classes = np.asarray([m['semantic_id'] for m in meta]); seed_frame = np.asarray([m['frame_id'] for m in meta])
        query = np.full((n,t),-1,np.int32); agreement = np.zeros((n,t),np.float32); score = np.zeros((n,t),np.float32); visible = np.zeros((n,t),np.int32); class_logits = np.zeros((n,t),np.float32); presence_logits = np.zeros((n,t),np.float32)
        prefix_checks = []
        for k,r in enumerate(info['frames']):
            data = np.load(r['file']); query_values = data['cell_ids']; query_offsets = data['offsets']
            qsets = [query_values[a:b] for a,b in zip(query_offsets[:-1],query_offsets[1:])]; q = membership(qsets,info['cell_count'])
            for lo in range(0,n,256):
                hi = min(lo+256,n); z = retrieve(h[lo:hi],classes[lo:hi],q,data['semantic_ids'],data['scores'],data['visible_cells']); selected = z['query']; active = (selected>=0)&(seed_frame[lo:hi]!=r['frame_id']); rows = np.flatnonzero(active)+lo; indices = selected[active]
                visible[lo:hi,k] = z['visible_cells']; query[rows,k] = data['query_ids'][indices]; agreement[rows,k] = z['agreement'][active]; score[rows,k] = data['scores'][indices]; class_logits[rows,k] = data['class_logits'][indices]; presence_logits[rows,k] = data['presence_logits'][indices]
            # Real prefix check: future hypotheses cannot affect current rows.
            if k in [t//3,2*t//3]:
                rows = np.flatnonzero(seed_frame<r['frame_id'])[:32]
                z = retrieve(h[rows],classes[rows],q,data['semantic_ids'],data['scores'],data['visible_cells']); active = z['query']>=0; expected = np.full(len(rows),-1,np.int32); expected[active] = data['query_ids'][z['query'][active]]
                np.testing.assert_array_equal(expected,query[rows,k]); np.testing.assert_array_equal(z['agreement'],agreement[rows,k]); prefix_checks.append(dict(frame_id=r['frame_id'],hypotheses=len(rows),exact=True))
            if (k+1)%15==0: print('EVIDENCE',key,k+1,t,flush=True)
        frames = np.asarray([r['frame_id'] for r in info['frames']]); self_frame = seed_frame[:,None]==frames[None,:]
        assert np.all(query[self_frame]==-1) and np.all(agreement[self_frame]==0)
        np.testing.assert_array_equal(visible.max(1)<=np.diff(h.indptr),np.ones(n,bool))
        dest = OUT/'evidence'/key; dest.mkdir(parents=True,exist_ok=False); path = dest/'readout.npz'
        np.savez_compressed(path,query_ids=query,agreement=agreement,scores=score,visible_cells=visible,class_logits=class_logits,presence_logits=presence_logits,seed_frames=seed_frame,witness_frames=frames,available_frames=np.maximum(seed_frame[:,None],frames[None,:])); files[str(path)] = sha(path)
        stats[key] = dict(hypotheses=n,witness_frames=t,hypotheses_with_other_measured_view=int(((visible>0)&~self_frame).any(1).sum()),hypotheses_with_other_query=int((query>=0).any(1).sum()),weak_correspondences_below_01=int(((score>0)&(score<=.1)).sum()),self_confirmation_removed=True,prefix_checks=prefix_checks,seconds=time.monotonic()-start)
    dump(OUT/'evidence_complete.json',dict(complete=True,files=files,stats=stats,index_sha256=sha(OUT/'index_complete.json'),protocol_sha256=sha(OUT/'protocol.json')))
    print('EVIDENCE COMPLETE',stats,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','index','evidence']);globals()[parser.parse_args().action]()
