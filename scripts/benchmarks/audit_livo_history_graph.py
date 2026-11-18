"""Exhaustive graph-edge evidence, pending independent identity annotation."""
import html
import json
from pathlib import Path

import cv2
import numpy as np

from audit_evidence_retention import OUT, REPORT, dump
from omni_semantic_map.flat_benchmark import sha256


def read(p):
    return json.loads(p.read_text())


def main():
    root = OUT / 'history_graph_v1'
    dest = REPORT / 'history_graph_v1'
    dest.mkdir(exist_ok=False)
    cache = OUT / 'livo_full_strong_v2/cache_supportcc'
    manifest = read(cache / 'manifest.json')
    lookup = {r['observation_offset']+k:(r,k) for r in manifest['records'] for k in range(r['masks'])}
    frames = {f['frame_id']: f for f in read(OUT / 'jobs/livo_full_v2.json')['frames']}
    variants = ['native_bridge_1','dino_bridge_1','native_bridge_2','dino_bridge_2']
    graphs = {v:read(root / v / 'graph.json') for v in variants}
    decisions = {v:{tuple(sorted([d['a'],d['b']])):d for d in graph['decisions']} for v,graph in graphs.items()}
    rows = {v:read(root / v / 'classification.json')['objects'] for v in ['baseline']+variants}
    proposals = read(root / 'proposals.json')
    pairs = sorted({tuple(sorted([p['a'],p['b']])) for p in proposals})
    def panel(oid, title):
        rec,k = lookup[oid]
        data = np.load(cache / 'frames' / rec['file'])
        rgb = cv2.imread(frames[rec['frame_id']]['rgb'])
        mask = data['segmentation']==k
        rgb[mask] = (rgb[mask]*.6+np.array([0,210,255])*.4).astype(np.uint8)
        cv2.rectangle(rgb,(0,0),(rgb.shape[1],28),(20,20,20),-1)
        cv2.putText(rgb,f'{title} | f{rec["frame_id"]} obs{oid+1}',(6,19),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
        return rgb
    records, cards, sheets = [], [], []
    for index, pair in enumerate(pairs):
        witnesses = sorted([p for p in proposals if tuple(sorted([p['a'],p['b']]))==pair],key=lambda p:(p['frame'],p['query']))
        p = witnesses[0]
        # Earliest witness and earliest historical members, without score selection.
        native,alternate = min(p['native_members']),min(p['alternate_members'])
        img = np.concatenate([panel(p['query'],f'Edge {pair[0]}-{pair[1]} query'),
                              panel(native,f'Native {p["a"]}'),panel(alternate,f'Alternate {p["b"]}')],axis=1)
        image = dest / f'edge_{pair[0]}_{pair[1]}.jpg'
        assert cv2.imwrite(str(image),img)
        sheets.append(cv2.resize(img,(1152,216)))
        record = dict(a=pair[0],b=pair[1],witness_queries=[w['query'] for w in witnesses],
                      frames=sorted({w['frame'] for w in witnesses}),
                      dino_frames=sorted({w['frame'] for w in witnesses if w['dino_pass']}),
                      decisions={v:decisions[v][pair] for v in variants}, evidence_image=image.name,
                      image_sha256=sha256(image), identity_label=None, confidence=None,
                      reviewer=None, review_notes=None)
        records.append(record)
        decisions_text=' · '.join(f'{v}: {decisions[v][pair]["reason"]}' for v in variants)
        cards.append(f'<article id="edge-{pair[0]}-{pair[1]}"><h2>{index+1}. 历史边 {pair[0]} ↔ {pair[1]}</h2>'
            f'<p>原生支持帧 {record["frames"]}；DINO 支持帧 {record["dino_frames"]}</p><p>{html.escape(decisions_text)}</p>'
            f'<a href="{image.name}"><img src="{image.name}" alt="当前查询与两条历史的最早成员"></a>'
            '<p>独立身份标注：待完成（same / different / uncertain）</p></article>')
    for start in range(0,len(sheets),4):
        assert cv2.imwrite(str(dest/f'contact_{start//4+1:02d}.jpg'),np.concatenate(sheets[start:start+4],axis=0))
    nodes={n['alias']:n for n in read(root/'snapshot/nodes.json')}
    for a,b in [(525,1000),(526,1013),(672,883)]:
        images=[panel(i,f'Fragment {node}') for node in [a,b] for i in nodes[node]['members']]
        while len(images)%3:
            images.append(np.full_like(images[0],25))
        grid=np.concatenate([np.concatenate(images[i:i+3],axis=1) for i in range(0,len(images),3)],axis=0)
        assert cv2.imwrite(str(dest/f'full_history_{a}_{b}.jpg'),grid)
    query=2860
    rec,k=lookup[query]
    descriptors=np.load(OUT/f'dino_livo_tokens_v2/{rec["frame_id"]:06d}.npz')['mean']
    scores=descriptors@descriptors[k]
    scores[k]=-np.inf
    scores[np.linalg.norm(descriptors,axis=1)<1e-12]=-np.inf
    competitor=rec['observation_offset']+int(scores.argmax())
    assert cv2.imwrite(str(dest/'dino_veto_2822.jpg'),np.concatenate([
        panel(query,'Query'),panel(2821,'Earlier fragment'),panel(competitor,'Strongest competitor')],axis=1))
    baseline_members={i:o for o in rows['baseline'] for i in o['observation_ids']}
    consequences=[]
    for v in variants:
        for obj in rows[v]:
            members=obj['observation_ids']
            old={min(baseline_members[i]['observation_ids'])+1 for i in members if i in baseline_members}
            added=[i for i in members if i not in baseline_members]
            if len(old)>1 or added:
                consequences.append(dict(variant=v,alias=min(members)+1,name=obj['predictions']['mean']['name'],
                    foreground=obj['predictions']['mean']['retained'],baseline_aliases=sorted(old),added_observations=added,
                    members=members,unique_frames=obj['unique_frames']))
    dump(dest/'annotation_template.json',dict(scope='All 25 fixed proposal edges; earliest-witness images only, inspect all member frames before independent labeling. Labels are deliberately blank, not model-generated GT.',edges=records))
    dump(dest/'consequences.json',consequences)
    dump(dest/'summary.json',read(root/'summary.json'))
    (dest/'gallery.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>历史合并全候选审查</title>
<style>body{font:16px/1.7 system-ui;max-width:1500px;margin:32px auto;padding:0 20px;background:#f4f6f8;color:#182431}article{background:white;padding:20px;margin:24px 0;border:1px solid #dce3e9;border-radius:8px}img{width:100%}h2{font-size:20px}a{color:#155b9a}</style>
<h1>历史图：全部 25 条候选边</h1><p>候选源自固定底座的全部 41 次歧义决策，共 57 条提议。左：最早查询；中、右：两条历史的最早成员。固定按时间选图，不按 DINO 分数选图。</p>
<p>native_bridge 是原底座几何＋SigLIP 候选，不是纯几何。1 / 2 表示至少一次 / 两次不同查询帧支持。joined 仅表示算法执行合并，不代表身份正确。coframe_conflict 是保守冲突规则，也不是异物真值。</p>
<p><a href="annotation_template.json">独立标注模板（空标签）</a> · <a href="consequences.json">最终历史变化</a> · <a href="../../../benchmarks/livo_history_graph_2026-11-11.md">报告</a></p>
'''+'\n'.join(cards)+'</html>\n')
    print('Edges:',len(records),'consequences:',len(consequences))
    for c in consequences:
        if c['foreground']:
            print(c['variant'],c['alias'],c['name'],'old',c['baseline_aliases'],'added',[i+1 for i in c['added_observations']])


if __name__ == '__main__':
    main()
