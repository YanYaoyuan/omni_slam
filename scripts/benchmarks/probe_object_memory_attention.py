"""One bounded, frozen-feature memory-reader probe on recorded causal decisions.

No new maps, inference, training, GT grouping, future history, or temperature
search. Predictions for ALL scenes are sealed before post-hoc GT diagnostics.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'omni_semantic_map'))
from omni_semantic_map.object_memory_attention import rerank

OUT = ROOT / 'build/benchmarks/object_memory_attention_20261112'
REPORT = ROOT / 'docs/reports/object_memory_attention_2026-11-12'
HISTORY = ROOT / 'build/benchmarks/evidence_retention_20261111/public_history_v1'
DINO = HISTORY.parent / 'public_dino_v1'
KEYS = ['replica_office_0', 'replica_office_1', 'scenenn_014', 'scenenn_015']
FEATURES = ['siglip', 'dino']
METHODS = ['native', 'mean', 'nearest', 'attention']


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes(hashes):
    for p, expected in hashes.items():
        assert sha(p) == expected, p


def freeze():
    assert not OUT.exists(), 'Never overwrite or tune a completed probe'
    parent = read(HISTORY / 'protocol.json')
    scenes = [next(s for s in parent['scenes'] if s['key'] == k) for k in KEYS]
    inputs = {str(HISTORY / 'protocol.json'): sha(HISTORY / 'protocol.json'),
              str(DINO / 'protocol.json'): sha(DINO / 'protocol.json')}
    references = {}
    for s in scenes:
        cache = Path(s['cache']); manifest = read(cache / 'manifest.json')
        assert manifest['complete'] and manifest['no_association']
        assert sha(cache / 'manifest.json') == s['hashes'][str(cache / 'manifest.json')]
        for name, h in manifest['files_sha256'].items():
            inputs[str(cache / name)] = h
        inputs[str(cache / 'manifest.json')] = sha(cache / 'manifest.json')
        for r in manifest['records']:
            inputs[str(cache / 'frames' / r['file'])] = r['sha256']
        run = HISTORY / 'runs' / s['key']; status = read(run / 'run.json')
        assert status['complete']
        inputs[str(run / 'run.json')] = sha(run / 'run.json')
        inputs[str(run / 'events.json')] = status['artifacts']['events.json']
        fm = DINO / 'features' / s['key'] / 'manifest.json'; d = read(fm)
        assert d['complete'] and d['protocol_sha256'] == sha(DINO / 'protocol.json')
        assert [(r['frame_id'], r['observation_offset'], r['masks']) for r in manifest['records']] == [
            (r['frame_id'], r['observation_offset'], r['masks']) for r in d['records']]
        inputs[str(fm)] = sha(fm)
        for r in d['records']:
            inputs[str(fm.parent / r['file'])] = r['sha256']
        references[s['reference']] = s['hashes'][s['reference']]
    verify_hashes(inputs); verify_hashes(references)
    sources = [Path(__file__), ROOT / 'omni_semantic_map/omni_semantic_map/object_memory_attention.py',
               ROOT / 'omni_semantic_map/test/test_object_memory_attention.py']
    p = dict(created_epoch=time.time(), scenes=scenes, features=FEATURES, readers=METHODS,
        temperature=.1, learned_parameters=0, trained=False, maximum_new_maps=0,
        input_sha256=inputs, reference_sha256=references,
        source_sha256={str(s): sha(s) for s in sources},
        memory='Actual CG candidate members at decision time; one largest-area mask per prior frame, lowest ID tie; all such prior frames, no GT selection.',
        mean='Normalize each view, mean then normalize; query cosine.',
        nearest='Maximum query cosine over exactly the same memory views.',
        attention='Identity Q=K=V on unit features; softmax(query dot view / 0.1), weighted view sum then normalize; query cosine. Fixed, untrained single-head retrieval, NOT trained Transformer.',
        candidates='Only original CG ambiguous events with >=2 native-eligible candidates. Same geometric candidate gate for all methods, appearance-only reranking; native score remains a separate control.',
        fallback='Any invalid query/view or zero pooled vector triggers native fallback for ALL readers of that feature/event. No candidate removal.',
        chronology='Every memory frame strictly earlier than query. Alternative choices never change later native histories: one-step teacher-forced diagnostic, not a causal mapping rollout.',
        gt='Evaluation only after all predictions sealed. Raw measured-support union per original candidate prefix; nearest original reference <=5cm, original valid mask, foreground dominant purity>=0.7, >=5 valid points. Background participates in purity denominator. Proxy identity labels, not exact instance ground truth assignments.',
        primary='Common SigLIP/DINO descriptor-supported events, query and ALL candidates have usable proxy identities, and both a same-identity candidate and a different-identity candidate exist. Report unresolved, unsupported, absent-positive and all-positive separately.',
        gate=dict(min_primary_events_per_dataset=50, delta_vs_mean_and_nearest_pp=2.,
                  not_below_native_each_dataset=True, nondecreasing_scenes_vs_both=3,
                  minimum_native_wrong_primary_per_dataset=10),
        gate_meaning='Each feature is a separate arm. Within BOTH datasets, equal-scene mean attention accuracy >= mean+2pp and nearest+2pp, >= native, >=3/4 scenes nondecreasing vs both; adequate coverage required. Only merits a separately designed rollout, never a PQ or generalization claim.',
        stop='One temperature, four fixed development scenes, two existing feature caches, three readers plus native. No sweeps, maps, training, new features or automatic expansion even if promising.',
        training_data='All 11 scenes in parent history protocol already examined; no independent train/validation/test split established for this pilot. No fitting on these four scenes.',
        limitations='Whole-mask pooled descriptors only, not patch-level DINO cross-attention. Existing memory can be contaminated. No abstention/new-object choice; no-positive failures exposed. Repeated events correlated; no significance/independent generalization claim.')
    dump(OUT / 'protocol.json', p)
    for s in sources:
        target = OUT / 'sources' / s.name
        target.parent.mkdir(exist_ok=True); target.write_bytes(s.read_bytes())
    print('FROZEN', len(inputs), 'input files', sha(OUT / 'protocol.json'), flush=True)


def protocol():
    p = read(OUT / 'protocol.json'); verify_hashes(p['source_sha256'])
    return p


def load_scene(s, supports=False):
    cache = Path(s['cache']); m = read(cache / 'manifest.json')
    siglip, frames, areas, points = [], [], [], []
    n = 0
    for r in m['records']:
        assert r['observation_offset'] == n
        with np.load(cache / 'frames' / r['file']) as z:
            siglip.append(z['features']); areas.extend(z['areas'].tolist())
            if supports:
                points.extend([np.unique(z['point_ids'][a:b]) for a, b in zip(z['offsets'][:-1], z['offsets'][1:])])
        frames.extend([r['frame_id']] * r['masks']); n += r['masks']
    fm = DINO / 'features' / s['key']; dino = []
    for r in read(fm / 'manifest.json')['records']:
        with np.load(fm / r['file']) as z:
            dino.append(z['mean'])
    features = dict(siglip=np.concatenate(siglip), dino=np.concatenate(dino))
    assert all(len(f) == n for f in features.values())
    return features, np.array(frames), np.array(areas), points


def predict():
    p = protocol(); verify_hashes(p['input_sha256'])
    assert not (OUT / 'predictions').exists()
    sealed = {}
    for s in p['scenes']:
        start = time.monotonic()
        features, frames, areas, _ = load_scene(s)
        events = read(HISTORY / 'runs' / s['key'] / 'events.json')
        assert len({e['query'] for e in events}) == len(events)
        for feature in FEATURES:
            rows = [rerank(e, features[feature], frames, areas, p['temperature']) for e in events]
            path = OUT / 'predictions' / s['key'] / (feature + '.json')
            dump(path, dict(complete=True, no_gt=True, rows=rows,
                           protocol_sha256=sha(OUT / 'protocol.json')))
            sealed[str(path)] = sha(path)
        print('PREDICTED', s['key'], len(events), 'events', round(time.monotonic()-start, 2), 's', flush=True)
    dump(OUT / 'predictions_complete.json', dict(complete=True, files_sha256=sealed,
         new_maps=0, trained=False, protocol_sha256=sha(OUT / 'protocol.json'), epoch=time.time()))


def signature(point_ids, projected, supported):
    ids = np.unique(point_ids)
    labels = projected[ids[supported[ids]]]
    values, counts = np.unique(labels[labels > 0], return_counts=True)
    dominant = int(values[np.argmax(counts)]) if len(values) else None
    purity = float(counts.max() / len(labels)) if len(values) else 0.
    return dict(dominant=dominant, purity=purity, valid_points=len(labels),
                usable=bool(len(labels) >= 5 and dominant is not None and purity >= .7))


def tally(rows, feature, selector):
    selected = [r for r in rows if selector(r)]
    methods = {}
    for name in METHODS:
        correct = [r['correct'][feature][name] for r in selected]
        comparisons = {}
        for base in ['native', 'mean', 'nearest']:
            comparisons[base] = dict(
                gained=sum(r['correct'][feature][name] and not r['correct'][feature][base] for r in selected),
                lost=sum(not r['correct'][feature][name] and r['correct'][feature][base] for r in selected))
        methods[name] = dict(correct=sum(correct), total=len(selected),
                             accuracy100=100*sum(correct)/len(selected) if selected else None,
                             transitions=comparisons)
    return methods


def scene_evaluate(s):
    from scipy.spatial import cKDTree
    key = s['key']; features, frames, areas, supports = load_scene(s, supports=True)
    events = read(HISTORY / 'runs' / key / 'events.json')
    predictions = {f: read(OUT / 'predictions' / key / (f+'.json'))['rows'] for f in FEATURES}
    ref = np.load(s['reference']); geo = np.load(Path(s['cache']) / 'geometry.npz')
    valid = ref['valid'] if 'valid' in ref.files else np.ones(len(ref['instance']), bool)
    dist, idx = cKDTree(ref['xyz']).query(geo['xyz'], workers=1)
    supported = (dist <= .05) & valid[idx]; projected = ref['instance'][idx]
    assert (projected[supported] >= 0).all()
    cache = {}
    def sig(members):
        t = tuple(sorted(set(members)))
        if t not in cache:
            cache[t] = signature(np.concatenate([supports[i] for i in t]), projected, supported)
        return cache[t]
    rows = []
    for i, e in enumerate(events):
        q = sig([e['query']]); candidates = [sig(c['members']) for c in e['candidates']]
        resolved = q['usable'] and all(c['usable'] for c in candidates)
        same = [bool(resolved and c['dominant'] == q['dominant']) for c in candidates]
        common = all(predictions[f][i]['supported'] for f in FEATURES)
        assert all(predictions[f][i]['query'] == e['query'] for f in FEATURES)
        correct = {f: {name: same[predictions[f][i]['choices'][name]] for name in METHODS} for f in FEATURES}
        row = dict(query=e['query'], frame=e['frame'], query_signature=q,
                   candidate_signatures=candidates, resolved=resolved, common_supported=common,
                   positive_candidates=sum(same), negative_candidates=len(same)-sum(same) if resolved else None,
                   primary=bool(resolved and common and any(same) and not all(same)),
                   correct=correct, choices={f: predictions[f][i]['choices'] for f in FEATURES},
                   feature_supported={f: predictions[f][i]['supported'] for f in FEATURES})
        rows.append(row)
    counts = dict(total=len(rows), primary=sum(r['primary'] for r in rows),
        unresolved=sum(not r['resolved'] for r in rows),
        resolved_no_positive=sum(r['resolved'] and r['positive_candidates']==0 for r in rows),
        resolved_all_positive=sum(r['resolved'] and r['negative_candidates']==0 for r in rows),
        resolved_discriminative_unsupported=sum(r['resolved'] and r['positive_candidates']>0 and r['negative_candidates']>0 and not r['common_supported'] for r in rows),
        unsupported_by_feature={f: sum(not r['feature_supported'][f] for r in rows) for f in FEATURES})
    assert sum(counts[k] for k in ['primary','unresolved','resolved_no_positive','resolved_all_positive','resolved_discriminative_unsupported']) == len(rows)
    result = dict(key=key, dataset=s['dataset'], counts=counts,
        primary={f: tally(rows, f, lambda r: r['primary']) for f in FEATURES},
        all_resolved={f: tally(rows, f, lambda r: r['resolved']) for f in FEATURES},
        primary_frames=len({r['frame'] for r in rows if r['primary']}),
        primary_query_objects=len({r['query_signature']['dominant'] for r in rows if r['primary']}),
        oracle_resolved=dict(total=sum(r['resolved'] for r in rows),
            has_positive=sum(r['resolved'] and r['positive_candidates']>0 for r in rows)),
        primary_native_wrong=sum(r['primary'] and not r['correct']['siglip']['native'] for r in rows))
    dump(OUT / 'evaluation' / key / 'events.json', rows)
    dump(OUT / 'evaluation' / key / 'metrics.json', result)
    print('SCORED', key, counts, flush=True)
    return result


def aggregate(scenes, p):
    datasets = {}; gates = {}
    for ds in ['Replica', 'SceneNN']:
        subset = [s for s in scenes if s['dataset'] == ds]; group = {}
        for feature in FEATURES:
            group[feature] = {}
            for method in METHODS:
                values = [s['primary'][feature][method] for s in subset]
                group[feature][method] = dict(
                    macro_accuracy100=float(np.mean([v['accuracy100'] for v in values])) if all(v['total'] for v in values) else None,
                    correct=sum(v['correct'] for v in values), total=sum(v['total'] for v in values),
                    transitions={base: {k: sum(v['transitions'][base][k] for v in values) for k in ['gained','lost']} for base in ['native','mean','nearest']})
        datasets[ds] = dict(methods=group, primary=sum(s['counts']['primary'] for s in subset),
            total=sum(s['counts']['total'] for s in subset),
            native_wrong=sum(s['primary_native_wrong'] for s in subset))
    for feature in FEATURES:
        sufficient = all(d['primary'] >= p['gate']['min_primary_events_per_dataset'] and
                         d['native_wrong'] >= p['gate']['minimum_native_wrong_primary_per_dataset'] and
                         d['methods'][feature]['attention']['macro_accuracy100'] is not None for d in datasets.values())
        checks = {}
        for ds, d in datasets.items():
            a = d['methods'][feature]
            checks[ds] = {base: (a['attention']['macro_accuracy100'] is not None and a[base]['macro_accuracy100'] is not None and
                            a['attention']['macro_accuracy100'] - a[base]['macro_accuracy100'] >= (0 if base=='native' else p['gate']['delta_vs_mean_and_nearest_pp'])) for base in ['native','mean','nearest']}
        nondecreasing = sum(all(s['primary'][feature]['attention']['accuracy100'] is not None and s['primary'][feature][b]['accuracy100'] is not None and
                          s['primary'][feature]['attention']['accuracy100'] >= s['primary'][feature][b]['accuracy100'] for b in ['mean','nearest']) for s in scenes)
        passed = sufficient and all(all(v.values()) for v in checks.values()) and nondecreasing >= p['gate']['nondecreasing_scenes_vs_both']
        gates[feature] = dict(sufficient=sufficient, checks=checks, nondecreasing_scenes=nondecreasing, passed=passed,
            decision='PROMISING_ONE_STEP_ONLY' if passed else ('FAIL_FIXED_READER_GATE' if sufficient else 'INSUFFICIENT_PROXY_COVERAGE'))
    return dict(complete=True, protocol_sha256=sha(OUT/'protocol.json'), scenes=scenes, datasets=datasets, gates=gates,
                new_maps=0, learned_parameters=0, official_benchmark_score=False,
                stop='No automatic sweep, training or mapping rollout. Preserve this one-step probe.')


def evaluate():
    p = protocol(); seal = read(OUT / 'predictions_complete.json')
    assert seal['complete']; verify_hashes(seal['files_sha256'])
    verify_hashes(p['input_sha256']); verify_hashes(p['reference_sha256'])
    assert not (OUT / 'evaluation').exists()
    results = [scene_evaluate(s) for s in p['scenes']]
    summary = aggregate(results, p); dump(OUT / 'summary.json', summary)
    print('GATES', summary['gates'], flush=True)


def audit():
    from scipy.spatial import cKDTree
    p = protocol(); verify_hashes(p['input_sha256']); verify_hashes(p['reference_sha256'])
    seal = read(OUT/'predictions_complete.json'); verify_hashes(seal['files_sha256'])
    assert len(seal['files_sha256']) == len(KEYS)*len(FEATURES)
    for s in p['scenes']:
        features, frames, areas, supports = load_scene(s, supports=True)
        events = read(HISTORY/'runs'/s['key']/'events.json')
        for feature in FEATURES:
            saved = read(OUT/'predictions'/s['key']/(feature+'.json'))['rows']
            assert len(saved)==len(events)
            # Independent direct formula, including tie policy and causal selection.
            for event, row in zip(events, saved):
                assert row == rerank(event, features[feature], frames, areas, p['temperature'])
                native = next(i for i,c in enumerate(event['candidates']) if c['is_native'])
                values = {m:[] for m in METHODS[1:]}
                for members, r in zip(row['memories'], row['reads']):
                    assert all(frames[i] < frames[event['query']] for i in members)
                    assert len({frames[i] for i in members})==len(members)
                    if r is None: continue
                    q = features[feature][event['query']].astype(float); q /= np.linalg.norm(q)
                    h = features[feature][members].astype(float); h /= np.linalg.norm(h, axis=1)[:,None]
                    logits = np.array([np.dot(q,v)/p['temperature'] for v in h]); w=np.exp(logits-logits.max());w/=sum(w)
                    avg = sum(h)/len(h); att = sum(v*weight for v,weight in zip(h,w))
                    expected = dict(mean=np.dot(q,avg)/np.linalg.norm(avg), nearest=max(np.dot(q,v) for v in h), attention=np.dot(q,att)/np.linalg.norm(att))
                    for method in values:
                        np.testing.assert_allclose(r['scores'][method], expected[method], atol=1e-12)
                        values[method].append(r['scores'][method])
                    np.testing.assert_allclose(r['weights'],w,atol=1e-12)
                if row['supported']:
                    for method,scores in values.items():
                        assert row['choices'][method] == max(range(len(scores)),key=lambda i:(scores[i],i==native,-i))
                else: assert set(row['choices'].values())=={native}
        rows=read(OUT/'evaluation'/s['key']/'events.json'); metric=read(OUT/'evaluation'/s['key']/'metrics.json')
        # Independently rebuild identity proxies from unique original supports,
        # rather than trusting the saved evaluator labels.
        ref=np.load(s['reference']); xyz=np.load(Path(s['cache'])/'geometry.npz')['xyz']
        distance,index=cKDTree(ref['xyz']).query(xyz,workers=1)
        mask=ref['valid'] if 'valid' in ref.files else np.ones(len(ref['xyz']),bool)
        usable=(distance<=.05)&mask[index]; projected=ref['instance'][index]
        signatures={}
        def independent_signature(members):
            key=tuple(sorted(set(members)))
            if key not in signatures:
                ids=np.unique(np.concatenate([supports[i] for i in key]))
                labels=projected[ids[usable[ids]]]
                counts=Counter(int(x) for x in labels if x>0)
                gt,n=max(counts.items(),key=lambda z:(z[1],-z[0])) if counts else (None,0)
                purity=n/len(labels) if len(labels) else 0.
                signatures[key]=dict(dominant=gt,purity=purity,valid_points=len(labels),
                    usable=bool(len(labels)>=5 and gt is not None and purity>=.7))
            return signatures[key]
        for event,row in zip(events,rows):
            assert independent_signature([event['query']])==row['query_signature']
            assert [independent_signature(c['members']) for c in event['candidates']]==row['candidate_signatures']
            resolved=row['query_signature']['usable'] and all(c['usable'] for c in row['candidate_signatures'])
            same=[resolved and c['dominant']==row['query_signature']['dominant'] for c in row['candidate_signatures']]
            assert row['resolved']==resolved
            assert row['primary']==bool(resolved and row['common_supported'] and any(same) and not all(same))
        for feature in FEATURES:
            assert metric['primary'][feature] == tally(rows,feature,lambda r:r['primary'])
            for r in rows:
                for method,choice in r['choices'][feature].items():
                    assert r['correct'][feature][method] == bool(r['resolved'] and r['candidate_signatures'][choice]['dominant']==r['query_signature']['dominant'])
    summary=read(OUT/'summary.json')
    assert aggregate(summary['scenes'],p)==summary
    dump(OUT/'audit.json',dict(complete=True, sources_and_inputs_unchanged=True,
        predictions_reproduced=True, independent_attention_formula_checked=True,
        identity_proxies_independently_recomputed=True,
        causal_unique_frame_memories=True, ground_truth_after_prediction_seal=True,
        counts_transitions_and_gates_reproduced=True, maps_created=0,
        summary_sha256=sha(OUT/'summary.json'), protocol_sha256=sha(OUT/'protocol.json')))
    print('AUDIT PASS',flush=True)


def report():
    p=protocol(); s=read(OUT/'summary.json'); a=read(OUT/'audit.json'); assert a['complete']
    assert a['summary_sha256']==sha(OUT/'summary.json')
    REPORT.mkdir(parents=True,exist_ok=False)
    lines=['# 对象记忆与注意力：一次固定特征关联诊断', '',
        '2026-11-12。四个已查看开发场景，SigLIP 与 DINOv2 两套已有特征，平均／最近邻／固定注意力三种读取方式。未训练、未提取新特征、未生成新地图。', '',
        '**这是原 CG 历史上的单步候选重排序，不是闭环建图实验，不产生实例 PQ 或论文泛化成绩。** 历史含原算法的错误；替代决策不会改变后续历史。', '',
        '## 预设结论', '']
    for feature,g in s['gates'].items():
        lines.append(f"- {feature}: **{g['decision']}**；覆盖充分={g['sufficient']}，四场景中同时不低于平均和最近邻的场景数={g['nondecreasing_scenes']}。")
    lines += ['', '## 固定可判定子集的正确选择率（%，两场景等权均值）', '',
        '| 数据集 | 特征 | 原生 CG | 平均 | 最近邻 | 注意力 | 注意力−平均 | 注意力−最近邻 | 可判定 / 全部事件 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v): return 'N/A' if v is None else f'{v:.2f}'
    for ds,d in s['datasets'].items():
        for feature,methods in d['methods'].items():
            v={m:methods[m]['macro_accuracy100'] for m in METHODS}
            delta=lambda base: None if v['attention'] is None or v[base] is None else v['attention']-v[base]
            lines.append(f"| {ds} | {feature} | "+' | '.join(fmt(v[m]) for m in METHODS)+f" | {fmt(delta('mean'))} | {fmt(delta('nearest'))} | {d['primary']} / {d['total']} |")
    lines += ['', '## 注意力纠正与破坏的关联次数', '',
        '| 数据集 | 特征 | 相对原生：纠正 / 破坏 | 相对平均：纠正 / 破坏 | 相对最近邻：纠正 / 破坏 |',
        '|---|---|---:|---:|---:|']
    for ds,d in s['datasets'].items():
        for feature,methods in d['methods'].items():
            t=methods['attention']['transitions']
            lines.append(f'| {ds} | {feature} | '+' | '.join(f"{t[b]['gained']} / {t[b]['lost']}" for b in ['native','mean','nearest'])+' |')
    lines += ['', '## 每个场景与覆盖', '',
        '| 场景 | 特征 | 原生 | 平均 | 最近邻 | 注意力 | 有效事件 / 帧 / 查询对象 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in s['scenes']:
        for f in FEATURES:
            lines.append(f"| {r['key']} | {f} | "+' | '.join(fmt(r['primary'][f][m]['accuracy100']) for m in METHODS)+f" | {r['counts']['primary']} / {r['primary_frames']} / {r['primary_query_objects']} |")
    lines += ['', '| 场景 | 全部 | 身份无法判定 | 没有正确候选 | 全是正确候选 | 有区分度但缺特征 | 主评价子集 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in s['scenes']:
        c=r['counts'];lines.append('| '+r['key']+' | '+' | '.join(str(c[k]) for k in ['total','unresolved','resolved_no_positive','resolved_all_positive','resolved_discriminative_unsupported','primary'])+' |')
    lines += ['', '## 公式与公平性', '',
        '- 现有 SigLIP384 掩码描述子和 DINOv2-small 支持点池化描述子分别测试。没有换前端、位姿或几何候选门槛；不涉及局部 patch 注意力。',
        '- 三种方法读取完全相同的历史：每个历史帧保留面积最大的候选成员，平局取最小观测 ID；不读取当前或未来帧，不按 GT 整理历史。',
        '- 平均：单位视图特征求均值、归一化，与单位查询取余弦。最近邻：所有历史视图与查询余弦的最大值。',
        '- 注意力：Q/K/V 均为已有单位特征，无训练投影；w=softmax(q·h/0.1)，对历史加权求和并归一化，再与查询取余弦。温度只取一次，未搜索。',
        '- 任一候选所需特征无效，三种读取方法在该事件均退回原生选择；不偷偷移除困难候选。相同得分优先原生候选。',
        '- GT 只在四场景的全部预测封存后读取。对查询支持与各候选的原始历史支持并集投影到原参考域，5cm 内有效点至少 5 个，前景主导实例纯度至少 70%；背景计入纯度分母。该标签只是诊断代理，混合候选不强行判对错。',
        '- 主比较使用两种特征共同有效、全部身份代理可判定、至少一个正确且一个错误候选的固定子集。不能把条件正确率当全体事件准确率；全部可判定事件（含无正确候选）另存 metrics.json。',
        '- 同一对象的多次关联事件相关，不以事件数冒充独立场景数；已查看场景不作为泛化测试。', '',
        '## 继续门槛与停止', '',
        '每个特征分支分别要求：两个数据集的场景均值均比平均和最近邻各高至少 2 个百分点、不低于原生 CG；至少三场景不低于平均及最近邻。每个数据集至少 50 个主评价事件及 10 个原生错误机会；不足时结论为覆盖不足。即使通过，也只支持另行设计实际建图回放，不能宣称 PQ 提升。', '',
        '**本轮到此停止；没有追加温度搜索、训练或地图。** 失败只能否定这版固定公式及特征条件，不能否定所有学习式注意力。平均 vs 最近邻反映是否值得保留多视角，注意力 vs 最近邻才反映软加权是否有额外价值。', '',
        '[冻结协议](protocol.json) · [完整结果](summary.json) · [复核](audit.json)', '']
    text='\n'.join(lines);(REPORT/'report.md').write_text(text)
    # Small self-contained HTML, preserving tables without an external renderer.
    body=[]; in_table=False;in_list=False
    for line in lines:
        if line.startswith('|'):
            if not in_table:body.append('<table>');in_table=True
            cells=[v.strip() for v in line.strip('|').split('|')]
            if all(set(c)<=set('-: ') for c in cells):continue
            body.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>');continue
        if in_table:body.append('</table>');in_table=False
        if line.startswith('- '):
            if not in_list:body.append('<ul>');in_list=True
            body.append('<li>'+html.escape(line[2:])+'</li>');continue
        if in_list:body.append('</ul>');in_list=False
        if line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line:body.append('<p>'+html.escape(line).replace('**','')+'</p>')
    if in_table:body.append('</table>')
    if in_list:body.append('</ul>')
    (REPORT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>对象记忆与注意力诊断</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#172d3b}table{border-collapse:collapse;width:100%;font-size:14px}td{border:1px solid #cbd5df;padding:8px}tr:first-child{background:#e8f0f5;font-weight:bold}h1{font-size:28px}h2{margin-top:36px}li{margin:8px 0}</style>'+''.join(body)+'<p><a href="report.md">Markdown</a> · <a href="protocol.json">协议</a> · <a href="summary.json">结果</a> · <a href="audit.json">复核</a></p></html>')
    for name in ['protocol.json','summary.json','audit.json']:
        (REPORT/name).write_bytes((OUT/name).read_bytes())
    print('REPORT',REPORT,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','predict','evaluate','audit','report'])
    args=parser.parse_args();globals()[args.action]()
