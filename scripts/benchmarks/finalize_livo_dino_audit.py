"""Archive reviewed decision evidence and verify frozen association artifacts.

Visual notes are assistant inspection of all eight generated image triptychs,
not independent ground truth or automatic identity labels.
"""
import hashlib
import html
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'build/benchmarks/evidence_retention_20261111'
DEST = ROOT / 'docs/reports/evidence_retention_2026-11-11/dino_association_v1'
NOTES = {
    630: '白色 SUV：两条候选历史的证据图看起来均为当前同一辆车。改选历史分支，尚未证明纠正了异车误合并。',
    1114: '地面：改选较近时刻的地面观测。没有地面实例边界真值，不能计作对象身份改善。',
    1207: '白色小车：两条候选历史的证据图看起来均为同车。最终保留成员从 5 个变为 4 个，未恢复两个分支的全部历史。',
    1350: '墙面：窄墙片与更大墙区域的分区选择，实例范围有歧义，不能判作准确率提升。',
    1743: '墙面：当前窄墙片在两条候选证据中都可见；DINO 改选更大墙区域。背景分区改变，未证明改善。',
    2397: '停车位车辆：立柱、入口及邻车位置支持两条候选证据均为同车。没有直接观察到异车误合并被纠正。',
    2843: '停放车辆：遮挡程度不同，但车身与邻车位置支持两条候选证据均为同车。仍是历史分支选择。',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def foreground_geometry(path):
    rows = read(path / 'classification.json')['objects']
    geometry = np.load(path / 'geometry.npz')
    return {tuple(sorted(o['observation_ids'])):
            geometry['xyz'][geometry['instance'] == o['object_id']]
            for o in rows if o['predictions']['mean']['retained']}


def main():
    root = OUT / 'dino_association_v1'
    spec = read(root / 'protocol.json')
    for path, expected in spec['sources'].items():
        assert digest(path) == expected, path
    assert digest(OUT / 'scale_ablation_v1/protocol.json') == spec['scale_protocol_sha256']
    assert digest(OUT / 'dino_livo_tokens_v2/manifest.json') == spec['dino_manifest_sha256']
    assert digest(OUT / spec['base'] / 'run.json') == spec['base_run_sha256']
    base = root / 'baseline/cache_supportcc/both'
    old = OUT / spec['base']
    assert read(base / 'classification.json') == read(old / 'classification.json')
    g, expected = np.load(base / 'geometry.npz'), np.load(old / 'geometry.npz')
    assert set(g.files) == set(expected.files)
    for key in g.files:
        np.testing.assert_array_equal(g[key], expected[key])
    a, b = foreground_geometry(base), foreground_geometry(root / 'dino_local/cache_supportcc/both')
    assert a.keys() == b.keys() and len(a) == 41
    assert all(len(points) > 0 for points in a.values())
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    summary = read(DEST / 'summary.json')
    for row in summary['summaries']:
        association = root / row['variant'] / 'association.json'
        assert digest(association) == row['association_sha256']
        run = read(association)
        assert run['complete'] and run['protocol_sha256'] == digest(root / 'protocol.json')
        assert digest(Path(run['output']) / 'run.json') == row['run_sha256']
    cases, fragments, cards = [], [], []
    rows_by_variant = {v: read(root / v / 'cache_supportcc/both/classification.json')['objects']
                       for v in ['baseline', 'dino_pooled']}
    for case in summary['cases']:
        oid = case['observation_id']
        image = Path(case['evidence_image'])
        assert digest(image) == case['image_sha256']
        note = NOTES[oid + 1]
        cases.append(dict(variant=case['variant'], observation_id=oid,
                          image_sha256=case['image_sha256'], note=note))
        cards.append(f'<article><h2>{html.escape(case["variant"])} · 观测 {oid+1}</h2>'
                     f'<p>{html.escape(note)}</p><a href="{image.name}">'
                     f'<img src="{image.name}" alt="当前观测、原候选和 DINO 候选"></a></article>')
        if case['variant'] == 'dino_pooled' and oid + 1 in [630, 1207, 2397, 2843]:
            item = dict(query_alias=oid+1, candidates_existed_before_query=case['candidates'])
            for variant, rows in rows_by_variant.items():
                match = next(o for o in rows if oid in o['observation_ids'])
                item[variant] = dict(alias=min(match['observation_ids'])+1,
                                     member_aliases=[i+1 for i in match['observation_ids']])
            fragments.append(item)
    assert len(cases) == 8
    review = dict(scope=__doc__, visual_review_complete=True, independent_ground_truth=False,
                  summary_sha256=digest(DEST / 'summary.json'),
                  frozen_source_hashes_verified=True, baseline_exact_reproduction=True,
                  local_foreground_geometry_exact=True, cases=cases, car_history_changes=fragments)
    (DEST / 'review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n')
    (DEST / 'gallery.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>DINO 关联全量改选审查</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;background:#f4f6f8;color:#182431}article{background:white;padding:20px;margin:24px 0;border:1px solid #dce3e9;border-radius:8px}img{width:100%;height:auto}h2{font-size:20px}a{color:#155b9a}</style>
<h1>DINO 关联：全部 8 个改选案例</h1>
<p>完整 330 帧重放；左：当前观测，中：原生候选，右：DINO 候选。候选图为实际打分中 DINO 得分最高的历史观测，并非原生 SigLIP 最相似观测。点击可看原图。</p>
<p>四个车辆改选的图像均支持“同车的不同历史分支”；未确认纠正异车误合并。局部 DINO 只改变一次墙面选择，41 个前景对象的成员和点云均与基线精确相同。以下为助手图像审查，不是独立实例真值。</p>
<p><a href="../../../benchmarks/livo_dino_association_2026-11-11.md">完整报告</a> · <a href="summary.json">机器统计</a> · <a href="review.json">审查与校验记录</a></p>
''' + '\n'.join(cards) + '</html>\n')
    print('Verified frozen sources, baseline reproduction, 41 identical local foreground geometries, 8 reviewed cases.')


if __name__ == '__main__':
    main()
