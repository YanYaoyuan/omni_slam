"""Presentation-only Ours labels; leave frozen algorithm/source IDs intact."""
from pathlib import Path
import argparse
import json
import hashlib
import re
import shutil
import time

ROOT=Path(__file__).resolve().parents[2]
DEV=ROOT/'docs/reports/visibility_cleanup_2026-11-12'
INDEPENDENT=ROOT/'docs/reports/visibility_independent_2026-11-12'
ARCHIVE=ROOT/'build/benchmarks/visibility_report_labels_20261112'
LABELS={
    'cg':'ConceptGraphs (shared input)',
    'history2':'omni_slam baseline',
    'sor':'Baseline + geometric SOR',
    'visibility1':'omni_slam + Visibility Cleanup (Ours)',
    'visibility3':'Ours — 3-frame control',
    'visibility1_no_depth':'Ours w/o visibility (ablation)',
    'maskclustering':'MaskClustering (shared input)',
}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    temporary=path.with_suffix(path.suffix+'.labels.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def snapshot(folder):
    target=ARCHIVE/folder.name
    if target.exists():return
    target.mkdir(parents=True)
    for p in folder.iterdir():
        if p.is_file():shutil.copyfile(p,target/p.name)


def independent():
    path=INDEPENDENT/'index.html'
    if not path.exists():return
    content=path.read_text()
    if 'id="ours-method-label"' in content:return
    snapshot(INDEPENDENT)
    for key,label in LABELS.items():
        marked=f'<strong>{label}</strong>' if key=='visibility1' else label
        content=content.replace(f'<td>{key}</td>',f'<td>{marked}<br><code>{key}</code></td>')
    banner='<div id="ours-method-label" style="padding:14px 18px;background:#e4f5ee;border-left:5px solid #169b74;margin:16px 0"><strong>我们的方法：omni_slam + Visibility Cleanup (Ours)</strong><br>固定实验 ID：<code>visibility1</code>。<code>history2</code> 是改进前的强基线；三帧版和去可见性版为对照 / 消融。</div>'
    content=content.replace('</h1>','</h1>'+banner,1)
    content=content.replace('<tr><td><strong>'+LABELS['visibility1'], '<tr style="background:#e4f5ee"><td><strong>'+LABELS['visibility1'])
    path.write_text(content)
    save(INDEPENDENT/'method_labels.json',dict(primary_method_id='visibility1',primary_label=LABELS['visibility1'],labels=LABELS,
        scope='Display labels only. Frozen protocol IDs, thresholds and metric values unchanged.'))
    save(INDEPENDENT/'display_revision.json',dict(updated_epoch=time.time(),html_sha256=sha(path),
        previous_html_sha256=sha(ARCHIVE/INDEPENDENT.name/'index.html'),change='Ours and baseline/ablation display labels only',algorithm_or_scores_changed=False))
    print('LABELED',path,flush=True)


def development():
    snapshot(DEV)
    original_summary=sha(DEV/'summary.json')
    import publish_visibility_cleanup as report
    report.LABELS.update(native_bridge_2='omni_slam 强基线',multiview='Ours（三帧开发版）',
        no_visibility='Ours w/o visibility（消融）',single_refutation='Ours（单帧反证简化版；开发消融）')
    report.ENGLISH[:]=['CG (shared input)','omni_slam baseline','Geometric SOR',
                      'Ours (3-frame pilot)','Ours w/o visibility','Ours (1-frame)']
    report.main()
    assert sha(DEV/'summary.json')==original_summary
    path=DEV/'index.html';content=path.read_text()
    banner='<div id="ours-method-label" class="callout"><strong>Ours 标注：</strong>本页原始主实验是 <code>multiview</code>，现标为 <b>Ours（三帧开发版）</b>；<code>single_refutation</code> 是 <b>Ours（单帧反证简化版）</b>，在后续独立测试中固定为主方法 <code>visibility1</code>。这里只补名称，没有更换本页原主结果或任何分数。</div>'
    content=content.replace('</h1>','</h1>'+banner,1);path.write_text(content)
    save(DEV/'method_labels.json',dict(original_primary_id='multiview',independent_primary_correspondence={'single_refutation':'visibility1'},labels=report.LABELS))
    save(DEV/'display_revision.json',dict(updated_epoch=time.time(),change='Ours labels in report tables, comparison bars and legends',
        original_artifacts_archived=str(ARCHIVE/DEV.name),metric_summary_sha256=original_summary,
        metric_summary_unchanged=True,algorithm_sources_unchanged=True))
    audit=json.loads((DEV/'audit.json').read_text())
    audit['presentation_revision']='display_revision.json'
    audit['outputs']={str(p.relative_to(DEV)):sha(p) for p in DEV.rglob('*') if p.is_file() and p.name not in ['audit.json','verification.json']}
    save(DEV/'audit.json',audit)
    verification=json.loads((ARCHIVE/DEV.name/'verification.json').read_text())
    verification.update(report_hash=sha(path),presentation_revision='display_revision.json',
        previous_verification_archived=str(ARCHIVE/DEV.name/'verification.json'))
    save(DEV/'verification.json',verification)


def watch():
    status=ROOT/'build/benchmarks/visibility_independent_20261112/queue_status.json'
    for _ in range(4320):
        independent()
        state=json.loads(status.read_text())
        if state.get('complete') or state.get('failed'):
            independent();return
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--development',action='store_true');args=parser.parse_args()
    if args.development:development()
    independent()
    if args.watch:watch()
