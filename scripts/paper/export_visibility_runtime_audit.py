"""Export existing kernel timers and map sizes, without rerunning any method."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'build/benchmarks/visibility_independent_20261112'
OUT = ROOT / 'docs/paper/visibility_revision/latex'


def main():
    inputs, rows = {}, []
    implementation = ROOT / 'scripts/benchmarks/visibility_independent_mapping.py'
    inputs[str(implementation.relative_to(ROOT))] = hashlib.sha256(implementation.read_bytes()).hexdigest()

    def read(path):
        data = path.read_bytes()
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    for scene in ['052', '054', '057', '061']:
        run = read(BASE / f'maps/{scene}/cleanup/run.json')
        events = read(BASE / f'maps/{scene}/cleanup/events.json')
        objects = read(BASE / f'maps/{scene}/cg/history2/classification.json')['objects']
        assert run['complete'] and len(events) == 200
        rows.append(dict(scene=scene, foreground_objects=sum(
            bool(o['predictions']['mean']['retained']) for o in objects),
            source_points=run['methods']['visibility1']['foreground_points'],
            frames=len(events), evidence_seconds=sum(e['seconds'] for e in events),
            cleanup_job_seconds=run['seconds']))
    audit = dict(new_experiments=False, source_sha256=inputs, rows=rows,
                 evidence_timer='frame_evidence plus accumulation; computes gated and FOV-only votes together; excludes mask/depth loading, SOR, rejection and export',
                 job_timer='complete cached cleanup job including loading, all four variants, SOR and exports; excludes frontend and initial mapping',
                 limitations='Single archived execution; not isolated Ours latency, repeated benchmark, or end-to-end SLAM runtime; original CPU and thread allocation not pinned by these logs.')
    (OUT / 'runtime_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    table = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Scene & Objects & Points & Evidence (s) & Job (s) \\', r'\midrule']
    for r in rows:
        table.append(f"{r['scene']} & {r['foreground_objects']} & {r['source_points']:,} & "
                     f"{r['evidence_seconds']:.3f} & {r['cleanup_job_seconds']:.3f}" + r' \\')
    table += [r'\bottomrule', r'\end{tabular}', '']
    (OUT / 'tables/runtime.tex').write_text('\n'.join(table))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
