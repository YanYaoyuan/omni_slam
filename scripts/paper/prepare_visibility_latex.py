"""Export frozen evidence into a portable LaTeX source package; no inference."""
from pathlib import Path
import hashlib
import json
import statistics
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'docs/paper/visibility_revision'
OUT = PAPER / 'latex'
BASE = ROOT / 'build/benchmarks/visibility_independent_20261112'
REVIEW = ROOT / 'docs/reports/visibility_mechanism_review_2026-11-12'
INPUTS = {}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def use(p):
    INPUTS[str(p.relative_to(ROOT))] = sha(p)
    return p

def read(p):
    return json.loads(use(p).read_text())

def table(name, columns, header, rows):
    content = [r'\begin{tabular}{' + columns + '}', r'\toprule',
               ' & '.join(header) + r' \\', r'\midrule']
    content.extend(' & '.join(map(str, r)) + r' \\' for r in rows)
    content.extend([r'\bottomrule', r'\end{tabular}', ''])
    (OUT / 'tables' / (name + '.tex')).write_text('\n'.join(content))

def main():
    if (OUT / 'merged_version.json').exists():
        raise SystemExit('The canonical manuscript now combines current and V1 evidence. '
                         'This legacy V1 exporter must not overwrite it. '
                         'See docs/paper/visibility_revision/latex/README.md; '
                         'the original V1 package is retained under docs/paper/archives/.')
    (OUT / 'tables').mkdir(exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    metrics = {s: read(BASE / 'evaluation' / s / 'metrics.json')['methods']
               for s in ['052', '054', '057', '061']}
    methods = [('cg', 'ConceptGraphs (shared input)'),
               ('maskclustering', 'MaskClustering (shared input)'),
               ('history2', 'Strong baseline'), ('sor', 'Baseline + SOR'),
               ('visibility1', r'\rowcolor{ourstint}\textbf{Ours}'),
               ('visibility3', 'Ours: three-frame control'),
               ('visibility1_no_depth', 'Ours: no depth gate')]
    scores = {}
    for k, _ in methods:
        values = [m[k]['score']['observed_voxel_instance_pq'] * 100 for m in metrics.values()]
        scores[k] = values + [statistics.mean(values)]
    best = [max(v[i] for v in scores.values()) for i in range(5)]
    rows = []
    stages = {'cg': 'C', 'maskclustering': 'M', 'history2': 'C', 'sor': 'C+S',
              'visibility1': 'C+V', 'visibility3': 'C+V', 'visibility1_no_depth': 'C+V'}
    for k, label in methods:
        values = [r'\textbf{' + f'{v:.3f}' + '}' if abs(v-best[i]) < 1e-10 else f'{v:.3f}'
                  for i, v in enumerate(scores[k])]
        rows.append([label, stages[k], *values])
    table('main', 'lcrrrrr', ['Method', 'Final stages', '052', '054', '057', '061', 'Mean'], rows)
    contrast = []
    for k, label in [('history2', 'Ours -- baseline'), ('sor', 'Ours -- SOR'),
                     ('visibility1_no_depth', 'Ours -- no depth'), ('visibility3', 'Ours -- three-frame')]:
        ds = [a-b for a,b in zip(scores['visibility1'][:4], scores[k][:4])]
        contrast.append([label, f'{statistics.mean(ds):+.3f}', f'{sum(d>1e-9 for d in ds)}/{sum(d< -1e-9 for d in ds)}'])
    for k,label in [('visibility1','Ours locked -- baseline'),('sor','SOR locked -- baseline')]:
        ds=[100*(m[k]['locked_assignment_score']['observed_voxel_instance_pq']-m['history2']['score']['observed_voxel_instance_pq']) for m in metrics.values()]
        contrast.append([label,f'{statistics.mean(ds):+.3f}',f'{sum(d>1e-9 for d in ds)}/{sum(d< -1e-9 for d in ds)}'])
    table('ablation','lrr',['Contrast',r'$\Delta$ PQ','Wins/losses'],contrast)
    objects=read(REVIEW/'per_object.json');rows=[]
    for scene in metrics:
        ds=[r['delta_ours_baseline'] for r in objects if r['scene']==scene]
        rows.append([scene,len(ds),sum(d>1e-10 for d in ds),sum(d< -1e-10 for d in ds),sum(abs(d)<=1e-10 for d in ds)])
    table('objects','lrrrr',['Scene','GT','Improved','Reduced','Same'],rows)
    diag=read(REVIEW/'scene057_diagnosis.json');rows=[]
    for k,label in [('history2','Baseline'),('visibility1','Ours'),('sor','SOR'),('visibility1_locked','Ours (locked)')]:
        st=diag['pair_stats'][k];rows.append([label,st['correct'],st['excess'],st['missing'],f"{st['iou']:.6f}"])
    table('case','lrrrr',['Method','Correct','Excess','Missed','IoU'],rows)
    supplement=use(PAPER/'supplement.md').read_text()
    pilot=[]
    for line in supplement.splitlines():
        if line.startswith(('| office_0 |','| office_1 |','| SN 014 |','| SN 015 |')):
            pilot.append([s.strip().replace('_',r'\_') for s in line.strip('|').split('|')])
    assert len(pilot)==4
    table('development','lrrrrr',['Scene','Baseline','SOR','Three-frame','No depth (3)','Ours (1)'],pilot)
    inventory=read(PAPER/'figures/rosbag_visual_provenance.json')['recordings']
    table('recordings','lrrr',['Recording','Minutes','Path (m)','Map points'],[
        [r['bag'].replace('_',r'\_'),f"{r['duration_s']/60:.1f}",f"{r['estimated_path_m']:.1f}",f"{r['points']:,}"] for r in inventory])
    figures=['method_architecture','object_comparison_real','mechanism_decomposition',
             'all_object_changes','057_threshold_case','development_vs_unseen',
             'dataset_scene_057','dataset_scene_061','garage_real_evidence',
             'rosbag_routes_1','rosbag_routes_2','rosbag_outdoor_semantics','rosbag_ground_details']
    jobs=[(use(PAPER/'figures'/(name+'.svg')),OUT/'figures'/(name+'.pdf')) for name in figures]
    def export(job):
        src,dst=job
        export_src = src
        if src.stem == 'method_architecture':
            content = src.read_text()
            for before, after in {'Support P': 'Support S', 'Refutation N': 'Refutation R',
                                  '>NNP<': '>RRS<',
                                  r'$N\geq 1\quad\wedge\quad N\geq 2P$':
                                  r'$R\geq 1\quad\wedge\quad R\geq 2S$'}.items():
                assert before in content, before
                content = content.replace(before, after)
            export_src = OUT / 'figures/method_architecture_notation.svg'
            export_src.write_text(content)
        result=subprocess.run(['inkscape',str(export_src),'--export-type=pdf','--export-filename='+str(dst)],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True,
            env={**os.environ,'XDG_CACHE_HOME':'/tmp/omni_inkscape_cache'})
        assert dst.is_file() and dst.stat().st_size>1000
        return {'source':str(src.relative_to(ROOT)),'pdf':str(dst.relative_to(OUT)),
                'sha256':sha(dst),'export':'Inkscape SVG-to-PDF; support/refutation symbols S/R; no image or geometry alteration'}
    with ThreadPoolExecutor(max_workers=3) as pool: exported=list(pool.map(export,jobs))
    from export_visibility_runtime_audit import main as export_runtime
    export_runtime()
    (OUT/'evidence_manifest.json').write_text(json.dumps(dict(
        no_new_experiments=True,primary='visibility1',scene_order=list(metrics),
        source_sha256=INPUTS,means_pq100={k:v[-1] for k,v in scores.items()},
        figures=exported,table_sha256={p.name:sha(p) for p in (OUT/'tables').glob('*.tex')}
    ),indent=2)+'\n')
    print(json.dumps({'tables':7,'figures':len(exported),'input_paths':len(INPUTS)}))

if __name__=='__main__':main()
