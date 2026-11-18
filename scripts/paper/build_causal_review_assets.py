"""Generate reviewer-response tables from sealed audits; no inference."""
from pathlib import Path
import json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'docs/paper/visibility_revision/latex';B=ROOT/'build/benchmarks';D=P/'data/review_20261117';D.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def copy(src,name):shutil.copy2(src,D/name)
for folder,prefix,files in [('reviewer_causal_revision_20261117','causal',['protocol.json','maps_complete.json','summary.json','per_object.json','history_diagnostics.json','validation.json']),('reviewer_maskclustering_20261117','external',['protocol.json','maps_complete.json','summary.json','score_recovery.json']),('reviewer_extension_20261117','extension',['protocol.json','preflight.json'])]:
 for f in files:copy(B/folder/f,prefix+'_'+f)
copy(ROOT/'build/datasets/reviewer_extension_20261117/window_selection.json','extension_selection.json')
for n in ['reviewer_extension_selection_attempt_20261117.json','reviewer_extension_selection_attempt2_20261117.json']:
 copy(B/n,n)
for f in ['review_causal_revision.py','review_maskclustering_control.py','recover_review_maskclustering_score.py','audit_review_causal_revision.py','freeze_reviewer_extension.py']:
 shutil.copy2(ROOT/'scripts/benchmarks'/f,P/'tools'/f)
s=read(D/'causal_summary.json');h=read(D/'causal_history_diagnostics.json');p=read(D/'causal_maps_complete.json')
def table(name,heads,rows,fmt):
 text='\\begin{tabular}{'+fmt+'}\n\\toprule\n'+' & '.join(heads)+' \\\\\n\\midrule\n'
 for row in rows:text+=' & '.join(map(str,row))+' \\\\\n'
 text+='\\bottomrule\n\\end{tabular}\n';(P/'tables/current'/name).write_text(text)
methods=[('ours_unconditional_history','Unconditional old IDs + Ours new'),('ours_no_history','Preserved old IDs + Ours new'),('witness_no_history','Preserved old IDs + witness'),('witness_history','Authorized old IDs + witness'),('nearest_witness_history','Authorized old IDs + nearest'),('full_authorization',r'\textbf{Ours (fixed)}')]
rows=[]
for m,label in methods:
 vals=[100*s['results'][k][m]['primary']['metrics']['allAp'] for k in ['kitti360_0009','kitti360_0010']];rows.append([label,*[f'{v:.3f}' for v in vals],f'{sum(vals)/2:.3f}'])
table('causal_controls.tex',['Export intervention','0009','0010','Mean'],rows,'@{}lrrr@{}')
rows=[]
for key,label in [('historical_edits_at_full_support_pp','Apply authorized old edits'),('history_protection_at_full_support_pp','Protect vs. unconditional'),('local_policy_vs_witness_pp','Local policy vs. witness'),('certificate_vs_nearest_pp','New certificate vs. nearest')]:
 vals=[s['contrasts'][k][key] for k in ['kitti360_0009','kitti360_0010']];rows.append([label,*[f'{v:+.3f}' for v in vals],f'{sum(vals)/2:+.3f}'])
table('causal_contrasts.tex',['AP difference (pp)','0009','0010','Mean'],rows,'@{}lrrr@{}')
rows=[]
for k in ['kitti360_0009','kitti360_0010']:
 for m,label in [('unconditional','Unconditional'),('full_authorization',r'\textbf{Ours}')]:
  z=h['results'][k][m];rows.append([k[-4:],label,f"{z['changed_source_points']:,}",*[f"{z['categories'][c]['confidence_mass']:.1f}" for c in ['correction','damage','same_object_consolidation','still_wrong','unknown_correspondence']]])
table('history_correctness.tex',['Window','Method','Old edits','Correction','Damage','Consolidation','Unresolved','Unknown'],rows,'@{}llrrrrrr@{}')
rows=[]
for k,z in p['results'].items():
 a=z['profile'];rows.append([k[-4:],f"{a['graph_nodes']:,}",f"{a['graph_directed_edges']:,}",f"{a['seed_rows']:,}",f"{1000*a['graph_certificate_compute_s']:.2f}",f"{1000*a['pair_evidence_compute_s']:.3f}",f"{a['cached_original_raw_evidence_prepare_s']:.3f}"])
table('runtime_review.tex',['Window','Nodes','Edges','Seeds','Graph (ms)','Pairs (ms)','Prior prep. (s)'],rows,'@{}lrrrrrr@{}')
x=read(D/'external_summary.json');rows=[]
for k,z in x['results'].items():rows.append([k[-4:],*[f"{100*z[m]['metrics']['allAp']:.4f}" for m in ['primary','raw_targets','uniform_scores']]])
table('external_core.tex',['Window','Primary AP','Canonical AP','Uniform-score AP'],rows,'@{}lrrr@{}')
shutil.copy2(__file__,P/'tools/build_causal_review_assets.py')
print('Generated 5 evidence tables and bundled frozen audit inputs')
