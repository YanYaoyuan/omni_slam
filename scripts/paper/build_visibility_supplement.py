"""Existing-result supplement: development split, mechanism provenance, environment."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/paper/visibility_revision'
DEV=ROOT/'build/benchmarks/visibility_cleanup_20261112'
TEST=ROOT/'build/benchmarks/visibility_independent_20261112'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def build():
    inputs={}
    def record(p):
        inputs[str(p.relative_to(ROOT))]=sha(p);return read(p)
    summary=record(DEV/'summary.json');protocol=record(DEV/'protocol.json')
    assert summary['protocol_sha256']==sha(DEV/'protocol.json')
    assert summary['prediction_gate_sha256']==sha(DEV/'predictions_complete.json')
    record(DEV/'predictions_complete.json')
    methods=['native_bridge_2','geometric_sor','multiview','no_visibility','single_refutation']
    rows=[]
    for scene,digest in summary['scene_results'].items():
        p=DEV/'evaluation'/scene/'metrics.json';assert sha(p)==digest,scene
        m=record(p)['methods'];vals=[m[n]['score']['observed_voxel_instance_pq']*100 for n in methods]
        rows.append([scene.replace('replica_','').replace('scenenn_','SN '),*[f'{x:.3f}' for x in vals]])
    test_summary=record(TEST/'summary.json');record(TEST/'protocol.json')
    grouped=[]
    for name,group in summary['datasets'].items():
        b=group['native_bridge_2']['pq100'];o=group['single_refutation']['pq100']
        grouped.append([name+' (development)',2,b,o,o-b,group['geometric_sor']['pq100']])
    ms=[record(TEST/'evaluation'/s/'metrics.json')['methods'] for s in ['052','054','057','061']]
    means=lambda n:float(np.mean([m[n]['score']['observed_voxel_instance_pq']*100 for m in ms]))
    grouped.append(['SceneNN (new sequences)',4,means('history2'),means('visibility1'),means('visibility1')-means('history2'),means('sor')])
    group_rows=[[r[0],r[1],*[f'{v:.3f}' for v in r[2:]]] for r in grouped]
    # A diagnostic identity, not a benchmark run or a new filtering variant.
    sys.path.insert(0,str(ROOT/'omni_semantic_map'))
    from omni_semantic_map.visibility_cleanup import rejection_mask,CleanupConfig
    pairs=np.array([(p,n) for p in range(201) for n in range(201-p)],dtype=np.int64)
    actual=rejection_mask(pairs[:,0],pairs[:,1],CleanupConfig(),single_frame=True)
    ratio=(pairs.sum(1)>0)&(3*pairs[:,0]<=pairs.sum(1))
    assert np.array_equal(actual,ratio)
    upstream=TEST/'upstream/MaskClustering';commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    assert commit==read(TEST/'protocol.json')['external']['commit']
    src=upstream/'utils/post_process.py';original=src.read_bytes()
    assert original==subprocess.check_output(['git','-C',str(upstream),'show',commit+':utils/post_process.py'])
    inputs[str(src.relative_to(ROOT))]=sha(src)
    original_config=record(TEST/'adapters/MaskClusteringCPU/configs/scannet.json')
    code_link=f'https://github.com/PKU-EPIC/MaskClustering/blob/{commit}/utils/post_process.py'
    mechanism=dict(complete=True,upstream_commit=commit,upstream_source_sha256=sha(src),upstream_source_url=code_link,
        original_mc_point_filter_threshold=original_config['point_filter_threshold'],
        ours_equivalence='For integer votes and P+N>0, reject iff P/(P+N)<=1/3; P=N=0 abstains.',
        equivalent_pairs_checked=len(pairs),vote_total_maximum=200,
        mc_denominator='Point presence in frames restricted by node.visible_frame; comment says whole video, implementation applies this restriction.',
        code_locations={'node_visible_frame_selection':84,'ratio_and_strict_threshold':94,'node_mask_count_guard':96},
        causal_claim='No isolated MC-filter-on-history2 comparison exists; full adapted-system gap cannot identify the filter alone.',
        no_new_maps_or_scored_variants=True)
    (OUT/'mechanism_audit.json').write_text(json.dumps(mechanism,indent=2)+'\n')
    distributions=['numpy','scipy','matplotlib','Pillow','reportlab','torch','open3d','opencv-python','opencv-python-headless','h5py']
    versions={}
    for name in distributions:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    evalpython=ROOT/'build/benchmarks/ovo_full_20261110/venv311/bin/python'
    reader=ROOT/'build/benchmarks/evidence_retention_20261111/tools/hdf5_reader'
    evaluation=json.loads(subprocess.check_output([str(evalpython),'-c',
        'import sys,json,importlib.metadata as m;sys.path.insert(0,sys.argv[1]);import h5py;print(json.dumps(dict(python=sys.version,hdf5_reader=h5py.__file__,packages={n:m.version(n) for n in ["numpy","scipy","h5py"]})))',str(reader)],text=True))
    for p in [reader/'h5py/__init__.py',reader/'h5py-3.15.1.dist-info/METADATA']:
        inputs[str(p.relative_to(ROOT))]=sha(p)
    env=dict(paper_python=sys.version,paper_executable=str(Path(sys.executable).relative_to(ROOT)),platform=platform.platform(),
        package_versions=versions,evaluation_environment=evaluation,
        font_files=['/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'],
        system_site_packages=True,scope='Observed working environments, not a clean-room lockfile or portability guarantee.',
        compatibility_note='Python 3.10 environment exposes SciPy 1.8.0 with NumPy 1.26.4 outside its declared range. Existing numerical checks pass; not certified compatible.',
        gpu_required_for_document_build=False)
    (OUT/'environment_observed.json').write_text(json.dumps(env,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(9,3.2),layout='constrained')
    for ax,row in zip(axes,grouped):
        values=[row[5]-row[2],row[4]];ax.bar([0,1],values,color=['#bf9861','#159b84'])
        ax.set_xticks([0,1],['SOR','Ours']);ax.set_ylim(0,2.5)
        ax.set_title(row[0].replace(' (','\n('),fontsize=10)
        for x,v in enumerate(values):ax.text(x,v+.04,f'+{v:.3f}',ha='center',fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('PQ-point gain over strong baseline')
    fig.savefig(OUT/'figures/development_vs_unseen.png',dpi=190);fig.savefig(OUT/'figures/development_vs_unseen.svg');plt.close(fig)
    appendix='''
<!-- page -->

## Appendix A. Development Results Are Separate Evidence

The pilot used four previously viewed scenes. Its original primary method required three refutation frames; its full gate failed. The one-refutation arm was then selected before the new-sequence run and is the current Ours. The independent protocol records this selection. These pilot results document development, not a second independent test dataset.

'''+table(['Development scene','Baseline','SOR','3-frame','No depth (3)','Ours (1)'],rows)+'''

Table A1. Pilot PQ x 100. Ours (1) is the existing single_refutation arm; the original primary multiview arm is shown as 3-frame. The pilot no_visibility control also requires three refutations, whereas the new-sequence no-depth control requires one. These two no-depth controls must not be treated as an identical ablation.

'''+table(['Split','Scenes','Baseline','Ours','Gain','SOR'],group_rows)+'''

Table A2. Scene-macro means within each split. No pooled development-plus-test mean is reported. The pilot one-frame Ours gained one match and lost none in each dataset; the independent run gained and lost zero matches. Development matching gains do not carry over to independent test recall claims.

![Figure A1. Gains over the corresponding strong baseline, separated by dataset and exposure status. The first two panels are development; the last contains the four new SceneNN sequences.](figures/development_vs_unseen.png)

The original three-frame pilot failed its dataset-gain and persistence checks. The new-sequence primary failed the beats-SOR check. Both decisions remain unchanged. No additional predictions or parameter choices are introduced by this supplement.

<!-- page -->

## Appendix B. Relationship to Existing Point Filtering

With nonnegative integer counts and P+N>0, the frozen primary rule has an equivalent support-share form:

    N >= 1 and N >= 2P   <=>   P / (P + N) <= 1/3

The zero-evidence case P=N=0 is retained. This identity was checked against the actual rejection function for all 20,301 count pairs satisfying P+N<=200. This is a mathematical/code consistency check, not a new benchmark experiment. The ratio itself is not a distinct learned model or a new principle of evidence fusion.

The pinned MaskClustering source [9] already computes point detection ratios and cites OVIR-3D for that filtering step. Its numerator counts cluster support; its denominator counts point presence restricted to the node's visible frames. The implementation retains ratios strictly above 0.5 in the configuration used here. It also operates with cluster splitting and object-level retention, so this is not an isolated fixed-identity deletion module.

| Aspect | Ours | Pinned MaskClustering filter |
|---|---|---|
| Decision statistic | Support among decisive votes | Cluster detection / point presence |
| Evidence domain | Own-object observed frames; direct depth agreement | Node-visible frames; cached point support |
| Border / unknown handling | 2 px band and uncertain pixels abstain | Determined by mask/support construction |
| Point decision | Remove support share <= 1/3; retain no-evidence points | Retain detection ratio > configured 0.5 |
| Object identity | Fixed during cleanup | Cluster splitting/filtering/export pipeline |

Table B1. Code-level distinctions, not performance attribution. Different visibility definitions, denominators, masks and object memberships prevent treating the numerical thresholds as directly equivalent across the systems.

The appropriate hypothesis is that an explicit abstention domain and depth-consistent refutation improve fixed-identity geometry. Our depth ablation tests one part of this hypothesis. The adapted full-system comparison does not isolate the competing point filters. No claim that the Ours rule alone beats MaskClustering's filter is justified without a controlled same-baseline filter comparison. This missing attribution is documented rather than silently filled with a new experiment.

<!-- page -->

## Appendix C. Reproduction and Artifact Scope

The paper package distinguishes verification, document regeneration and rerunning mapping. Verification and document regeneration do not start mapping, frontend inference, training, downloads or GPU jobs. The one-command entry point defaults to verification:

    bash scripts/paper/reproduce_visibility_paper.sh verify
    bash scripts/paper/reproduce_visibility_paper.sh build

Verification checks the frozen experimental inputs, generated artifact hashes, primary method identity, arithmetic in the main result table, the development/test separation and local document links. It fails on missing or changed evidence. Rebuilding regenerates this supplement, figures, Markdown, HTML and PDF from existing inputs; it marks PDF visual review as pending until the new PDF is inspected.

### Environment

The working document environment is Python 3.10 with inherited system packages; the original HDF5 evaluation used a separate Python 3.11 environment and a locally injected h5py wheel under evidence_retention_20261111/tools/hdf5_reader. Importing h5py from the Python 3.11 environment alone is insufficient. Exact observed versions and the loaded wheel path are recorded in environment_observed.json. This is an environment inventory rather than a clean-room dependency lock. The existing SciPy/NumPy declared-version mismatch is recorded; successful result checks do not certify every operation in that environment.

The builder requires NumPy, Matplotlib, Pillow and ReportLab, the DejaVu fonts, and local frozen RGB/mask/map caches. Supplement verification also uses the existing cleanup module and its NumPy/SciPy/OpenCV dependencies. Poppler is used for PDF inspection. No model checkpoints are loaded by the document build.

### Data and portability

The repository keeps absolute paths in the historical experiment records. Verification resolves their hashed local evidence in this checkout; copying only the paper folder does not recreate the experimental environment. The verification tool reports missing evidence and never downloads substitutes. Rerunning all original mapping is a separate GPU/data operation and is not implied by the document-build command.

### Remaining submission evidence

Original external pipeline reproduction, a genuinely unexposed second dataset, class-aware semantic/query metrics and end-to-end robot measurements remain outside the completed evaluation. They are not filled with estimated results. The present package is a complete auditable research draft, while a submission claiming those broader properties requires additional evidence under a new bounded protocol.
'''
    (OUT/'supplement.md').write_text(appendix)
    manifest=dict(complete=True,development_gate_passed=summary['gate_passed'],independent_gate_passed=test_summary['primary_validation_passed'],
        rows_by_exposure=grouped,inputs_sha256=inputs,no_new_scored_variants=True,
        source_sha256=sha(__file__),generated_sha256={n:sha(OUT/n) for n in ['supplement.md','mechanism_audit.json','environment_observed.json','figures/development_vs_unseen.png','figures/development_vs_unseen.svg']})
    (OUT/'supplement_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    print(json.dumps(build()['rows_by_exposure'],indent=2))
