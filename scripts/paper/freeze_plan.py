"""Create the one-time paper-v1 source/config/evidence manifest (no experiments)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'docs/paper/frozen_v1'
OUT=ROOT/'build/benchmarks/evidence_retention_20261111'
SCENES=['036','038','041','045','047','049']


def read(p):return json.loads(Path(p).read_text())
def dump(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def command(args):return subprocess.check_output(args,cwd=ROOT,text=True).strip()


def prepare():
    assert not (DEST/'manifest.json').exists(),'A freeze already exists; do not overwrite it.'
    config=DEST/'config';config.mkdir(exist_ok=True)
    stage=read(OUT/'stage3_selection.json')
    cfg=Path(stage['cg_config']);assert sha(cfg)==stage['cg_config_sha256']
    shutil.copyfile(cfg,config/'cg_rgbd.json')
    livo=read(OUT/'scale_ablation_v1/cache_supportcc/both/run.json')
    dump(config/'livo_profile.json',{'effective_config':livo['effective_config'],
        'overlap_radius_m':livo['overlap_radius_m'],'support':'cache_supportcc',
        'note':'Sparse LiDAR only; original overlap monkeypatch changes query radius, not downsample size.'})
    job=read(OUT/'category_holdout_v1/jobs/025.json')
    dump(config/'frontend_scenenn.json',job['config'])
    ref=Path(job['reference_cache'])
    for name in ['vocabulary.json','text_features.npy']:
        assert sha(ref/name)==job['input_sha256'][str(ref/name)]
        shutil.copyfile(ref/name,config/name)
    parent=read(OUT/'category_holdout_v1/selection.json')
    dump(config/'nyu_prediction_mapping.json',parent['nyu_prediction_mapping'])
    entries=read(OUT/'official_annotation_audit/cvpr18_zip_contents.json')
    members=[e for e in entries if any(e['name'].endswith('_'+s+'.hdf5') for s in SCENES)]
    assert len(members)==12
    dump(config/'final_test_annotation_members.json',{'scenes':SCENES,'members':members,
        'source':str(OUT/'official_annotation_audit/cvpr18_zip_contents.json'),
        'source_sha256':sha(OUT/'official_annotation_audit/cvpr18_zip_contents.json'),
        'semantic_values_read':False,'scores_read':False})
    model=read(ROOT/'build/benchmarks/ovo_full_20261110/model_artifacts.json')
    for name,entry in model['files'].items():
        p=Path(name);assert p.stat().st_size==entry['bytes']
        assert sha(p)==entry['sha256'],name
    model['verified_again_for_paper_v1']=True
    dump(config/'models.json',model)
    for name in ['runtime_manifest.json','environment_packages.txt']:
        shutil.copyfile(ROOT/'build/benchmarks/ovo_full_20261110'/name,config/name)
    venv=ROOT/'build/benchmarks/ovo_full_20261110/venv/bin/python'
    (config/'current_environment_packages.txt').write_text(command([str(venv),'-m','pip','freeze'])+'\n')
    hw=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],
                      cwd=ROOT,text=True,capture_output=True)
    dump(config/'repository_state.json',{
        'head':command(['git','rev-parse','HEAD']),
        'status_porcelain':command(['git','status','--porcelain']),
        'note':'Dirty/untracked code is identified by content hashes and included in the source archive; no commit created.',
        'cg_head':command(['git','-C','build/benchmarks/expanded_semantic_20261109/concept-graphs','rev-parse','HEAD']),
        'ovo_head':command(['git','-C','build/benchmarks/ovo_full_20261110/upstream','rev-parse','HEAD']),
        'hardware_query':{'returncode':hw.returncode,'stdout':hw.stdout.strip(),'stderr':hw.stderr.strip()},
        'hardware_from_completed_run':read(OUT/'category_holdout_v1/frontend/025/features_run.json')['device'],
        'hardware_note':'Live NVML may be unavailable in this sandbox; cached run reports RTX 5090. No GPU availability claim.'})
    variants=[
        ('B0','cg','mean','strong reference'),('B1','ovo_3views','mean','external shared frontend baseline'),
        ('B2','history2','mean','association control'),('B3','history2','area','strong category baseline'),
        ('M','history2','consensus_sqrt_area','only main version'),
        ('A1','history1','consensus_sqrt_area','repeat support ablation'),
        ('A2','history2_no_veto','consensus_sqrt_area','component veto ablation'),
        ('A3','history2','sqrt_area','consensus ablation')]
    dump(DEST/'protocol.json',{
        'version':'omni_slam-paper-v1','date':'2026-11-11','status':'main implementation and evaluation design frozen; final execution pending',
        'primary_venue':'ICRA (CCF B)','target_regular_cycle':'2028; deadline not verified',
        'main_variant':'M','no_new_training':True,'no_dino':True,'no_sam3':True,
        'experiments_started_by_this_freeze':False,'final_test_scenes':SCENES,
        'historical_development':{'Replica':['office_0','office_1','office_2','office_3','office_4','room_0','room_1','room_2'],
            'SceneNN':['005','011','014','015','016','021']},
        'previously_frozen_now_viewed':['025','032'],'map_only_missing_annotation':['030'],
        'sampling':{'geometry_frames':400,'semantic_frames':200,'geometry_index':'floor(j*(N-1)/399)',
            'semantic_selection':'even j','insufficient_frames':'report ineligible; do not adapt'},
        'variants':[dict(id=i,geometry=g,category=c,purpose=p) for i,g,c,p in variants],
        'geometry_runs_per_scene':5,'max_mapping_runs':30,'method_scene_rows':48,'max_timing_repeats':3,
        'shared_frontend':{'sam':'SAM2.1 Hiera L','descriptor':'SigLIP384 learned','dimensions':1152,
            'seed':0,'feature_microbatch':8,'support_precision':'BF16 native projection; verify cached implementation'},
        'main_graph':{'min_distinct_query_frames':2,'component_coframe_veto':True,'dino':False,
            'edge_order':'native distinct-frame count descending; endpoint IDs ascending','offline':True,
            'native_final_merge_also_vetoed':False},
        'category':{'views':'largest mask per frame; ties minimum global observation ID',
            'centrality':'median pair cosine, diagonal excluded','keep':'centrality >= median centrality, all ties',
            'weights':'sqrt(raw mask area)','fallback':'<3 views or zero consensus norm: sqrt_area',
            'foreground':'freeze own geometry branch original unique-frame mean retention',
            'nonfinite_features':'reject','vocabulary':'config/vocabulary.json'},
        'metrics':{'instance':['observed_voxel_PQ','SQ','RQ','TP','FP','FN','precision','recall'],
            'semantic':['pooled_class_aware_PQ','matched_class_accuracy','corrections','regressions',
                        'observed_domain_per_class_PQ_SQ_RQ','observed_domain_macro_PQ','foreground_mIoU'],
            'mechanism':['edge_GT_precision_and_coverage','mixed_object_rate','GT_fragment_count',
                         'coframe_rejections','post_native_merge_conflicts','effective_views'],
            'cost':['frontend_s','causal_mapping_s','offline_finalization_s','category_s','CPU_RSS_peak',
                    'GPU_peak','cache_disk_bytes','frame_median_P95','cold_start'],
            'domain':['observed_voxels','valid_fraction','category_valid_fraction','unknown_predictions'],
            'projection_distance_m':0.05,'voxel_size_m':0.05,'match_iou_strictly_greater_than':0.5,
            'GT_class_purity_min':0.7,'GT_fragment_overlap_fraction_min':0.1,
            'official_leaderboard':False},
        'statistics':{'bootstrap_unit':'scene','paired_bootstrap_draws':10000,'seed':0,'interval':0.95,
            'primary_contrasts':['M-B0 instance PQ','M-B3 category PQ','M-A3 category PQ'],
            'effect_gate':'mean delta >0 and nonnegative in >=ceil(2n/3) scored scenes',
            'minimum_scored_scenes_for_generalization_claim':4},
        'pending_bounded_implementation':['A2 no-veto switch','OVO fixed-test adapter','macro metrics and mechanism exports'],
        'stop':'one fixed round; retain all results; narrow claims if controls fail; no adaptive branches',
        'missing_data':'no replacement; report failure; do not change sampling',
        'references':['evaluation_protocol.md','method.md','evidence.md']})
    print('Prepared configs; verified all referenced model weight hashes. No experiments run.')


def seal(timestamp):
    assert not (DEST/'manifest.json').exists(),'Do not overwrite frozen version.'
    files=set()
    # Preserve source closure broadly; enabled experimental methods are ONLY protocol variants.
    dirs=['omni_semantic_map','omni_slam_interfaces','omni_slam_manager','FAST_LIVO2',
          'scripts/benchmarks','scripts/paper',
          'build/benchmarks/expanded_semantic_20261109/concept-graphs/conceptgraph',
          'build/benchmarks/ovo_full_20261110/upstream/ovo',
          'build/benchmarks/shared_frontend_20261110/tools']
    extensions={'.py','.cpp','.cc','.c','.h','.hpp','.yaml','.yml','.xml','.msg','.srv','.cfg','.toml'}
    for name in dirs:
        base=ROOT/name
        if not base.exists():continue
        for p in base.rglob('*'):
            if not p.is_file() or any(x in p.parts for x in ['.git','__pycache__','node_modules','venv']):continue
            if p.suffix in extensions or p.name in ['CMakeLists.txt','LICENSE','LICENSE.txt','setup.cfg','requirements.txt']:
                files.add(p)
    evidences=[
        OUT/'public_history_v1/protocol.json',OUT/'public_history_v1/summary.json',
        OUT/'public_category_evidence_v2/protocol.json',OUT/'public_category_evidence_v2/summary.json',
        OUT/'category_holdout_v1/selection.json',OUT/'category_holdout_v1/annotation_amendment.json',
        OUT/'category_holdout_v1/summary.json',OUT/'category_holdout_v1/completion_audit/audit.json',
        OUT/'category_holdout_v1/predictions_complete.json',OUT/'category_holdout_v1/queue_status.json',
        OUT/'consensus_category_candidate_v1/manifest.json',OUT/'scale_ablation_v1/protocol.json',
        ROOT/'build/benchmarks/shared_frontend_20261110/protocol_backends.json',
        ROOT/'build/benchmarks/shared_frontend_20261110/protocol_cache.json',
        ROOT/'build/benchmarks/shared_frontend_20261110/summary.json']
    for p in evidences:
        assert p.exists(),p
        files.add(p)
    files.add(ROOT/'README.md')
    for base in ['build/benchmarks/expanded_semantic_20261109/concept-graphs',
                 'build/benchmarks/ovo_full_20261110/upstream']:
        for name in ['LICENSE','LICENSE.txt','requirements.txt','environment.yml','ReadMe.md','README.md']:
            p=ROOT/base/name
            if p.is_file():files.add(p)
    # All delivered documents/configs are fixed by the same manifest.
    for p in DEST.rglob('*'):
        if p.is_file() and p.name not in ['manifest.json','source_snapshot.tar.gz']:files.add(p)
    pdf=ROOT/'output/pdf/omni_slam_paper_plan_v1.pdf';assert pdf.exists();files.add(pdf)
    records=[]
    for p in sorted(files):
        archived=p.suffix not in ['.png','.svg','.pdf']
        records.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,
                        'sha256':sha(p),'in_source_snapshot':archived})
    archive=DEST/'source_snapshot.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for row in records:
            if row['in_source_snapshot']:tf.add(ROOT/row['path'],arcname=row['path'],recursive=False)
    dump(DEST/'manifest.json',{'version':'omni_slam-paper-v1','sealed_at_utc':timestamp,
        'git_head':command(['git','rev-parse','HEAD']),'dirty_tree_expected':True,
        'scope':'Current main implementation, configuration, historical evidence, bounded future protocol. Not a standalone container.',
        'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'files':records,
        'external_assets':'config/models.json records model hashes reverified during prepare; dataset/cache references are in historical protocols',
        'main_entrypoints':['scripts/benchmarks/run_public_history_graph.py',
            'scripts/benchmarks/category_holdout_finish.py',
            'omni_semantic_map/omni_semantic_map/consensus_category.py'],
        'supplemental_sources':'Archive includes dependency and historical code for reproducibility; inclusion does not enable extra method variants.',
        'final_test_executed':False})
    print(f'Sealed {len(records)} files; archive {archive.stat().st_size/1024/1024:.2f} MiB.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','seal']);p.add_argument('--timestamp')
    args=p.parse_args()
    if args.action=='prepare':prepare()
    else:
        assert args.timestamp,'Record trusted clock UTC at final sealing.'
        seal(args.timestamp)
