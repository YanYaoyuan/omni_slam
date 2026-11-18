"""Two missing controlled interventions separating history from new ownership.

This is a bounded mechanism audit, not a new tuned algorithm. Existing
development failures remain failures; all new maps are sealed before scoring.
"""
import argparse
import time
from pathlib import Path
import runpy
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from run_revision_authorization import OUT as PUBLIC, PARENT, verify as verify_public
from run_revision_robot import OUT as ROBOT, verify as verify_robot

OUT = ROOT/'build/benchmarks/revision_factorial_v1'
NEW = ['authorized_history_only', 'authorized_history_write']


def contexts():
    result=[]
    for c in read(PUBLIC/'protocol.json')['contexts']:
        result.append(dict(**c, parent=str(PARENT), authorization=str(PUBLIC)))
    for c in read(ROBOT/'protocol.json')['contexts']:
        result.append(dict(**c, parent=str(ROBOT/'cover'), authorization=str(ROBOT/'authorization')))
    return result


def freeze():
    verify_public(); verify_robot(); assert not OUT.exists()
    assert read(ROBOT/'completion_record.json')['complete']
    cs=contexts();files=[PUBLIC/'protocol.json',PUBLIC/'summary.json',PUBLIC/'delivery_audit.json',
        PUBLIC/'per_object.json',ROBOT/'protocol.json',ROBOT/'completion_record.json',
        ROBOT/'delivery_audit.json',ROOT/'build/benchmarks/revision_loss_sources_v1/summary.json']
    for c in cs:
        k=c['key'];parent=Path(c['parent']);auth=Path(c['authorization'])
        files += [parent/'maps_complete.json',auth/'maps_complete.json',
                  auth/'prepared'/k/'write_decisions.npz',Path(c['ours'])/'geometry.npz',
                  Path(c['ours'])/'classification.json',Path(c['full'])/'geometry.npz']
        for directory in [parent/'maps'/k/'cover_history',parent/'maps'/k/'cover_current',
                          auth/'maps'/k/'ours',auth/'maps'/k/'write_only',auth/'maps'/k/'identity_only']:
            files += [directory/'geometry.npz',directory/'classification.json']
        if c['kind']=='outdoor':
            files += [Path(c['reference']),Path(c['raw_reference'])]
            for m in ['baseline','unconditional','identity_only','write_only','current_only','ours']:
                files.append(PUBLIC/'evaluation'/k/m/'reference_prediction.npz')
    src=[Path(__file__),ROOT/'scripts/benchmarks/evaluate_kitti360_development.py',
         ROOT/'omni_semantic_map/omni_semantic_map/kitti360_evaluation.py',
         ROOT/'omni_semantic_map/omni_semantic_map/kitti360_reference_identity.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),contexts=cs,
        inputs={str(f):sha(f) for f in files},sources={str(f):sha(f) for f in src},
        scope='Post-hoc finite factorial mechanism audit, motivated by already-exposed source attribution. Two exposed public windows and same330framegarage. NOT a new novel method, unused evaluation, or modification of failed public gate.',
        authorized_history_only='Parent cover_history exact XYZ/additions/new labels/metadata; replace ONLY old-point labels by already-frozen authorized-history labels from Ours. No new point ownership competition or witness filtering.',
        authorized_history_write='Existing write_only exact XYZ/additions/new labels/metadata; replace ONLY old-point labels with frozen authorized-history labels. Thus support is direct-component+cellwitness, without new local ownership/certificate.',
        contrasts='unconditional->authorized_history_only isolates old history authorization; authorized_history_only->identity_only adds local new-label competition/certificate. write_only->authorized_history_write isolates old history authorization at witnessed support; authorized_history_write->ours adds local new-label competition/certificate. New-label certificate also abstains, so do not call the latter a label-only contrast.',
        safety_of_interpretation='No choosing regions/objects/classes with GT, no fitting or thresholds. Exact controlled interventions may not be the optimal production transaction. Report all outcomes and per-object changes. New metadata scores and unused metadata entries stay unchanged.',
        budget='4newpublic map exports+2robot map exports;12newpublic scores(primary/raw/uniform). Reuse all original controls and validate reference readouts; no models/training/parameter search or unused scenes.',
        decision='Do not retroactively rescue original Ours. Report whether gains come from old history protection, new region reassignment or witness restriction. Neither this audit nor any small positive mean alone promotes the combination; same inherited mean+1pp, each-windowAP/AP50 nonregression, no raw match loss remain efficacy requirements for an independent assessment.'))
    print('FACTORIAL AUDIT FROZEN / 6 maps / 12 scores',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['inputs'],**p['sources']}.items():assert sha(f)==h,f
    return p


def maps():
    p=verify();assert not (OUT/'maps_complete.json').exists();results={};files={}
    for c in p['contexts']:
        k=c['key'];parent=Path(c['parent']);auth=Path(c['authorization'])
        base=dict(np.load(Path(c['ours'])/'geometry.npz'));nold=len(base['xyz'])
        frozen=dict(np.load(auth/'maps'/k/'ours/geometry.npz'))
        decisions=dict(np.load(auth/'prepared'/k/'write_decisions.npz'))
        prefix=frozen['instance'][:nold]
        np.testing.assert_array_equal(prefix,decisions['authorized_history'])
        results[k]={}
        for method,source in [('authorized_history_only',parent/'maps'/k/'cover_history'),
                              ('authorized_history_write',auth/'maps'/k/'write_only')]:
            data=dict(np.load(source/'geometry.npz'));original=data['instance'].copy()
            np.testing.assert_array_equal(data['xyz'][:nold],base['xyz'])
            data['instance']=np.r_[prefix,original[nold:]]
            dest=OUT/'maps'/k/method;dest.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(dest/'geometry.npz',**data)
            dump(dest/'classification.json',read(source/'classification.json'))
            stats=dict(points=len(data['xyz']),added_sources=len(data['added_full_source_ids']),
                       old_changes_vs_parent_control=int((original[:nold]!=prefix).sum()),
                       old_changes_vs_strong=int((base['instance']!=prefix).sum()),
                       source_control=str(source),new_geometry_labels_unchanged=True)
            results[k][method]=stats;dump(dest/'run.json',stats)
            for f in dest.iterdir():files[str(f)]=sha(f)
            print('FACTORIAL COMPLETE MAP',k,method,stats,flush=True)
    dump(OUT/'maps_complete.json',dict(complete=True,files=files,results=results,new_maps=6,
         new_public_maps=4,new_robot_maps=2,gt_read=False,protocol_sha256=sha(OUT/'protocol.json')))


def audit():
    p=verify();seal=read(OUT/'maps_complete.json');assert seal['complete']
    assert not (OUT/'prediction_audit.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    results={}
    for c in p['contexts']:
        k=c['key'];parent=Path(c['parent']);auth=Path(c['authorization'])
        base=dict(np.load(Path(c['ours'])/'geometry.npz'));n=len(base['xyz'])
        orig=dict(np.load(parent/'maps'/k/'cover_history/geometry.npz'))
        write=dict(np.load(auth/'maps'/k/'write_only/geometry.npz'))
        ours=dict(np.load(auth/'maps'/k/'ours/geometry.npz'))
        identity=dict(np.load(auth/'maps'/k/'identity_only/geometry.npz'))
        decisions=dict(np.load(auth/'prepared'/k/'write_decisions.npz'))
        source=orig['added_full_source_ids'];full=np.load(Path(c['full'])/'geometry.npz')['xyz']
        keep=decisions['direct']&decisions['witness']
        np.testing.assert_array_equal(write['added_full_source_ids'],source[keep])
        np.testing.assert_array_equal(write['instance'][n:],orig['instance'][n:][keep])
        bymethod={}
        for m,control,target in [(NEW[0],orig,identity),(NEW[1],write,ours)]:
            g=dict(np.load(OUT/'maps'/k/m/'geometry.npz'))
            np.testing.assert_array_equal(g['xyz'],control['xyz'])
            np.testing.assert_array_equal(g['added_full_source_ids'],control['added_full_source_ids'])
            np.testing.assert_array_equal(g['instance'][n:],control['instance'][n:])
            np.testing.assert_array_equal(g['instance'][:n],ours['instance'][:n])
            np.testing.assert_array_equal(g['xyz'][n:],full[g['added_full_source_ids']])
            assert read(OUT/'maps'/k/m/'classification.json')==read(parent/'maps'/k/'cover_history/classification.json')
            # Same old prefix, different new-region decision only.
            np.testing.assert_array_equal(g['instance'][:n],target['instance'][:n])
            local=np.zeros(len(full),np.int64);other=local.copy()
            local[g['added_full_source_ids']]=g['instance'][n:]
            other[target['added_full_source_ids']]=target['instance'][n:]
            bymethod[m]=dict(complete_export_replay=True,
                new_source_label_differences_vs_local_competition=int((local!=other).sum()),
                geometry_matches_shared_control=True,new_labels_match_shared_control=True,
                old_labels_match_authorized=True)
            if c['kind']=='robot' and m==NEW[0]:
                old=dict(np.load(parent/'maps'/k/'cover_current/geometry.npz'))
                for key in ['xyz','instance','added_full_source_ids']:np.testing.assert_array_equal(g[key],old[key])
                bymethod[m]['identical_to_existing_cover_current']=True
        results[k]=bymethod
    dump(OUT/'prediction_audit.json',dict(complete=True,results=results,new_map_replays=6,
         source_sha256=sha(__file__),maps_seal_sha256=sha(OUT/'maps_complete.json'),gt_read=False))
    print('FACTORIAL EXPORT AUDIT PASSED',results,flush=True)


def score():
    from evaluate_kitti360_development import transfer,read_static_ply,clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from omni_semantic_map.kitti360_reference_identity import encode_reference
    from run_discovery_query_benchmark import matched_ids
    p=verify();assert read(OUT/'prediction_audit.json')['complete']
    assert not (OUT/'summary.json').exists()
    seal=read(OUT/'maps_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    previous=read(PUBLIC/'summary.json');tax=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    evaluator=ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py';results={};count=0
    for c in p['contexts']:
        if c['kind']!='outdoor':continue
        k=c['key'];ref=dict(np.load(c['reference']));raw=read_static_ply(c['raw_reference'])
        truth,_=encode_reference(raw['semantic'],raw['instance'],tax,'target_only');rows={}
        for method in NEW:
            path=OUT/'maps'/k/method;g=dict(np.load(path/'geometry.npz'))
            pred={r['object_id']:r['predictions']['mean'] for r in read(path/'classification.json')['objects']}
            labels=transfer(g['xyz'],g['instance'],ref['xyz'],.2);dest=OUT/'evaluation'/k/method
            dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'reference_prediction.npz',instance=labels);row={}
            for mode,target,uniform in [('primary',ref['train_instance'],False),('raw_targets',truth,False),('uniform_scores',ref['train_instance'],True)]:
                info={o:dict(v,score=1.) for o,v in pred.items()} if uniform else pred
                r=evaluate(target,ref['confidence'],labels,info,evaluator)
                dump(dest/(mode+'_matches.json'),clean_json(r['matches']))
                matches=set(matched_ids(r['matches']));old=set(previous['results'][k]['baseline'][mode]['matched_gt_ids'])
                row[mode]=dict(metrics=clean_json(r['averages']),matched_gt_ids=sorted(matches),
                               changes_vs_baseline=dict(gained=sorted(matches-old),lost=sorted(old-matches)))
                count+=1
            rows[method]=row
            print('FACTORIAL SCORE',k,method,{m:100*r['metrics']['allAp'] for m,r in row.items()},flush=True)
        results[k]=rows
    mean={mode:{m:float(np.mean([r[m][mode]['metrics']['allAp'] for r in results.values()])) for m in NEW}
          for mode in ['primary','raw_targets','uniform_scores']}
    checks={}
    for m in NEW:
        changes=[r[m]['raw_targets']['changes_vs_baseline'] for r in results.values()]
        checks[m]=dict(each_window_ap_ap50_nonregression=all(r[m]['primary']['metrics'][metric]>=previous['results'][k]['baseline']['primary']['metrics'][metric] for k,r in results.items() for metric in ['allAp','allAp50%']),
            mean_at_least_one_pp_above_strong=mean['primary'][m]>=previous['mean_ap']['primary']['baseline']+.01,
            raw_gained=sum(len(r['gained']) for r in changes),raw_lost=sum(len(r['lost']) for r in changes))
    assert count==12
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_ap=mean,efficacy_requirements=checks,
         original_development_gate=False,original_gate_preserved=True,candidate_promoted=False,new_scores=count,new_public_maps=4,new_robot_maps=2,
         new_independent_scenes=0,new_model_calls=0,new_training=0,goal_complete=False,
         protocol_sha256=sha(OUT/'protocol.json'),maps_seal_sha256=sha(OUT/'maps_complete.json')))
    print('FACTORIAL AUDIT SCORED',mean,checks,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','maps','audit','score'])
    globals()[parser.parse_args().action]()
