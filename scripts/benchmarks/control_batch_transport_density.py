"""Posthoc necessary density-only control; does not modify frozen candidate/gate."""
import argparse,shutil,time
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_batch_stable_transport import OUT as PARENT,FULL,SEM,BASE,verify,seal
OUT=ROOT/'build/benchmarks/batch_transport_density_control_v1'


def freeze():
    p=verify();seal('prepared');seal('predictions_complete');s=read(PARENT/'summary.json');assert not OUT.exists()
    inputs={str(PARENT/n):sha(PARENT/n) for n in ['protocol.json','prepared.json','predictions_complete.json','summary.json']}
    for scene in p['scenes']:
        key=scene['key']
        for path in [PARENT/'fields'/key/'full/field.npz',FULL/key/'geometry.npz',FULL/key/'old_to_full.npy',SEM/'aggregate'/key/'source.npz',BASE/key/'geometry_graph/geometry.npz',BASE/key/'geometry_graph/classification.json']:inputs[str(path)]=sha(path)
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),source_sha256=sha(__file__),input_hashes=inputs,scenes=p['scenes'],
        decision_history='Added AFTER all10development scores. Original candidate gate remainsfalse:0newAP50matches. Ours gains4.878889points vsoldbaseline but only0.229560vsstrongest sameinputcontrol. Many added points havezero graphdistance; test whether simple within-cell measurement lifting explains the improvement. This is a posthoc diagnostic control, not prespecified confirmatory evidence or a new candidate.',
        rule='Keep all original map coordinates/IDs/classscores. Append previously unowned fullrawsources only when their20cmcell contains a unique foreground seed owner with no competing owner/background seed. Equivalent nearest_distance0 and unique nearest_owner>0. No graph traversal, newmodel, tuning, newID, historicalrevision or postprocessing.',
        budget='Exactly2extra diagnosticmaps on same exposed windows, one fixed rule. Never choose or tune a distance cutoff. Bothmapssealedbefore scores; donotchange prior10maps,protocol,gate or bestmethod naming.',
        scope='Common-input density ablation. No independent or robot inference. If it matches/exceedsOurs, the claimed benefit of cross-celltransport must be narrowed accordingly.'))


def checked():
    p=read(OUT/'protocol.json');assert sha(__file__)==p['source_sha256'];verify()
    for f,h in p['input_hashes'].items():assert sha(f)==h,f
    return p


def maps():
    p=checked();assert not (OUT/'predictions_complete.json').exists();files={};stats={}
    for scene in p['scenes']:
        key=scene['key'];field=np.load(PARENT/'fields'/key/'full/field.npz');full=np.load(FULL/key/'geometry.npz');mapping=np.load(FULL/key/'old_to_full.npy');sem=np.load(SEM/'aggregate'/key/'source.npz');base=np.load(BASE/key/'geometry_graph/geometry.npz')
        owners=np.zeros(len(full['xyz']),np.int64);owners[mapping]=sem['owners'];cell=field['source_to_cell'];winner=field['nearest_owner'];zero=field['label_distance'][:,0]==0
        valid=zero[cell]&(winner[cell]>0)&(owners==0);added=np.flatnonzero(valid);pred=winner[cell[added]]
        # Independently reproduce same-cell decision from seed memberships.
        seed=field['seeds'];minimum=np.full(len(winner),np.iinfo(np.int64).max,np.int64);maximum=np.full(len(winner),-1,np.int64)
        np.minimum.at(minimum,seed[:,0],seed[:,1]);np.maximum.at(maximum,seed[:,0],seed[:,1])
        direct=np.where((minimum==maximum)&(minimum>0),minimum,-1)
        np.testing.assert_array_equal(winner[zero&(winner>0)],direct[zero&(winner>0)])
        np.testing.assert_array_equal(np.flatnonzero((direct[cell]>0)&(owners==0)),added)
        folder=OUT/'maps'/key;folder.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(folder/'geometry.npz',xyz=np.r_[base['xyz'],full['xyz'][added]],instance=np.r_[base['instance'],pred],added_full_source_ids=added)
        shutil.copyfile(BASE/key/'geometry_graph/classification.json',folder/'classification.json')
        stats[key]=dict(added_sources=len(added),old_map_unchanged=True,direct_seed_membership_replayed=True,graph_traversal=False)
        dump(folder/'run.json',stats[key])
        for f in folder.iterdir():files[str(f)]=sha(f)
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,stats=stats,protocol_sha256=sha(OUT/'protocol.json')))


def score():
    from evaluate_kitti360_development import transfer,clean_json
    from omni_semantic_map.kitti360_evaluation import evaluate
    from run_discovery_query_benchmark import matched_ids
    p=checked();seal=read(OUT/'predictions_complete.json');assert not (OUT/'summary.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    parent=read(PARENT/'summary.json');results={}
    for scene in p['scenes']:
        key=scene['key'];folder=OUT/'maps'/key;g=np.load(folder/'geometry.npz');ref=np.load(scene['reference']);objects=read(folder/'classification.json')['objects']
        pred=transfer(g['xyz'],g['instance'],ref['xyz'],.2);r=evaluate(ref['train_instance'],ref['confidence'],pred,{o['object_id']:o['predictions']['mean'] for o in objects},ROOT/'build/datasets/kitti360_prepare/evalInstanceLevelSemanticLabeling.py')
        dump(folder/'matches.json',clean_json(r['matches']));np.savez_compressed(folder/'reference_prediction.npz',instance=pred)
        ids=set(matched_ids(r['matches']));old=set(parent['results'][key]['methods']['baseline']['matched_gt_ids'])
        results[key]=dict(metrics=clean_json(r['averages']),matched_gt_ids=sorted(ids),gained=sorted(ids-old),lost=sorted(old-ids));print('DENSITY CONTROL',key,results[key],flush=True)
    mean=float(np.mean([r['metrics']['allAp'] for r in results.values()]))
    dump(OUT/'summary.json',dict(complete=True,results=results,mean_primary_ap=mean,posthoc_development_diagnostic=True,
        ours_mean_minus_density=parent['mean_primary_ap']['ours_frame']-mean,original_gate_changed=False,protocol_sha256=sha(OUT/'protocol.json')))
    print('DENSITY MEAN',mean,'oursdelta',parent['mean_primary_ap']['ours_frame']-mean,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','maps','score']);globals()[a.parse_args().action]()
