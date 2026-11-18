"""Prospective new-window quality assessment selected after development.

Does not change the failed discovery gate or execute/score a new scene.
Adapter code must receive a separate execution seal before new RGB inference.
"""
from pathlib import Path
import time
from paper_final_common import ROOT,read,dump,sha
from run_batch_stable_transport import OUT as DEV,verify,seal
from control_batch_transport_density import OUT as DENSITY

OUT=ROOT/'build/benchmarks/batch_transport_assessment_v1'
DATA=ROOT/'build/datasets/kitti360_reserved_v3'


def main():
    assert not OUT.exists();dev=verify();seal('prepared');seal('predictions_complete')
    s=read(DEV/'summary.json');density=read(DENSITY/'summary.json');audit=read(DEV/'artifact_audit.json')
    assert audit['complete'] and not s['advance_gate'] and s['gained']==s['lost']==0
    assert s['mean_primary_ap']['ours_frame']>max(v for k,v in s['mean_primary_ap'].items() if k!='ours_frame')
    assert s['mean_primary_ap']['ours_frame']>density['mean_primary_ap']
    data=read(DATA/'delivery_audit.json');assert data['complete'] and not data['reference_labels_decoded'] and not data['method_evaluated']
    selection=read(DATA/'window_selection.json');inputs={};scenes=[]
    for path in [DEV/'protocol.json',DEV/'summary.json',DEV/'artifact_audit.json',DENSITY/'protocol.json',DENSITY/'summary.json',DATA/'window_selection.json',DATA/'delivery_audit.json',DATA/'geometry_separation_audit.json']:
        inputs[str(path)]=sha(path)
    for path in (DATA/'data').glob('*_download.json'):
        download=read(path);assert download['complete'];inputs[str(path)]=sha(path)
        for row in download['files']:
            f=ROOT/row['path'];assert sha(f)==row['sha256'];inputs[str(f)]=row['sha256']
    for w in selection['windows']:
        key='kitti360_'+w['scene'].split('_')[-2]
        scenes.append(dict(key=key,kind='outdoor',window=w,job=str(OUT/'prepared'/key/'jobs'/(w['scene']+'.json')),
            opaque_reference=str(DATA/'data/opaque_references'/w['member'])))
        for f in (ROOT/'build/datasets/kitti360_prepare/extracted'/w['scene']).rglob('*.txt'):inputs[str(f)]=sha(f)
    for f in (ROOT/'build/datasets/kitti360_prepare/extracted/calibration').glob('*.txt'):inputs[str(f)]=sha(f)
    front=read(ROOT/'build/benchmarks/discovery_query_benchmark_v1/protocol.json')
    model=ROOT/'build/benchmarks/intrinsic_surface_ownership_v1/models/rosbag2245/intrinsic_geometry/model.pt'
    for f in [model,Path(front['sam3_checkpoint']),ROOT/'build/benchmarks/ovo_full_20261110/model_artifacts.json',ROOT/'build/models/segformer_b5_cityscapes/source_manifest.json']:
        inputs[str(f)]=sha(f)
    assert inputs[front['sam3_checkpoint']]==front['sam3_sha256']
    robot=ROOT/'build/benchmarks/robot_geometry_reference_v1';robot_input=ROOT/'build/benchmarks/discovery_measured_inputs_v1/rosbag2245/manifest.json'
    for f in [robot/'maps_complete.json',robot/'maps/geometry_graph_reference/geometry.npz',robot/'maps/geometry_graph_reference/classification.json',robot/'maps/geometry_graph_reference/groups.json',robot_input,Path(read(robot_input)['geometry'])]:
        inputs[str(f)]=sha(f)
    sources=dict(front['source_hashes'])
    for name in ['run_batch_stable_transport.py','control_batch_transport_density.py','geometry_reference_backend.py','run_robot_geometry_reference.py','run_intrinsic_public_holdout.py','run_surface_ownership.py','run_intrinsic_surface_ownership.py','prepare_full_lidar_surface_inputs.py','prepare_kitti360_reserved.py','capture_prethreshold_candidates.py','capture_closed_vocabulary_semantics.py']:
        f=ROOT/'scripts/benchmarks'/name;sources[str(f)]=sha(f)
    for name in ['batch_stable_transport.py','batch_stable_transport.cpp','intrinsic_surface_geometry.py','kitti360_input.py']:
        f=ROOT/'omni_semantic_map/omni_semantic_map'/name;sources[str(f)]=sha(f)
    sources[str(Path(__file__))]=sha(__file__)
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=scenes,robot_key='rosbag2245',methods=['baseline','same_cell','full_nearest','full_two_cells','ours_frame'],
        input_hashes=inputs,source_hashes=sources,selected_model=str(model),sam3_checkpoint=front['sam3_checkpoint'],sam3_sha256=front['sam3_sha256'],
        development_selection='POSTHOC hypothesis selection after all development results. Original recovery gate failed because0newAP50matches and remainsfalse. Ours mean28.011884 vsbaseline23.133995,nearest27.131881,two_cells27.782324,samecell24.790812. Select the exact unchanged frame-certificate for a NEW independent geometry-quality test; do not claim prespecified developmental success or object discovery.',
        explicit_scope_change='Supersedes only the prior self-imposed expansion stop for this bounded prospective quality assessment; no previous gate/result is edited. New-object recovery remains unproven. Original task of a useful mechanism, robot comparison and new evaluation remains incomplete until the actual runs and analysis exist.',
        mechanism='Exact frozen graph_and_seeds,nearest_owner,certify implementation.20cmfirst-actual-sourcecells,symmetric16NNstrict0.4m,Euclideanpathcost,1e-10strictcomparison. Foregroundpredictedsourceanchors plus originalcamera-source SegFormerbackground. Second distinct acquisitionframe must beat every originalcompetingowner. No parameter,seed,class,threshold or finalprocessing change.',
        backbone='Same SAM3car/building native0.5 masks, frozen SigLIP+CGhistory2 frontend. Intrinsic_geometry deployment weights already trained only on0000+0002. Same raw-positive geometric reference expansion, cellIoU>0.5, seed/birth/score rules as oldgeometry_graph and robotgolden replay. No learned association fit or testlabels. Reuse cached robotbase, no substitution of a weaker baseline.',
        input='Public:full rawpairedLiDARsamecalibration/poses/5cmfirstreturns plus exact camera-source preservation. Robot:existing archived LIVO fulltrackingcloudsource ledger and acquisitionframeprovenance; points areprocessedLiDAR,not claimed originalbeams. This method needs no ray origins/timestamps. No syntheticpoints, densifieddepth orGTselection.',
        external_semantics='Exact frozen Cityscapes SegFormerB5 FP32,1024nativeprocessor,all19classes,mean actualvisible-sourceprobabilities, nofinetuning; used as sharedbackgroundevidence inallfourmodifiedconditions. No unknown-source-as-background shortcut.',
        reference_policy='Only metadata/hash audits beforefreeze. Bothnewwindows all5finalpredictions sealed before decoding either reference. Independent development selection but same dataset/domain; foundationpretrainingoverlapunknown. No looking at partial scores to modify candidate, replacewindows or tuneparameters.',
        evaluation='Same fullstaticreference,strict20cm nearestmap readout,confidenceweightedKITTIbuilding/car AP50:90/AP50/AP25; allGTincludinguncovered retained. Localadaptation,notofficialserver. Report eachwindow/class/method,fullbestIoUandallgained/lostmatches.',
        independent_gate='Ours meanprimaryAP>=baseline+.02 and >each ofsamecell,full_nearest,full_two_cells; perwindowprimaryAPandAP50>=baseline. These are new prospective geometry-quality criteria,not retroactive discoverycriteria. No new-instanceclaim from this gate. Report allmatchlosses evenifgatepasses; no forcedbestmethodswitch.',
        robot='Public evaluation first. Same frozenframecertificate andfourcontrols onall330existingrobotframes. No manualGTavailable: reportactualexecution,source/IDinvariants,objectgeometrychanges,RGBand3Dcomparisons,runtime. Do notreportsemanticAP/accuracyorS100realtime. No robot-specificparameterchoice.',
        budget='2newwindows89RGBpairs,10finalscoredmaps(plus2CGand2intrinsicbackboneintermediates); unchangedrobot330frames,5finalmapcomparisons ifpublicquality gatepasses.0newfits,0hyperparametersweeps,0replacementscenes. SharedSAM3highandrawcapture mayuse2encodesperpublicimage;sharedSigLIP/SegFormer onceperpublicimage. RobotrawSAM3andbackbonecached;only330sharedSegFormerforwardsneeded.',
        execution_gate='Before any newRGBinference, implementisolatedadapters,sealtheirsourcehashes,andcheckreusedbackboneagainstexistinggoldenoutputs. This filepre-registersmethod/data/selection/gate; itisnotanexecutionorcompletionmanifest.',
        status='registered_not_executed',goal_complete=False))
    print('REGISTERED QUALITY ASSESSMENT',OUT/'protocol.json',[(s['key'],len(s['window']['input_frames'])) for s in scenes],flush=True)


if __name__=='__main__':main()
