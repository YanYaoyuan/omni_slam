"""Return to full object maps: stage coverage, actual readout control, dense labels.

No new learned mapper, map reconstruction, threshold sweep or claimed oracle
algorithm. Frozen v1 maps remain untouched. GT only enters this OFFLINE audit.
"""
from collections import Counter
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree

from probe_object_memory_attention import ROOT,HISTORY,DINO,read,dump,sha,verify_hashes
from omni_semantic_map.map_support_audit import covered_by,object_coverage,token_groups,label_distribution
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels

OUT=ROOT/'build/benchmarks/full_object_mainline_20261112'
REPORT=ROOT/'docs/reports/full_object_mainline_2026-11-12'
LEARNED=ROOT/'build/benchmarks/learned_object_memory_20261112'
KEYS=['replica_office_0','replica_office_1','scenenn_014','scenenn_015']
STAGES=['raw_geometry','mask_support','accepted_observation_support','native_fused_geometry',
        'final_v1_all_geometry','final_v1_foreground']


def freeze():
    assert not OUT.exists()
    parent=read(LEARNED/'protocol.json');verify_hashes(parent['input_sha256']);verify_hashes(parent['reference_sha256'])
    pool_source=ROOT/'omni_semantic_map/omni_semantic_map/livo_dino_support.py'
    assert sha(pool_source)==read(DINO/'protocol.json')['source_sha256'][str(pool_source)]
    artifacts={}
    for key in KEYS:
        base=HISTORY/'runs'/key;run=read(base/'run.json')
        assert run['complete']
        for name in ['nodes.json','snapshot.npz']:
            assert sha(base/name)==run['artifacts'][name]
        for name in ['run.json','nodes.json','snapshot.npz','native_bridge_2/geometry.npz','native_bridge_2/classification.json']:
            artifacts[str(base/name)]=sha(base/name)
        for name in ['metrics.json','native_bridge_2.npz']:
            path=HISTORY/'evaluation'/key/name;artifacts[str(path)]=sha(path)
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/map_support_audit.py',
             ROOT/'omni_semantic_map/test/test_map_support_audit.py',
             ROOT/'omni_semantic_map/omni_semantic_map/livo_dino_support.py',
             ROOT/'omni_semantic_map/omni_semantic_map/full_ovo_evaluation.py',
             ROOT/'omni_semantic_map/omni_semantic_map/mesh_instance_evaluation.py',
             ROOT/'omni_semantic_map/omni_semantic_map/flat_benchmark.py',
             ROOT/'scripts/benchmarks/probe_object_memory_attention.py']
    p=dict(created_epoch=time.time(),scenes=parent['scenes'],splits=parent['splits'],map_scenes=KEYS,
        source_sha256={str(x):sha(x) for x in sources},input_sha256=parent['input_sha256'],
        reference_sha256=parent['reference_sha256'],map_artifact_sha256=artifacts,
        parent_training_protocol_sha256=sha(LEARNED/'protocol.json'),
        full_map_domain='Original valid observed 5cm reference, all GT instances, no >=70% purity filtering of evaluation objects.',
        stage_order=STAGES,
        coverage='For every GT instance, fraction of reference vertices within 5cm of stage geometry. More than 50% is necessary but NOT sufficient for IoU>0.5. A coverage ceiling counts geometrically reachable objects, NOT predicted true positives or attainable PQ.',
        interpretation='Different sampling/denoising can make adjacent stage coverage nonmonotonic. Record gained/lost voxels and eligibility both ways. First failed checkpoint is a diagnostic routing label, not isolated causal attribution.',
        control='Keep all final v1 regions regardless of predicted background class. Geometry, memberships and instance IDs unchanged. Predefined readout control; not a proposed learned method. Score whole map with FP/FN and retained/lost GT matches.',
        supervision='All original masks and selected local DINO tokens retain full support-label histograms (-1 unknown, 0 background, >0 instance). No dominant-purity filtering of export. GT used only as offline target; input features remain unchanged.',
        token_provenance='Reconstruct the exact camera projection/bin selection of the frozen pool function from given poses, intrinsics and cached DINO resize. Verify selected token count for every observation; store source point IDs and CSR label counts. Do not claim pixel-level segmentation GT.',
        limitations='Already examined data. RGB-D provided poses, not a fresh LIVO experiment. Sampled supports can be sparse, displaced or occluded. A token target is a distribution of projected measured supports, not complete pixel GT. Per-scene instance IDs are local.',
        stop='Four full-map audits and all eleven fixed-scene supervision exports only. No new inference, training, maps, dataset expansion or deployment.')
    dump(OUT/'protocol.json',p)
    for x in sources:
        dest=OUT/'sources'/x.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(x.read_bytes())
    print('FROZEN',sha(OUT/'protocol.json'),flush=True)


def protocol():
    p=read(OUT/'protocol.json');verify_hashes(p['source_sha256']);return p


def data(s):
    cache=Path(s['cache']);m=read(cache/'manifest.json');geo=np.load(cache/'geometry.npz')
    supports=[];frames=[]
    for r in m['records']:
        assert len(supports)==r['observation_offset']
        z=np.load(cache/'frames'/r['file'])
        for a,b in zip(z['offsets'][:-1],z['offsets'][1:]):supports.append(z['point_ids'][a:b].astype(np.int64))
        frames.extend([r['frame_id']]*r['masks'])
    return cache,m,geo,supports,np.array(frames)


def point_targets(s,xyz):
    ref=np.load(s['reference']);valid=ref['valid'] if 'valid' in ref.files else np.ones(len(ref['xyz']),bool)
    dist,idx=cKDTree(ref['xyz']).query(xyz,workers=1)
    labels=np.where((dist<=.05)&valid[idx],ref['instance'][idx],-1).astype(np.int64)
    return ref,valid,labels


def map_audit(s):
    key=s['key'];cache,m,geo,supports,frames=data(s);xyz=geo['xyz']
    ref,valid,point_labels=point_targets(s,xyz);truth=ref['instance'][valid];vertices=ref['xyz']
    base=HISTORY/'runs'/key;nodes=read(base/'nodes.json');snap=np.load(base/'snapshot.npz')
    final=np.load(base/'native_bridge_2/geometry.npz');rows=read(base/'native_bridge_2/classification.json')['objects']
    all_ids=np.unique(np.concatenate(supports));members=[i for n in nodes for i in n['members']]
    assert len(members)==len(set(members))
    accepted=np.unique(np.concatenate([supports[i] for i in members]))
    kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
    fg=np.isin(final['instance'],kept)
    clouds=[xyz,xyz[all_ids],xyz[accepted],snap['xyz'],final['xyz'],final['xyz'][fg]]
    masks={name:covered_by(points,vertices) for name,points in zip(STAGES,clouds)}
    stages={};bygt={}
    for name,points in zip(STAGES,clouds):
        records=object_coverage(truth,masks[name][valid]);bygt[name]={r['gt']:r for r in records}
        stages[name]=dict(points=len(points),gt_instances=len(records),
            foreground_voxel_coverage=float(masks[name][valid][truth>0].mean()),
            reachable_gt_over_half=sum(r['can_exceed_iou_half'] for r in records))
    old=read(HISTORY/'evaluation'/key/'metrics.json')['methods']['native_bridge_2']
    pred,_=transfer_labels(final['xyz'],np.where(fg,final['instance'],0),vertices)
    np.testing.assert_array_equal(pred,np.load(HISTORY/'evaluation'/key/'native_bridge_2.npz')['instance'])
    score=voxel_metrics(truth,pred[valid])
    for metric in ['observed_voxel_instance_pq','tp','fp','fn','sq','rq']:assert score[metric]==old[metric]
    all_pred,_=transfer_labels(final['xyz'],final['instance'],vertices)
    all_score=voxel_metrics(truth,all_pred[valid])
    np.testing.assert_array_equal(pred>0,masks['final_v1_foreground'])
    np.testing.assert_array_equal(all_pred>0,masks['final_v1_all_geometry'])
    matched={r['gt_id'] for r in score['matches']};all_matched={r['gt_id'] for r in all_score['matches']}
    transitions=[]
    for before,after in zip(STAGES[:-1],STAGES[1:]):
        a,b=masks[before][valid],masks[after][valid];fgtruth=truth>0
        oldset={g for g,r in bygt[before].items() if r['can_exceed_iou_half']};newset={g for g,r in bygt[after].items() if r['can_exceed_iou_half']}
        transitions.append(dict(before=before,after=after,foreground_voxels_lost=int((a&~b&fgtruth).sum()),
            foreground_voxels_gained=int((~a&b&fgtruth).sum()),reachable_objects_lost=sorted(oldset-newset),reachable_objects_gained=sorted(newset-oldset)))
    objects=[]
    for gt in sorted(bygt[STAGES[0]]):
        failed=[name for name in STAGES if not bygt[name][gt]['can_exceed_iou_half']]
        reason='matched' if gt in matched else ('first_coverage_failure:'+failed[0] if failed else 'coverage_sufficient_but_identity_or_contamination_wrong')
        objects.append(dict(gt=gt,matched=gt in matched,all_regions_matched=gt in all_matched,
            coverage={name:bygt[name][gt]['fraction'] for name in STAGES},routing=reason))
    obs=[]
    for oid,ids in enumerate(supports):
        d=label_distribution(point_labels[ids]);known=d['valid'];fgcounts=[c for g,c in zip(d['ids'],d['counts']) if g>0]
        dominant=max(fgcounts,default=0)/known if known else 0.
        obs.append(dict(observation_id=oid,frame=int(frames[oid]),distribution=d,dominant_foreground_purity=dominant,
            multiple_foreground_instances=d['foreground_instances']>=2,
            old_purity_proxy_usable=bool(known>=5 and dominant>=.7)))
    folder=OUT/'maps'/key;folder.mkdir(parents=True)
    np.savez_compressed(folder/'coverage.npz',**masks)
    np.savez_compressed(folder/'predictions.npz',v1=pred,all_regions_control=all_pred)
    np.savez_compressed(folder/'point_targets.npz',instance=point_labels)
    dump(folder/'objects.json',objects);dump(folder/'observations.json',obs)
    result=dict(key=key,dataset=s['dataset'],complete=True,gt_instances=len(objects),stages=stages,
        actual_v1=score,actual_all_regions_control=all_score,
        readout_transition=dict(gained=sorted(all_matched-matched),lost=sorted(matched-all_matched)),
        routing_counts=dict(Counter(o['routing'] for o in objects)),stage_transitions=transitions,
        observations=len(obs),observations_multiple_gt=sum(o['multiple_foreground_instances'] for o in obs),
        observations_old_proxy_usable=sum(o['old_purity_proxy_usable'] for o in obs),
        no_new_mapping=True,coverage_is_not_accuracy=True)
    dump(folder/'summary.json',result)
    print('MAP',key,'GT',len(objects),'TP',score['tp'],'routing',result['routing_counts'],flush=True)
    return result


def maps():
    p=protocol();verify_hashes(p['input_sha256']);verify_hashes(p['reference_sha256']);verify_hashes(p['map_artifact_sha256'])
    assert not (OUT/'maps').exists()
    results=[map_audit(next(s for s in p['scenes'] if s['key']==key)) for key in p['map_scenes']]
    dump(OUT/'map_summary.json',dict(complete=True,scenes=results,
        gt_instances=sum(r['gt_instances'] for r in results),matched=sum(r['actual_v1']['tp'] for r in results),
        routing_counts=dict(sum((Counter(r['routing_counts']) for r in results),Counter())),
        actual_score_scope='Observed-domain instance PQ; all-regions readout is a predefined control, not a new algorithm.',
        diagnostic_scope='Coverage counts are optimistic geometric eligibility, not predicted TP or attainable PQ.'))


def export_scene(s,p):
    key=s['key'];role=next(k for k,v in p['splits'].items() if key in v)
    cache,m,geo,supports,frames=data(s);xyz=geo['xyz'];_,_,labels=point_targets(s,xyz)
    dpath=DINO/'features'/key;dm=read(dpath/'manifest.json');poses=np.load(cache/'poses.npy')
    assert len(poses)==len(m['records'])==len(dm['records'])
    cam=m['config']['cam'];width,height=cam['W'],cam['H']
    K=np.array([[cam['fx'],0,cam['cx']],[0,cam['fy'],cam['cy']],[0,0,1.]])
    gw,gh=dm['resize'][0]//14,dm['resize'][1]//14
    folder=OUT/'supervision'/role/key;folder.mkdir(parents=True)
    np.savez_compressed(folder/'point_targets.npz',instance=labels)
    records=[];totals=Counter()
    for index,(r,dr) in enumerate(zip(m['records'],dm['records'])):
        assert all(r[k]==dr[k] for k in ['frame_id','observation_offset','masks'])
        source=dpath/dr['file'];z=np.load(source);T=np.linalg.inv(poses[index])
        token_bins=[];point_ids=[];point_offsets=[0];hist_ids=[];hist_counts=[];hist_offsets=[0];stats=[]
        observations=[]
        for j in range(r['masks']):
            oid=r['observation_offset']+j;ids=supports[oid]
            bins,groups=token_groups(xyz[ids],T,K,width,height,gw,gh)
            assert len(bins)==int(z['offsets'][j+1]-z['offsets'][j]),(key,oid,'cached token count mismatch')
            for b,g in zip(bins,groups):
                original=ids[g];d=label_distribution(labels[original])
                token_bins.append(int(b));point_ids.extend(original.tolist());point_offsets.append(len(point_ids))
                hist_ids.extend(d['ids']);hist_counts.extend(d['counts']);hist_offsets.append(len(hist_ids))
                stats.append([d['total'],d['valid'],d['foreground'],d['foreground_instances']])
                totals['tokens']+=1;totals['tokens_with_valid_support']+=int(d['valid']>0)
                totals['tokens_with_foreground']+=int(d['foreground']>0)
                totals['tokens_multiple_foreground_instances']+=int(d['foreground_instances']>=2)
                totals['token_support_occurrences']+=d['total']
            distribution=label_distribution(labels[ids]);observations.append(dict(observation_id=oid,distribution=distribution))
            totals['observations']+=1;totals['observations_multiple_foreground_instances']+=int(distribution['foreground_instances']>=2)
        assert len(token_bins)==len(z['local_features'])
        target=folder/dr['file']
        np.savez_compressed(target,features=z['local_features'],offsets=z['offsets'],
            observation_ids=np.arange(r['observation_offset'],r['observation_offset']+r['masks']),
            patch_bins=np.array(token_bins,np.int32),point_ids=np.array(point_ids,np.int64),point_offsets=np.array(point_offsets,np.int64),
            target_instance_ids=np.array(hist_ids,np.int64),target_counts=np.array(hist_counts,np.int32),target_offsets=np.array(hist_offsets,np.int64),
            target_stats=np.array(stats,np.int64).reshape(-1,4))
        meta=target.with_suffix('.json');dump(meta,dict(frame=r['frame_id'],observations=observations,
            feature_source=str(source),feature_source_sha256=dr['sha256'],contains_gt_targets=True))
        records.append(dict(frame_id=r['frame_id'],file=target.name,sha256=sha(target),meta=meta.name,meta_sha256=sha(meta)))
    manifest=dict(complete=True,key=key,role=role,records=records,counts=dict(totals),
        point_targets_sha256=sha(folder/'point_targets.npz'),reference_sha256=p['reference_sha256'][s['reference']],
        features_unchanged=True,all_observations_exported=True,gt_for_offline_supervision_only=True,
        instance_scope='scene-local; -1 unknown, 0 background, positive IDs from fixed reference',
        target_kind='histogram of projected measured supports, not full image/pixel labels',
        correlated_samples=True,new_inference=False,new_training=False)
    assert totals['observations']==len(supports)
    dump(folder/'manifest.json',manifest);print('SUPERVISION',key,role,dict(totals),flush=True)
    return dict(key=key,role=role,counts=dict(totals),manifest=str(folder/'manifest.json'),sha256=sha(folder/'manifest.json'))


def supervision():
    p=protocol();verify_hashes(p['input_sha256']);verify_hashes(p['reference_sha256']);assert not (OUT/'supervision').exists()
    results=[export_scene(s,p) for s in p['scenes']]
    dump(OUT/'supervision_summary.json',dict(complete=True,scenes=results,
        totals_by_role={role:dict(sum((Counter(r['counts']) for r in results if r['role']==role),Counter())) for role in p['splits']},
        no_purity_based_target_dropping=True,no_new_training=True,features_from_previous_cache=True))


def audit():
    p=protocol();verify_hashes(p['input_sha256']);verify_hashes(p['reference_sha256']);verify_hashes(p['map_artifact_sha256'])
    summary=read(OUT/'map_summary.json');dataset=read(OUT/'supervision_summary.json')
    for r in summary['scenes']:
        entry=next(s for s in p['scenes'] if s['key']==r['key']);ref=np.load(entry['reference'])
        valid=ref['valid'] if 'valid' in ref.files else np.ones(len(ref['xyz']),bool);truth=ref['instance'][valid]
        coverage=np.load(OUT/'maps'/r['key']/'coverage.npz');pred=np.load(OUT/'maps'/r['key']/'predictions.npz')
        objects=read(OUT/'maps'/r['key']/'objects.json')
        for name in STAGES:
            count=0
            for row in objects:
                hit=int(coverage[name][valid][truth==row['gt']].sum());n=int((truth==row['gt']).sum())
                assert row['coverage'][name]==hit/n;count+=int(2*hit>n)
            assert count==r['stages'][name]['reachable_gt_over_half']
        for name,outname in [('v1','actual_v1'),('all_regions_control','actual_all_regions_control')]:
            assert voxel_metrics(truth,pred[name][valid])==r[outname]
    for row in dataset['scenes']:
        assert sha(row['manifest'])==row['sha256'];manifest=read(row['manifest']);folder=Path(row['manifest']).parent
        entry=next(s for s in p['scenes'] if s['key']==row['key']);cache,m,geo,supports,frames=data(entry)
        assert sha(folder/'point_targets.npz')==manifest['point_targets_sha256']
        labels=np.load(folder/'point_targets.npz')['instance']
        _,_,independent_labels=point_targets(entry,geo['xyz']);np.testing.assert_array_equal(labels,independent_labels)
        counts=Counter()
        for rec in manifest['records']:
            assert sha(folder/rec['file'])==rec['sha256'];assert sha(folder/rec['meta'])==rec['meta_sha256']
            z=np.load(folder/rec['file']);meta=read(folder/rec['meta']);orig=np.load(meta['feature_source'])
            np.testing.assert_array_equal(z['features'],orig['local_features']);np.testing.assert_array_equal(z['offsets'],orig['offsets'])
            for oid,start,end in zip(z['observation_ids'],z['offsets'][:-1],z['offsets'][1:]):
                original_ids=supports[int(oid)]
                for i in range(int(start),int(end)):
                    ids=z['point_ids'][z['point_offsets'][i]:z['point_offsets'][i+1]]
                    assert np.isin(ids,original_ids).all()
                    values,n=np.unique(labels[ids],return_counts=True)
                    sl=slice(z['target_offsets'][i],z['target_offsets'][i+1])
                    np.testing.assert_array_equal(values,z['target_instance_ids'][sl]);np.testing.assert_array_equal(n,z['target_counts'][sl])
                    stat=[len(ids),int((labels[ids]>=0).sum()),int((labels[ids]>0).sum()),int((values>0).sum())]
                    np.testing.assert_array_equal(stat,z['target_stats'][i])
                    counts['tokens']+=1;counts['tokens_with_valid_support']+=int(stat[1]>0)
                    counts['tokens_with_foreground']+=int(stat[2]>0);counts['tokens_multiple_foreground_instances']+=int(stat[3]>=2)
                    counts['token_support_occurrences']+=stat[0]
                counts['observations']+=1;counts['observations_multiple_foreground_instances']+=int(len(np.unique(labels[original_ids][labels[original_ids]>0]))>=2)
            for ob in meta['observations']:assert ob['distribution']==label_distribution(labels[supports[ob['observation_id']]])
        assert dict(counts)==manifest['counts']==row['counts']
    dump(OUT/'audit.json',dict(complete=True,all_inputs_and_original_maps_unchanged=True,
        all_full_map_metrics_recomputed=True,coverage_counts_recomputed=True,
        token_features_bytewise_array_equal=True,all_token_target_histograms_recomputed=True,
        source_support_membership_checked=True,all_scene_point_targets_recomputed=True,
        no_new_inference_training_or_mapping=True,
        map_summary_sha256=sha(OUT/'map_summary.json'),supervision_summary_sha256=sha(OUT/'supervision_summary.json')))
    print('AUDIT PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','maps','supervision','audit'])
    args=parser.parse_args();globals()[args.action]()
