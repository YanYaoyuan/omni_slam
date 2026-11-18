"""GT-assisted diagnostics only. Never import these assignments into mapping."""
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT, read, dump, sha
from run_vggt_fullmap import OUT as MAPS, verify
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.instance_error_audit import audit_instances, filter_transition

OUT=ROOT/'build/benchmarks/association_ceiling_v1'


def relabel_atoms(truth, atoms, native):
    """One label per indivisible atom; preserve every assigned point.

    Assign positive GT plurality (including background in the vote). When
    background wins, retain its native positive identity, never suppress it.
    This is a constructive oracle, not a combinatorial optimum or upper bound.
    """
    truth,atoms,native=map(np.asarray,(truth,atoms,native))
    assert truth.shape==atoms.shape==native.shape
    assert np.all(native[atoms<0]==0) and np.all(native[atoms>=0]>0)
    result=native.copy();records=[];offset=int(native.max(initial=0))+1
    for atom in np.unique(atoms[atoms>=0]):
        mask=atoms==atom;prior=np.unique(native[mask]);assert len(prior)==1
        ids,counts=np.unique(truth[mask],return_counts=True);winner=int(ids[np.argmax(counts)])
        label=offset+winner if winner>0 else int(prior[0])
        result[mask]=label
        records.append(dict(atom=int(atom),native=int(prior[0]),assigned=int(label),
            gt_plurality=winner,gt_ids=ids.tolist(),counts=counts.tolist()))
    assert np.array_equal(result>0,native>0)
    return result,records


def coverage_ceiling(truth,covered):
    """Optimistic evaluation-space PQ ceiling within a fixed covered domain.

    Grant perfect point identity AND background rejection. Suppress GT objects
    whose reachable fraction is <=.5 since they cannot be true positives and
    would only increase FP. No claim of a realizable whole-observation map.
    """
    prediction=np.zeros_like(truth);rows=[]
    for gid in np.unique(truth[truth>0]):
        mask=truth==gid;size=int(mask.sum());n=int(np.count_nonzero(mask&covered));ratio=n/size
        if ratio>.5:prediction[mask&covered]=gid
        rows.append(dict(gt_id=int(gid),reference_points=size,covered=n,coverage=ratio,recoverable_at_iou50=ratio>.5))
    return prediction,rows


def freeze():
    assert not OUT.exists()
    parent=verify(True)
    sources=[Path(__file__),Path(__file__).with_name('test_association_ceiling.py'),
        ROOT/'omni_semantic_map/omni_semantic_map/full_ovo_evaluation.py',
        ROOT/'omni_semantic_map/omni_semantic_map/mesh_instance_evaluation.py',
        ROOT/'omni_semantic_map/omni_semantic_map/instance_error_audit.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=[s['key'] for s in parent['scenes']],
        input_seal_sha256=sha(MAPS/'predictions_complete.json'),parent_protocol_sha256=sha(MAPS/'protocol.json'),
        source_hashes={str(p):sha(p) for p in sources},baseline='vggt_single',gt_access=True,
        whole_map='Keep exact final VGGT-single predicted reference partition. Relabel entire predicted objects using GT plurality, including background in voting; keep native identity if background wins. Merge-only diagnostic, never splits/deletes points.',
        whole_observation='Use only observations retained as foreground in final VGGT-single map. Keep original source points and supports. Fix physical-point ownership by earliest observation and separately by latest observation; project nearest source within5cm. Compare native object IDs and GT plurality observation reassignment on the exact same owner partition. Background-dominant atoms retain native IDs. No within-observation split, no point deletion, no semantic reclassification.',
        overlap_sensitivity='Both earliest and latest ownership must be reported; no selecting best. Canonical point ownership clips overlaps equally for native/oracle. This replay renderer omits CG denoising and differs from the native final geometry, so its paired delta is the valid association intervention.',
        coverage='Four fixed support domains: native final foreground points; raw retained-observation supports; all raw mask supports; all measured geometry. Within each domain grant perfect per-reference-point identity and background rejection. This is optimistic evaluation-space coverage ceiling, not an association algorithm score or a proven achievable 3D map.',
        gate='Invest in one bounded observation-ownership revision prototype only if BOTH overlap policies improve their matched replay control by >=5 PQ points and >=2 net true positives in EACH scene. This is a resource-allocation criterion, not a paper acceptance criterion. Otherwise report exact bottleneck without automatic search.',
        budget='Two exposed scenes, CPU-only diagnostics, zero new model inference/training or native mapping, no threshold sweep. GT-derived artifacts stay in this separate diagnostics folder.'))


def run():
    p=read(OUT/'protocol.json');parent=verify(True)
    for f,h in p['source_hashes'].items():assert sha(f)==h
    assert sha(MAPS/'protocol.json')==p['parent_protocol_sha256']
    assert sha(MAPS/'predictions_complete.json')==p['input_seal_sha256']
    for f,h in read(MAPS/'predictions_complete.json')['files'].items():assert sha(f)==h
    assert not (OUT/'summary.json').exists()
    scenes=[];started=time.monotonic()
    for s in parent['scenes']:
        key=s['key'];dest=OUT/key;dest.mkdir(exist_ok=False)
        assert sha(s['reference'])==s['reference_sha256']
        ref=np.load(s['reference']);valid=ref['valid'] if 'valid' in ref else np.ones(len(ref['instance']),bool)
        truth=ref['instance'][valid];vertices=ref['xyz'][valid]
        folder=MAPS/'maps'/key/p['baseline']/'history2';rows=read(folder/'classification.json')['objects']
        fg=[r for r in rows if r['predictions']['mean']['retained']];fgids=[r['object_id'] for r in fg]
        geom=np.load(folder/'geometry.npz');keep=np.isin(geom['instance'],fgids)
        native,distance=transfer_labels(geom['xyz'],np.where(keep,geom['instance'],0),vertices,mode='nearest_5cm')
        expected=next(r for r in read(MAPS/'evaluation.json')['results'] if r['scene']==key)['methods'][p['baseline']]['metrics']
        assert abs(voxel_metrics(truth,native)['observed_voxel_instance_pq']-expected['observed_voxel_instance_pq'])<1e-12
        methods={};audits={};predictions={}
        def record(name,pred):
            methods[name]=voxel_metrics(truth,pred);audits[name]=audit_instances(truth,pred);predictions[name]=pred
            dump(dest/(name+'_audit.json'),audits[name])
        record('native',native)
        merged,assignment=relabel_atoms(truth,np.where(native>0,native,-1),native)
        record('final_merge_oracle',merged);dump(dest/'final_merge_assignment.json',assignment)
        cache=Path(s['cache']);source=np.load(cache/'geometry.npz')['xyz'];n=len(source)
        nobs=sum(r['masks'] for r in s['records']);object_for_obs=np.zeros(nobs,np.int64)
        for r in fg:
            ids=np.array(r['observation_ids']);assert not np.any(object_for_obs[ids]);object_for_obs[ids]=r['object_id']
        first=np.full(n,-1,np.int32);last=np.full(n,-1,np.int32);all_support=np.zeros(n,bool)
        for rec in s['records']:
            z=np.load(cache/'frames'/rec['file']);all_support[z['point_ids']]=True
            for j,(lo,hi) in enumerate(zip(z['offsets'][:-1],z['offsets'][1:])):
                oid=rec['observation_offset']+j
                if not object_for_obs[oid]:continue
                ids=z['point_ids'][lo:hi];unknown=first[ids]<0;first[ids[unknown]]=oid;last[ids]=oid
        active=first>=0;assert np.array_equal(active,last>=0);assert np.all(all_support[active])
        activeids=np.flatnonzero(active);tree=cKDTree(source[activeids]);dist,idx=tree.query(vertices,workers=1)
        covered=dist<=.05;nearest=activeids[idx];transitions={}
        for order,owners in [('earliest',first),('latest',last)]:
            atoms=np.where(covered,owners[nearest],-1);base=np.where(covered,object_for_obs[np.maximum(atoms,0)],0)
            oracle,assignments=relabel_atoms(truth,atoms,base)
            record(order+'_native',base);record(order+'_oracle',oracle)
            dump(dest/(order+'_assignment.json'),assignments)
            transitions[order]=filter_transition(audits[order+'_native'],audits[order+'_oracle'])
            np.save(dest/(order+'_reference_observation.npy'),atoms)
        ceilings={}
        domains={'native_final':distance<=.05,'retained_observations':covered,
            'all_masks':cKDTree(source[all_support]).query(vertices,workers=1)[0]<=.05,
            'all_measurements':cKDTree(source).query(vertices,workers=1)[0]<=.05}
        assert np.all(~domains['retained_observations']|domains['all_masks'])
        assert np.all(~domains['all_masks']|domains['all_measurements'])
        for name,domain in domains.items():
            pred,coverage=coverage_ceiling(truth,domain);record('ceiling_'+name,pred);ceilings[name]=coverage
        np.savez_compressed(dest/'reference_predictions.npz',**predictions)
        counts=dict(source_points=n,mask_supported_points=int(all_support.sum()),retained_supported_points=int(active.sum()),
            observations=nobs,foreground_retained_observations=int(np.count_nonzero(object_for_obs)))
        merge_transition=filter_transition(audits['native'],audits['final_merge_oracle'])
        gate=all(methods[o+'_oracle']['observed_voxel_instance_pq']-methods[o+'_native']['observed_voxel_instance_pq']>=.05
            and methods[o+'_oracle']['tp']-methods[o+'_native']['tp']>=2 for o in ['earliest','latest'])
        matched={k:{r['gt_id'] for r in v['matches']} for k,v in methods.items()};objects=[]
        for i,g in enumerate(ceilings['all_measurements']):
            gid=g['gt_id'];coverage={k:next(x for x in v if x['gt_id']==gid) for k,v in ceilings.items()}
            objects.append(dict(gt_id=gid,coverage=coverage,matched={k:gid in v for k,v in matched.items()},
                raw_association_recovered_both=all(gid not in matched[o+'_native'] and gid in matched[o+'_oracle'] for o in ['earliest','latest']),
                native_missing_recovered_both=gid not in matched['native'] and all(gid in matched[o+'_oracle'] for o in ['earliest','latest'])))
        row=dict(scene=key,counts=counts,methods=methods,coverage=ceilings,transitions=transitions,
            merge_transition=merge_transition,objects=objects,gate=gate)
        dump(dest/'results.json',row);scenes.append(row)
        print(key,{k:(round(v['observed_voxel_instance_pq']*100,2),v['tp']) for k,v in methods.items()},flush=True)
    dump(OUT/'summary.json',dict(complete=True,gt_diagnostic_only=True,scenes=scenes,
        invest_gate=all(s['gate'] for s in scenes),protocol_sha256=sha(OUT/'protocol.json'),wall_s=time.monotonic()-started,
        new_model_inference=0,new_native_mapping=0,training_runs=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args();globals()[args.action]()
