"""Post-hoc provenance of all reference changes in the sealed development maps.

No predictions, map variants, scores, fitting or threshold selection. Gate
categories overlap; use a declared priority only to tabulate a disjoint sum.
"""
from pathlib import Path
import runpy
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from run_revision_authorization import verify, OUT as PUBLIC, PARENT
from evaluate_kitti360_development import read_static_ply
from omni_semantic_map.kitti360_reference_identity import encode_reference

OUT = ROOT / 'build/benchmarks/revision_loss_sources_v1'


def main():
    p = verify(); assert not OUT.exists()
    rows = read(PUBLIC / 'per_object.json')['results']; inputs = {}
    files = [PUBLIC/'protocol.json', PUBLIC/'summary.json', PUBLIC/'per_object.json']
    for c in p['contexts']:
        k = c['key']
        files += [Path(c['reference']),Path(c['raw_reference']),Path(c['ours'])/'geometry.npz',
                  PARENT/'maps'/k/'cover_history/geometry.npz',
                  PUBLIC/'prepared'/k/'write_decisions.npz',
                  ROOT/'build/benchmarks/object_revision_provenance_v1'/k/'reference_provenance.npz']
        files += [PUBLIC/'evaluation'/k/m/'reference_prediction.npz' for m in ['unconditional','ours']]
    inputs = {str(f):sha(f) for f in files}
    dump(OUT/'protocol.json',dict(inputs=inputs,source_sha256=sha(__file__),
        scope='Post-hoc diagnostic of BOTH exposed windows and ALL91raw-target objects, no new map/AP/model/training. Ground truth only explains sealed outcomes.',
        assignment='Hold each GT object parent-best predictedID fixed. This is a reference influence analysis, NOT correspondence-aware semantic matching or AP. A changed bestID may retain semantic quality, so preserve all method bestID/IoU alongside counts.',
        categories='For the previously nearest parent-added source: outside_direct_component, no_cell_witness, no_local_certificate, relabelled, retained. Priority in that order. Parent old/absent winner separately recorded. Source predicates are exact; they are not counterfactual causal AP contributions.',
        missing='No cell witness is lack of observed selected-mask support, not a certified background/occlusion label. Reference weighted area is not source-point accuracy.'))
    taxonomy = runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    result={}
    for c in p['contexts']:
        k=c['key'];ref=np.load(c['reference']);raw=read_static_ply(c['raw_reference'])
        gt,_=encode_reference(raw['semantic'],raw['instance'],taxonomy,'target_only');weight=ref['confidence']
        parent=np.load(PARENT/'maps'/k/'cover_history/geometry.npz');nold=len(np.load(Path(c['ours'])/'geometry.npz')['xyz'])
        index=np.load(ROOT/'build/benchmarks/object_revision_provenance_v1'/k/'reference_provenance.npz')['nearest_index']
        d=np.load(PUBLIC/'prepared'/k/'write_decisions.npz')
        before=np.load(PUBLIC/'evaluation'/k/'unconditional/reference_prediction.npz')['instance']
        after=np.load(PUBLIC/'evaluation'/k/'ours/reference_prediction.npz')['instance']
        replay=np.zeros(len(index),np.int64);present=index>=0;replay[present]=parent['instance'][index[present]]
        np.testing.assert_array_equal(replay,before)
        reason=np.full(len(d['direct']),4,np.int8)
        reason[d['local_label']!=parent['instance'][nold:]]=3
        reason[d['local_label']<0]=2;reason[~d['witness']]=1;reason[~d['direct']]=0
        names=['outside_direct_component','no_cell_witness','no_local_certificate','relabelled','retained','old_or_absent_parent_winner']
        refreason=np.full(len(index),5,np.int8);new=index>=nold;refreason[new]=reason[index[new]-nold]
        def mass(t):return float(weight[t].sum())
        objects=[]
        for row in rows[k]['raw_targets']:
            gid=row['gt_id'];oid=row['methods']['unconditional']['object_id'];target=gt==gid
            lost=target&(before==oid)&(after!=oid);gained=target&(before!=oid)&(after==oid)
            lost_by={name:mass(lost&(refreason==j)) for j,name in enumerate(names)}
            assert abs(sum(lost_by.values())-mass(lost))<1e-7
            replacements={int(i):mass(lost&(after==i)) for i in np.unique(after[lost])}
            objects.append(dict(gt_id=gid,class_name=row['class_name'],fixed_parent_id=oid,
                best_matches=row['methods'],parent_tp=mass(target&(before==oid)),ours_fixed_id_tp=mass(target&(after==oid)),
                fixed_id_tp_lost=mass(lost),fixed_id_tp_gained=mass(gained),
                fixed_id_fp_removed=mass(~target&(before==oid)&(after!=oid)),
                fixed_id_fp_added=mass(~target&(before!=oid)&(after==oid)),
                lost_parent_reference_area_by_source_decision=lost_by,replacement_ids=replacements))
        result[k]=dict(objects=objects,parent_additions=len(reason),
            source_decision_counts={name:int((reason==j).sum()) for j,name in enumerate(names[:5])},
            exact_reference_index_replay=True)
    for f,h in inputs.items():assert sha(f)==h,f
    assert sha(__file__)==read(OUT/'protocol.json')['source_sha256']
    dump(OUT/'summary.json',dict(complete=True,results=result,objects=sum(len(v['objects']) for v in result.values()),
         source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.json'),new_maps=0,new_scores=0,goal_complete=False))
    print('REVISION LOSS SOURCE AUDIT COMPLETE / all91objects / no new predictions',flush=True)


if __name__=='__main__':
    main()
