"""Separate semantic contamination from same-class identity errors, post hoc.

No candidate/map/parameter is changed. Explicit original semantic taxonomy
keeps road (train0) distinct from ignored (train255) reference regions.
"""
import runpy
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from evaluate_kitti360_development import transfer, read_static_ply

OUT = ROOT/'build/benchmarks/mask_observation_code_probe_v1'
FIELD = ROOT/'build/benchmarks/surface_evidence_field_v1'
PARENT = ROOT/'build/benchmarks/signed_query_association_v1'


def main():
    assert read(OUT/'predictions_complete.json')['complete']
    assert not (OUT/'field_error_types.json').exists()
    p = read(FIELD/'protocol.json')
    refs = read(ROOT/'build/benchmarks/surface_ownership_v1/protocol.json')['references']
    taxonomy_path = ROOT/'build/datasets/kitti360_prepare/labels.py'
    sources = [__file__, ROOT/'scripts/benchmarks/evaluate_kitti360_development.py', taxonomy_path]
    inputs = {}
    for f in [FIELD/'predictions_complete.json', FIELD/'error_provenance.json',
              PARENT/'object_diagnostics.json', OUT/'predictions_complete.json']:
        inputs[str(f)] = sha(f)
    for scene in p['scenes']:
        k = scene['key']
        assert sha(refs[k]['raw']) == refs[k]['sha256']
        for f in [scene['reference'], refs[k]['raw'], FIELD/'maps'/k/'baseline/geometry.npz', FIELD/'maps'/k/'ours/geometry.npz']:
            inputs[str(f)] = sha(f)
    dump(OUT/'field_error_type_protocol.json', dict(created_epoch=time.time(), input_hashes=inputs,
         source_hashes={str(f):sha(f) for f in sources},
         scope='Existing frozen surface-field failure only. Baseline AP50-correct objects as in previous provenance audit. Full reference confidence-weighted areas, same20cm nearest-map transfer; no source-point accuracy/AP/causal effect interpretation.',
         categories='Same target; other same semantic class (including unsplit instance0); different known semantic class including road; ignored semantic train255 separately. Raw reference row order verified against original train_instance and confidence.',
         direction='Post-score evidence to decide whether foreground acceptance or within-class identity needs work; never fed back to frozen predictions. Not a new mapping experiment.'))
    taxonomy = runpy.run_path(str(taxonomy_path))['id2label']
    details = read(PARENT/'object_diagnostics.json')['results']
    old_audit = read(FIELD/'error_provenance.json')['results']
    results = {}
    for scene in p['scenes']:
        key = scene['key']; ref = dict(np.load(scene['reference'])); raw = read_static_ply(refs[key]['raw'])
        train = np.array([taxonomy[int(s)].trainId for s in raw['semantic']], np.int64)
        truth = train*1000 + raw['instance'].astype(np.int64)%1000; truth[train == 255] = 0
        np.testing.assert_array_equal(truth, ref['train_instance'])
        np.testing.assert_allclose(raw['confidence'], ref['confidence'])
        w = ref['confidence'].astype(float)
        base = dict(np.load(FIELD/'maps'/key/'baseline/geometry.npz'))
        ours = dict(np.load(FIELD/'maps'/key/'ours/geometry.npz'))
        ix = transfer(ours['xyz'], np.arange(1,len(ours['xyz'])+1), ref['xyz'], .2)
        added = ix > len(base['xyz'])
        owner = np.where(ix>0, ours['instance'][np.maximum(ix-1,0)], 0)
        anchors = {o['object_id']:o['overlaps'][0]['gt_id'] for o in details[key]['geometry_graph']
                   if o['overlaps'] and o['overlaps'][0]['iou'] > .5}
        rows=[]; totals={n:0. for n in ['same_target','same_class_other','different_known_class','ignored']}
        classes={}
        for oid,gid in anchors.items():
            take = added & (owner == oid)
            masks = dict(same_target=take&(truth==gid),
                         same_class_other=take&(train==gid//1000)&(truth!=gid),
                         different_known_class=take&(train!=gid//1000)&(train!=255),
                         ignored=take&(train==255))
            parts = {n:float(w[m].sum()) for n,m in masks.items()}
            np.testing.assert_allclose(sum(parts.values()), w[take].sum())
            for n,v in parts.items():totals[n]+=v
            for sem in np.unique(raw['semantic'][masks['different_known_class']]):
                name=taxonomy[int(sem)].name
                classes[name]=classes.get(name,0.)+float(w[masks['different_known_class']&(raw['semantic']==sem)].sum())
            rows.append(dict(object_id=int(oid),gt_id=int(gid),new_reference_area=parts))
        old = old_audit[key]['totals']['new_measurement']
        np.testing.assert_allclose(totals['same_target'],old['correct'],rtol=1e-6)
        np.testing.assert_allclose(sum(totals[n] for n in totals if n!='same_target'),old['outside_target'],rtol=1e-6)
        results[key] = dict(totals=totals,different_known_classes=dict(sorted(classes.items(),key=lambda a:-a[1])),objects=rows)
        print('FIELD ERROR TYPES',key,totals,classes,flush=True)
    dump(OUT/'field_error_types.json',dict(complete=True,results=results,
         protocol_sha256=sha(OUT/'field_error_type_protocol.json'),no_new_map=True,no_new_prediction=True,
         warning='Only previously AP50-correct baseline objects. Reference coverage area, not source-point count. Same-class-other can include unsplit instance0. No causal intervention.'))


if __name__=='__main__':main()
