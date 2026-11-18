"""Versioned pre-inference correction; original protocols/results stay intact."""
import argparse
import shutil
import time
import numpy as np
import run_batch_transport_assessment as original
from audit_transport_unknown_seeds import OUT as AUDIT, corrected_graph_and_seeds, known_semantics
from paper_final_common import ROOT, read, dump, sha

OUT = original.OUT
AMENDMENT = OUT / 'unknown_semantic_amendment.json'


def freeze():
    original.verify(True)
    assert not AMENDMENT.exists()
    assert not (OUT / 'frontend').exists(), 'Amendment must precede new image inference'
    audit = read(AUDIT / 'complete.json'); assert audit['complete']
    # A background-to-unknown change can alter a field but cannot by itself
    # change the exported foreground map. Check every cell, not a sample.
    export_checks = {}
    for keydomain, row in audit['results'].items():
        key, domain = keydomain.split('/')
        changes = {}
        if not row['identical_seeds']:
            fixed = np.load(AUDIT / (key + '_' + domain + '.npz'))
            old = np.load(ROOT / 'build/benchmarks/batch_stable_transport_v1/fields' / key / domain / 'field.npz')
            for name in row['changed_cells']:
                a = fixed[name]
                b = old[name] if name != 'same_cell_owner' else np.where(old['label_distance'][:, 0] == 0, old['nearest_owner'], -1)
                changes[name] = int(np.count_nonzero(np.where(a > 0, a, 0) != np.where(b > 0, b, 0)))
                assert changes[name] == 0, (keydomain, name, 'Requires separate corrected development evaluation')
        export_checks[keydomain] = changes
    # Boundary contract, including conflict ownership and valid background.
    sample = dict(owners=np.array([0,0,0,0,0,7,-1]), seen=np.ones(7, dtype=int), mean_response=np.array([-1,255,0,2,13,-1,-1]))
    checked = known_semantics(sample)
    np.testing.assert_array_equal(checked['seen'], [0,0,1,1,1,0,0])
    np.testing.assert_array_equal(sample['seen'], np.ones(7))
    np.testing.assert_array_equal(checked['owners'], sample['owners'])
    sources = {str(p):sha(p) for p in [ROOT / 'scripts/benchmarks/audit_transport_unknown_seeds.py', ROOT / 'scripts/benchmarks/run_batch_transport_assessment_corrected.py', AUDIT / 'complete.json']}
    deps = ROOT / 'build/benchmarks/sam3_probe_runtime/deps/pycocotools'
    runtime = {str(p):sha(p) for p in deps.iterdir() if p.is_file() and p.suffix in ['.py','.so']}
    assert runtime
    dump(AMENDMENT, dict(complete=True, created_epoch=time.time(), original_execution_sha256=sha(OUT/'execution_protocol.json'), original_registration_sha256=sha(OUT/'protocol.json'), source_hashes=sources, runtime_dependency_hashes=runtime,
        correction='Suppress unknown/invalid semantic winners before background seed extraction; foreground ownership unchanged. Same correction shared by all controls and future robot run. No threshold/model/scene/gate change.',
        timing='Before any new RGB model inference or reference decoding; initial mask attempt failed at import.',
        development_field_changes=audit['results'], development_foreground_export_changes=export_checks, development_maps_and_scores_unchanged=True,
        runtime='Existing pycocotools2.0.10 CPython3.11 extension via isolated PYTHONPATH; no installation or shared runtime modification.',
        reference_labels_read=False))
    print('PRE-INFERENCE CORRECTION FROZEN; ALL DEVELOPMENT FOREGROUND EXPORTS IDENTICAL', flush=True)


def verify_amendment():
    p=read(AMENDMENT); assert p['complete']
    assert sha(OUT/'execution_protocol.json') == p['original_execution_sha256']
    assert sha(OUT/'protocol.json') == p['original_registration_sha256']
    for path, value in {**p['source_hashes'], **p['runtime_dependency_hashes']}.items(): assert sha(path)==value, path


def recover_failed_import():
    verify_amendment()
    # Caller confirmed the old execution handle missing; require its terminal
    # import failure and absence of inference products before moving metadata.
    assert not (OUT/'frontend').exists() and not (OUT/'masks_complete.json').exists()
    log=ROOT / 'build/benchmarks/batch_transport_assessment_v1/runtime_attempt1.log'
    shutil.copyfile('/tmp/omni_assessment_masks.log',log)
    assert "ModuleNotFoundError: No module named 'pycocotools'" in log.read_text()
    dest=OUT/'runtime_attempts/attempt1'; dest.mkdir(parents=True,exist_ok=False)
    shutil.move(str(OUT/'stages/masks.json'),str(dest/'masks.json'))
    shutil.move(str(log),str(dest/'stderr.log'))
    dump(dest/'terminal.json',dict(terminal=True, handle=88777, handle_query='Unknown process id', inference_outputs_absent=True, reason='Import failure before model initialization', retry='Same stage/model/inputs with existing local dependency; use correction amendment wrapper'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','recover_import','masks','features','supports','base_maps','candidates','intrinsic','raw','geometry','anchors','semantics','aggregate','fields','maps','score'])
    action=parser.parse_args().action
    if action=='freeze': return freeze()
    if action=='recover_import': return recover_failed_import()
    verify_amendment()
    native_start=original.stage_start
    def stage_start(name):
        verify_amendment();native_start(name)
        path=OUT/'stages'/(name+'.json');record=read(path);record['correction_amendment_sha256']=sha(AMENDMENT);dump(path,record)
    original.stage_start=stage_start
    original.graph_and_seeds=corrected_graph_and_seeds
    original.frontend(action) if action in ['masks','features','supports'] else getattr(original,action)()


if __name__=='__main__':main()
