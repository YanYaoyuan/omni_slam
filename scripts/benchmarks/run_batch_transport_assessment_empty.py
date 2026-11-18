"""Bounded empty-input compatibility fixes; no model or selection rule changes.

Apply three checked source substitutions in isolated namespaces. Frozen source
files remain intact and normal-input outputs are replayed before registration.
"""
import inspect
import sys
import time
import shutil
import numpy as np
import run_batch_transport_assessment as original
import run_batch_transport_assessment_corrected as corrected
from paper_final_common import ROOT, read, dump, sha

OUT=original.OUT
RECORD=OUT/'empty_input_amendment.json'
PATCHES={
 'run_surface_ownership.py':('features', [('np.repeat(np.r_[0,np.cumsum(count)[:-1]],count)', 'np.repeat(np.r_[0,np.cumsum(count)][:len(count)],count)')]),
 'run_intrinsic_surface_ownership.py':('features', [('np.max(np.abs(tensor[valid]-transformed[k][valid]))', 'np.max(np.abs(tensor[valid]-transformed[k][valid]),initial=0.)')]),
 'run_robot_geometry_reference.py':('export_reference', [('np.concatenate([sets[j] for j,m in enumerate(metadata) if m[\'old_object_id\']>=0])', 'np.concatenate([sets[j] for j,m in enumerate(metadata) if m[\'old_object_id\']>=0]+[np.empty(0,np.int64)])'), ('max(old_ids)+1','max(old_ids,default=0)+1')]),
}
native_adapter=original.isolated_adapter


def adapted(filename):
    module=native_adapter(filename)
    if filename in PATCHES:
        name,replacements=PATCHES[filename];fn=getattr(module,name);source=inspect.getsource(fn)
        for old,new in replacements:
            assert source.count(old)==1,(filename,old);source=source.replace(old,new)
        namespace=dict(fn.__globals__)
        exec(compile(source,str(__file__)+':'+filename,'exec'),namespace)
        replacement=namespace[name]
        # Feature functions must use the isolated module's redirected OUT/INPUT.
        if name=='features':
            def call(*args,**kwargs):
                namespace.update(module.__dict__)
                return replacement(*args,**kwargs)
            setattr(module,name,call)
        else:setattr(module,name,replacement)
    if filename=='run_intrinsic_public_holdout.py':module.isolated_adapter=adapted
    return module


def freeze():
    corrected.verify_amendment();original.verify(True)
    assert not RECORD.exists() and not (OUT/'evaluation').exists()
    n=adapted('run_robot_geometry_reference.py');checks={}
    for key in ['kitti360_0000','kitti360_0002']:
        d=np.load(n.PARENT/'nodes'/key/'nodes.npz');sets=n.pieces(d['source_ids'],d['source_offsets']);meta=read(n.PARENT/'nodes'/key/'nodes.json');source=np.load(read(n.INPUT/key/'manifest.json')['geometry'])['xyz'];cells=np.load(n.INPUT/'query_cells'/key/'source_to_cell.npy');base=dict(np.load(n.BASE/'maps'/key/'intrinsic_geometry/geometry.npz'));cls=read(n.BASE/'maps'/key/'intrinsic_geometry/classification.json')
        g,c,groups,stat=n.export_reference(base,cls,source,sets,meta,cells);expected=np.load(n.PARENT/'maps'/key/'geometry_graph/geometry.npz')
        for k in g:np.testing.assert_array_equal(g[k],expected[k])
        assert c==read(n.PARENT/'maps'/key/'geometry_graph/classification.json')
        groups['old_id_assignment']={str(k):v for k,v in groups['old_id_assignment'].items()};assert groups==read(n.PARENT/'maps'/key/'geometry_graph/groups.json')
        checks[key]=dict(full_geometry_classification_groups_exact=True)
    source=np.arange(60,dtype=float).reshape(20,3);cells=np.arange(20);base=dict(xyz=np.empty((0,3)),instance=np.empty(0,np.int64),added_source_ids=np.empty(0,np.int64));cls=dict(objects=[])
    g,c,_,_=n.export_reference(base,cls,source,[],[],cells);assert len(g['xyz'])==0 and c['objects']==[]
    # Empty native core must not disable the already specified raw-query birth.
    meta=[dict(old_object_id=-1,hypothesis_index=f,seed_frame=f,semantic_id=26,score=.2) for f in range(3)]
    g,c,_,_=n.export_reference(base,cls,source,[np.arange(20)]*3,meta,cells);np.testing.assert_array_equal(g['xyz'],source);assert len(c['objects'])==1
    for count in [np.array([],int),np.array([1]),np.array([3,1,5])]:
        new=np.repeat(np.r_[0,np.cumsum(count)][:len(count)],count)
        if len(count):np.testing.assert_array_equal(new,np.repeat(np.r_[0,np.cumsum(count)[:-1]],count))
        else:assert len(new)==0
    dump(RECORD,dict(complete=True,created_epoch=time.time(),source_sha256=sha(__file__),prior_amendment_sha256=sha(corrected.AMENDMENT),changes=PATCHES,golden=checks,empty_tests=['empty candidate slots','empty geometry export','three-frame raw birth from empty core'],reason='Actual0007nativeCGfilteredallobjects; empty candidate reduction crashed. Keep scene and allGT, propagate empty inputs without fake owners or changed acceptance thresholds.',reference_labels_decoded=False))
    print('EMPTY INPUT AMENDMENT SEALED; OLD FULL MAPS EXACT',flush=True)


def recover():
    assert sha(__file__)==read(RECORD)['source_sha256']
    dest=OUT/'runtime_attempts/candidates_attempt1';dest.mkdir(parents=True,exist_ok=False)
    shutil.move(str(OUT/'stages/candidates.json'),str(dest/'candidates.json'));shutil.copyfile('/tmp/omni_assessment_candidates.log',dest/'stderr.log')
    for sub in ['ownership_axis','ownership_intrinsic']:
        folder=OUT/sub/'features'
        if folder.exists():shutil.move(str(folder),str(dest/sub))
        assert not (OUT/sub/'features_complete.json').exists()
    dump(dest/'terminal.json',dict(handle=65309,exit_code=1,reason='Empty candidate slot array broadcast',partial_outputs_preserved=True,scientific_parameters_unchanged=True))


if __name__=='__main__':
    if sys.argv[1]=='freeze_empty':freeze()
    elif sys.argv[1]=='recover_empty':recover()
    else:
        r=read(RECORD);assert r['complete'] and sha(__file__)==r['source_sha256'] and sha(corrected.AMENDMENT)==r['prior_amendment_sha256']
        original.isolated_adapter=adapted
        corrected.main()
