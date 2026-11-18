"""Check all ten actual maps and distinguish no effect from cached outputs.

Only prediction/source artifacts are read. This must run before assessment
scoring, and does not require methods to differ when evidence gives equality.
"""
import itertools
import numpy as np
from paper_final_common import read, dump, sha
from run_batch_transport_assessment import OUT, FULL, SEM, BASE, METHODS, seal, verify
from run_batch_transport_assessment_corrected import verify_amendment, AMENDMENT
from run_batch_transport_assessment_empty import RECORD as EMPTY_AMENDMENT


def main():
    verify_amendment();p=verify(True);pred=seal(OUT/'predictions_complete.json')
    empty=read(EMPTY_AMENDMENT)
    assert empty['complete'] and sha(__import__('run_batch_transport_assessment_empty').__file__)==empty['source_sha256']
    assert not (OUT/'summary.json').exists(), 'Run pre-score audit before decoding references'
    assert not (OUT/'prediction_audit.json').exists()
    for name in ['masks_complete','features_complete','supports_complete','baseline_complete','fields_complete']:
        seal(OUT/(name+'.json'))
    results={}
    for scene in p['scenes']:
        key=scene['key'];base=np.load(BASE/key/'geometry_graph/geometry.npz');full=np.load(FULL/key/'geometry.npz');sem=np.load(SEM/'aggregate'/key/'source.npz');mapping=np.load(FULL/key/'old_to_full.npy');field=np.load(OUT/'fields'/key/'field.npz')
        owners=np.zeros(len(full['xyz']),np.int64);owners[mapping]=sem['owners'];n=len(base['xyz'])
        source_labels={};maps={}
        for method in METHODS:
            folder=OUT/'final_maps'/key/method;geometry=np.load(folder/'geometry.npz');assert sha(folder/'classification.json')==sha(BASE/key/'geometry_graph/classification.json')
            np.testing.assert_array_equal(geometry['xyz'][:n],base['xyz']);np.testing.assert_array_equal(geometry['instance'][:n],base['instance'])
            added=np.empty(0,np.int64);assignment=np.zeros(len(owners),np.int64)
            if method!='baseline':
                name={'same_cell':'same_cell','full_nearest':'nearest','full_two_cells':'cell','ours_frame':'frame'}[method]
                decision=field[name+'_owner'][field['source_to_cell']]
                expected=np.flatnonzero((owners==0)&(decision>0));added=geometry['added_full_source_ids']
                np.testing.assert_array_equal(added,expected);assert len(np.unique(added))==len(added)
                np.testing.assert_array_equal(geometry['xyz'][n:],full['xyz'][added]);np.testing.assert_array_equal(geometry['instance'][n:],decision[added]);assignment[added]=decision[added]
            else:assert len(geometry['xyz'])==n
            assert set(np.unique(geometry['instance']))==set(np.unique(base['instance']))
            assert np.isfinite(geometry['xyz']).all()
            source_labels[method]=assignment
            maps[method]=dict(points=len(geometry['xyz']),added_sources=len(added),objects=len(np.unique(geometry['instance'])),geometry_sha256=sha(folder/'geometry.npz'),original_geometry_identity_scores_preserved=True,added_actual_sources_exact=True)
        comparisons={}
        for a,b in itertools.combinations(METHODS,2):
            x,y=source_labels[a],source_labels[b]
            comparisons[a+' vs '+b]=dict(different_source_assignments=int(np.count_nonzero(x!=y)),only_first=int(((x>0)&(y==0)).sum()),only_second=int(((x==0)&(y>0)).sum()),different_positive_owner=int(((x>0)&(y>0)&(x!=y)).sum()),identical_export=bool(np.array_equal(x,y)))
        results[key]=dict(maps=maps,comparisons=comparisons)
        print('MAP DIFFERENCE AUDIT',key,comparisons,flush=True)
    dump(OUT/'prediction_audit.json',dict(complete=True,results=results,predictions_seal_sha256=sha(OUT/'predictions_complete.json'),correction_amendment_sha256=sha(AMENDMENT),empty_input_amendment_sha256=sha(EMPTY_AMENDMENT),source_sha256=sha(__file__),reference_labels_read=False,all_ten_maps_checked=True))


if __name__=='__main__':main()
