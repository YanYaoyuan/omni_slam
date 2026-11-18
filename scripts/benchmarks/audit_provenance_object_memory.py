"""Causal-image/provenance checks and exact map replay for identified memory."""
import argparse
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import read,dump,sha
from run_provenance_object_memory import OUT,INPUT,SEM,POOL,BASE,PREV,METHODS,verify,seal,unit_rows
from audit_mask_observation_code import source_anchors
from omni_semantic_map.persistent_source_memory import sample_patch_features


def inputs():
    p=verify();seal('features_complete');results={}
    for s in p['scenes']:
        key=s['key'];m=read(INPUT/key/'manifest.json');records=m['records'];source=np.load(m['geometry'])['xyz']
        base=np.load(BASE/key/'geometry_graph/geometry.npz');data=np.load(OUT/'features'/key/'candidates.npz');bank=np.load(OUT/'features'/key/'bank.npz')
        anchors=source_anchors(key,len(source));cell=np.load(INPUT/'query_cells'/key/'source_to_cell.npy')
        events=read(OUT/'features'/key/'provenance.json');by_frame={i:[] for i in range(len(records))}
        for e in events:by_frame[e['frame_index']].append(e)
        object_ids=bank['object_ids'];columns={int(o):i for i,o in enumerate(object_ids)}
        classes={o['object_id']:o['predictions']['mean']['train_id'] for o in read(BASE/key/'geometry_graph/classification.json')['objects']}
        old_last=np.full(len(base['xyz']),-1,int);old_xy=np.zeros((len(old_last),2),int);old_prob=np.zeros((len(old_last),19),np.float32)
        sid=data['new_source_ids'];lookup=np.full(len(source),-1,int);lookup[sid]=np.arange(len(sid))
        new_last=np.full(len(sid),-1,int);new_xy=np.zeros((len(sid),2),int);new_prob=np.zeros((len(sid),19),np.float32)
        grids=[];observed=np.zeros(bank['observed'].shape,bool);bank_rows=0
        for i,r in enumerate(records):
            d=np.load(SEM/'frames'/key/f'{r["frame_id"]:06d}.npz');grid=np.load(PREV/'features'/key/f'{r["frame_id"]:06d}.npz')['grid'];grids.append(grid)
            order=np.argsort(d['source_ids']);sorted_ids=d['source_ids'][order]
            for e in by_frame[i]:
                ids=np.asarray(e['source_ids'],np.int64);oid=e['object_id'];col=columns[oid]
                assert len(np.unique(ids))==len(ids) and np.all(anchors[ids]==oid)
                ix=order[np.searchsorted(sorted_ids,ids)];np.testing.assert_array_equal(d['source_ids'][ix],ids)
                features=unit_rows(sample_patch_features(grid,d['source_pixel_xy'][ix],(376,1408)))
                unique,inverse,count=np.unique(cell[ids],return_inverse=True,return_counts=True);weights=1/count[inverse];weights/=weights.sum()
                desc=unit_rows(np.sum(features*weights[:,None],axis=0,keepdims=True))[0]
                np.testing.assert_allclose(bank['descriptors'][col,i],desc,rtol=1e-6,atol=2e-7)
                np.testing.assert_allclose(bank['metadata'][col,i],[np.log1p(len(ids)),np.sum(d['source_probabilities'][ix,classes[oid]]*weights)],rtol=1e-6,atol=1e-7)
                assert e['cell_count']==len(unique);assert not observed[col,i];observed[col,i]=True;bank_rows+=len(ids)
            mi=d['map_ids'];old_last[mi]=i;old_xy[mi]=d['map_pixel_xy'];old_prob[mi]=d['map_probabilities']
            ix=lookup[d['source_ids']];good=ix>=0;selected=ix[good]
            new_last[selected]=i;new_xy[selected]=d['source_pixel_xy'][good];new_prob[selected]=d['source_probabilities'][good]
        np.testing.assert_array_equal(observed,bank['observed'])
        oi=data['old_indices'];n_old=len(oi);np.testing.assert_array_equal(oi,np.flatnonzero(old_last>=0))
        qf=np.r_[old_last[oi],new_last];qxy=np.r_[old_xy[oi],new_xy];qp=np.r_[old_prob[oi],new_prob]
        np.testing.assert_array_equal(data['query_frame'],qf);np.testing.assert_array_equal(data['query_xyz'],np.r_[base['xyz'][oi],source[sid]])
        np.testing.assert_array_equal(data['static'][:,:,12:31],np.broadcast_to(qp[:,None],data['static'][:,:,12:31].shape))
        candidate=data['owners'];valid=data['valid'];available=data['available'];separated=data['separated']
        col=np.zeros_like(candidate)
        for oid,c in columns.items():col[candidate==oid]=c
        expected=observed[col]&(candidate>0)[...,None];disjoint=np.abs(qf[:,None,None]-np.arange(len(records))[None,None])>=4
        expected_valid=(expected&disjoint).any(-1);expected_valid[:n_old][~expected_valid[:n_old,0]]=False
        np.testing.assert_array_equal(valid,expected_valid);np.testing.assert_array_equal(available,expected&valid[...,None])
        np.testing.assert_array_equal(separated,available&disjoint)
        np.testing.assert_array_equal(candidate[:n_old,0],base['instance'][oi])
        proposals=np.load(POOL/'features'/key/'candidates.npz');np.testing.assert_array_equal(sid,proposals['source_ids'])
        np.testing.assert_array_equal(candidate[n_old:,:proposals['owners'].shape[1]],np.where(proposals['valid'],proposals['owners'],0))
        assert not np.any(data['views'][~available])
        # Verify all added alternative old owners obey the frozen measured radius.
        for oid in object_ids:
            ii,jj=np.where((candidate[:n_old]==oid)&(np.arange(candidate.shape[1])[None]>0))
            if len(ii):assert np.all(cKDTree(source[anchors==oid]).query(data['query_xyz'][ii],workers=1)[0]<=.4)
        # Fixed uniformly spaced rows, with old/new namespaces both represented.
        samples=np.unique(np.r_[np.linspace(0,n_old-1,min(n_old,256),dtype=int),n_old+np.linspace(0,len(sid)-1,min(len(sid),256),dtype=int)])
        cosine_checks=0
        for fr in np.unique(qf[samples]):
            ii=samples[qf[samples]==fr];q=unit_rows(sample_patch_features(grids[fr],qxy[ii],(376,1408)))
            for j in range(candidate.shape[1]):
                good=valid[ii,j];rows=ii[good]
                if not len(rows):continue
                cos=np.einsum('bd,bvd->bv',q[good],bank['descriptors'][col[rows,j]])
                vv=available[rows,j]
                np.testing.assert_allclose(data['views'][rows,j,:,0][vv],cos[vv],rtol=8e-4,atol=5e-4)
                cosine_checks+=int(vv.sum())
        results[key]=dict(objects=len(object_ids),view_records=len(events),source_observations_replayed=bank_rows,
            query_rows=len(qf),old_namespace=n_old,new_namespace=len(sid),sampled_visual_cosines_verified=cosine_checks,
            all_eligibility_masks_exact=True,old_alternative_distances_verified=True,last_frame_semantics_exact=True,
            all_bank_descriptors_replayed=True,no_gt_read=True)
        print('PROVENANCE INPUT AUDIT',key,results[key],flush=True)
    dump(OUT/'input_audit.json',dict(complete=True,results=results,source_sha256=sha(__file__),features_sha256=sha(OUT/'features_complete.json')))


def maps():
    p=verify();prediction=seal('predictions_complete');results={}
    for s in p['scenes']:
        key=s['key'];base=np.load(BASE/key/'geometry_graph/geometry.npz');data=np.load(OUT/'features'/key/'candidates.npz')
        source=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];old_indices=data['old_indices'];n_old=len(old_indices)
        checked=np.zeros(len(base['xyz']),bool);checked[old_indices[data['valid'][:n_old].any(1)]]=True
        original={o['object_id']:o for o in read(BASE/key/'geometry_graph/classification.json')['objects']};rows={};decisions={}
        for method in METHODS:
            folder=OUT/'maps'/key/method
            if method=='baseline':
                for name in ['geometry.npz','classification.json']:assert sha(folder/name)==sha(BASE/key/'geometry_graph'/name)
                continue
            g=np.load(folder/'geometry.npz');choice=np.load(folder/'decisions.npz')['owners'];revised=base['instance'].copy()
            if method!='ours_append_only':revised[old_indices]=choice[:n_old]
            np.testing.assert_array_equal(g['revised_base_owners'],revised)
            np.testing.assert_array_equal(revised[~checked],base['instance'][~checked])
            keep=np.flatnonzero(revised>0);new=choice[n_old:]>0;ids=data['new_source_ids'][new]
            np.testing.assert_array_equal(g['retained_base_indices'],keep);np.testing.assert_array_equal(g['added_source_ids'],ids)
            np.testing.assert_array_equal(g['xyz'],np.r_[base['xyz'][keep],source[ids]])
            np.testing.assert_array_equal(g['instance'],np.r_[revised[keep],choice[n_old:][new]])
            assert len(np.unique(ids))==len(ids)
            objects=read(folder/'classification.json')['objects'];assert all(o==original[o['object_id']] for o in objects)
            assert {o['object_id'] for o in objects}==set(g['instance'])
            decisions[method]=choice
            rows[method]=dict(points=len(g['xyz']),unchallenged_old_preserved=True,coordinates_provenance_exact=True,identity_class_scores_unchanged=True)
        np.testing.assert_array_equal(decisions['ours_append_only'],decisions['ours_revision'])
        rows['differences_vs_ours']={m:int((v!=decisions['ours_revision']).sum()) for m,v in decisions.items() if m!='ours_revision'}
        results[key]=rows;print('PROVENANCE MAP AUDIT',key,rows,flush=True)
    dump(OUT/'map_audit.json',dict(complete=True,results=results,source_sha256=sha(__file__),prediction_sha256=sha(OUT/'predictions_complete.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['inputs','maps']);globals()[ap.parse_args().action]()
