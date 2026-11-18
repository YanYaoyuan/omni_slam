"""Post-seal measured-observation diagnostics; no complete-map AP."""
import runpy
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from probe_world_anchor_tracking import OUT,INPUT,PARENT,KEYS,METHODS,verify,prepared
from evaluate_kitti360_development import transfer,read_static_ply


def rates(row):
    t,f,n=row['tp'],row['fp'],row['fn']
    return dict(precision=t/(t+f) if t+f else 0.,recall=t/(t+n) if t+n else 0.,f1=2*t/(2*t+f+n) if 2*t+f+n else 0.)


def main():
    p=verify();cases=prepared()['cases'];seal=read(OUT/'predictions_complete.json');assert seal['complete'];assert not (OUT/'summary.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    baseline=read(PARENT/'object_diagnostics.json')['results'];taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label'];results={};gate=len(cases)==8
    for scene in p['scenes']:
        key=scene['key'];ref=dict(np.load(scene['reference']));raw=read_static_ply(p['references'][key]['raw']);train=np.array([taxonomy[int(v)].trainId for v in raw['semantic']]);expected=train*1000+raw['instance'].astype(np.int64)%1000;expected[train==255]=0;np.testing.assert_array_equal(expected,ref['train_instance'])
        xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];nn=transfer(ref['xyz'],np.arange(1,len(train)+1),xyz,.2);ix=np.maximum(nn-1,0);known=(nn>0)&(train[ix]!=255)&(ref['confidence'][ix]>0);truth=ref['train_instance'][ix];weight=ref['confidence'][ix];anchor=np.load(OUT/f'{key}_anchors.npz');owners=anchor['owners'];cal=anchor['calibration'];rows=[];new={m:set() for m in METHODS}
        for case in [c for c in cases if c['key']==key]:
            oid=case['object_id'];base=next(o for o in baseline[key]['baseline'] if o['object_id']==oid);gid=base['overlaps'][0]['gt_id'] if base['overlaps'] else None;gate&=gid is not None;methods={}
            for method in METHODS:
                masks=np.load(OUT/'cases'/case['name']/(method+'.npz'));counts=dict(tp=0.,fp=0.,fn=0.,known_observations=0,unknown_selected=0,heldout_own_seen=0,heldout_own_selected=0,heldout_foreign_seen=0,heldout_foreign_selected=0);unique_new=set()
                for j in case['common_frames']:
                    assert masks['valid'][j];obs=np.load(case['records'][j]['file']);sid=obs['visible_source_ids'];xy=obs['pixel_xy'];packed=masks['masks_packed'][j];inside=((packed[xy[:,1],xy[:,0]//8]>>(7-xy[:,0]%8))&1).astype(bool);k=known[sid];correct=truth[sid]==gid if gid is not None else np.zeros(len(sid),bool);w=weight[sid]
                    if gid is not None:
                        counts['tp']+=float(w[k&inside&correct].sum());counts['fp']+=float(w[k&inside&~correct].sum());counts['fn']+=float(w[k&~inside&correct].sum());counts['known_observations']+=int(k.sum());unique_new.update(sid[k&inside&correct&(owners[sid]==0)].tolist())
                    counts['unknown_selected']+=int((inside&~k).sum());own=(owners[sid]==oid)&~cal[sid];foreign=(owners[sid]>0)&(owners[sid]!=oid)&~cal[sid]
                    for name,take in [('own',own),('foreign',foreign)]:counts['heldout_'+name+'_seen']+=int(take.sum());counts['heldout_'+name+'_selected']+=int((take&inside).sum())
                methods[method]=dict(**counts,**rates(counts),new_correct_unowned_sources=len(unique_new),gt_target_known=gid is not None);new[method].update(unique_new)
            rows.append(dict(case=case['name'],object_id=oid,semantic_id=case['semantic_id'],gt_target=gid,common_nonseed_frames=len(case['common_frames']),methods=methods));print('TRACKING DIAGNOSTIC',case['name'],gid,{m:round(v['f1'],4) for m,v in methods.items()},flush=True)
        valid=[r for r in rows if r['gt_target'] is not None];macro={m:float(np.mean([r['methods'][m]['f1'] for r in valid])) if valid else 0. for m in METHODS}
        gate&=macro['video_world_feedback']>macro['video_seed'] and macro['video_world_feedback']>=macro['image_reprompt'] and len(new['video_world_feedback'])>0
        results[key]=dict(cases=rows,macro_case_f1=macro,new_correct_unowned_sources={m:len(ids) for m,ids in new.items()},missing_gt_cases=len(rows)-len(valid))
    dump(OUT/'summary.json',dict(complete=True,results=results,capability_screen_pass=bool(gate),prediction_seal_sha256=sha(OUT/'predictions_complete.json'),source_sha256=sha(__file__),
        scope='Small predetermined object clips, GT used only after masks sealed. Confidence-weighted measured source-frame observations on common eligible nonseed frames; repeated sources are repeated observations. Fixed baseline best GT diagnostic, not GT prompts. Not full-map AP, independent evaluation or novel-method success. Feedback uses more decoder calls; no compute-matched novelty claim.'))
    print('CAPABILITY SCREEN',bool(gate),flush=True)


if __name__=='__main__':main()
