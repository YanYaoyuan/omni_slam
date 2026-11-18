"""Post-score prompt correctness and mask compliance; no new model calls."""
import runpy
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from probe_world_anchor_tracking import OUT,INPUT,KEYS,METHODS,verify,prepared
from evaluate_kitti360_development import transfer,read_static_ply


def main():
    p=verify();cases=prepared()['cases'];s=read(OUT/'summary.json');seal=read(OUT/'predictions_complete.json')
    assert s['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    tax=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label'];results={}
    for scene in p['scenes']:
        key=scene['key'];ref=np.load(scene['reference']);raw=read_static_ply(p['references'][key]['raw']);train=np.array([tax[int(v)].trainId for v in raw['semantic']]);xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz']
        nn=transfer(ref['xyz'],np.arange(1,len(train)+1),xyz,.2);ix=np.maximum(nn-1,0);known=(nn>0)&(train[ix]!=255)&(ref['confidence'][ix]>0);truth=ref['train_instance'][ix];rows=[]
        for c in [v for v in cases if v['key']==key]:
            gt=next(v['gt_target'] for v in s['results'][key]['cases'] if v['case']==c['name']);masks={m:np.load(OUT/'cases'/c['name']/(m+'.npz')) for m in METHODS};frames=[]
            for j in [0]+c['common_frames']:
                q=c['prompts'][j];ids=np.array(q['source_ids']);labels=np.array(q['labels']);xy=np.array(q['pixels']);valid=known[ids]&(gt is not None);correct=(truth[ids]==gt)==(labels==1) if gt is not None else np.zeros(len(ids),bool)
                compliance={};areas={}
                for m,d in masks.items():
                    packed=d['masks_packed'][j];inside=((packed[xy[:,1],xy[:,0]//8]>>(7-xy[:,0]%8))&1).astype(bool);compliance[m]=int((inside==(labels==1)).sum());areas[m]=int(np.unpackbits(packed,axis=1)[:,:int(d['width'])].sum())
                contradictions=int(sum(labels[i]!=labels[k] and np.array_equal(xy[i],xy[k]) for i in range(len(ids)) for k in range(i)))
                frames.append(dict(frame_index=j,frame_id=c['records'][j]['frame_id'],prompts=len(ids),known=int(valid.sum()),wrong_positive=int((valid&~correct&(labels==1)).sum()),wrong_negative=int((valid&~correct&(labels==0)).sum()),unknown=int((~valid).sum()),pixel_contradictions=contradictions,compliance=compliance,mask_area=areas))
            rows.append(dict(case=c['name'],gt_target=gt,frames=frames))
        results[key]=rows
    dump(OUT/'prompt_audit.json',dict(complete=True,source_sha256=sha(__file__),results=results,new_inference=False,new_map=False,
         scope='Post-score nearest measured-source GT and prompt adherence, including seed separately. Predicted anchors are not truth. This diagnostic alone does not prove a cause for temporal correction failure; unequal decoding and native current-mask/memory interactions remain possible factors.'))
    print('PROMPT AUDIT COMPLETE',flush=True)


if __name__=='__main__':main()
