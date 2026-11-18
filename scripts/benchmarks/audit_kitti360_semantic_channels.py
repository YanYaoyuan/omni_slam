"""Diagnose cached semantic channels on ALL final objects, no new inference.

The two already-exposed windows are post-hoc diagnostic data, not independent
validation. Channel readouts are ordinary controls, not a new method.
"""
from pathlib import Path
import time
import numpy as np
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_kitti360_reserved import OUT,META,verify
from evaluate_kitti360_development import transfer,clean_json
from omni_semantic_map.kitti360_evaluation import evaluate

DEST=ROOT/'build/benchmarks/kitti360_semantic_channel_audit_v1'
REPORT=ROOT/'output/kitti360_retention_audit_v1'
CHANNELS=['learned','global','masked','bbox','uniform_triplets']


def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def main():
    assert not (DEST/'summary.json').exists()
    protocol=verify();seal=read(OUT/'predictions_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    dump(DEST/'protocol.json',dict(created_epoch=time.time(),source_sha256=sha(__file__),
        sealed_input_sha256=sha(OUT/'predictions_complete.json'),posthoc=True,channels=CHANNELS,
        scope='Readout-only diagnostic on two exposed windows. Same baseline geometry/ownership/selected observation IDs/19 class text vocabulary. Not independent validation or end-to-end remapping.',
        control='Uniform mean of original three cached triplets per observation, normalized, then same across-view mean. Each single channel uses stored vectors without new inference. No prompts, thresholds or scene selection changed.',
        order='Seal all channel readouts for both windows before this script scores references. GT already exposed by prior audits; this seal does not restore independence.',
        gt_policy='No GT-dependent class reassignment. Existing retention GT diagnostic used only after predictions to caption visual errors.',
        decision='Diagnose information loss and establish mandatory masked/bbox controls for subsequent proposals. Do not promote simple channel replacement as innovation.'))
    files={};scenes=[]
    for w in protocol['windows']:
        scene=w['scene'];root=OUT/scene;cache=root/'connected/cache';m=read(cache/'manifest.json')
        rows=read(root/'base/maps/cg/history2/classification.json')['objects']
        trip=np.concatenate([np.load(root/'frontend'/scene/'triplets'/r['file'])['triplets'] for r in m['records']]).astype(np.float64)
        learned=np.concatenate([np.load(cache/'frames'/r['file'])['features'] for r in m['records']]).astype(np.float64)
        channels=dict(learned=learned,global_=trip[:,0],masked=trip[:,1],bbox=trip[:,2],uniform_triplets=norm(trip.mean(1)))
        channels['global']=channels.pop('global_')
        text=norm(np.load(cache/'text_features.npy').astype(np.float64));vocab=read(cache/'vocabulary.json')
        outcomes=[]
        for r in rows:
            selected=r['selected_observation_ids'];by_channel={}
            for name in CHANNELS:
                descriptor=norm(channels[name][selected].mean(0));scores=text@descriptor
                order=np.argsort(-scores,kind='stable');best=int(order[0]);sem=int(vocab['semantic_ids'][best])
                by_channel[name]=dict(name=vocab['names'][best],semantic_id=sem,score=float(scores[best]),
                    retained=sem in vocab['target_semantic_ids'],top3=[dict(name=vocab['names'][int(k)],score=float(scores[k])) for k in order[:3]])
            assert by_channel['learned']['name']==r['predictions']['mean']['name']
            assert abs(by_channel['learned']['score']-r['predictions']['mean']['score'])<1e-6
            # Keep original score exactly for the reproduction control.
            by_channel['learned'].update(r['predictions']['mean'])
            outcomes.append(dict(object_id=r['object_id'],selected_observation_ids=selected,
                observation_ids=r['observation_ids'],readouts=by_channel))
        path=DEST/scene/'readouts.json';dump(path,outcomes);files[str(path)]=sha(path)
        scenes.append(dict(scene=scene,rows=outcomes))
    dump(DEST/'readouts_complete.json',dict(complete=True,files=files,protocol_sha256=sha(DEST/'protocol.json')))
    for s in scenes:
        scene=s['scene'];root=OUT/scene;geom=np.load(root/'base/maps/cg/history2/geometry.npz')
        ref=np.load(root/'evaluation/reference.npz')
        # Transfer ALL object IDs once; target filtering can then be applied
        # without changing nearest-neighbor selection or nontarget blocking.
        all_ids=transfer(geom['xyz'],geom['instance'],ref['xyz'],.2);scores={}
        for name in CHANNELS:
            pred={r['object_id']:r['readouts'][name] for r in s['rows'] if r['readouts'][name]['retained']}
            projected=np.where(np.isin(all_ids,list(pred)),all_ids,0)
            result=evaluate(ref['train_instance'],ref['confidence'],projected,pred,META/'evalInstanceLevelSemanticLabeling.py')
            scores[name]=dict(averages=clean_json(result['averages']),
                eligible_predictions={k:len(v) for k,v in result['matches']['window']['prediction'].items()})
            dump(DEST/scene/(name+'_matches.json'),clean_json(result['matches']))
            if name=='learned':
                np.testing.assert_array_equal(projected,np.load(root/'evaluation/history2_prediction.npz')['instance'])
                assert scores[name]['averages']==read(root/'evaluation/summary.json')['methods']['history2']['full_window']['averages']
            print(scene,name,scores[name],flush=True)
        s['scores']=scores;dump(DEST/scene/'results.json',s)
    dump(DEST/'summary.json',dict(complete=True,posthoc=True,scenes=scenes,
        protocol_sha256=sha(DEST/'protocol.json'),readout_seal_sha256=sha(DEST/'readouts_complete.json')))
    # Include ALL final objects, one deterministic largest-area selected view each.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    REPORT.mkdir(parents=True,exist_ok=True)
    for s in scenes:
        root=OUT/s['scene'];m=read(root/'connected/cache/manifest.json');job=read(root/'base/job.json')
        obs={rec['observation_offset']+j:(i,rec,j) for i,rec in enumerate(m['records']) for j in range(rec['masks'])}
        fig,axes=plt.subplots((len(s['rows'])+3)//4,4,figsize=(16,3.4*((len(s['rows'])+3)//4)),squeeze=False)
        captions=[]
        for ax,r in zip(axes.flat,s['rows']):
            choices=[]
            for oid in r['selected_observation_ids']:
                i,rec,j=obs[oid];f=np.load(root/'connected/cache/frames'/rec['file'])
                choices.append((int(f['areas'][j]),-oid,oid))
            oid=max(choices)[2];i,rec,j=obs[oid];f=np.load(root/'connected/cache/frames'/rec['file'])
            image=np.array(Image.open(job['frames'][i]['rgb']).convert('RGB'))
            mask=np.unpackbits(f['masks_packed'][j],axis=1)[:,:image.shape[1]].astype(bool)
            y,x=np.where(mask);x0=max(0,x.min()-30);x1=min(image.shape[1],x.max()+31)
            y0=max(0,y.min()-30);y1=min(image.shape[0],y.max()+31)
            crop=image[y0:y1,x0:x1].astype(float);b=mask[y0:y1,x0:x1]
            crop[b]=.72*crop[b]+.28*np.array([0,220,200]);ax.imshow(crop.astype(np.uint8))
            c=r['readouts'];ax.set_title(f"Object {r['object_id']} | frame {rec['frame_id']}\nLearned: {c['learned']['name']} | Global: {c['global']['name']}\nMasked: {c['masked']['name']} | Box: {c['bbox']['name']}",fontsize=9);ax.axis('off')
            captions.append(dict(object_id=r['object_id'],observation_id=oid,frame_id=rec['frame_id'],rgb=job['frames'][i]['rgb'],mask_index=j))
        for ax in axes.flat[len(s['rows']):]:ax.axis('off')
        fig.suptitle(s['scene']+' — ALL final objects; fixed largest-area selected view',fontsize=14);fig.tight_layout()
        fig.savefig(REPORT/(s['scene']+'_all_objects.png'),dpi=130);plt.close(fig)
        dump(REPORT/(s['scene']+'_captions.json'),captions)


if __name__=='__main__':main()
