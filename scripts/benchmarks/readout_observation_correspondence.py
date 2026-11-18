"""Sealed correspondence readout; native projection and LK remain controls."""
from pathlib import Path
import time
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from probe_observation_correspondence import OUT, inside, verify

DEST = ROOT / 'output/observation_correspondence_probe_v1'
METHODS = ['geometry', 'lk', 'vggt']


def describe(values):
    a=np.asarray(values);a=a[np.isfinite(a)]
    return dict(n=len(a),median=float(np.median(a)) if len(a) else None,
                p90=float(np.quantile(a,.9)) if len(a) else None)


def semantic_stats(labels, truth, confidence, domain):
    take=domain & (truth>=0) & (truth<19) & (confidence>0)
    mass=float(confidence[take].sum());result=dict(observations=int(take.sum()),confidence_mass=mass,methods={})
    for name in METHODS:
        correct=labels[name]==truth
        row=dict(correct=int((correct&take).sum()),
                 accuracy=float(confidence[take&correct].sum()/mass) if mass else None)
        if name!='geometry':
            old=labels['geometry']==truth
            row.update(corrections=int((take&correct&~old).sum()),regressions=int((take&~correct&old).sum()),
                       changed=int((take&(labels[name]!=labels['geometry'])).sum()))
        result['methods'][name]=row
    return result


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy.spatial import cKDTree
    p=verify();prep=read(OUT/'prepared.json');pred=read(OUT/'predictions_complete.json')
    assert pred['complete'] and not (OUT/'readout.json').exists()
    sealed={**prep['files'],**pred['files']}
    for f,h in sealed.items():assert sha(f)==h,f
    # Freeze this readout and reference hashes before looking at reference arrays.
    refs={c['key']:c['reference'] for c in p['contexts'] if c['reference']}
    refhash={f:sha(f) for f in refs.values()}
    if not (OUT/'readout_protocol.json').exists():
        dump(OUT/'readout_protocol.json',dict(created_epoch=time.time(),source_sha256=sha(__file__),references=refhash,
            predictions_sha256=sha(OUT/'predictions_complete.json'),
            statistic='Confidence-weighted source-observation class accuracy; frozen strict20cm reference nearest-neighbor, known positive train_instance and confidence. Repeated frames/sources are not independent. Class correctness is not instance association correctness or full-map AP.',
            common='Non-seed observations, measured-depth-visible in existing cache, positive camera z, both complete-clip tracker validity masks. Same labels and source GT for all methods. Also report per-tracker valid domains and assigned/unassigned strata; no GT-dependent selection.',
            gate='On common domain in EACH public window, native accuracy strictly greater than geometry and LK, and >=1 corrected class observation relative to geometry. No domain means fail, not replacement. Necessary information screen only.',
            display='All9clips, final frame, at most32 uniformly indexed sources regardless of validity/outcome. Geometry projection, LK and VGGT arrows in original RGB. All raw data retained.'))
    rp=read(OUT/'readout_protocol.json');assert rp['source_sha256']==sha(__file__)
    assert rp['references']==refhash
    DEST.mkdir(parents=True,exist_ok=True);results=[];aggregate={};files={};figures=[];cache={}
    predictions={r['name']:r for r in pred['results']}
    for row in prep['results']:
        name=row['name'];key=row['key'];d=np.load(row['input']);o=np.load(predictions[name]['file'])
        n=len(d['source_ids']);coords=dict(geometry=d['projected_xy'],lk=o['lk_xy'],vggt=o['vggt_xy'])
        labels={};integer={};inbounds={}
        offset=.5 if row['kind']=='outdoor' else 0
        for method in METHODS:
            px=np.floor(coords[method]+offset).astype(np.int64)
            ok=inside(px,d['size']) & (d['camera_z']>0)
            a=np.full((3,n),-1,np.int16)
            for i,r in enumerate(row['observations']):
                cls=np.load(r['file'])['class_map'];assert cls.shape==tuple(d['size'][::-1])
                a[i,ok[i]]=cls[px[i,ok[i],1],px[i,ok[i],0]]
            labels[method]=a;integer[method]=px;inbounds[method]=ok
        expected=np.floor(d['projected_xy'][0]+offset).astype(int)
        np.testing.assert_array_equal(expected,d['seed_xy'])
        base=d['measured_visible'].copy() & inbounds['geometry'];base[0]=False
        vg=base & o['vggt_valid'][None] & inbounds['vggt']
        lk=base & o['lk_valid'][None] & inbounds['lk'];common=vg&lk
        truth=np.full(n,-1,np.int16);confidence=np.zeros(n);reference_index=np.full(n,-1,np.int64)
        reference_distance=np.full(n,np.inf)
        if key in refs:
            if key not in cache:
                gt=np.load(refs[key]);cache[key]=(gt,cKDTree(gt['xyz']))
            gt,tree=cache[key];distance,idx=tree.query(d['xyz'],workers=1)
            raw=gt['train_instance'][idx];conf=gt['confidence'][idx]
            known=(distance<.2)&(raw>0)&(conf>0)&(raw//1000<19)
            truth[known]=raw[known]//1000;confidence[known]=conf[known]
            reference_index=idx;reference_distance=distance
        tt=np.broadcast_to(truth,(3,n));cc=np.broadcast_to(confidence,(3,n))
        stats=dict(name=name,key=key,frames=[f['frame_id'] for f in row['frames']],seeds=n,
                   assigned=int((d['owner']>0).sum()),source_initialization_max_error_native=float(np.max(np.linalg.norm(o['vggt_native'][0]-d['seed_native'],axis=1))),
                   vggt_valid=int(o['vggt_valid'].sum()),lk_valid=int(o['lk_valid'].sum()),
                   vggt_cycle=describe(o['vggt_cycle']),lk_cycle=describe(o['lk_cycle']),
                   measured_nonseed=int(base.sum()),common_nonseed=int(common.sum()),semantics={})
        for method,dom in [('vggt',vg),('lk',lk)]:
            residual=np.linalg.norm((coords[method]-coords['geometry'])*d['scale'],axis=-1)
            stats[method+'_projection_disagreement_all_native']=describe(residual[base])
            stats[method+'_projection_disagreement_valid_native']=describe(residual[dom])
        if key in refs:
            for domain,dom in [('common',common),('vggt_valid',vg),('lk_valid',lk),
                               ('common_assigned',common&(d['owner']>0)[None]),('common_unassigned',common&(d['owner']==0)[None])]:
                stats['semantics'][domain]=semantic_stats(labels,tt,cc,dom)
        results.append(stats)
        path=OUT/'readout'/(name+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,source_ids=d['source_ids'],truth=truth,confidence=confidence,
            reference_index=reference_index,reference_distance=reference_distance,owner=d['owner'],
            common=common,base=base,vggt_valid_domain=vg,lk_valid_domain=lk,**{m+'_class':v for m,v in labels.items()})
        files[str(path)]=sha(path)
        aggregate.setdefault(key,[]).append(dict(labels=labels,truth=tt,confidence=cc,common=common,vg=vg,lk=lk,owner=d['owner']))
        with Image.open(row['frames'][-1]['rgb']) as im: rgb=np.asarray(im.convert('RGB'))
        fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        ix=np.linspace(0,n-1,min(32,n),dtype=int);seed=d['seed_xy'][ix];g=coords['geometry'][-1,ix]
        for ax,method in zip(axes,METHODS):
            ax.imshow(rgb);q=coords[method][-1,ix];vis=inside(q,d['size'])&inside(g,d['size'])
            ax.scatter(g[vis,0],g[vis,1],s=12,facecolors='none',edgecolors='#fff447',linewidths=.7)
            if method!='geometry':
                delta=q-g
                ax.quiver(g[vis,0],g[vis,1],delta[vis,0],delta[vis,1],color='#ff377d',angles='xy',scale_units='xy',scale=1,width=.0025)
                ax.scatter(q[vis,0],q[vis,1],s=9,c='#1af2c2')
            ax.set(xlim=(0,d['size'][0]),ylim=(d['size'][1],0),title=method+' / all sampled tracks, including rejected')
            ax.axis('off')
        fig.suptitle(f'{name} / frames {stats["frames"]} / final RGB: yellow=geometry, green=tracked\nDisagreement is NOT correspondence ground truth or map improvement')
        f=DEST/(name+'.png');fig.savefig(f,dpi=140);plt.close(fig);figures.append(str(f));files[str(f)]=sha(f)
        print('CORRESPONDENCE READOUT',name,'common',stats['common_nonseed'],stats['semantics'].get('common'),flush=True)
    public_gate=True;summary={}
    for key,rows in aggregate.items():
        s=dict(clips=len(rows));summary[key]=s
        if key not in refs: s.update(has_reference=False,screen_gate=None);continue
        lab={m:np.concatenate([r['labels'][m].ravel() for r in rows]) for m in METHODS}
        t=np.concatenate([r['truth'].ravel() for r in rows]);cf=np.concatenate([r['confidence'].ravel() for r in rows])
        for domain in ['common','vg','lk']:
            dom=np.concatenate([r[domain].ravel() for r in rows]);s[domain]=semantic_stats(lab,t,cf,dom)
        met=s['common']['methods'];native=met['vggt']['accuracy']
        ok=native is not None and native>met['geometry']['accuracy'] and native>met['lk']['accuracy'] and met['vggt']['corrections']>=1
        s['screen_gate']=bool(ok);public_gate &= ok
    for f,h in sealed.items():assert sha(f)==h,f
    dump(OUT/'readout.json',dict(complete=True,results=results,summary=summary,screen_gate=bool(public_gate),files=files,
        figures=figures,new_maps=0,new_ap=0,independent_scenes=0,robot_scope='3fixedclips only; no semanticGT',
        protocol_sha256=sha(OUT/'protocol.json'),predictions_sha256=sha(OUT/'predictions_complete.json')))
    import html
    blocks=''.join(f'<h2>{html.escape(r["name"])}</h2><img src="{Path(f).name}"><pre>{html.escape(str(r["semantics"].get("common", "Robot: no semantic GT")))}</pre>' for r,f in zip(results,figures))
    (DEST/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Observation correspondence audit</title><style>body{max-width:1400px;margin:40px auto;font-family:sans-serif}img{width:100%}pre{white-space:pre-wrap}</style><h1>Measured projection vs native point correspondence</h1><p>All 9 predefined clips. Existing exposed windows and robot clips; not full-map AP or independent assessment. Green tracks / yellow geometry / pink difference. Rejected tracks also shown.</p><pre>'+html.escape(str(summary))+'</pre>'+blocks)
    print('CORRESPONDENCE SCREEN GATE',bool(public_gate),flush=True)


if __name__=='__main__': main()
