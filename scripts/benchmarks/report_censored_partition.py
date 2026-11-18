"""Actual maps/RGB and every object from the frozen full-window comparison."""
import html
from pathlib import Path
import runpy
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from evaluate_censored_partition import OUT,verify_maps,METHODS
from evaluate_kitti360_development import transfer,read_static_ply
from omni_semantic_map.kitti360_reference_identity import encode_reference

DEST=ROOT/'output/censored_object_partition_v1'
NAMES={'baseline':'Frozen strong baseline','pairwise_history':'Fixed pairwise + history',
       'region_current':'Region / current only','region_history':'Ours candidate / region + history'}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=verify_maps();s=read(OUT/'map_summary.json');audit=read(OUT/'delivery_audit.json')
    assert s['complete'] and audit['complete'];assert not (OUT/'report_audit.json').exists()
    DEST.mkdir(parents=True,exist_ok=False)
    taxonomy=runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['id2label']
    details={};figures=[];parts=[];visuals={}
    for c in p['contexts']:
        key=c['key'];ref=dict(np.load(c['reference']));raw=read_static_ply(c['raw_reference'])
        physical,_=encode_reference(raw['semantic'],raw['instance'],taxonomy,'target_only')
        gtsets={'primary':ref['train_instance'],'raw_targets':physical};tables={}
        for mode,truth in gtsets.items():
            mass=np.bincount(truth,weights=ref['confidence'])
            gt=np.flatnonzero(np.isin(np.arange(len(mass))//1000,[2,13])&(mass>=100))
            tables[mode]={int(g):dict(gt_id=int(g),class_name='building' if g//1000==2 else 'car',methods={}) for g in gt}
        maps={};predictions={}
        for method in METHODS:
            path=Path(c['ours']) if method=='baseline' else OUT/'maps'/key/method
            maps[method]=dict(np.load(path/'geometry.npz'));records=read(path/'classification.json')['objects']
            classes={r['object_id']:r['predictions']['mean']['train_id'] for r in records}
            predicted=np.load(OUT/'evaluation'/key/method/'reference_prediction.npz')['instance'];predictions[method]=predicted
            pm=np.bincount(predicted,weights=ref['confidence'],minlength=max(classes)+1)
            for mode,truth in gtsets.items():
                mass=np.bincount(truth,weights=ref['confidence'])
                for gid,row in tables[mode].items():
                    take=truth==gid;inter=np.bincount(predicted[take],weights=ref['confidence'][take],minlength=len(pm))
                    ids=np.array([o for o,cls in classes.items() if cls==gid//1000],int)
                    values=inter[ids]/np.maximum(1e-12,mass[gid]+pm[ids]-inter[ids])
                    best=int(np.argmax(values));row['methods'][method]=dict(best_iou=float(values[best]),object_id=int(ids[best]))
        details[key]={mode:list(rows.values()) for mode,rows in tables.items()}
        bounds=np.stack([np.min([g['xyz'][:,:2].min(0) for g in maps.values()],axis=0),
                         np.max([g['xyz'][:,:2].max(0) for g in maps.values()],axis=0)])
        fig,axes=plt.subplots(1,4,figsize=(18,5),constrained_layout=True)
        for ax,method in zip(axes,METHODS):
            g=maps[method];ids=np.arange(0,len(g['xyz']),max(1,len(g['xyz'])//100000))
            ax.scatter(g['xyz'][ids,0],g['xyz'][ids,1],c=g['instance'][ids]%20,cmap='tab20',s=.3,vmin=0,vmax=19,rasterized=True)
            ax.set_xlim(bounds[:,0]);ax.set_ylim(bounds[:,1]);ax.set_aspect('equal');ax.set_title(NAMES[method],fontsize=10)
            ax.set_xlabel('World x / m');ax.set_ylabel('World y / m')
        fig.suptitle(key+' / complete actual exported maps / common bounds / palette repeats; new IDs are not aligned across methods')
        f=DEST/(key+'_maps.png');fig.savefig(f,dpi=160);plt.close(fig);figures.append(str(f))
        views=read(OUT/'prepared'/key/'views.json');selected=views[len(views)//2];v=np.load(selected['file'])
        frame=next(x for x in read(c['job'])['frames'] if x['frame_id']==selected['frame_id'])
        rgb=np.asarray(Image.open(frame['rgb']).convert('RGB'));full=np.load(Path(c['full'])/'geometry.npz')['xyz']
        source=full[v['source_ids']];xy=v['pixel_xy'];fig,axes=plt.subplots(1,3,figsize=(18,3),constrained_layout=True)
        for ax,method in zip(axes,['baseline','region_current','region_history']):
            g=maps[method];label=transfer(g['xyz'],g['instance'],source,.05);take=label>0
            ax.imshow(rgb);ax.scatter(xy[take,0],xy[take,1],c=label[take]%20,cmap='tab20',s=1,vmin=0,vmax=19,alpha=.75)
            ax.set_title(NAMES[method],fontsize=11);ax.axis('off')
        fig.suptitle(f'{key} / actual RGB {selected["frame_id"]}, fixed median input frame / visible measured-source dots, NOT dense masks or GT')
        f=DEST/(key+'_rgb.png');fig.savefig(f,dpi=160);plt.close(fig);figures.append(str(f))
        visuals[key]=dict(frame_id=selected['frame_id'],selection='Fixed median original input frame, no GT-based selection',
                           display_transfer_radius_m=.05,evaluation_transfer_radius_m=.2)
        parts.append(f'<h2>{key}</h2><img src="{key}_maps.png"><img src="{key}_rgb.png">')
        rows=''
        for m in METHODS:
            r=s['results'][key][m]
            rows+=('<tr><td>'+NAMES[m]+'</td>'+''.join(f'<td>{r[mode]["metrics"]["allAp"]*100:.4f}</td>' for mode in ['primary','raw_targets','uniform_scores'])+
                   '<td>'+html.escape(str(r['raw_targets']['changes_vs_baseline']))+'</td></tr>')
        parts.append('<table><tr><th>Method</th><th>Primary AP</th><th>Raw-target AP</th><th>Uniform-score AP</th><th>Raw AP50 gains/losses</th></tr>'+rows+'</table>')
        for mode,rows in tables.items():
            body=''
            for gid,r in rows.items():
                delta=r['methods']['region_history']['best_iou']-r['methods']['baseline']['best_iou']
                body+='<tr><td>'+str(gid)+'</td><td>'+r['class_name']+'</td>'+''.join(f'<td>{r["methods"][m]["best_iou"]:.5f}</td>' for m in METHODS)+f'<td>{delta:+.5f}</td></tr>'
            parts.append('<details><summary>Every eligible '+mode+' object: best same-class IoU, not AP matching</summary><table><tr><th>GT</th><th>Class</th>'+''.join('<th>'+NAMES[m]+'</th>' for m in METHODS)+'<th>Ours delta</th></tr>'+body+'</table></details>')
    table=''.join('<tr><td>'+NAMES[m]+'</td>'+''.join(f'<td>{s["mean_ap"][mode][m]*100:.6f}</td>' for mode in ['primary','raw_targets','uniform_scores'])+'</tr>' for m in METHODS)
    page='''<!doctype html><meta charset="utf-8"><title>omni_slam object partition experiment</title>
<style>body{max-width:1500px;margin:40px auto;padding:0 20px;font:16px/1.5 sans-serif;color:#153443}img{width:100%}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #d8e0e4;text-align:right}td:first-child,th:first-child{text-align:left}details{margin:20px 0;overflow:auto}.note{background:#fff0db;padding:16px}</style>
<h1>omni_slam: whole-region object partition and historical revision</h1>
<p class="note">Development gate: '''+str(s['development_gate'])+'''. Six new complete maps and 24 complete evaluations. A lower partition objective or more object IDs is not an accuracy improvement. The frozen strong baseline remains the mainline. These two windows are exposed development data, not independent tests or official leaderboard scores.</p>
<p>The candidate stores measured-cell / view / mask evidence, treats unknown views as absent evidence, and recomputes complete-region categorical compatibility when proposing merges. Historical IDs are not immutable. The additive control shares all observations and proposals, but sums frozen initial pair costs. Current-only and history variants share the exact new geometry, identities, scores and transport; only old point labels differ.</p>
<table><tr><th>Method</th><th>Primary mean AP50:90 (%)</th><th>Raw-target diagnostic</th><th>Uniform-score diagnostic</th></tr>'''+table+'</table>'+''.join(parts)+'''
<p>Pre-map capacity was a necessary coverage diagnostic, not expected map accuracy. Its reach model uses hard semantic-background barriers and only full raw-source geometry; the final exporter uses witnessed background seeds and also retains legacy XYZ. Do not treat those reach values as a strict upper bound for every final exported geometry. The direct measured-component coverage and actual new/lost AP50 matches are reported separately. No GT-assisted subset, seed or mask selection is used by inference.</p>
<p>Six complete exports replayed; eight actual single-frame-batch removals preserve every certified label. These checks establish the stated implementation, not semantic correctness. No new robot or independent-scene evaluation is included in this stage.</p>'''
    (DEST/'index.html').write_text(page)
    dump(OUT/'per_object.json',dict(complete=True,results=details,scope='All eligible objects under both fixed reference definitions; independent best-IoU diagnostic'))
    dump(OUT/'report_audit.json',dict(complete=True,figures=figures,visuals=visuals,source_sha256=sha(__file__),
        summary_sha256=sha(OUT/'map_summary.json'),files={str(f):sha(f) for f in DEST.iterdir()},
        per_object_sha256=sha(OUT/'per_object.json')))
    print('OBJECT PARTITION REPORT',DEST/'index.html',flush=True)


if __name__=='__main__':main()
