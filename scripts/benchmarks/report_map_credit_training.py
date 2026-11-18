"""Actual complete-map and per-object evidence for the fixed map-credit run."""
import html
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_map_credit_training import OUT,METHODS,verify

DEST=ROOT/'output/map_credit_training_v1'
NAMES={'baseline':'Frozen strong baseline','point':'Point supervision','object_balanced':'Object-balanced supervision',
       'mean_neighbor':'Map loss / mean neighbors','ordered_map':'Ours candidate / ordered map credit'}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from omni_semantic_map.surface_ownership import allocate
    from omni_semantic_map.spatial_acceptance_readout import hard_readout
    p=verify();s=read(OUT/'summary.json');assert s['complete']
    assert not (OUT/'delivery_audit.json').exists();DEST.mkdir(parents=True,exist_ok=True)
    seal=read(OUT/'predictions_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    allrows={};audits={};figures=[];sections=[]
    fig,axes=plt.subplots(2,4,figsize=(16,6),constrained_layout=True)
    for i,c in enumerate(p['contexts']):
        for j,method in enumerate(METHODS):
            t=read(OUT/'models'/c['key']/method/'training.json');axes[i,j].plot(np.arange(1,121),t['loss'],color='#008b84')
            axes[i,j].set_title(c['key']+' / '+method);axes[i,j].set_xlabel('Fixed full-batch update');axes[i,j].set_ylabel('Own training objective')
    fig.suptitle('All 8 fits / different loss values are not directly comparable / final step only')
    f=DEST/'training.png';fig.savefig(f,dpi=150);plt.close(fig);figures.append(str(f))
    for c in p['contexts']:
        key=c['key'];ref=np.load(c['reference']);truth=ref['train_instance'];weight=ref['confidence'];raw=np.load(Path(c['full'])/'geometry.npz')
        d=np.load(OUT/'inputs'/key/'candidates.npz');adj=dict(np.load(OUT/'inputs'/key/'adjacency.npz'))
        base=np.load(Path(c['ours'])/'geometry.npz');records=read(Path(c['ours'])/'classification.json')['objects']
        classes={r['object_id']:r['predictions']['mean']['train_id'] for r in records}
        mass_gt=np.bincount(truth,weights=weight);eligible=np.flatnonzero(np.isin(np.arange(len(mass_gt))//1000,[2,13])&(mass_gt>=100))
        rows={int(g):dict(gt_id=int(g),class_name='building' if g//1000==2 else 'car',methods={}) for g in eligible}
        maps={};checks={};methods=['baseline']+METHODS
        for method in methods:
            folder=Path(c['ours']) if method=='baseline' else OUT/'maps'/key/method
            g=np.load(folder/'geometry.npz');maps[method]=g
            predicted=np.load(OUT/'evaluation'/key/method/'reference_prediction.npz')['instance']
            if method!='baseline':
                z=np.load(folder/'decisions.npz');choice,confidence=allocate(z['logits'],d['valid'],d['owners'])
                np.testing.assert_array_equal(choice,z['owners'])
                # exp/normalization across the two NumPy builds can differ by
                # roundoff; require 4 ULP while choices remain bit-exact.
                np.testing.assert_array_max_ulp(confidence,z['confidence'],maxulp=4)
                selected=choice>0
                np.testing.assert_array_equal(g['xyz'][:len(base['xyz'])],base['xyz']);np.testing.assert_array_equal(g['instance'][:len(base['xyz'])],base['instance'])
                np.testing.assert_array_equal(g['xyz'][len(base['xyz']):],raw['xyz'][d['source_ids'][selected]])
                np.testing.assert_array_equal(g['instance'][len(base['xyz']):],choice[selected])
                np.testing.assert_array_equal(g['added_full_source_ids'],d['source_ids'][selected])
                np.testing.assert_array_equal(hard_readout(adj,choice),predicted)
                assert sha(folder/'classification.json')==sha(Path(c['ours'])/'classification.json')
                checks[method]=dict(old_map_exact=True,measured_additions_exact=True,full_projection_exact=True,
                                    decisions_exact=True,confidence_max_abs_difference=float(np.abs(confidence-z['confidence']).max()),
                                    confidence_within_4ulp=True,added_sources=int(selected.sum()))
            labels=int(max(classes))+1;table=np.bincount(truth*labels+predicted,weights=weight,minlength=len(mass_gt)*labels).reshape(-1,labels)
            pm=table.sum(0)
            for gid in eligible:
                ids=np.array([oid for oid,cls in classes.items() if cls==gid//1000],int)
                inter=table[gid,ids];iou=inter/np.maximum(1e-12,mass_gt[gid]+pm[ids]-inter)
                best=int(np.argmax(iou));rows[int(gid)]['methods'][method]=dict(best_iou=float(iou[best]),object_id=int(ids[best]))
        for row in rows.values():
            row['delta_vs_baseline']=row['methods']['ordered_map']['best_iou']-row['methods']['baseline']['best_iou']
        allrows[key]=list(rows.values());audits[key]=checks
        # Whole-window actual XY point maps, common coordinate bounds and sampling.
        fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
        displayed=['baseline','mean_neighbor','ordered_map']
        bounds=np.array([np.min([g['xyz'][:,:2].min(0) for g in maps.values()],axis=0),
                         np.max([g['xyz'][:,:2].max(0) for g in maps.values()],axis=0)])
        for ax,method in zip(axes,displayed):
            g=maps[method];idx=np.arange(0,len(g['xyz']),max(1,len(g['xyz'])//100000))
            ax.scatter(g['xyz'][idx,0],g['xyz'][idx,1],c=g['instance'][idx]%20,cmap='tab20',s=.2,vmin=0,vmax=19,rasterized=True)
            ax.set_title(NAMES[method]);ax.set_xlim(bounds[:,0]);ax.set_ylim(bounds[:,1]);ax.set_aspect('equal');ax.set_xlabel('World x / m');ax.set_ylabel('World y / m')
        fig.suptitle(key+' / complete predicted map, XY view / actual measured coordinates / colors may repeat')
        f=DEST/(key+'_maps.png');fig.savefig(f,dpi=160);plt.close(fig);figures.append(str(f))
        # A single shared RGB chosen by most visible added Ours sources (no GT).
        choice=np.load(OUT/'maps'/key/'ordered_map/decisions.npz')['owners'];added=d['source_ids'][choice>0]
        manifest=read(ROOT/'build/benchmarks/full_source_visual_visibility_audit_v1'/key/'manifest.json')
        views=[]
        for v in manifest['records']:
            vi=np.load(v['file']);views.append((int(np.isin(vi['visible_source_ids'],added).sum()),-v['frame_id'],v))
        _,_,selected=max(views,key=lambda x:x[:2]);view=np.load(selected['file'])
        frame=next(r for r in read(c['job'])['frames'] if r['frame_id']==selected['frame_id'])
        rgb=np.asarray(Image.open(frame['rgb']).convert('RGB'));fig,axes=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
        for ax,method in zip(axes,['point','mean_neighbor','ordered_map']):
            dc=np.load(OUT/'maps'/key/method/'decisions.npz');sid=d['source_ids'][dc['owners']>0];keep=np.isin(view['visible_source_ids'],sid)
            px=view['pixel_xy'][keep];ax.imshow(rgb)
            if len(px):ax.scatter(px[:,0],px[:,1],s=3,c='#ff3377',alpha=.7)
            ax.set_title(NAMES[method]+f' / {len(px)} added-source observations');ax.axis('off')
        fig.suptitle(f'{key} / actual RGB {selected["frame_id"]} / magenta=added measured sources\nFrame chosen by largest visible Ours addition count, not reference quality')
        f=DEST/(key+'_rgb.png');fig.savefig(f,dpi=150);plt.close(fig);figures.append(str(f))
        body=''.join('<tr><td>'+str(row['gt_id'])+'</td><td>'+row['class_name']+'</td>'+''.join(f'<td>{row["methods"][m]["best_iou"]:.4f}</td>' for m in methods)+f'<td>{row["delta_vs_baseline"]:+.4f}</td></tr>' for row in rows.values())
        sections.append(f'<h2>{key}</h2><img src="{key}_maps.png"><img src="{key}_rgb.png"><h3>Every eligible reference object; best same-class IoU, not AP matching</h3><table><tr><th>GT</th><th>Class</th>'+''.join(f'<th>{html.escape(NAMES[m])}</th>' for m in methods)+'<th>Ours delta</th></tr>'+body+'</table>')
    dump(OUT/'per_object.json',dict(complete=True,results=allrows,scope='Best same-class IoU for all eligibleGT, separate from AP matching'))
    table=''.join('<tr><td>'+html.escape(NAMES[m])+'</td>'+''.join(f'<td>{s["results"][c["key"]]["methods"][m]["metrics"]["allAp"]*100:.4f}</td>' for c in p['contexts'])+f'<td>{s["mean_primary_ap"][m]*100:.4f}</td></tr>' for m in ['baseline']+METHODS)
    page='<!doctype html><meta charset="utf-8"><title>Object-map credit training</title><style>body{max-width:1450px;margin:40px auto;font:16px sans-serif;color:#173345}img{width:100%}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border-bottom:1px solid #ccd6db;padding:9px;text-align:right}td:first-child,th:first-child{text-align:left}p{line-height:1.6}</style><h1>omni_slam / map-credit training experiment</h1><p>Eight fixed fits, current strong baseline, identical candidates/network/update count. Ours candidate is ordered spatial credit. Fixed-ID geometry admission only: no identity birth, historical reassignment, new independent scene or robot validation.</p><p>Exposed KITTI-360 development windows; inherited stacking prevents independent-cross-validation claims. Local adapted AP50:90, not leaderboard. Every prediction sealed before new AP scoring. Development gate: '+str(s['development_gate'])+'.</p><table><tr><th>Method</th><th>0000 AP%</th><th>0002 AP%</th><th>Mean AP%</th></tr>'+table+'</table><img src="training.png">'+''.join(sections)
    (DEST/'index.html').write_text(page)
    dump(OUT/'delivery_audit.json',dict(complete=True,results=audits,all_objects=sum(len(v) for v in allrows.values()),figures=figures,
        files={str(f):sha(f) for f in [OUT/'per_object.json',DEST/'index.html',*map(Path,figures)]},
        full_projection_replays=8,old_baseline_unchanged=True,goal_complete=False))
    print('MAP CREDIT REPORT',len(figures),'figures',sum(len(v) for v in allrows.values()),'GT objects;8exactprojectionreplays',flush=True)


if __name__=='__main__':main()
