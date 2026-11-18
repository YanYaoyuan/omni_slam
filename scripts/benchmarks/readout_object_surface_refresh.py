"""Shared-domain diagnosis of newly acquired masks; never exports a new map."""
import time
from pathlib import Path
import html
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from probe_object_surface_refresh import OUT, verify

DEST=ROOT/'output/object_surface_refresh_probe_v1'


def freeze():
    assert not (OUT/'readout_protocol.json').exists()
    p=verify();files={str(OUT/'prepared.json'):sha(OUT/'prepared.json')}
    for c in p['contexts']:files[c['reference']]=sha(c['reference'])
    dump(OUT/'readout_protocol.json',dict(created_epoch=time.time(),inputs=files,source_sha256=sha(__file__),
        scope='Diagnose all16 frozen acquisition sites on EXACT same visible source domain inside crop. No new map/AP or independent benchmark claim. All predictions sealed before reference arrays read. No output chosen from GT.',
        prompt='Native highest-quality mask selection is retained even if a prompt fails. Report satisfaction; do not replace it with another decoder output or score fit.',
        reference='Nearest fixed annotated reference within strict20cm per real measured source. Zero/unknown label and nonpositive confidence excluded from known-support IoU but explicitly count selected unknown points. Repeated raw sources do not constitute independent evidence or reference area; report source-support quantities only.',
        target='Posthoc diagnostic target is largest confidence mass GT among original object anchor sources in the SAME crop. Require>=16 known anchor sources,strictly>half known anchor mass, and target trainclass matching declared car/building. Otherwise keep case with N/A, no fallback target. Selection uses noGT; this diagnostic correspondence cannot be used for inference or reused as held-out truth.',
        metrics='Both full and crop: weighted source-support IoU, supported correct/other/unknown unassigned sources, loss of old correct source support. Also all mask pixels and source disagreements, regardlessGT availability. Aggregate by window and category, include every N/A. No threshold search or promotion from these local quantities.'))
    print('SURFACE REFRESH READOUT FROZEN',flush=True)


def unpack(path):
    d=np.load(path);w,h=map(int,d['original_size'])
    masks=np.unpackbits(d['masks_packed'],axis=2,count=w).astype(bool)
    assert masks.shape[1:]==(h,w)
    selected=int(d['selected']);assert selected==int(np.argmax(d['scores']))
    return masks[selected],d


def run():
    from scipy.spatial import cKDTree
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=verify();rp=read(OUT/'readout_protocol.json');assert sha(__file__)==rp['source_sha256']
    assert not (OUT/'readout.json').exists()
    for f,h in rp['inputs'].items():assert sha(f)==h
    prep=read(OUT/'prepared.json');predictions=read(OUT/'predictions_complete.json');assert predictions['complete']
    for seal in [prep,predictions]:
        for f,h in seal['files'].items():assert sha(f)==h
    DEST.mkdir(parents=True,exist_ok=False);results={};all_rows=[];figures=[];files={}
    for c in p['contexts']:
        key=c['key'];ref=np.load(c['reference']);tree=cKDTree(ref['xyz']);full=np.load(Path(c['full'])/'geometry.npz')
        outputs={r['object_id']:r for r in predictions['results'][key]};results[key]=[]
        for row in prep['results'][key]['selected']:
            oid=row['object_id'];d=np.load(row['input']);ids=d['source_ids'];xy=d['pixel_xy'];owner=d['owner']
            distance,nearest=tree.query(full['xyz'][ids],workers=2)
            truth=ref['train_instance'][nearest].copy();weight=ref['confidence'][nearest].astype(float)
            known=(distance<.2)&(truth>0)&(weight>0);truth[~known]=0;weight[~known]=0
            anchor=(d['anchor_owner']==oid)&known;mass=np.bincount(truth[anchor],weights=weight[anchor])
            target=int(mass.argmax()) if len(mass) and mass.sum()>0 else None
            purity=float(mass[target]/mass.sum()) if target is not None else None
            eligible=bool(target is not None and anchor.sum()>=16 and purity>.5 and target//1000==row['train_id'])
            masks={};measurements={};bounds=row['crop'];x0,y0,x1,y1=bounds
            for branch in ['full','crop']:
                mask,raw=unpack(outputs[oid]['branches'][branch]['file']);coords=xy if branch=='full' else xy-[x0,y0]
                assert (coords>=0).all() and (coords[:,0]<mask.shape[1]).all() and (coords[:,1]<mask.shape[0]).all()
                expected_points=np.asarray(row['points'])-(np.array([x0,y0]) if branch=='crop' else 0)
                np.testing.assert_array_equal(raw['points'],expected_points)
                pts=expected_points.astype(int);satisfied=mask[pts[:,1],pts[:,0]]
                chosen=mask[coords[:,1],coords[:,0]];masks[branch]=mask[y0:y1,x0:x1] if branch=='full' else mask
                m=dict(prompt_satisfied=satisfied.tolist(),all_prompts_satisfied=bool(satisfied.all()),
                       selected_sources=int(chosen.sum()),selected_unassigned=int((chosen&(owner==0)).sum()),
                       selected_other_owners=int((chosen&(owner>0)&(owner!=oid)).sum()),
                       excluded_current_owner=int((~chosen&(owner==oid)).sum()),
                       selected_unknown=int((chosen&~known).sum()),source_support_iou=None)
                if eligible:
                    positive=truth==target;intersection=float(weight[chosen&positive].sum());union=float(weight[known&(chosen|positive)].sum())
                    m.update(source_support_iou=intersection/union if union>0 else None,
                        new_correct_sources=int((chosen&(owner==0)&positive).sum()),
                        new_other_known_sources=int((chosen&(owner==0)&known&~positive).sum()),
                        new_unknown_sources=int((chosen&(owner==0)&~known).sum()),
                        excluded_old_correct_sources=int((~chosen&(owner==oid)&positive).sum()),
                        correct_mass=intersection,other_known_mass=float(weight[chosen&known&~positive].sum()))
                measurements[branch]=m
                path=OUT/'readout'/key/f'object_{oid}_{branch}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,source_ids=ids,selected=chosen,reference_index=nearest,reference_distance=distance,
                                    truth=truth,confidence=weight,known=known)
                files[str(path)]=sha(path)
            diff=masks['full']!=masks['crop'];source_diff=diff[xy[:,1]-y0,xy[:,0]-x0]
            record=dict(object_id=oid,frame_id=row['frame_id'],declared_class=row['train_id'],
                target_gt=target,anchor_known_sources=int(anchor.sum()),anchor_target_purity=purity,diagnostic_eligible=eligible,
                ineligible_reason=None if eligible else 'Unknown/small/mixed anchor support or category mismatch; no target substitution',
                changed_mask_pixels=int(diff.sum()),changed_source_support=int(source_diff.sum()),methods=measurements)
            results[key].append(record)
            with Image.open(row['rgb']) as im:rgb=np.asarray(im.convert('RGB'))[y0:y1,x0:x1]
            fig,axes=plt.subplots(1,4,figsize=(14,4),constrained_layout=True)
            axes[0].imshow(rgb);points=np.asarray(row['points'])-[x0,y0];axes[0].scatter(points[:,0],points[:,1],c='#ffd94b',s=35,edgecolors='black')
            axes[0].set_title('Original crop + shared prompts')
            for i,branch in enumerate(['full','crop'],start=1):
                color=np.zeros_like(rgb);color[:]=[0,175,155];blend=rgb.copy();inside=masks[branch]
                blend[inside]=(.55*rgb[inside]+.45*color[inside]).astype(np.uint8);axes[i].imshow(blend)
                value=measurements[branch]['source_support_iou'];text='N/A' if value is None else f'{value:.3f}'
                axes[i].set_title(f'{branch}: native selected mask\nconditional source IoU {text}')
            axes[3].imshow(rgb);rgba=np.zeros((*diff.shape,4));rgba[diff]=[.95,.1,.35,.7];axes[3].imshow(rgba)
            axes[3].set_title(f'Actual difference\n{int(source_diff.sum())} measured sources')
            for ax in axes:ax.axis('off')
            fig.suptitle(f'{key} / old object {oid} / frame {row["frame_id"]} — not full-map AP')
            path=DEST/f'{key}_object_{oid}.png';fig.savefig(path,dpi=145);plt.close(fig);figures.append(str(path))
            a,b=measurements['full'],measurements['crop']
            fmt=lambda x:'N/A' if x is None else f'{x:.6f}'
            all_rows.append(f'<tr><td>{key}/{oid}</td><td>{target}</td><td>{eligible}</td><td>{source_diff.sum()}</td><td>{fmt(a["source_support_iou"])}</td><td>{fmt(b["source_support_iou"])}</td><td>{a["all_prompts_satisfied"]}/{b["all_prompts_satisfied"]}</td></tr>')
            print('SURFACE REFRESH READOUT',key,oid,'eligible',eligible,'IoU',a['source_support_iou'],b['source_support_iou'],'changed',int(source_diff.sum()),flush=True)
    aggregates={}
    for key,rows in results.items():
        aggregates[key]={}
        for name,cls in [('all',None),('building',2),('car',13)]:
            subset=[r for r in rows if (cls is None or r['declared_class']==cls)]
            valid=[r for r in subset if r['diagnostic_eligible']]
            deltas=[r['methods']['crop']['source_support_iou']-r['methods']['full']['source_support_iou'] for r in valid]
            aggregates[key][name]=dict(cases=len(subset),eligible=len(valid),unscored=len(subset)-len(valid),
                full_mean_iou=float(np.mean([r['methods']['full']['source_support_iou'] for r in valid])) if valid else None,
                crop_mean_iou=float(np.mean([r['methods']['crop']['source_support_iou'] for r in valid])) if valid else None,
                better=sum(x>1e-12 for x in deltas),worse=sum(x< -1e-12 for x in deltas),equal=sum(abs(x)<=1e-12 for x in deltas))
    dump(OUT/'readout.json',dict(complete=True,results=results,aggregates=aggregates,files=files,figures=figures,
        predictions_sha256=sha(OUT/'predictions_complete.json'),new_maps=0,new_ap=0,
        scope='Conditional source-support IoU on fixed crop, targets defined by posthoc anchor majority; NOT full-map instance AP, new-object recall, all-surface correctness or independent evaluation.'))
    (DEST/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>omni_slam — surface refresh evidence probe</title>
<style>body{font:16px/1.6 system-ui;color:#183245;max-width:1440px;margin:40px auto;padding:0 25px}img{width:100%}table{width:100%;border-collapse:collapse}td,th{padding:7px;text-align:left;border-bottom:1px solid #ddd}th{background:#edf7f5}pre{white-space:pre-wrap}</style>
<h1>新掩码取证：旧对象与未归属实测表面</h1><p>两个已暴露室外窗口，每处4建筑＋4车辆。全体候选由原锚点和未归属实测单元自动选择；两组使用同一物理位置的3个正点、同模型与原生质量选择。裁剪同时改变上下文和采样分辨率，不能只归因为分辨率。</p>
<p>全部模型输出在参考标注读入前封存。表中仅为共同裁剪域上的条件源点IoU，未知标签排除且选中未知点另有记录。目标由旧锚点GT多数定义，类别不符或多数不足保留N/A。不是完整地图AP或独立评测；0新地图/新AP。</p>
<table><tr><th>对象</th><th>诊断GT</th><th>可评分</th><th>实测支持变化</th><th>原图IoU</th><th>裁剪IoU</th><th>提示满足 原图/裁剪</th></tr>'''+''.join(all_rows)+'''</table><h2>分组结果</h2><pre>'''+html.escape(str(aggregates))+'''</pre><h2>全部真实案例</h2>'''+''.join(f'<img src="{Path(f).name}">' for f in figures))
    dump(OUT/'delivery_audit.json',dict(complete=True,all_sealed_input_prediction_hashes_verified=True,
        all_prompt_coordinates_and_shared_source_domains_verified=True,
        files={str(f):sha(f) for f in [DEST/'index.html',*map(Path,figures),OUT/'readout.json']},
        new_maps=0,new_ap=0))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);globals()[parser.parse_args().action]()
