"""Posthoc reference agreement of added supports for baseline-correct objects."""
from collections import Counter
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_fixed_identity_recovery_control import OUT,PARENT,verify
from run_discovery_query_benchmark import INPUT,OUT as FRONT
from evaluate_kitti360_development import transfer


def main():
    p=verify(True);results={};parent_summary=read(PARENT/'summary.json')
    for s in p['scenes']:
        if not s['reference']:continue
        key=s['key'];ref=np.load(s['reference']);manifest=read(INPUT/key/'manifest.json');source=np.load(manifest['geometry'])['xyz']
        nearest=transfer(ref['xyz'],np.arange(1,len(ref['xyz'])+1,dtype=np.int64),source,.2);known=nearest>0
        gt=np.zeros(len(source),np.int64);weight=np.zeros(len(source));gt[known]=ref['train_instance'][nearest[known]-1];weight[known]=ref['confidence'][nearest[known]-1]
        base=PARENT/'maps'/key/'largest';g=np.load(base/'geometry.npz');projected=transfer(g['xyz'],g['instance'],ref['xyz'],.2)
        rows=read(base/'classification.json')['objects'];mass=np.bincount(projected,weights=ref['confidence'],minlength=max(r['object_id'] for r in rows)+1)
        eligible=[r['object_id'] for r in sorted(rows,key=lambda r:r['object_id']) if mass[r['object_id']]>=100];anchors={}
        for category,objects in read(base/'matches.json')['window']['groundTruth'].items():
            for obj in objects:
                if obj['instance_id']<1000 or obj['vert_count']<100:continue
                for pred in obj['matched_pred']:
                    if pred['intersection']/(obj['vert_count']+pred['vert_count']-pred['intersection'])>.5:anchors[eligible[pred['pred_id']]]=obj['instance_id']
        assert set(anchors.values())==set(parent_summary['results'][key]['methods']['largest']['matched_gt_ids'])
        by_obs={o:r['object_id'] for r in rows if r['object_id'] in anchors for o in r['observation_ids']};records=read(FRONT/'masks_complete.json')['scenes'][key]['records'];methods={}
        supports={m:{oid:set() for oid in anchors} for m in ['largest','all','pointwise','component']}
        for method in supports:
            for r in records:
                z=np.load(PARENT/'supports'/key/method/f'{r["frame_id"]:06d}.npz')
                for j in range(r['masks']):
                    oid=r['observation_offset']+j
                    if oid in by_obs:supports[method][by_obs[oid]].update(map(int,z['point_ids'][z['offsets'][j]:z['offsets'][j+1]]))
        for method in ['all','pointwise','component']:
            objects=[];counter=Counter();total=Counter()
            for oid,target in sorted(anchors.items()):
                added=np.asarray(sorted(supports[method][oid]-supports['largest'][oid]),np.int64);counter.update(map(int,added))
                w=weight[added];labels=gt[added];same=labels==target;other_car=(labels//1000==13)&~same;building=labels//1000==2
                row=dict(object_id=oid,gt_id=target,added_source_points=len(added),known_nearest_points=int(known[added].sum()),
                    same_gt_mass=float(w[same].sum()),other_car_mass=float(w[other_car].sum()),building_mass=float(w[building].sum()),
                    other_or_ignored_mass=float(w[~(same|other_car|building)].sum()),total_nearest_confidence_mass=float(w.sum()))
                row['same_gt_fraction']=row['same_gt_mass']/row['total_nearest_confidence_mass'] if row['total_nearest_confidence_mass'] else None
                objects.append(row)
                for name in ['added_source_points','known_nearest_points','same_gt_mass','other_car_mass','building_mass','other_or_ignored_mass','total_nearest_confidence_mass']:total[name]+=row[name]
            totals=dict(total);totals['same_gt_fraction']=totals['same_gt_mass']/totals['total_nearest_confidence_mass'] if totals['total_nearest_confidence_mass'] else None
            totals['unique_source_points_with_multiple_anchor_claims']=sum(n>1 for n in counter.values())
            methods[method]=dict(totals=totals,objects=objects);print(key,method,totals,flush=True)
        results[key]=methods
    result=dict(complete=True,posthoc_only=True,results=results,source_sha256=sha(__file__),
        scope='Only baseline AP50-correct anchors. Count unique added source IDs per anchor; nearest reference within20cm supplies label/confidence. This is source agreement, not AP, accuracy on all objects, or a feasible oracle map. Other/ignored labels are not automatically known negatives. Conflicting claims count exact shared source IDs, not nearby points. No predictor or training data is modified.')
    dump(OUT/'source_ownership_audit.json',result);report=ROOT/'output/fixed_identity_recovery_control_v1';dump(report/'source_ownership_audit.json',result)
    table=''
    for key,methods in results.items():
        for method,value in methods.items():
            row=value['totals'];fraction='—' if row['same_gt_fraction'] is None else f'{100*row["same_gt_fraction"]:.2f}%'
            table+=f'<tr><td>{key}</td><td>{method}</td><td>{row["added_source_points"]}</td><td>{fraction}</td><td>{row["unique_source_points_with_multiple_anchor_claims"]}</td></tr>'
    block=f'<h2>补充：新增支持是否与原正确对象的参考身份一致</h2><table><tr><th>窗口</th><th>策略</th><th>新增点—对象配对</th><th>同 GT 的邻近参考置信度占比</th><th>被多个原正确对象认领的源点</th></tr>{table}</table><p>仅针对原 AP50 正确对象；每个对象内部去重源点，再查询 20 cm 内最近参考的标签与置信度。不等于完整对象准确率或 AP。未找到邻近参考的点不参与比例，其他/忽略标签不能自动当作可靠负样本。GT 只用于本项事后审计，未进入任何预测或训练。<a href="source_ownership_audit.json">逐对象来源与分项</a></p>'
    html=(report/'index.html').read_text();(report/'index.html').write_text(html.replace('</html>',block+'</html>'))


if __name__=='__main__':main()
