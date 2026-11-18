"""Post-seal source allocation and equal-score audit, no prediction edits."""
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_surface_ownership import OUT,verify,sealed


def best_iou(folder):
    result={}
    for category,objects in read(folder/'matches.json')['window']['groundTruth'].items():
        for obj in objects:
            if obj['instance_id']<1000 or obj['vert_count']<100:continue
            result[obj['instance_id']]=dict(category=category,iou=max((r['intersection']/(obj['vert_count']+r['vert_count']-r['intersection']) for r in obj['matched_pred']),default=0.))
    return result


def main():
    p=verify(True);sealed('predictions_complete');results={}
    for scene in p['scenes']:
        key=scene['key'];folder=OUT/'maps'/key
        a=np.load(folder/'independent/decisions.npz');b=np.load(folder/'ours_competitive/decisions.npz')
        np.testing.assert_array_equal(a['source_ids'],b['source_ids'])
        diff=a['owners']!=b['owners'];row=dict(changed_source_decisions=int(diff.sum()),
            added_vs_independent=int(((a['owners']==0)&(b['owners']>0)).sum()),
            removed_vs_independent=int(((a['owners']>0)&(b['owners']==0)).sum()),
            reassigned_vs_independent=int(((a['owners']>0)&(b['owners']>0)&diff).sum()),
            independent_geometry_sha256=sha(folder/'independent/geometry.npz'),competitive_geometry_sha256=sha(folder/'ours_competitive/geometry.npz'))
        assert row['independent_geometry_sha256']!=row['competitive_geometry_sha256'] or not diff.any()
        if scene['reference']:
            base=best_iou(folder/'baseline');left=best_iou(folder/'independent');right=best_iou(folder/'ours_competitive')
            row['objects']=[dict(gt_id=gid,category=base[gid]['category'],baseline_iou=base[gid]['iou'],independent_iou=left[gid]['iou'],competitive_iou=right[gid]['iou']) for gid in sorted(base)]
            row['changed_gt_best_iou_vs_independent']=sum(abs(left[g]['iou']-right[g]['iou'])>1e-12 for g in base)
            row['threshold_crossings_vs_independent']={str(round(t,2)):int(sum((left[g]['iou']>t)!=(right[g]['iou']>t) for g in base)) for t in np.arange(.5,.91,.05)}
            row['scope']='Best IoU threshold crossings diagnose geometric changes; official AP additionally includes rankings, ignored regions and one-to-one matches.'
            d=np.load(OUT/'features'/key/'candidates.npz');y=np.load(OUT/'training_labels'/key/'TRAINING_ONLY.npz');valid=y['eligible'];w=y['weight'];selection={}
            for method in ['nearest_core','learned_geometry','independent','ours_competitive']:
                decision=np.load(folder/method/'decisions.npz');owner=decision['owners'];selected=owner>0;correct=((d['owners']==owner[:,None])&y['positive']&d['valid']).any(1);known=selected&valid;den=w[known].sum();possible=valid&y['positive'].any(1)
                selection[method]=dict(selected=int(selected.sum()),evaluable_selected=int(known.sum()),correct_weighted_fraction=float(w[known&correct].sum()/den) if den else None,positive_recall=float(w[valid&correct].sum()/w[possible].sum()))
            row['conditional_source_selection']=selection
        results[key]=row;print(key,{k:v for k,v in row.items() if k not in ['objects','conditional_source_selection']},flush=True)
    result=dict(complete=True,posthoc_only=True,results=results,source_sha256=sha(__file__),prediction_seal_sha256=sha(OUT/'predictions_complete.json'),
        scope='Conditional source selection uses eligible training-teacher anchor definition only after all target maps are sealed. It is not full-map accuracy. Never used for fitting, tuning or modifying predictions.')
    dump(OUT/'result_audit.json',result)
    report=ROOT/'output/surface_ownership_v1';dump(report/'result_audit.json',result)
    if (report/'index.html').exists():
        table=''
        for key,row in results.items():table+=f'<tr><td>{key}</td><td>{row["changed_source_decisions"]}</td><td>{row.get("changed_gt_best_iou_vs_independent","—")}</td><td>{row["independent_geometry_sha256"]!=row["competitive_geometry_sha256"]}</td></tr>'
        block=f'<h2>相同分数核查：独立判断与竞争判断</h2><table><tr><th>场景</th><th>源点决策不同</th><th>GT 最佳 IoU 不同</th><th>地图文件不同</th></tr>{table}</table><p>两个方法独立训练并各自导出地图，绝非重复读取同一份预测。完整 AP 由离散阈值、排序与匹配决定，几何不同也可能同分。<a href="result_audit.json">逐对象 IoU、阈值越界和源点选择诊断</a>。源点正确比例仅在可评估锚点子集统计，不是全地图准确率，也未用来改变模型。</p>'
        html=(report/'index.html').read_text();(report/'index.html').write_text(html.replace('</html>',block+'</html>'))


if __name__=='__main__':main()
