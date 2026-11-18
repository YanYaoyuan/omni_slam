"""Posthoc anchor-identity contamination audit, without new map predictions."""
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_deferred_surface_support import OUT,verify
from evaluate_kitti360_development import transfer


def main():
    p=verify(True);summary=read(OUT/'summary.json');result={}
    for s in p['scenes']:
        key=s['key'];base=OUT/'maps'/key/'largest';rows=read(base/'classification.json')['objects'];label='baseline predicted object IDs'
        object_anchor={r['object_id']:r['object_id'] for r in rows}
        if s['reference']:
            ref=np.load(s['reference']);g=np.load(base/'geometry.npz');projected=transfer(g['xyz'],g['instance'],ref['xyz'],.2)
            mass=np.bincount(projected,weights=ref['confidence'].astype(float),minlength=max(object_anchor,default=0)+1)
            eligible=[oid for oid in sorted(object_anchor) if mass[oid]>=100];matches=read(base/'matches.json')['window']['groundTruth'];object_anchor={}
            for category,objects in matches.items():
                for gt in objects:
                    if gt['instance_id']<1000 or gt['vert_count']<100:continue
                    for pred in gt['matched_pred']:
                        iou=pred['intersection']/(gt['vert_count']+pred['vert_count']-pred['intersection'])
                        if iou>.5:object_anchor[eligible[pred['pred_id']]]=gt['instance_id']
            assert set(object_anchor.values())==set(summary['results'][key]['methods']['largest']['matched_gt_ids'])
            label='baseline AP50-correct GT anchors; diagnostic only'
        observation_anchor={}
        for r in rows:
            if r['object_id'] in object_anchor:
                for oid in r['observation_ids']:
                    assert oid not in observation_anchor;observation_anchor[oid]=object_anchor[r['object_id']]
        methods={}
        for method in ['all','pointwise','component']:
            groups=[]
            for row in read(OUT/'maps'/key/method/'classification.json')['objects']:
                anchors=sorted({observation_anchor[o] for o in row['observation_ids'] if o in observation_anchor})
                if len(anchors)>1:groups.append(dict(object_id=row['object_id'],anchor_ids=anchors,
                    observations=len(row['observation_ids']),semantic=row['predictions']['mean']['name']))
            mixed={i for g in groups for i in g['anchor_ids']}
            if s['reference']:
                lost=set(object_anchor.values())-set(summary['results'][key]['methods'][method]['matched_gt_ids'])
                detail=dict(lost_anchor_ids=sorted(lost),lost_anchors_present_in_mixed_groups=sorted(lost&mixed))
            else:detail={}
            methods[method]=dict(mixed_groups=groups,mixed_anchor_count=len(mixed),**detail)
        result[key]=dict(anchor_scope=label,methods=methods)
        print(key,{m:dict(mixed_groups=len(x['mixed_groups']),mixed_anchors=x['mixed_anchor_count'],lost_in_mixed=len(x.get('lost_anchors_present_in_mixed_groups',[]))) for m,x in methods.items()},flush=True)
    dump(OUT/'identity_audit.json',dict(complete=True,posthoc=True,results=result,source_sha256=sha(__file__),
        prediction_seal_sha256=sha(OUT/'predictions_complete.json'),
        scope='Tracks shared observation IDs from baseline-correct object anchors into candidate final groups. Multiple correct GT anchors in one group show identity mixing. This is not a causal ablation separating geometry contamination from association effects; robot uses unlabelled baseline IDs and cannot establish erroneous merging. No prediction changed.'))
    report=ROOT/'output/deferred_surface_support_v1';dump(report/'identity_audit.json',read(OUT/'identity_audit.json'))
    html=(report/'index.html').read_text();block='<h2>补充：补点后，原正确身份是否被混合</h2><table><tr><th>场景</th><th>变体</th><th>混合多个原正确 GT 的组数</th><th>丢失匹配中进入混合组的 GT 数</th></tr>'
    for key,r in result.items():
        if not key.startswith('kitti'):continue
        for method,row in r['methods'].items():block+=f'<tr><td>{key}</td><td>{method}</td><td>{len(row["mixed_groups"])}</td><td>{len(row["lost_anchors_present_in_mixed_groups"])}/{len(row["lost_anchor_ids"])}</td></tr>'
    block+='</table><p>按共享原始观测编号追踪原 AP50 正确对象，避免对象编号重排造成误判。混合多个不同正确 GT 的组证明发生身份混合，但本审计不能单独分离新增错误点与关联变化的因果效应。车库只追踪原预测 ID，无准确率结论。<a href="identity_audit.json">完整身份审计</a></p>'
    (report/'index.html').write_text(html.replace('</html>',block+'</html>'))


if __name__=='__main__':main()
