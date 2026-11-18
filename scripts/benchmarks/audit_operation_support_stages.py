"""Localize fixed-input loss before the frozen outdoor revision candidate.

This is a reference-based support diagnostic only, not a map or AP prediction.
"""
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_object_operation_gate import OUT,verify
from audit_kitti360_object_retention import covered


def main():
    p=verify(True);s=next(s for s in p['scenes'] if s['kind']=='outdoor');cache=Path(s['cache']);manifest=read(cache/'manifest.json')
    raw=ROOT/'build/benchmarks/kitti360_development_v1/frontend'/s['key']/'cache'
    assert sha(raw/'geometry.npz')==sha(cache/'geometry.npz')
    xyz=np.load(cache/'geometry.npz')['xyz'];rows=read(Path(s['fixed_map'])/'classification.json')['objects']
    graphroot=ROOT/'build/benchmarks/kitti360_connected_development_v1/maps/cg'
    nodes=read(graphroot/'nodes.json');alias={n['alias']:n for n in nodes};groups=read(graphroot/'history2/graph.json')['groups'];history=set()
    for group in groups:
        if len({f for name in group for f in alias[name]['frames']})>=3:history.update(o for name in group for o in alias[name]['members'])
    final={o for row in rows for o in row['observation_ids']};target={o for row in rows if row['predictions']['mean']['retained'] for o in row['observation_ids']}
    assert target<=final<=history
    stages=['measurements','raw_mask_support','connected_support','three_frame_history','final_objects','semantic_foreground']
    domains={k:np.zeros(len(xyz),bool) for k in stages};domains['measurements'][:]=True;source_hashes={}
    for rec in manifest['records']:
        src=raw/'frames'/rec['file'];con=cache/'frames'/rec['file'];assert sha(con)==rec['sha256']
        source_hashes[str(src)]=sha(src);source_hashes[str(con)]=sha(con)
        with np.load(src) as a,np.load(con) as b:
            ids=a['point_ids'];offs=a['offsets'];points=b['point_ids'];off=b['offsets']
            domains['raw_mask_support'][ids]=True;domains['connected_support'][points]=True
            for j in range(rec['masks']):
                support=points[off[j]:off[j+1]];assert np.isin(support,ids[offs[j]:offs[j+1]]).all();oid=rec['observation_offset']+j
                for name,selected in [('three_frame_history',history),('final_objects',final),('semantic_foreground',target)]:
                    if oid in selected:domains[name][support]=True
    for a,b in zip(stages,stages[1:]):assert not np.any(domains[b]&~domains[a]),(a,b)
    ref=np.load(s['reference']);assert sha(s['reference'])==p['reference_hashes'][s['reference']]
    select=np.isin(ref['train_instance']//1000,[2,13]);truth=ref['train_instance'][select];weight=ref['confidence'][select];query=ref['xyz'][select]
    objects=[]
    for gid in np.unique(truth):
        mass=float(weight[truth==gid].sum())
        if mass>=100:objects.append(dict(gt_id=int(gid),category='car' if gid//1000==13 else 'building',confidence_mass=mass,coverage={}))
    for stage in stages:
        hit=covered(xyz,domains[stage],query)
        for obj in objects:obj['coverage'][stage]=float(weight[(truth==obj['gt_id'])&hit].sum()/obj['confidence_mass'])
        print(stage,int(domains[stage].sum()),{c:sum(o['category']==c and o['coverage'][stage]>.5 for o in objects) for c in ['car','building']},flush=True)
    totals={c:{stage:sum(o['category']==c and o['coverage'][stage]>.5 for o in objects) for stage in stages} for c in ['car','building']}
    result=dict(complete=True,diagnostic_only=True,scene=s['key'],stages=stages,objects=objects,coverage_over_half=totals,
        source_points={k:int(v.sum()) for k,v in domains.items()},observation_counts=dict(all=manifest['observations'],history=len(history),final=len(final),retained=len(target)),
        source_sha256=sha(__file__),input_sha256=source_hashes,parent_prediction_seal_sha256=sha(OUT/'predictions_complete.json'),
        scope='Necessary pointwise oracle coverage, not AP or achievable recall. Radius0.2m and confidence mass>=100 from existing diagnostic. Source domains are nested; this diagnostic uses independent nearest support per domain, NOT production background blockers.')
    path=ROOT/'build/benchmarks/operation_evidence_audit_v1/support_stages.json';assert not path.exists();dump(path,result)
    report=ROOT/'output/operation_evidence_audit_v1';dump(report/'support_stages.json',result)
    rows=''.join(f'<tr><td>{stage}</td><td>{result["source_points"][stage]:,}</td><td>{totals["car"][stage]} / 31</td><td>{totals["building"][stage]} / 18</td></tr>' for stage in stages)
    block=f'<h2>补充：目标在哪个阶段失去支持</h2><table><tr><th>阶段</th><th>源点数</th><th>车辆覆盖超过一半</th><th>建筑覆盖超过一半</th></tr>{rows}</table><p>各阶段源点域逐项验证嵌套。此表只作理想点级覆盖诊断，按各阶段支持独立做 20 cm 最近邻查询；不含正式预测的背景阻挡竞争，因此不能当作实际 AP。<a href="support_stages.json">逐目标支持与来源</a></p>'
    doc=(report/'index.html').read_text();doc=doc.replace('</html>',block+'</html>');(report/'index.html').write_text(doc)


if __name__=='__main__':main()
