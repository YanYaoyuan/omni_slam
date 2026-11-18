"""Score sealed unseen-sequence outputs, plus a fixed-assignment diagnostic."""
from pathlib import Path
import html
import time
import numpy as np
from scipy.spatial import cKDTree
from paper_final_common import ROOT,read,dump,sha
from omni_semantic_map.mesh_instance_evaluation import transfer_labels
from omni_semantic_map.full_ovo_evaluation import voxel_metrics
from omni_semantic_map.instance_error_audit import audit_instances,filter_transition

OUT=ROOT/'build/benchmarks/visibility_independent_20261112'
REPORT=ROOT/'docs/reports/visibility_independent_2026-11-12'


def locked_projection(xyz,labels,vertices,retained_indices):
    """Freeze baseline nearest-point association. Rejected owners become zero."""
    selection=np.flatnonzero(labels>0)
    if not len(selection):return np.zeros(len(vertices),np.int64)
    distance,index=cKDTree(xyz[selection]).query(vertices,workers=1)
    original=selection[index];keep=np.zeros(len(xyz),bool);keep[retained_indices]=True
    return np.where((distance<=.05)&keep[original],labels[original],0)


def evaluate():
    from visibility_independent_queue import verify
    from paper_final_metrics import reference
    from publish_visibility_cleanup import independent_score
    protocol=verify();barrier=read(OUT/'predictions_complete.json');assert barrier['complete']
    for scene,h in barrier['files'].items():assert sha(OUT/'predictions'/(scene+'.json'))==h
    results=[]
    for scene in protocol['scenes']:
        output=OUT/'evaluation'/scene;output.mkdir(parents=True,exist_ok=False)
        state=read(OUT/'predictions'/(scene+'.json'))
        if not state.get('complete'):
            result=dict(scene=scene,complete=False,reason='prediction_failure',detail=state)
            dump(output/'metrics.json',result);results.append(result);continue
        # Exact-coordinate assertion is inherited. Failures are N/A, never fitted.
        try:ref=reference(scene,root=OUT)
        except Exception as error:
            import traceback
            result=dict(scene=scene,complete=False,reason='reference_or_coordinate_failure',error=repr(error),traceback=traceback.format_exc())
            dump(output/'metrics.json',result);results.append(result);continue
        vertices=ref['xyz'];valid=ref['valid'];truth=ref['truth'][valid]
        np.savez_compressed(output/'reference.npz',xyz=vertices,instance=ref['truth'],valid=valid)
        methods={};audits={};baseline_geometry=None;baseline_labels=None
        order=['history2','cg','sor','visibility1','visibility3','visibility1_no_depth','maskclustering']
        for name in order:
            record=state['methods'][name]
            if not record.get('complete',True):methods[name]=record;continue
            for field in ['geometry','classification']:
                assert sha(record[field+'_path'])==record[field+'_sha256']
            g=np.load(record['geometry_path']);rows=read(record['classification_path'])['objects']
            kept=[r['object_id'] for r in rows if r['predictions']['mean']['retained']]
            labels=np.where(np.isin(g['instance'],kept),g['instance'],0)
            projected,_=transfer_labels(g['xyz'],labels,vertices,mode='nearest_5cm')
            if name=='history2':baseline_geometry=g['xyz'];baseline_labels=labels
            score=voxel_metrics(truth,projected[valid]);independent=independent_score(truth,projected[valid])
            for k in ['tp','fp','fn']:assert score[k]==independent[k]
            assert np.isclose(score['observed_voxel_instance_pq'],independent['pq'],atol=1e-12,rtol=0)
            audit=audit_instances(truth,projected[valid]);audits[name]=audit
            item=dict(complete=True,score=score,transition=filter_transition(audits['history2'],audit),
                assigned_valid_fraction=float((projected[valid]>0).mean()))
            if name in ['sor','visibility1','visibility3','visibility1_no_depth']:
                ids=np.load(Path(record['geometry_path']).with_name('retained_source_indices.npy'))
                locked=locked_projection(baseline_geometry,baseline_labels,vertices,ids)
                locked_score=voxel_metrics(truth,locked[valid])
                item['locked_assignment_score']=locked_score
                item['projection_reassignment_voxels']=int(((locked!=projected)&valid).sum())
                np.savez_compressed(output/(name+'_locked.npz'),instance=locked)
            methods[name]=item
            np.savez_compressed(output/(name+'.npz'),instance=projected);dump(output/(name+'_audit.json'),audit)
        result=dict(scene=scene,complete=True,domain=ref['domain'],methods=methods,
            reference_scope='Same fixed observed 5cm valid domain; strictly IoU>0.5; not official full-mesh leaderboard.',
            all_scores_independently_recomputed=True)
        dump(output/'metrics.json',result);results.append(result)
        print('UNSEEN SCORED',scene,{k:round(v['score']['observed_voxel_instance_pq']*100,4) for k,v in methods.items() if v.get('complete')},flush=True)
    available=[r for r in results if r['complete']];contrasts={}
    for name,base in [('visibility1','history2'),('visibility1','sor'),('visibility1','visibility1_no_depth'),('visibility1','maskclustering')]:
        paired=[r for r in available if r['methods'].get(name,{}).get('complete') and r['methods'].get(base,{}).get('complete')]
        delta=[(r['methods'][name]['score']['observed_voxel_instance_pq']-r['methods'][base]['score']['observed_voxel_instance_pq'])*100 for r in paired]
        contrasts[name+'__'+base]=dict(scenes=[r['scene'] for r in paired],mean_delta_pq100=float(np.mean(delta)) if delta else None,
            per_scene_delta=dict(zip([r['scene'] for r in paired],delta)),wins=sum(x>1e-8 for x in delta),ties=sum(abs(x)<=1e-8 for x in delta),losses=sum(x< -1e-8 for x in delta))
    checks={};main=contrasts['visibility1__history2']
    primary_ready=len(main['scenes'])==len(protocol['scenes'])
    if primary_ready:
        locked_delta=[(r['methods']['visibility1']['locked_assignment_score']['observed_voxel_instance_pq']-r['methods']['history2']['score']['observed_voxel_instance_pq'])*100 for r in available]
        checks=dict(all_four_scenes=True,positive_mean_gain=main['mean_delta_pq100']>0,
            at_least_three_nondecreasing=main['wins']+main['ties']>=3,
            total_tp_not_lower=sum(r['methods']['visibility1']['score']['tp']-r['methods']['history2']['score']['tp'] for r in available)>=0,
            locked_assignment_mean_positive=float(np.mean(locked_delta))>0,
            beats_sor_mean=contrasts['visibility1__sor']['mean_delta_pq100']>0,
            visibility_ablation_positive=contrasts['visibility1__visibility1_no_depth']['mean_delta_pq100']>0)
    external_complete=len(contrasts['visibility1__maskclustering']['scenes'])==len(protocol['scenes'])
    result=dict(complete=True,selected_scenes=protocol['scenes'],scored_scenes=[r['scene'] for r in available],contrasts=contrasts,
        checks=checks,primary_validation_passed=primary_ready and all(checks.values()),external_comparison_complete=external_complete,
        unavailable=[r for r in results if not r['complete']],protocol_sha256=sha(OUT/'protocol.json'),
        new_parameter_search=False,independence='Previously unprocessed SceneNN sequences in this project; same-dataset generalization only.',
        warning='A pass is evidence under this protocol, not proof of novelty or paper acceptance.')
    dump(OUT/'summary.json',result);publish(result,results)


def publish(summary,results):
    REPORT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for r in results:
        if not r['complete']:rows.append([r['scene'],'N/A',r['reason'],'','','']);continue
        base=r['methods']['history2']['score']['observed_voxel_instance_pq']*100
        for name,v in r['methods'].items():
            if not v.get('complete'):rows.append([r['scene'],name,'失败','','','']);continue
            s=v['score'];l=v.get('locked_assignment_score')
            rows.append([r['scene'],name,f"{s['observed_voxel_instance_pq']*100:.4f}",f"{s['observed_voxel_instance_pq']*100-base:+.4f}",f"{s['tp']}/{s['fp']}/{s['fn']}",f"{l['observed_voxel_instance_pq']*100:.4f}" if l else '—'])
    cells=lambda tag,items:''.join(f'<{tag}>{html.escape(str(v))}</{tag}>' for v in items)
    table='<table><tr>'+cells('th',['场景','方法','PQ×100','Δ 对强基线','TP/FP/FN','冻结归属 PQ'])+'</tr>'+''.join('<tr>'+cells('td',r)+'</tr>' for r in rows)+'</table>'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>可见性清理独立验证</title><style>body{max-width:1200px;margin:36px auto;font:16px/1.75 system-ui}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}a{color:#167492}</style><h1>可见性清理：独立 SceneNN 验证</h1><p>固定新场景 052 / 054 / 057 / 061；所有预测封存后评分。主方法 visibility1 是开发阶段已测过的简化版，非测试后挑选。MaskClustering 为官方核心＋共享 SAM2/SigLIP 输入适配，非原论文完整系统。</p>'+table+'<h2>配对结果与验收</h2><pre>'+html.escape(__import__('json').dumps(summary,ensure_ascii=False,indent=2))+'</pre><p><a href="protocol.json">运行前协议</a> · <a href="summary.json">汇总</a></p><p>官方对照来源：<a href="https://github.com/PKU-EPIC/MaskClustering">MaskClustering</a>。原方法已有跨视角点过滤，不能将点过滤本身写成新增贡献。本轮固定归属诊断用于区分边界清理和最近邻重新分配的收益。</p></html>'
    (REPORT/'index.html').write_text(page)
    import shutil
    for name in ['protocol.json','selection.json','summary.json','predictions_complete.json']:shutil.copyfile(OUT/name,REPORT/name)


if __name__=='__main__':evaluate()
