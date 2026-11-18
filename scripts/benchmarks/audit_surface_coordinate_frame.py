"""No-GT diagnostic: exact feature transform under a 90-degree world yaw.

For x'=-y,y'=x,z'=z, AABB outside/extent x/y channels swap. Every other
feature is invariant. This reveals coordinate dependence without creating or
scoring additional maps, retraining or selecting a better parameter.
"""
import numpy as np
import torch
from paper_final_common import ROOT,read,dump,sha
from run_surface_ownership import OUT,INPUT,verify,sealed
from omni_semantic_map.surface_ownership import OwnershipScorer,allocate


def main():
    p=verify();sealed('features_complete');sealed('trained');sealed('predictions_complete');torch.set_num_threads(1);result={};geometry_checks={}
    for scene in p['scenes']:
        key=scene['key'];d=np.load(OUT/'features'/key/'candidates.npz');x=d['features'].copy();x[:,:,[4,5]]=x[:,:,[5,4]];x[:,:,[7,8]]=x[:,:,[8,7]];methods={}
        xyz=np.load(read(INPUT/key/'manifest.json')['geometry'])['xyz'];core=read(OUT/'features'/key/'core_sources.json');rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        checked=0
        for owner,ids in core.items():
            groups,slots=np.where(d['owners']==int(owner))
            if not len(groups):continue
            own=xyz[np.asarray(ids)]@rotation.T;points=xyz[d['source_ids'][groups]]@rotation.T;lo=own.min(0);hi=own.max(0);extent=np.maximum(hi-lo,.05)
            outside=np.maximum(np.maximum(lo-points,points-hi),0)/extent
            np.testing.assert_allclose(outside,x[groups,slots,4:7],rtol=2e-6,atol=2e-6)
            np.testing.assert_allclose(np.broadcast_to(np.log1p(extent),outside.shape),x[groups,slots,7:10],rtol=2e-6,atol=2e-6)
            np.testing.assert_allclose(np.log1p(np.linalg.norm((points-(lo+hi)/2)/extent,axis=1)),x[groups,slots,3],rtol=2e-6,atol=2e-6)
            checked+=len(groups)
        geometry_checks[key]=checked
        for method in ['learned_geometry','independent','ours_competitive']:
            m=torch.load(OUT/'models'/key/method/'model.pt',map_location='cpu',weights_only=False);model=OwnershipScorer(m['dimensions']);model.load_state_dict(m['state']);model.eval()
            with torch.no_grad():logits=model(torch.tensor((x[:,:,:m['dimensions']]-m['mean'])/m['std'])).numpy()
            owner,_=allocate(logits,d['valid'],d['owners'],independent=method=='independent');old=np.load(OUT/'maps'/key/method/'decisions.npz')['owners']
            methods[method]=dict(candidate_sources=len(owner),changed_decisions=int((old!=owner).sum()),changed_fraction=float((old!=owner).mean()),original_accepted=int((old>0).sum()),rotated_accepted=int((owner>0).sum()))
        result[key]=methods;print(key,methods,flush=True)
    report=dict(complete=True,results=result,source_sha256=sha(__file__),posthoc_only=True,no_gt_read=True,new_maps=0,actual_rotated_geometry_checks=geometry_checks,
        explanation='World yaw by90degrees swaps AABB normalized outside x/y and logextent x/y, checked against actual rotated source geometry. Physical support/cell memberships stay fixed. Nearest distances, scaled center norm, semantics and visibility counts unchanged. Standardization/model stay fixed. This measures model coordinate-frame invariance, not accuracy, AP, a rerun of upstream grid construction, or robustness to physical viewpoint changes. No model is modified.')
    dump(OUT/'coordinate_audit.json',report);dest=ROOT/'output/surface_ownership_v1';dump(dest/'coordinate_audit.json',report)
    table=''
    for key,methods in result.items():
        for method,r in methods.items():table+=f'<tr><td>{key}</td><td>{method}</td><td>{r["changed_decisions"]}</td><td>{100*r["changed_fraction"]:.2f}%</td></tr>'
    html=(dest/'index.html').read_text();block=f'<h2>补充稳定性核查：只旋转世界坐标系 90°</h2><p>物理点、掩码、位姿关系和对象均未变化；轴对齐包围盒的 x/y 外部距离和尺寸特征交换。只做原模型推理，不读取 GT、不新建地图、不评分，也未训练或选择新参数。归属若改变，表示该版本依赖任意世界坐标轴。</p><table><tr><th>场景</th><th>模型</th><th>归属决策变化</th><th>占全部候选源点</th></tr>{table}</table><p><a href="coordinate_audit.json">复核明细</a>。这是当前模型的机制缺口，不能把坐标变化当作真实视角变化。</p>'
    heading='<h2>补充稳定性核查：只旋转世界坐标系 90°</h2>'
    if heading in html:html=html[:html.index(heading)]+'</html>'
    (dest/'index.html').write_text(html.replace('</html>',block+'</html>'))


if __name__=='__main__':main()
