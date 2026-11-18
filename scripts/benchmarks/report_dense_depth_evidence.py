"""Verify the sensor-depth split and render actual held-out depth errors."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni_slam_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from probe_dense_depth_evidence import OUT,METHODS,verify
from omni_semantic_map.depth_evidence_interval import fit_intervals,intervals,candidate_depth_agreement

DEST=ROOT/'output/dense_depth_evidence_probe_v1'
NAMES=dict(nearest='Nearest measured anchor',linear_inverse='Linear inverse-depth',vggt_global='VGGT: global metric scale',vggt_local_shape='VGGT: LiDAR-anchored local shape')


def main():
    p=verify();summary=read(OUT/'summary.json');capture=read(OUT/'capture_complete.json');completion=read(OUT/'completion_complete.json');assert summary['complete']
    for seal in [capture,completion]:
        for f,h in seal['files'].items():assert sha(f)==h
    DEST.mkdir(parents=True,exist_ok=True);table=[];images=[];checks={}
    for s in p['scenes']:
        key=s['key'];cal=s['records'][1];depth=np.load(cal['measured'])['depth'];yy,xx=np.indices(depth.shape);held=(xx//32+yy//32)%2==1;mask=held&np.isfinite(depth)&(depth>0);c=np.load(OUT/'completion'/key/f'{cal["frame_id"]:06d}.npz')
        test_depth=[];preds={m:[] for m in METHODS};dist=[]
        for i,r in enumerate(s['records']):
            z=np.load(r['measured'])['depth'];v=np.load(OUT/'completion'/key/f'{r["frame_id"]:06d}.npz');xy=v['anchor_xy'];assert np.all((xy[:,0]//32+xy[:,1]//32)%2==0)
            np.testing.assert_array_equal(v['anchor_depth'],z[xy[:,1],xy[:,0]])
            for m in ['nearest','linear_inverse','vggt_local_shape']:np.testing.assert_allclose(v[m][xy[:,1],xy[:,0]],v['anchor_depth'],atol=1e-4,rtol=1e-6)
            if i==1:continue
            take=held&np.isfinite(z)&(z>0);test_depth.append(z[take]);dist.append(v['anchor_distance'][take])
            for m in METHODS:preds[m].append(v[m][take])
        truth=np.concatenate(test_depth);distance=np.concatenate(dist)
        for method in METHODS:
            model=fit_intervals(c[method][mask],depth[mask],c['anchor_distance'][mask]);assert model==read(OUT/'calibration'/key/(method+'.json'))
            d=np.load(OUT/'evaluated'/key/(method+'.npz'));np.testing.assert_array_equal(d['measured'],truth);np.testing.assert_array_equal(d['predicted'],np.concatenate(preds[method]))
            lo,hi,ok=intervals(d['predicted'],distance,model)
            np.testing.assert_allclose(lo,d['lower'],equal_nan=True);np.testing.assert_allclose(hi,d['upper'],equal_nan=True);np.testing.assert_array_equal(ok,d['precise']);np.testing.assert_array_equal(candidate_depth_agreement(truth,lo,hi,ok),d['agree'])
            r=summary['results'][key]['methods'][method];np.testing.assert_allclose(np.median(np.abs(d['predicted']-truth)),r['median_absolute_m'])
            table.append(f'<tr><td>{key[-4:]}</td><td>{NAMES[method]}</td><td>{100*r["median_absolute_m"]:.2f}</td><td>{r["p90_absolute_m"]:.3f}</td><td>{100*r["raw_within_original_relative_gate"]:.2f}%</td><td>{r["precise"]}</td></tr>')
        checks[key]=dict(fit_pixels_disjoint_from_calibration_and_test=True,calibration_frame_excluded_from_test=True,calibration_models_recomputed_exact=True,all_evaluated_arrays_replayed=True,test_measurements=len(truth))
        # Fixed third frame in the four-frame capture, before inspecting errors.
        r=s['records'][2];rgb=np.asarray(Image.open(r['rgb']).convert('RGB'));z=np.load(r['measured'])['depth'];v=np.load(OUT/'completion'/key/f'{r["frame_id"]:06d}.npz');take=held&np.isfinite(z)&(z>0);y,x=np.nonzero(take)
        fig,axes=plt.subplots(3,1,figsize=(13,10.5),layout='constrained')
        axes[0].imshow(rgb);axes[0].set_title(f'{key} / frame {r["frame_id"]}: actual RGB')
        for ax,method in zip(axes[1:],['linear_inverse','vggt_local_shape']):
            ax.imshow(rgb);error=np.abs(v[method][take]-z[take]);sc=ax.scatter(x,y,c=error,s=2,vmin=0,vmax=2,cmap='magma')
            ax.set_title(NAMES[method]+': held-out measured pixels only; error colours clipped at 2 m',fontsize=11)
        for ax in axes:ax.axis('off')
        fig.colorbar(sc,ax=list(axes[1:]),label='Absolute depth error [m]',shrink=.6,pad=.015)
        filename=key+'.png';fig.savefig(DEST/filename,dpi=125,bbox_inches='tight');plt.close(fig)
        images.append(dict(file=filename,key=key,frame_id=r['frame_id'],rgb_sha256=sha(r['rgb']),measured_sha256=sha(r['measured']),selection='third of four fixed frames,not error selection'))
    doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>VGGT深度证据能力验证</title><style>body{{max-width:1200px;margin:35px auto;padding:0 24px;color:#23384a;font:16px/1.7 system-ui}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #dbe5eb;padding:8px}}th{{background:#eef5f8}}img{{width:100%}}.note{{padding:18px;background:#fff1dc}}a{{color:#137f91}}</style>
<h1>VGGT深度形状是否能提供新增观测证据</h1><p>两个已暴露室外窗口 · 固定8张RGB · 80,952个留出LiDAR像素 · 不使用对象GT</p>
<p class="note"><b>局部深度形状有增益，但可靠观测筛选未通过。</b> LiDAR锚定后的VGGT局部形状，比最近点和线性逆深度插值的中位数/P90误差更低；直接使用单一全局尺度的VGGT更差。所有方法在当前保守区间规则下均没有通过“足够窄”的条件，因此没有据此新增观测、改地图或报AP。</p>
<h2>留出测量结果</h2><table><tr><th>窗口</th><th>方法</th><th>中位绝对误差[cm]</th><th>P90误差[m]</th><th>误差≤0.05+0.01z</th><th>精确区间数量</th></tr>{''.join(table)}</table>
<p>最后一列是当前区间校准的结果，不能解释为所有像素深度都不准确。约67%/72%的局部形状预测满足所列相对深度容差，但剩余大误差尚无法由当前距离分箱规则可靠识别；不能直接把这些预测当作几何可见性真值。</p>
<h2>输入、对照和泄漏核查</h2><p>VGGT官方已有权重，仅严格加载聚合器与原生深度头。官方预处理得到140×518图像，原生深度双线性回到376×1408。窗口0000：232/237/242/247；0002：4501/4506/4511/4516。此前VGGT实验只使用特征，本轮实际运行了深度头。</p>
<p>32像素棋盘格的偶数块提供实际LiDAR锚点；所有方法使用完全相同的锚点。线性控制在逆深度域做Delaunay插值，凸包外回退最近测量。全局VGGT只拟合一个窗口尺度；局部形状使用最近锚点实测深度乘以VGGT在目标像素与锚点的深度比。未生成任何新XYZ。</p>
<p>第二帧奇数块的测量用于校准，其他三帧奇数块仅评测。按最近锚点像素距离分三箱，校准5%/95%残差区间；区间宽度必须≤两倍既定深度容差。校准模型、留出数组、锚点数据和像素互斥均已重放。邻近帧与像素仍相关，不宣称独立泛化、IID覆盖保证或真实遮挡标签。合成前后错深度检查因没有精确区间而记不可评价。</p>
<p>5项核心测试通过。两次四帧GPU推理及保存共{capture['seconds']:.2f}秒，PyTorch峰值{capture['peak_cuda_gib']:.2f}GiB；不含加载、哈希、CPU插值与核查，不是SLAM帧率。没有新训练、语义GT访问、全图AP、机器人或独立场景结果。</p>
<h2>真实图像与留出深度误差</h2><p>固定展示各窗口第三帧。彩色点仅为未用于拟合或校准的实测像素；颜色上限2米，其上的误差统一饱和显示。不是检测mask。</p>{''.join('<img src="'+r['file']+'">' for r in images)}
<h2>去留与主线</h2><p>本验证要求两个窗口均有至少25%测试像素通过窄区间，区间内测量覆盖至少90%，且正确通过比例超过两个简单控制。当前为0，停止这个固定的硬几何证据候选，不放宽区间或换帧补分。局部形状的正向误差改善保留为能力证据，不能计为语义建图贡献。</p>
<p>回查现有代码与状态记录：最初的JointObjectMemory仍标记为prototype_untrained，当前训练入口没有调用它。它与此前训练过的局部提示注意力不同。后续应明确这个原主线模块的数据、持续槽位和历史修订训练协议，而不是继续向已失败表面规则添加条件。本轮不把未训练原型记为有效新机制；机器人对照和新增独立评测仍未完成。</p>
<p><a href="summary.json">全部指标</a> · <a href="audit.json">独立重放核查</a> · <a href="../../build/benchmarks/dense_depth_evidence_probe_v1/protocol.json">冻结协议</a></p></html>'''
    (DEST/'index.html').write_text(doc);dump(DEST/'summary.json',summary);dump(DEST/'images.json',images)
    audit=dict(complete=True,checks=checks,source_sha256=sha(__file__),capture_sha256=sha(OUT/'capture_complete.json'),completion_sha256=sha(OUT/'completion_complete.json'),summary_sha256=sha(OUT/'summary.json'),goal_complete=False)
    dump(OUT/'audit.json',audit);dump(DEST/'audit.json',audit);dump(DEST/'manifest.json',dict(files={f.name:sha(f) for f in DEST.iterdir() if f.is_file() and f.name!='manifest.json'}))
    print('DEPTH REPORT VERIFIED',DEST/'index.html',flush=True)


if __name__=='__main__':main()
