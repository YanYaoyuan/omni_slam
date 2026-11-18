"""Report every fixed case; real RGB and unchanged source map geometry."""
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from probe_surface_witness_transport import OUT,TRACK,AUDIT,BASE,METHODS,verify
from probe_world_anchor_tracking import prepared

DEST=ROOT/'output/surface_witness_transport_probe_v1'
NAMES=['Exact point','20 cm cell','Surface transport (candidate)']


def main():
    verify();s=read(OUT/'summary.json');seal=read(OUT/'predictions_complete.json');assert s['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    history=read(AUDIT/'summary.json');cell=read(AUDIT/'cell_summary.json');cases=prepared()['cases'];DEST.mkdir(parents=True,exist_ok=True)
    table='';case_table='';error_table='';figures='';images=[]
    for key,result in s['results'].items():
        for method,name in zip(METHODS,NAMES):
            m=result['metrics'][method]; table+='<tr>'+''.join(f'<td>{v}</td>' for v in [key,name,m['incorrect'],m['correct'],m['unknown'],f'{100*m["error_detection_precision"]:.2f}',f'{100*m["error_detection_recall"]:.2f}'])+'</tr>'
        sem=cell['results'][key]['error_semantics']['video_seed']['point'];other=sum(v['other_class'] for v in sem.values());same=sum(v['same_class_other_instance'] for v in sem.values());unseen=history['results'][key]['totals']['map']['video_seed']['unseen'];support=history['results'][key]['totals']['map']['video_seed']['supported_only']
        error_table+='<tr>'+''.join(f'<td>{v}</td>' for v in [key,other,same,unseen['incorrect'],support['incorrect']])+'</tr>'
        for r in result['cases']:
            for method,name in zip(METHODS,NAMES):
                g=r['methods'][method]['excluded_repeated'];values=[r['case'],r['gt_target'] if r['gt_target'] is not None else 'N/A',name,g['incorrect'] if r['gt_target'] is not None else 'N/A',g['correct'] if r['gt_target'] is not None else 'N/A',g['unknown']]
                case_table+='<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in values)+'</tr>'
    for case in cases:
        key=case['key'];name=case['name'];j=case['common_frames'][len(case['common_frames'])//2];rec=case['records'][j];rgb=np.asarray(Image.open(rec['rgb']).convert('RGB'));base=np.load(BASE/key/'geometry_graph/geometry.npz');a=np.load(AUDIT/'cases'/(name+'.npz'));b=np.load(AUDIT/'cell_cases'/(name+'.npz'));c=np.load(OUT/'cases'/(name+'.npz'));ids=c['map_ids'];xyz=base['xyz'][ids];np.testing.assert_array_equal(ids,a['map_ids']);np.testing.assert_array_equal(ids,b['map_ids'])
        counts=[(a['map_video_seed_positive'],a['map_video_seed_negative']),(b['video_seed_positive'],b['video_seed_negative']),(c['positive'],c['negative'])]
        # All geometry is retained for display; common PCA projection depends
        # solely on baseline coordinates, never the outcome or GT label.
        centered=xyz-xyz.mean(0);_,_,axes=np.linalg.svd(centered,full_matrices=False);xy=centered@axes[:2].T
        fig=plt.figure(figsize=(14,7),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1.1,1]);ax=fig.add_subplot(grid[0,:]);ax.imshow(rgb);ax.axis('off');ax.set_title(f'{name} / RGB frame {rec["frame_id"]} (fixed middle eligible nonseed frame)')
        for k,((pos,neg),label) in enumerate(zip(counts,NAMES)):
            ax=fig.add_subplot(grid[1,k]);rejected=(pos==0)&(neg>=2);ax.scatter(xy[:,0],xy[:,1],s=1,c='#a9c3c5',rasterized=True);ax.scatter(xy[rejected,0],xy[rejected,1],s=9,c='#d93660',rasterized=True);ax.set_title(f'{label}\n{rejected.sum()} diagnostic candidates');ax.set_aspect('equal');ax.set_xlabel('Common baseline PCA axis 1 (m)');ax.set_ylabel('Axis 2 (m)');ax.grid(alpha=.15)
        fig.suptitle('Actual old object geometry. Red = candidate exclusion, NOT GT error; no map edits applied.',fontsize=11)
        file=name+'.png';fig.savefig(DEST/file,dpi=120,bbox_inches='tight',pad_inches=.12);plt.close(fig);images.append(dict(case=name,frame=rec['frame_id'],rgb=rec['rgb'],rgb_sha256=sha(rec['rgb']),file=file,geometry_sha256=sha(BASE/key/'geometry_graph/geometry.npz'),selection='middle common nonseed frame; all8cases shown; common baseline PCA'))
        figures+=f'<h3>{name}</h3><img src="{file}">'
    style='<style>body{font:16px/1.7 system-ui;color:#233746;max-width:1250px;margin:30px auto;padding:0 24px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d8e1e6;padding:7px}img{width:100%}.note{padding:18px;background:#fff1db}a{color:#137785}</style>'
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam 历史地图证据与表面传递</title>{style}
<h1>历史地图修订：证据覆盖与表面传递验证</h1>
<p class="note"><b>候选未通过两个窗口的共同门槛，本版本停止。</b>0002 的局部错误检出有改善，0000 退步。强基线保持不变。没有新完整地图、AP/PQ、机器人或独立场景成绩；不将本轮写作已成立的论文贡献。</p>
<h2>原始问题：新掩码是否提供足够的历史修订证据？</h2>
<p>检查全部8个既定对象片段，在与前次相同的合格非种子帧中，把普通视频掩码作用于真正的融合地图点。精确点须通过原5厘米实测深度门槛；20厘米单元使用当帧所有实测可见源点，同一单元内掩码意见混合则保留未知。两种观测定义不同，不能把单元可见说成每个点都可见。</p>
<table><tr><th>窗口</th><th>其他类别污染点</th><th>同类其他实例点</th><th>未再观察到的错误点</th><th>仍被视频掩码支持的错误点</th></tr>{error_table}</table>
<p>这是8个片段中、固定基线GT对应下的已知错误点分类，不是全数据集错误分布。0002一个对象没有GT对应，保留N/A。旧图已含跨窗口监督，窗口也已暴露；不称为无监督或独立测试。</p>
<h2>实现的候选：局部表面上的实测证据传递</h2>
<p>仅用实测三维几何拟合局部平面：20厘米内16近邻，至少6点，排除线状、曲率过高及偏离平面的支持。旧点与当前实测点需法向夹角不超过30度，并同时满足双向5厘米点到平面距离。兼容见证全部落在掩码内才支持，全部在外才排除，冲突和缺测保留未知；每帧最多一票。</p>
<p>5项针对性测试通过，包括相邻/相交表面隔离、冲突处理、退化邻域、无测量不投票及刚体变换一致性。局部PCA、表面关联和多视图筛选均是已有工具；当前研究检验其能否解决稀疏观测中的证据传递，不据此宣称独立创新已成立。</p>
<h2>固定对照结果</h2>
<p>三组共享原视频掩码和帧。诊断候选统一为“至少2帧排除且没有支持”，并未实际删点。精度=检出的错误点/(检出的错误点+会误删的正确点)，召回=检出的错误点/原已知错误点。不是3D实例AP、PQ或完整图收益。</p>
<table><tr><th>窗口</th><th>观测方式</th><th>检出错误点</th><th>会误删正确点</th><th>未知点</th><th>错误检出精度%</th><th>错误检出召回%</th></tr>{table}</table>
<p>预设必要条件：两个窗口中均比粗单元有更高精度、比精确点有更高召回，且检出错误点多于误删正确点。0000不满足，整体未通过。不追加半径、邻居数、法向角度或帧数搜索，不以0002局部优势代替整体验收。</p>
<h2>全部案例</h2><table><tr><th>案例</th><th>固定GT对应</th><th>方式</th><th>检出错误</th><th>误删正确</th><th>未知</th></tr>{case_table}</table>
<h2>真实图像与历史对象几何</h2><p>全部8例均展示，RGB固定取合格非种子帧中间一帧，几何显示原完整对象。红色只是各观测规则产生的诊断候选，并非已知错误；三图共享只由原几何计算的投影视角。</p>{figures}
<h2>主线去留</h2><p>停止本版表面传递筛选，不将它接入机器人或扩大评测。现有证据不支持把历史修订简单等同于“重复看见同一个点再删掉”；更多观测覆盖也不能抵消错误掩码。下一项研究需要能区分对象与其他类别表面的新证据，并与强基线、普通融合和简单几何方法比较完整地图；新机制、同版本机器人对照和新增独立评测仍未完成。</p>
<p>局限：窗口末几何参与平面拟合，非在线因果；邻域表面兼容并不证明目标点自身可见或不存在遮挡；掩码之间相关；GT经20厘米最近参考转移，仅用于事后诊断。源点、融合点和体素单元不能互换。首次历史审计因位姿字段位置误读中止，修复后同输入重算，失败日志保留；模型输出没有重跑。</p>
<p><a href="summary.json">候选完整结果</a> · <a href="history_summary.json">精确点历史审计</a> · <a href="cell_summary.json">单元与错误类别审计</a> · <a href="../../build/benchmarks/surface_witness_transport_probe_v1/protocol.json">冻结条件</a> · <a href="../world_anchor_tracking_probe_v1/index.html">原RGB掩码对照</a></p></html>'''
    (DEST/'index.html').write_text(body)
    for src,name in [(OUT/'summary.json','summary.json'),(AUDIT/'summary.json','history_summary.json'),(AUDIT/'cell_summary.json','cell_summary.json')]:dump(DEST/name,read(src))
    dump(DEST/'images.json',images);dump(DEST/'manifest.json',dict(generator_sha256=sha(__file__),files={f.name:sha(f) for f in DEST.iterdir() if f.is_file() and f.name!='manifest.json'}))
    print('REPORT',DEST/'index.html',flush=True)


if __name__=='__main__':main()
