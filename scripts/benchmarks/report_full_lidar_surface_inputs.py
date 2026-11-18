"""Real input geometry, necessary coverage, bounded mechanism probe and readiness."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/omni_full_surface_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from prepare_full_lidar_surface_inputs import OUT as INPUT,OLD,verify

OUT=ROOT/'output/full_lidar_surface_inputs_v1'
PROBE=ROOT/'build/benchmarks/surface_gap_witness_v1'
RESERVED=ROOT/'build/datasets/kitti360_reserved_v3'


def run():
    p=verify();prepared=read(INPUT/'prepared.json');coverage=read(INPUT/'coverage.json');audit=read(INPUT/'raw_replay_audit.json')
    probe=read(PROBE/'summary.json');reserved=read(RESERVED/'delivery_audit.json');robot=read(INPUT/'robot_ray_input_audit.json')
    assert audit['complete'] and reserved['complete'];OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10});images=[];rows=[];categories=[]
    for scene in p['scenes']:
        key=scene['key'];g=np.load(INPUT/key/'geometry.npz');xyz=g['xyz'];old=g['old_source_id']>=0
        low=np.quantile(xyz[:,:2],.005,axis=0);high=np.quantile(xyz[:,:2],.995,axis=0)
        m=read(OLD/key/'manifest.json');frame=m['records'][len(m['records'])//2]
        fig=plt.figure(figsize=(13,8),layout='constrained');grid=fig.add_gridspec(2,2,height_ratios=[.75,1.5]);ax=fig.add_subplot(grid[0,:])
        ax.imshow(Image.open(frame['rgb']));ax.axis('off');ax.set_title(f'{key} | existing development RGB frame {frame["frame_id"]}',loc='left',weight='bold')
        for j,title in enumerate(['Previous camera-selected measurements','Same scans: complete measured geometry']):
            ax=fig.add_subplot(grid[1,j])
            ids=np.flatnonzero(old);ids=ids[::max(1,(len(ids)+89999)//90000)]
            ax.scatter(xyz[ids,0],xyz[ids,1],s=.3,c='#5c6875',label='Original sources')
            if j:
                ids=np.flatnonzero(~old);ids=ids[::max(1,(len(ids)+149999)//150000)]
                ax.scatter(xyz[ids,0],xyz[ids,1],s=.3,c='#009e9a',label='Additional measured sources')
            ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_aspect('equal');ax.set_xlabel('x / m');ax.set_ylabel('y / m');ax.set_title(title);ax.grid(alpha=.12)
            ax.legend(fontsize=8,markerscale=6)
        name=f'{key}_input.png';fig.savefig(OUT/name,dpi=150);plt.close(fig);images.append(dict(file=name,rgb=frame['rgb'],rgb_sha256=sha(frame['rgb'])))
        for category,r in coverage['results'][key]['category'].items():
            categories.append((key.replace('kitti360_','')+' / '+category,r))
            rows.append(f'<tr><td>{key}</td><td>{category}</td><td>{r["gt"]}</td><td>{r["old_over_half"]}</td><td>{r["full_over_half"]}</td></tr>')
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');x=np.arange(len(categories));a=[r['old_over_half'] for _,r in categories];b=[r['full_over_half'] for _,r in categories]
    ax.bar(x-.18,a,.36,label='Previous camera-selected sources',color='#687b8a');ax.bar(x+.18,b,.36,label='Complete same-scan sources',color='#009e9a')
    ax.set_xticks(x,[n for n,_ in categories]);ax.set_ylabel('GT objects with measured coverage >50%');ax.set_title('Necessary coverage only — not instance AP or predicted recall');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.savefig(OUT/'coverage.png',dpi=160);plt.close(fig)
    diagnostics=[]
    for key,r in probe['results'].items():
        for method,s in r['methods'].items():
            precision='N/A' if s['precision'] is None else f'{s["precision"]*100:.1f}%'
            diagnostics.append(f'<tr><td>{key}</td><td>{method}</td><td>{s["cut_total"]}</td><td>{s["known_different_cut"]}</td><td>{s["same_object_cut"]}</td><td>{precision}</td></tr>')
    selection=read(RESERVED/'window_selection.json');newrows=''.join(f'<tr><td>{w["scene"]}</td><td>{w["start"]}–{w["end"]}</td><td>{len(w["input_frames"])}</td></tr>' for w in selection['windows'])
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam 完整测量与新增评测准备</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#17354a;background:#f4f7fa;max-width:1200px;margin:32px auto;padding:0 24px}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #dce5ec;text-align:left}}img{{width:100%;height:auto}}.status{{padding:16px;border-left:5px solid #078a8a;background:#edf8f6}}</style>
<h1>omni_slam：完整实测表面、机制预检与新窗口准备</h1><p>2026-11-14</p><p class="status">完整扫描恢复了此前未进入室外语义缓存的测量；新窗口已下载并核验输入几何分离。空隙见证候选未通过局部判别门槛。本报告没有新增实例AP，也不代表机器人或独立评测已完成。</p>
<section><h2>发现了什么输入缺口</h2><p>原室外适配器只保存投到前置相机内、通过像素深度选择的激光返回。机器人缓存保留的跟踪坐标系点云范围不同。现在使用原来相同的60／48次扫描及标定，恢复完整激光测量，并精确保留原有2064298个源点。</p>
<p>两窗口共7468489个实际来源点，比原输入增加5404191个。保留每5厘米单元首个真实返回，再补保原来的相机来源返回，按帧号与原始点索引去重；没有平均或合成坐标。全部来源已从原始扫描重放。图中青色表示新增测量，不是语义标签；颜色也不是相机RGB。</p>
{''.join(f'<img src="{r["file"]}" alt="{r["file"]}">' for r in images)}<p>统一按完整测量XY的0.5%–99.5%分位显示，作确定性抽样；覆盖计算不裁剪。</p></section>
<section><h2>完整几何能覆盖多少对象</h2><img src="coverage.png" alt="Necessary measured surface coverage"><table><tr><th>窗口</th><th>类别</th><th>全部合格GT</th><th>原输入覆盖&gt;50%</th><th>完整输入覆盖&gt;50%</th></tr>{''.join(rows)}</table>
<p>按完整参考表面、20厘米严格距离和参考置信度计算。假定完美类别与身份，仅检查有没有测量可用；不是实例IoU、算法召回、AP或官方成绩。全部94个合格GT保留，不能删掉未覆盖对象抬分。后续比较必须让所有方法共享完整输入，把传感器覆盖收益与算法收益分开。</p></section>
<section><h2>空隙见证：实现及有限预检</h2><p>候选从未归属实测表面出发，连接5–40厘米内的对象来源锚点。每窗口先按源点索引均匀抽取最多4096个原相机来源、4096个新增来源，再用同一个局部平面条件筛选。得到828／463条连接；没有用GT选点。</p>
<p>射线规则要求同一扫描支持连接两端，并在中点穿过空闲空间；遮挡和缺测均不算反证。它只拒绝传播连接，不删除端点，不把中点写入地图。比较简单中点自由空间、首个共同观测和全部有效观测。</p>
<table><tr><th>窗口</th><th>条件</th><th>全部切断</th><th>已知跨对象切断</th><th>同对象内部切断</th><th>已知对象边界精度</th></tr>{''.join(diagnostics)}</table>
<p>严格条件两窗口分别切对0／1条、切断同对象内部15／3条。原合格连接中仅7／6条跨对象边，因此此样本的边界召回与精度都有明显样本限制。物理空隙不等于实例边界；同对象内部切边也不必然造成整图实例分裂，因为可能存在其他连接。本次未通过预设必要门槛，不扩大完整建图，不据此声称所有射线方法无效。</p>
<p>五项核心测试通过；全部25个首次空隙见证用全扫描直接点积独立重算通过。当前使用帧级已发布位姿，没有新增逐点去畸变或不确定性校准主张。</p></section>
<section><h2>新增公开窗口已准备</h2><table><tr><th>序列</th><th>静态窗口</th><th>输入帧</th></tr>{newrows}</table>
<p>按官方验证划分、字典序、至少100个有效位姿和轨迹包围盒间隔大于200米选择，完整排除已暴露0000/0002/0003/0004/0005。共89幅图像、89份扫描、2份未解码标注，180个文件共393242406字节，校验通过。没有查看新图像或使用新标签选模型。</p>
<p>逐扫描检查完整原始几何，两新窗口与五个旧窗口、以及彼此的测量包围盒均分离，最小空间距离下界{reserved['minimum_raw_geometry_separation_m']:.3f}米。它证明当前输入几何不重叠，不代表跨域泛化已得到证明。这些是保留的官方验证窗口，不是隐藏测试服务器成绩；尚未执行候选预测或评分。</p></section>
<section><h2>机器人侧的真实输入条件</h2><p>330份归档语义点云均仅保存XYZ。当前LIVO发布代码使用体素滤波后的点，缓存没有逐点时间或激光射线原点，不能直接当原始射线使用。原rosbag2245仍在，/lidar_points有x、y、z、intensity、ring、FLOAT64 timestamp字段；首条消息9600点覆盖约{robot['raw_time_span_s']:.3f}秒。</p>
<p>因此将来若有效方法需要射线，必须从原始数据与位姿历史恢复相应测量来源。此次只核查了接口，没有完成该恢复或机器人方法对照。失败的空隙分支不额外扩展到机器人。</p></section>
<section><h2>主线与已有工作边界</h2><p>目标继续是有充分证据的对象级语义建图。下一机制必须面对完整实测表面，并区分局部表面连接与对象身份；只有相同输入下超越简单几何/语义传播，才讨论独立贡献。新增测量本身、经验归一化或射线剔除本身均不直接算创新。</p>
<p><a href="https://arxiv.org/abs/2403.09412">OpenGraph</a>已研究图像与LiDAR结合的室外对象图；<a href="https://arxiv.org/html/2507.08851v1">OTAS</a>使用视觉token聚类和几何条件化构建室外语义特征场。它们分别提示需要区分对象身份与语义场，不能把普通投影或token接入作为新贡献。本次没有复现它们，也没有将它们的成绩与本地AP直接混比。</p>
<p>有效新机制、同版本机器人对照和真正执行的新增评测仍待完成；强基线与历史实验保持不变。</p></section>
<section><h2>核查材料</h2><p><a href="coverage.json">全部94个对象覆盖</a> · <a href="gap_summary.json">全部预检条件</a> · <a href="../../build/benchmarks/full_lidar_surface_inputs_v1/raw_replay_audit.json">来源与见证重放</a> · <a href="../../build/datasets/kitti360_reserved_v3/window_selection.json">新窗口选择</a> · <a href="../../build/datasets/kitti360_reserved_v3/geometry_separation_audit.json">输入空间分离</a> · <a href="../../build/benchmarks/full_lidar_surface_inputs_v1/robot_ray_input_audit.json">机器人接口核查</a></p></section></html>'''
    (OUT/'index.html').write_text(page);dump(OUT/'coverage.json',coverage);dump(OUT/'gap_summary.json',probe);dump(OUT/'images.json',images)
    dump(OUT/'manifest.json',dict(complete=True,files={str(f):sha(f) for f in OUT.iterdir() if f.is_file() and f.name!='manifest.json'},source_sha256=sha(__file__),new_map_ap=False,goal_complete=False))
    print('FULL LIDAR REPORT',OUT/'index.html',flush=True)


if __name__=='__main__':run()
