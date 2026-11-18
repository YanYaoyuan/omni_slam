"""Attribute frozen frame-balanced decisions and check sensor/model readout."""
from pathlib import Path
import time
import html
import runpy
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from audit_object_observations import OUT, DEST, PARENT


def interpolated_probabilities(logits, xy, height, width):
    # Independent align_corners=False sampling at original integer pixels.
    c, h, w = logits.shape
    x = np.clip((xy[:, 0] + .5) * w / width - .5, 0, w-1)
    y = np.clip((xy[:, 1] + .5) * h / height - .5, 0, h-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1,w-1), np.minimum(y0+1,h-1)
    wx, wy = x-x0, y-y0
    values = (logits[:, y0, x0] * (1-wx)*(1-wy) + logits[:, y0, x1]*wx*(1-wy)
              + logits[:, y1, x0]*(1-wx)*wy + logits[:, y1, x1]*wx*wy).T
    values -= values.max(1)[:, None]; p = np.exp(values)
    return p/p.sum(1)[:, None]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert not (OUT / 'influence_protocol.json').exists()
    prior = read(OUT / 'protocol.json'); summary = read(OUT / 'summary.json')
    contexts = read(PARENT / 'protocol.json')['contexts']
    inputs = {str(OUT/n): sha(OUT/n) for n in ['protocol.json','summary.json']}
    for c in contexts:
        folder = PARENT / 'evidence' / c['key']
        for n in ['observations.npz','before.npz']: inputs[str(folder/n)] = sha(folder/n)
    dump(OUT/'influence_protocol.json',dict(created_epoch=time.time(),source_sha256=sha(__file__),input_hashes=inputs,
         purpose='Posthoc attribution of same seven preselected failures, no new method or score. Reconstruct actual cell-balanced per-frame probabilities and equal frame weights, exhaustively describe leave-one-frame sensitivity without choosing a prediction. Verify selected probability vectors independently from native logits; independently project all robot case pixels using original camera pose and floor convention.',
         tolerance='FP64 independent bilinear logits/softmax vs saved GPU FP32: maxabs<=2e-5. Pixel reprojection exact. Equal-frame reconstruction maxabs<=1e-11.',
         scope='No GT, model forward, fit, class/geometry edit or AP. Small projected support and model mistakes are descriptive evidence, not a validated view-quality estimator.'))
    names={}
    for l in runpy.run_path(str(ROOT/'build/datasets/kitti360_prepare/labels.py'))['labels']:
        if 0<=l.trainId<19:names.setdefault(l.trainId,l.name)
    fig, axes = plt.subplots(7,2,figsize=(14,19),constrained_layout=True)
    results={}; row_idx=0
    colors={2:'#666666',8:'#4e9c43',11:'#df4372',13:'#126bac',14:'#6d50ae',17:'#17a6a0',18:'#e19d39'}
    for c in contexts:
        key=c['key']; results[key]={}
        folder=PARENT/'evidence'/key; obs=np.load(folder/'observations.npz'); before=np.load(folder/'before.npz')
        normalized=obs['probabilities']/obs['probabilities'].sum(1)[:,None]
        recs={r['frame_id']:r for r in read(c['observations'])['results'][key]}
        frames={r['frame_id']:r for r in read(c['job'])['frames']}
        full=np.load(Path(c['full'])/'geometry.npz'); mapping=np.load(Path(c['full'])/'old_to_full.npy'); owners=np.load(c['sem'])['owners']
        for oid_s, case in summary['results'][key].items():
            oid=int(oid_s); take=obs['cell_owner'][:,1]==oid; fids=np.unique(obs['observation_frame'][take])
            per_frame=np.array([normalized[take&(obs['observation_frame']==f)].mean(0) for f in fids])
            mean=per_frame.mean(0); expected=before['object_probability'][before['object_ids']==oid][0]
            np.testing.assert_allclose(mean,expected,rtol=0,atol=1e-11)
            counts={r['frame_id']:r for r in case['frames']}; records=[]; robot_pixels=0
            for i,f in enumerate(fids):
                record=counts[int(f)]
                leave=(per_frame.sum(0)-per_frame[i])/(len(fids)-1) if len(fids)>1 else None
                records.append(dict(frame_id=int(f),source_count=record['source_count'],unique_pixels=record['unique_pixels'],
                    cells=int((take&(obs['observation_frame']==f)).sum()),equal_frame_weight=1/len(fids),
                    cell_balanced_probability=per_frame[i].tolist(),leave_one_frame_argmax=int(leave.argmax()) if leave is not None else None))
                if c['kind']=='robot':
                    d=np.load(recs[int(f)]['file']); selected=owners[d['source_ids']]==oid
                    xyz=full['xyz'][mapping[d['source_ids'][selected]]]; frame=frames[int(f)]
                    inv=np.linalg.inv(np.asarray(frame['camera_pose'])); camera=xyz@inv[:3,:3].T+inv[:3,3]
                    k=np.asarray(frame['intrinsic']); uv=camera[:,:2]/camera[:,2:3]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
                    np.testing.assert_array_equal(np.floor(uv).astype(int),d['source_pixel_xy'][selected]); robot_pixels+=len(xyz)
            d=np.load(recs[case['selected_frame']]['file']); selected=np.load(case['data'])
            width,height=map(int,d['original_size'])
            independent=interpolated_probabilities(d['logits'].astype(np.float64),selected['pixel_xy'],height,width)
            error=float(np.abs(independent-selected['probabilities']).max())
            assert error<=2e-5,(key,oid,error)
            results[key][oid_s]=dict(final_probability=mean.tolist(),final_class=int(mean.argmax()),frames=records,
                selected_native_logits_max_abs_error=error,independently_projected_robot_pixels=robot_pixels,
                leave_one_frame_class_flips=sum(r['leave_one_frame_argmax']!=int(mean.argmax()) for r in records),
                max_min_pixel_count_ratio=max(r['unique_pixels'] for r in records)/min(r['unique_pixels'] for r in records))
            x=np.arange(len(fids)); ax=axes[row_idx,0]
            ax.bar(x,[r['unique_pixels'] for r in records],color='#14988f');ax.set_yscale('log')
            ax.set_xticks(x,[str(f) for f in fids],rotation=45);ax.set_ylabel('Distinct image pixels (log)')
            ax.set_title(f'{key}, object {oid}: every frame weight = 1/{len(fids)}')
            ax=axes[row_idx,1]
            chosen=[i for i in colors if (per_frame[:,i]>.15).any()]
            for i in chosen:ax.plot(x,per_frame[:,i],'-o',ms=3,color=colors[i],label=names[i])
            ax.set_xticks(x,[str(f) for f in fids],rotation=45);ax.set_ylim(0,1);ax.set_ylabel('Cell-balanced class probability')
            ax.set_title('Actual per-frame evidence; final = '+names[int(mean.argmax())]);ax.legend(fontsize=7,ncol=3)
            row_idx+=1
    fig.suptitle('Frozen decision attribution — visual support varies, but every observed frame receives equal weight',fontsize=14)
    fig.savefig(DEST/'frame_influence.png',dpi=150);plt.close(fig)
    for f,h in inputs.items(): assert sha(f)==h
    for f,h in prior['input_hashes'].items(): assert sha(f)==h
    dump(OUT/'influence.json',dict(complete=True,results=results,gt_read=False,new_maps=0,new_scores=0,model_calls=0,
         protocol_sha256=sha(OUT/'influence_protocol.json'),figure_sha256=sha(DEST/'frame_influence.png')))
    rows=[]; images=[]
    for key,cases in results.items():
        for oid,r in cases.items():
            rows.append(f'<tr><td>{key}/{oid}</td><td>{len(r["frames"])}</td><td>{r["max_min_pixel_count_ratio"]:.1f}</td><td>{names[r["final_class"]]}</td><td>{r["selected_native_logits_max_abs_error"]:.2g}</td></tr>')
            src=Path(summary['results'][key][oid]['figure']).name
            images.append(f'<h3>{key} / object {oid}</h3><img src="{html.escape(src)}">')
    (DEST/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>omni_slam — object observation audit</title>
<style>body{max-width:1400px;margin:40px auto;padding:0 25px;font:16px/1.6 system-ui;color:#173245}img{width:100%}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #ddd;padding:8px}th{background:#eef7f6}</style>
<h1>对象证据是否真的看到了对象？</h1><p>全部7个上一轮类别变化对象，固定选择最大可见来源点帧；不按GT或视觉效果挑图。0新模型/训练/地图/AP。所有原预测保持，未证明新方法收益。</p>
<p>室外案例每次观测的投影与缓存完全一致，机器人案例也以原camera_pose和floor规则独立复算。所展示帧的概率从保存的原生logits独立插值、softmax重建。尚未据此证明完整物理标定或像素遮挡处理绝对正确。</p>
<p>可见证据包含两种问题：远处或稀少观测与清晰观测等权；单帧识别本身错误。卡车近处可识别为车辆，摩托车最密观测却被识别为行人，车库暗车被识别成建筑/交通标志。重复投票不能自动生成缺失的正确语义。</p>
<table><tr><th>对象</th><th>观测帧</th><th>最大/最小像素数</th><th>冻结类别</th><th>独立概率重建误差</th></tr>'''+''.join(rows)+'''</table>
<h2>每帧观测支持与真实权重</h2><p>曲线是原算法实际使用的单元均衡帧概率，不是新的加权方案。只显示至少一次概率超过0.15的常见类别，其余概率仍完整保存在JSON；没有重新选择类别或输出地图。</p><img src="frame_influence.png">
<h2>原图、同区域原图、已有预测与投影点</h2><p>黄色点是原地图对象的实测来源投影；颜色不等于GT。各图包含未经叠加的相同原图区域，便于人工检查。</p>'''+''.join(images)+'''
<h2>决定</h2><p>不继续上一轮语义兼容阈值分支；需要在完整对象建图目标下检验观测分辨率/覆盖与对象更新的关系，并包含简单像素加权或最佳视图等强控制。观测加权本身不能自动成为论文创新，也不能声称已解决当前前端识别错误。本报告只提供下一研究决定的具体证据。</p>''')
    dump(OUT/'delivery_audit.json',dict(complete=True,original_inputs_preserved=True,
         files={str(f):sha(f) for f in [DEST/'index.html',DEST/'frame_influence.png',OUT/'summary.json',OUT/'influence.json']},
         source_sha256=sha(__file__),new_maps=0,new_scores=0,model_calls=0))
    print('OBSERVATION INFLUENCE VERIFIED',flush=True)


if __name__=='__main__':main()
