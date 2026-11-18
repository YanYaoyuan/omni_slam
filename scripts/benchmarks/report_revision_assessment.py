"""Actual outdoor images/maps, full scores and all-object gains/losses."""
from pathlib import Path
import html
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from run_revision_assessment import OUT, FULL, verify, sealed
from evaluate_revision_assessment import METHODS, MODES
from evaluate_kitti360_development import transfer

DEST = ROOT / 'output/revision_assessment_v1'
NAMES = dict(strong_baseline='Strong baseline', unconditional='Unconditional revision',
             history_only='History-only control', full_authorization='Ours candidate')


def fmt(value, scale=1.):
    return 'NA' if value is None else f'{value*scale:.4f}'


def table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>'+html.escape(str(h))+'</th>' for h in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table>'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p = verify()
    s = read(OUT / 'summary.json')
    assert s['complete'] and not DEST.exists()
    sealed(OUT / 'comparison_maps_complete.json')
    assert sha(OUT / 'comparison_maps_complete.json') == s['maps_seal_sha256']
    objects = read(OUT / 'per_object.json')['results']
    DEST.mkdir(parents=True)
    blocks = ['<h1>omni_slam · 新增室外对象地图评测</h1>',
        '<p>固定四组方法；KITTI-360 0009/0010，84对采样图像/雷达。主协议为完整静态参考上的 building/car AP50:90。'
        '公开数据使用官方位姿，车库使用既有LIVO结果；本轮评测对象建图。'
        '本地适配评测，非官方榜单。Ours candidate 尚未自动晋级；旧开发门槛保持失败。</p>',
        '<h2>固定门槛与实际结果</h2>',
        table(['项目','结果'], [('新增评测门槛',s['prospective_gate']), *s['efficacy'].items()]),
        '<p>所有8份地图封存后才读取参考。缺少有效目标时保留NA；不替换窗口或根据结果选方法。</p>']
    rows=[]
    for key, methods in s['results'].items():
        for m in METHODS:
            metrics=methods[m]['primary']['metrics']
            changes=methods[m]['raw_targets']['changes_vs_baseline']
            rows.append([key,NAMES[m],fmt(metrics['allAp'],100),fmt(metrics['allAp50%'],100),
                         ', '.join(map(str,changes['gained'])) or '—',', '.join(map(str,changes['lost'])) or '—'])
    blocks.append(table(['窗口','方法','AP50:90 (%)','AP50 (%)','新增原始目标匹配','损失原始目标匹配'],rows))
    blocks.append(table(['窗口','参考协议','有效建筑实例','有效车辆实例'],
        [[key,mode,counts['2'],counts['13']] for key,modes in s['reference_target_counts'].items()
         for mode,counts in modes.items()]))
    blocks.append(table(['协议']+[NAMES[m] for m in METHODS],
        [[mode]+[fmt(s['mean_ap'][mode][m],100) for m in METHODS] for mode in MODES]))
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,(key,methods) in zip(axes,s['results'].items()):
        vals=[methods[m]['primary']['metrics']['allAp'] for m in METHODS]
        y=[v*100 if v is not None else 0 for v in vals]
        bars=ax.bar(np.arange(4),y,color=['#8594a5','#d29b68','#3c81b0','#119c83'])
        ax.bar_label(bars,labels=[fmt(v,100) for v in vals],fontsize=9)
        ax.set_xticks(np.arange(4),['Strong\nbaseline','Unconditional','History-only','Ours\ncandidate'])
        ax.set_ylim(0,max(y+[1])*1.2);ax.set_ylabel('AP50:90 (%)');ax.set_title(key)
    fig.savefig(DEST/'scores.png',dpi=160);plt.close(fig)
    blocks.append('<img src="scores.png" alt="All fixed methods on both windows">')
    visuals=[]
    colors=plt.get_cmap('turbo')
    def rgba(ids):
        return colors(((ids.astype(np.int64)*37)%251)/250.)
    for scene in p['scenes']:
        key=scene['key'];job=read(scene['job']);full=np.load(FULL/key/'geometry.npz')['xyz']
        maps={m:dict(np.load(OUT/'comparison_maps'/key/m/'geometry.npz')) for m in METHODS}
        blocks.append('<h2>'+html.escape(key)+'：真实地图与原始图像</h2>')
        bounds=np.array([full[:,:2].min(0),full[:,:2].max(0)])
        context=full[::max(1,int(np.ceil(len(full)/60000)))];fig,axes=plt.subplots(1,4,figsize=(16,5),constrained_layout=True)
        for ax,m in zip(axes,METHODS):
            g=maps[m];ids=np.arange(0,len(g['xyz']),max(1,int(np.ceil(len(g['xyz'])/40000))))
            ax.scatter(context[:,0],context[:,1],c='#d8dde2',s=.2,rasterized=True)
            ax.scatter(g['xyz'][ids,0],g['xyz'][ids,1],c=rgba(g['instance'][ids]),s=.6,rasterized=True)
            ax.set(xlim=bounds[:,0],ylim=bounds[:,1],xlabel='x (m)',ylabel='y (m)',title=NAMES[m]);ax.set_aspect('equal')
        image=key+'_maps.png';fig.savefig(DEST/image,dpi=160);plt.close(fig)
        blocks.append('<img src="'+image+'" alt="Shared-coordinate object maps">')
        blocks.append('<p>俯视图使用完整实测来源的共同范围，灰色为实测环境，颜色为预测对象ID。'
                      '固定步长绘制最多6万环境点和4万对象点；评分使用完整地图。</p>')
        views=read(OUT/'visibility'/key/'manifest.json')['records']
        for rank in sorted({0,len(views)//2,len(views)-1}):
            view=views[rank];frame=next(f for f in job['frames'] if f['frame_id']==view['frame_id'])
            vis=dict(np.load(view['file']));source=full[vis['visible_source_ids']];xy=vis['pixel_xy']
            rgb=np.asarray(Image.open(frame['rgb']).convert('RGB'))
            fig,axes=plt.subplots(1,4,figsize=(16,2.6),constrained_layout=True)
            for ax,m in zip(axes,METHODS):
                g=maps[m];labels=transfer(g['xyz'],g['instance'],source,.05);take=labels>0
                ax.imshow(rgb);ax.scatter(xy[take,0],xy[take,1],c=rgba(labels[take]),s=1.,alpha=.85)
                ax.set_xlim(0,rgb.shape[1]);ax.set_ylim(rgb.shape[0],0);ax.set_title(NAMES[m],fontsize=10);ax.axis('off')
            fig.suptitle(key+' / actual RGB frame '+str(frame['frame_id'])+' / predicted source projection',fontsize=11)
            image=f'{key}_rgb_{frame["frame_id"]}.png';fig.savefig(DEST/image,dpi=150);plt.close(fig)
            blocks.append('<img src="'+image+'" alt="Actual RGB with predicted object projections">')
            visuals.append(dict(key=key,frame_id=frame['frame_id'],file=image,selection='first,median,last cached frame',
                                image_sha256=sha(frame['rgb']),display_transfer_m=.05,score_transfer_m=.2))
        blocks.append('<p>首帧、中位帧、末帧固定选取；所有方法共享同一原始RGB和实测深度可见域。'
                      '彩点为预测投影，不是像素分割GT；展示转移半径5cm，评分半径20cm。</p>')
        blocks.append('<h3>全部对象得失（参考置信度加权IoU）</h3>')
        for mode in ['primary','raw_targets']:
            obj=objects[key][mode]
            rows=[]
            for item in obj:
                values=item['methods'];delta=values['full_authorization']['best_iou']-values['strong_baseline']['best_iou']
                rows.append([item['gt_id']]+[fmt(values[m]['best_iou'])+' / ID '+str(values[m]['object_id']) for m in METHODS]+[f'{delta:+.4f}'])
            blocks.append('<details><summary>'+html.escape(mode)+f' · {len(rows)} objects</summary>'+
                table(['GT ID']+[NAMES[m]+' IoU / matched ID' for m in METHODS]+['Ours−strong IoU'],rows)+'</details>')
    robot=read(OUT/'robot_correspondence_audit.json')
    blocks.append('<h2>同版本车库对照</h2>'+table(['方法','点数','非空ID','旧点ID改动'],
        [[NAMES[m],robot['methods'][m]['points'],robot['methods'][m]['nonempty_object_ids'],robot['methods'][m]['old_point_id_changes']] for m in METHODS]))
    blocks.append('<p>复用已验证的330帧车库四版输出；无独立语义GT，不报告机器人AP或S100实时性。'
                  '<a href="../revision_robot_v1/index.html">查看车库真实RGB与地图</a>。</p>')
    blocks.append('<p>本轮不在新结果上调参。参考映射另以每图16个固定索引穷举检查，共'+str(s['brute_force_reference_samples'])+
                  '个样本；这是抽样核对。所有对象与失配均保留，模型运行失败及恢复记录也保留。</p>')
    css='body{font-family:system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 24px;color:#183344}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}th,td{border-bottom:1px solid #dae2e7;padding:9px;text-align:left}th{background:#edf5f4}img{width:100%;margin:14px 0}details{margin:16px 0}summary{cursor:pointer;font-weight:bold}p{line-height:1.7}h2{margin-top:36px}'
    (DEST/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>omni_slam outdoor revision assessment</title><style>'+css+'</style><body>'+''.join(blocks)+'</body></html>')
    dump(OUT/'report_audit.json',dict(complete=True,source_sha256=sha(__file__),summary_sha256=sha(OUT/'summary.json'),
        per_object_sha256=sha(OUT/'per_object.json'),visuals=visuals,files={str(f):sha(f) for f in DEST.iterdir() if f.is_file()},
        visual_review_pending=True,goal_complete=False))
    print('NEW OUTDOOR REPORT',DEST/'index.html',flush=True)


if __name__=='__main__':main()
