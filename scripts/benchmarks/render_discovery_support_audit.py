"""Actual RGB/support examples selected posthoc for diagnosing missed geometry."""
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from run_discovery_query_benchmark import OUT,INPUT,verify
from audit_discovery_full_map import DEST,STAGES
from evaluate_kitti360_development import transfer


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    p=verify(True);summary=read(DEST/'summary.json');visible=read(DEST/'visible_coverage.json')
    masks=read(OUT/'masks_complete.json');report=ROOT/'output/discovery_full_map_audit_v1';report.mkdir(parents=True,exist_ok=True)
    body='';provenance={}
    for s in p['scenes']:
        if not s['reference']:continue
        key=s['key'];result=summary['results'][key];v={o['gt_id']:o['visible_coverage'] for o in visible['results'][key]['objects']}
        obj=max((o for o in result['objects'] if o['category']=='building'),key=lambda o:(v[o['gt_id']]-o['coverage']['raw_masks'],-o['gt_id']))
        ref=np.load(s['reference']);manifest=read(INPUT/key/'manifest.json');source=np.load(manifest['geometry'])['xyz']
        # Diagnostic GT transfer only; never written to or read by any predictor.
        source_gt=transfer(ref['xyz'],ref['train_instance'],source,.2);target=source_gt==obj['gt_id']
        candidates=[]
        for r in manifest['records']:
            ids=np.load(r['file'])['visible_source_ids'];candidates.append((int(target[ids].sum()),-r['frame_id'],r))
        count,_,record=max(candidates,key=lambda r:r[:2]);fid=record['frame_id']
        cached=next(r for r in masks['scenes'][key]['records'] if r['frame_id']==fid);z=np.load(cached['file'])
        m=np.load(record['file']);ids=m['visible_source_ids'];px=m['pixel_xy'];valid=target[ids];labels=z['segmentation'][px[:,1],px[:,0]];accepted=labels>=0
        rgb=np.asarray(Image.open(record['rgb']).convert('RGB'));seg=z['segmentation'];color=plt.get_cmap('tab20')(np.maximum(seg,0)%20)[...,:3]*255
        overlay=rgb.astype(float).copy();covered=seg>=0;overlay[covered]=.55*overlay[covered]+.45*color[covered]
        fig,axes=plt.subplots(1,3,figsize=(17,3))
        axes[0].imshow(rgb);axes[1].imshow(overlay.astype(np.uint8));axes[2].imshow(rgb)
        for ok,c,label in [(True,'#00d9aa','Inside a predicted mask'),(False,'#ff3864','Outside all predicted masks')]:
            take=valid&(accepted==ok);axes[2].scatter(px[take,0],px[take,1],s=8,c=c,alpha=.85,label=label)
        axes[2].legend(fontsize=8,loc='lower right')
        for ax,title in zip(axes,['Original RGB','Actual shared SAM3 masks','GT-labelled measured source projection (audit only)']):ax.set_title(title,fontsize=10);ax.axis('off')
        fig.suptitle(f'{key} / GT building {obj["gt_id"]} / frame {fid} | {int(np.sum(valid&accepted))}/{count} measured supports inside masks')
        fig.tight_layout();fig.savefig(report/(key+'_missing_support.png'),dpi=160,bbox_inches='tight');plt.close(fig)
        provenance[key]=dict(gt_id=obj['gt_id'],frame_id=fid,visible_target_sources=count,inside_mask_sources=int(np.sum(valid&accepted)),
            rgb=record['rgb'],rgb_sha256=sha(record['rgb']),mask_file=cached['file'],mask_sha256=sha(cached['file']),reference_sha256=sha(s['reference']),
            selection='Posthoc: building with largest verified-visible-to-raw-mask coverage loss; frame with most target measured points, earliest tie. GT projection for explanation only, not a new prediction or evaluation.')
        table=''
        stages=['measurements','visible','raw_masks','connected_support','three_frame_groups','final_members']
        for stage in stages:
            counts=visible['results'][key]['coverage_over_half'] if stage=='visible' else {c:result['totals'][c]['coverage_over_half'][stage] for c in ['car','building']}
            table+=f'<tr><td>{stage}</td><td>{counts["car"]}/{result["totals"]["car"]["gt"]}</td><td>{counts["building"]}/{result["totals"]["building"]["gt"]}</td></tr>'
        body+=f'<h2>{key}</h2><table><tr><th>支持阶段</th><th>车辆覆盖 &gt; 50%</th><th>建筑覆盖 &gt; 50%</th></tr>{table}</table><p>最终单对象正确匹配 / 理想同类碎片合并：车辆 {result["totals"]["car"]["single_matched"]} / {result["totals"]["car"]["oracle_merge_matched"]}；建筑 {result["totals"]["building"]["single_matched"]} / {result["totals"]["building"]["oracle_merge_matched"]}。</p><img src="{key}_missing_support.png"><p>此图按 GT 支持损失选择，用于诊断，不是代表性随机样本。青色是落在预测掩码内的该建筑实测点，红色是未被任何预测掩码覆盖的实测点。GT 通过参考域 20 cm 最近邻转移，仅供显示。</p>'
    dump(report/'examples.json',provenance);dump(report/'summary.json',summary);dump(report/'visible_coverage.json',visible)
    (report/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 完整对象地图支持审计</title><style>body{{font:16px/1.7 system-ui;max-width:1400px;margin:35px auto;padding:0 24px;color:#19334a}}table{{border-collapse:collapse}}td,th{{padding:8px 30px;border:1px solid #cdd6df}}img{{width:100%}}.note{{padding:20px;background:#edf4f7}}</style><h1>完整地图：对象支持在哪里丢失</h1><p class="note">本报告只复核上一轮已封存的全帧 SAM3 + CG/history2 地图，无新模型推理、训练或参数选择。现有碎片即使按真值理想合并，也不能新增正确匹配；下游关联不是这两个窗口的主要可恢复缺口。</p><p>阶段覆盖是逐目标理想点级支持上界，允许完美区分前景与背景，并对各源域独立执行 20 cm 最近邻查询；它不是 AP 或可实现的召回率。所有比率由官方参考置信度加权。理想合并只改变现有同类预测区域的分组，不删除点、不补点；每个目标分别优化，不能解释为一张可实现的联合预测地图。</p>{body}<p><a href="summary.json">逐对象诊断</a> · <a href="visible_coverage.json">分割之前的可见性覆盖</a> · <a href="examples.json">图片选择与来源</a></p></html>''')
    print('REPORT',report/'index.html',flush=True)


if __name__=='__main__':main()
