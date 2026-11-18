"""Posthoc examples: all newly matched targets plus the worst car regression."""
import numpy as np
from paper_final_common import ROOT,read,dump
from run_surface_ownership import OUT,verify,sealed
from evaluate_kitti360_development import transfer


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=verify();sealed('predictions_complete');s=read(OUT/'summary.json');audit=read(OUT/'result_audit.json');report=ROOT/'output/surface_ownership_v1';manifest=[];html=''
    for scene in p['scenes']:
        key=scene['key']
        if not scene['reference']:continue
        ref=np.load(scene['reference']);clouds={};rows={};eligible={};matches={}
        for method in ['baseline','ours_competitive']:
            folder=OUT/'maps'/key/method;clouds[method]=np.load(folder/'geometry.npz');rows[method]=read(folder/'classification.json')['objects'];g=clouds[method]
            projected=transfer(g['xyz'],g['instance'],ref['xyz'],.2);mass=np.bincount(projected,weights=ref['confidence'],minlength=max(r['object_id'] for r in rows[method])+1)
            eligible[method]=[r['object_id'] for r in sorted(rows[method],key=lambda r:r['object_id']) if mass[r['object_id']]>=100]
            matches[method]={r['instance_id']:r for cat in read(folder/'matches.json')['window']['groundTruth'].values() for r in cat}
        gained=s['results'][key]['methods']['ours_competitive']['changes_vs_baseline']['gained'];cars=[r for r in audit['results'][key]['objects'] if r['category']=='car']
        worst=min(cars,key=lambda r:r['competitive_iou']-r['baseline_iou'])['gt_id'];cases=[(gid,'New AP50 match') for gid in gained]+[(worst,'Largest car IoU regression')]
        for gid,reason in cases:
            truth=ref['xyz'][ref['train_instance']==gid];center=truth.mean(0);coords=[];panels=[];details=[]
            for method in ['baseline','ours_competitive']:
                match=matches[method][gid];preds=match['matched_pred']
                if preds:
                    best=max(preds,key=lambda r:r['intersection']/(match['vert_count']+r['vert_count']-r['intersection']));oid=eligible[method][best['pred_id']];iou=best['intersection']/(match['vert_count']+best['vert_count']-best['intersection'])
                    g=clouds[method];indices=np.flatnonzero(g['instance']==oid);points=g['xyz'][indices];old=indices<len(clouds['baseline']['xyz'])
                else:oid=None;iou=0.;points=np.empty((0,3));old=np.empty(0,bool)
                coords.append(points);panels.append((method,points,old,iou));details.append(dict(method=method,object_id=oid,best_iou=iou,points=len(points)))
            union=np.concatenate([truth]+coords);lo=union.min(0);hi=union.max(0);span=np.maximum(hi-lo,.1);mid=(hi+lo)/2
            fig,axes=plt.subplots(1,2,figsize=(10,4.5),subplot_kw=dict(projection='3d'))
            sample=np.linspace(0,len(truth)-1,min(len(truth),6000),dtype=int)
            for ax,(method,points,old,iou) in zip(axes,panels):
                ax.scatter(*(truth[sample]-center).T,s=.4,c='#c6cdd3',alpha=.25)
                if len(points):
                    ix=np.linspace(0,len(points)-1,min(len(points),6000),dtype=int);ax.scatter(*(points[ix]-center).T,s=2,c=np.where(old[ix],'#137eab','#ec943c'),alpha=.9)
                ax.set_xlim(lo[0]-center[0]-.1,hi[0]-center[0]+.1);ax.set_ylim(lo[1]-center[1]-.1,hi[1]-center[1]+.1);ax.set_zlim(lo[2]-center[2]-.1,hi[2]-center[2]+.1);ax.set_box_aspect(span);ax.view_init(elev=22,azim=-55);ax.set_title(('Baseline' if method=='baseline' else 'Ours candidate')+f' | IoU {iou:.3f}');ax.set_xlabel('x / m');ax.set_ylabel('y / m');ax.set_zlabel('z / m')
            fig.suptitle(f'{key} / GT {gid}: {reason}\nGrey: reference; blue: core; orange: measured additions',fontsize=10);fig.tight_layout(rect=[0,0,1,.9]);name=f'{key}_gt{gid}.png';fig.savefig(report/name,dpi=160,bbox_inches='tight');plt.close(fig)
            manifest.append(dict(scene=key,gt_id=gid,reason=reason,methods=details,image=name));html+=f'<img src="{name}" alt="{key} GT {gid}">'
    dump(report/'object_cases.json',manifest)
    page=(report/'index.html').read_text();block='<h2>真实三维对象：全部新增匹配与每窗口最大车辆退步</h2><p>GT 仅用于事后解释选例。各对图共享视角和范围；范围包含完整参考对象及两组最佳匹配对象，避免裁掉错误扩展。显示每层最多 6,000 点，坐标相对参考对象中心。</p>'+html+'<p><a href="object_cases.json">对象编号、IoU 与选例规则</a></p>'
    (report/'index.html').write_text(page.replace('</html>',block+'</html>'));print('RENDERED',len(manifest),'cases',flush=True)


if __name__=='__main__':main()
