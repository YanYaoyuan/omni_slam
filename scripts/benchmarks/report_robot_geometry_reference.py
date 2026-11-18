"""Auditable real-robot reference comparison and actual-image-colored PLYs."""
from html import escape
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from paper_final_common import ROOT,read,dump,sha
from run_robot_geometry_reference import OUT,INPUT,BASE,KEY,verify

REPORT=ROOT/'output/robot_geometry_reference_v1'
METHODS=['native_core','intrinsic_geometry','geometry_graph_reference']
NAMES=['原始对象核心（既有结果）','不变几何归属（既有结果）','几何关联图（本次执行的参考）']


def write_ply(path,xyz,colors,labels,valid,added):
    dtype=np.dtype([('x','<f8'),('y','<f8'),('z','<f8'),('red','u1'),('green','u1'),('blue','u1'),('object_id','<i4'),('color_valid','u1'),('new_point','u1')]);data=np.empty(len(xyz),dtype)
    for j,name in enumerate(['x','y','z']):data[name]=xyz[:,j]
    for j,name in enumerate(['red','green','blue']):data[name]=colors[:,j]
    data['object_id']=labels;data['color_valid']=valid;data['new_point']=added
    header='ply\nformat binary_little_endian 1.0\ncomment RGB sampled from depth-verified real images; unobserved color gray\nelement vertex '+str(len(data))+'\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty int object_id\nproperty uchar color_valid\nproperty uchar new_point\nend_header\n'
    with path.open('wb') as stream:stream.write(header.encode());data.tofile(stream)
    assert path.stat().st_size==len(header.encode())+len(data)*dtype.itemsize


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=verify();seal=read(OUT/'maps_complete.json');assert seal['complete']
    for f,h in seal['files'].items():assert sha(f)==h,f
    raw=read(OUT/'raw/predictions_complete.json');assert raw['old_replay_exact'];REPORT.mkdir(parents=True,exist_ok=True)
    manifest=read(INPUT/KEY/'manifest.json');source=np.load(manifest['geometry'])['xyz'];colors=np.full((len(source),3),160,np.uint8);color_frame=np.full(len(source),-1,np.int64);color_xy=np.full((len(source),2),-1,np.int32)
    for rec in manifest['records']:
        view=np.load(rec['file']);ids=view['visible_source_ids'];pixels=view['pixel_xy'];fresh=color_frame[ids]<0;ids,pixels=ids[fresh],pixels[fresh]
        rgb=np.asarray(Image.open(rec['rgb']).convert('RGB'));assert np.all((pixels[:,0]>=0)&(pixels[:,0]<rgb.shape[1])&(pixels[:,1]>=0)&(pixels[:,1]<rgb.shape[0]))
        colors[ids]=rgb[pixels[:,1],pixels[:,0]];color_frame[ids]=rec['frame_id'];color_xy[ids]=pixels
    colored=np.flatnonzero(color_frame>=0);tree=cKDTree(source[colored]);maps={};audits={};rows='';boxes={};ply=[]
    native=np.load(OUT/'maps/native_core/geometry.npz');intrinsic=np.load(OUT/'maps/intrinsic_geometry/geometry.npz');n=len(intrinsic['xyz']);groups=read(OUT/'maps/geometry_graph_reference/groups.json');assignment={int(k):v for k,v in groups['old_id_assignment'].items()}
    for method,title in zip(METHODS,NAMES):
        folder=OUT/'maps'/method;g=dict(np.load(folder/'geometry.npz'));records=read(folder/'classification.json');maps[method]=g
        distance,ix=tree.query(g['xyz'],workers=1);valid=distance<=.05;color_ids=colored[ix];rgb=np.full((len(g['xyz']),3),160,np.uint8);rgb[valid]=colors[color_ids[valid]]
        if method=='geometry_graph_reference':
            np.testing.assert_array_equal(g['xyz'][:n],intrinsic['xyz']);np.testing.assert_array_equal(g['instance'][:n],[assignment[int(i)] for i in intrinsic['instance']])
            ids=g['added_source_ids'];assert len(np.unique(ids))==len(ids);np.testing.assert_array_equal(g['xyz'][n:],source[ids]);rgb[n:]=colors[ids];valid[n:]=color_frame[ids]>=0;color_ids[n:]=ids
            assert len(g['xyz'])-n==seal['stats'][method]['added_points']
        else:
            original=BASE/'maps'/KEY/('baseline' if method=='native_core' else 'intrinsic_geometry')
            for name in ['geometry.npz','classification.json']:assert sha(folder/name)==sha(original/name)
        added=np.arange(len(g['xyz']))>=len(native['xyz']);path=REPORT/(method+'_rgb.ply');write_ply(path,g['xyz'],rgb,g['instance'],valid,added);ply.append(path.name)
        np.savez_compressed(REPORT/(method+'_color_provenance.npz'),source_ids=np.where(valid,color_ids,-1),source_frame=np.where(valid,color_frame[color_ids],-1),source_pixel_xy=np.where(valid[:,None],color_xy[color_ids],-1),color_valid=valid)
        maps[method]['rgb']=rgb;boxes[method]=[]
        for obj in records['objects']:
            oid=obj['object_id'];points=g['xyz'][g['instance']==oid]
            assert len(points)
            boxes[method].append(dict(object_id=oid,semantic=obj['predictions']['mean'],points=len(points),min=points.min(0).tolist(),max=points.max(0).tolist()))
        audits[method]=dict(points=len(g['xyz']),objects=len(records['objects']),true_rgb_points=int(valid.sum()),color_missing_points=int((~valid).sum()),geometry_sha256=sha(folder/'geometry.npz'),ply_sha256=sha(path),old_coordinates_exact=True)
        rows+='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [title,len(records['objects']),len(g['xyz']),len(g['xyz'])-len(native['xyz']),f'{valid.mean()*100:.1f}%'])+'</tr>'
    np.savez_compressed(REPORT/'source_color_provenance.npz',source_frame=color_frame,source_pixel_xy=color_xy)
    fig,axes=plt.subplots(1,3,figsize=(16,6),sharex=True,sharey=True);background=source[::max(1,len(source)//20000)]
    for ax,method in zip(axes,METHODS):
        g=maps[method];xyz=g['xyz'];labels=g['instance'];sample=np.arange(0,len(xyz),max(1,len(xyz)//45000));palette=np.stack([(labels*67%211+25)/255,(labels*103%211+25)/255,(labels*151%211+25)/255],axis=1)
        ax.scatter(background[:,0],background[:,1],s=.1,c='#d6dce2',rasterized=True);ax.scatter(xyz[sample,0],xyz[sample,1],s=.7,c=palette[sample],rasterized=True);ax.set_aspect('equal');ax.set_title(method+'\n'+str(audits[method]['objects'])+' predicted objects');ax.set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)');fig.suptitle('rosbag2245 / same real LIVO input / colors are predicted instance IDs, not GT');fig.tight_layout();fig.savefig(REPORT/'comparison.png',dpi=160);plt.close(fig)
    g=maps['geometry_graph_reference'];xyz=g['xyz'];fig,axes=plt.subplots(1,2,figsize=(15,6));sample=np.arange(0,len(xyz),max(1,len(xyz)//100000))
    for ax,(x,y),title in zip(axes,[(0,1),(0,2)],['Top view: x/y','Side view: x/z']):
        ax.scatter(xyz[sample,x],xyz[sample,y],s=.8,c=g['rgb'][sample]/255,rasterized=True);ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)' if y==1 else 'z (m)')
    fig.suptitle('Actual-image RGB / unobserved points gray / no invented surface');fig.tight_layout();fig.savefig(REPORT/'actual_rgb_map.png',dpi=160);plt.close(fig)
    added=g['added_source_ids'];choices=[]
    for rec in manifest['records']:
        v=np.load(rec['file']);take=np.isin(v['visible_source_ids'],added);choices.append((int(take.sum()),-rec['frame_id'],rec,v['pixel_xy'][take]))
    count,_,rec,pixels=max(choices,key=lambda x:x[:2]);rgb=np.asarray(Image.open(rec['rgb']).convert('RGB'));fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax in axes:ax.imshow(rgb);ax.axis('off')
    axes[1].scatter(pixels[:,0],pixels[:,1],s=4,c='#fa973b');axes[0].set_title('Actual RGB / frame '+str(rec['frame_id']));axes[1].set_title(str(count)+' newly appended measurements visible here');fig.tight_layout();fig.savefig(REPORT/'actual_evidence.png',dpi=150);plt.close(fig)
    stat=seal['stats']['geometry_graph_reference'];changed=[dict(old_id=k,new_id=v) for k,v in assignment.items() if k!=v]
    dump(REPORT/'object_boxes.json',dict(kind='Axis-aligned bounds derived from predicted object points, not learned3Ddetections or GT',methods=boxes));dump(REPORT/'audit.json',dict(complete=True,maps=audits,old_identity_changes=changed,raw_frontend_replayed_frames=len(raw['results'][KEY]),all_high_outputs_exact=True,new_reference_stats=stat,protocol_sha256=sha(OUT/'protocol.json'),maps_seal_sha256=sha(OUT/'maps_complete.json'),color_rule='First actual measured-visible RGB per source; old processed points use nearest colored source within5cm for visualization. New source points use exact source IDs. Unobserved gray with color_valid0. Placeholder geometry colors/pixels never used.',evidence_image=dict(frame_id=rec['frame_id'],file=rec['rgb'],sha256=sha(rec['rgb']),visible_added=count),robot_accuracy_available=False,new_method_validated=False))
    downloads=' · '.join(f'<a href="{name}">{title}：RGB点云</a>' for name,title in zip(ply,NAMES))
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>omni_slam · 车库稳定参考对照</title><style>body{{font:16px/1.7 system-ui;color:#193349;max-width:1450px;margin:32px auto;padding:0 24px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d5dfe8;padding:9px}}.note{{background:#edf5f6;padding:20px}}</style>
<h1>车库330帧：稳定几何关联参考对照</h1><p class="note">原始对象核心和不变几何归属为明确复用的既有结果；几何关联图为本次新增执行。它是室外更强参考链路的机器人移植，不是新机制获胜证明。没有独立机器人标签，不报告语义准确率、正确对象新增数或S100实时性能。</p>
<table><tr><th>版本</th><th>预测对象</th><th>地图点</th><th>相对原核心追加点</th><th>真实RGB覆盖</th></tr>{rows}</table>
<p>几何关联参考合并{stat['old_ids_merged']}个旧ID，改写{stat['old_point_ids_changed']}个旧点归属，形成{stat['new_objects']}个新预测对象；相对不变几何底图追加{stat['added_points']}个真实测量点。上述是行为统计，不代表正确性。逐对象包围框和旧ID变化均保留供核查。</p>
<h2>同输入地图对照</h2><img src="comparison.png"><h2>真实RGB点云</h2><img src="actual_rgb_map.png"><p>颜色来自实测可见图像；未观测点为灰色，并有color_valid字段。实例图的彩色是ID显示，RGB点云的颜色才是相机取色。旧处理点使用5cm内最近有颜色源点进行显示；新测量点按精确源ID取色。取色不改变算法坐标或归属。</p>
<p>{downloads}</p><h2>原始图像与新增测量证据</h2><img src="actual_evidence.png"><p>选择新增测量可见数最多的帧，同数取最早；橙点是测量投影，不是人工车辆轮廓或正确性标签。</p>
<h2>复现与限制</h2><p>适配器先精确复现两个已有室外窗口的完整XYZ、实例、分类和分组；330帧原高分前端输出全部精确复现。对象来源、旧坐标、新增源ID及颜色来源均可追踪。本次330图像编码/660类别解码另有固定合成预热；下游离线窗口末执行，无新增训练或阈值搜索。两条失效新机制没有在本次运行。</p><p><a href="audit.json">核查记录</a> · <a href="object_boxes.json">全部对象及三维包围框</a> · <a href="../../build/benchmarks/robot_geometry_reference_v1/protocol.json">冻结协议</a></p></html>'''
    (REPORT/'index.html').write_text(page);dump(REPORT/'manifest.json',dict(generator_sha256=sha(__file__),files={f.name:sha(f) for f in REPORT.iterdir() if f.name!='manifest.json'}));print('ROBOT REPORT',REPORT/'index.html',audits,flush=True)


if __name__=='__main__':main()
