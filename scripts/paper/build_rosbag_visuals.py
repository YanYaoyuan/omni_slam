"""Additional real ROSBAG plates from archived runs; no inference or mapping."""
from pathlib import Path
from itertools import product
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import open3d as o3d

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/paper/visibility_revision/figures'
EXP=ROOT/'build/experiments'
GEOM=EXP/'boxes_ground_20261106/geometry'
LAND=EXP/'large_semantic_20261106/landmarks_final'
INK='#183247';GRAY='#738597';GREEN='#149e79';BLUE='#488dec'
PROV=[]

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(OUT/(name+'.'+ext),dpi=220,facecolor='white')
    plt.close(fig)

def read(p,use):return json.loads(use(p).read_text())
def cloud(p,use):
    c=o3d.io.read_point_cloud(str(use(p)));a=np.asarray(c.points)
    assert len(a)>0 and np.isfinite(a).all()
    return a

def im(ax,p,title,use):
    ax.imshow(Image.open(use(p)));ax.axis('off');ax.set_title(title,fontsize=11,color=INK,weight='bold',loc='left',pad=6)

def six_bags(use):
    audit=read(ROOT/'docs/experiments/fast_livo2_vbot_map_audit_2026-11-05/summary.json',use)
    inventory=read(EXP/'full_20261105/inventory.json',use);inventory={r['name']:r for r in inventory}
    rows=audit['bags'];records=[]
    for part,group in enumerate([rows[:3],rows[3:]]):
        fig=plt.figure(figsize=(12,10));fig.text(.025,.97,'Real ROSBAG recordings  |  '+('garage routes' if part==0 else 'outdoor routes'),fontsize=19,weight='bold',color=INK,va='top')
        fig.text(.025,.93,'Archived LIVO + reviewed offline backend; geometry examples, not visibility1 semantic tests',fontsize=10,color=GRAY)
        for row,r in enumerate(group):
            name=r['bag'];mapfile=Path(r['reviewed_maps']['LIVO']['map_path']);p=cloud(mapfile,use)
            stride=max(1,int(np.ceil(len(p)/160000)));q=p[::stride]
            trajectory=np.loadtxt(use(mapfile.parent/'trajectory_corrected.tum'))[:,1:4]
            source=EXP/'z_20261105/si_full'/name/'sample_undistorted.jpg'
            y=.67-row*.28;distance=r['trajectory_metrics']['LIVO_before']['path_m'];duration=inventory[name]['duration_s']
            fig.text(.025,y+.221,name,fontsize=12,weight='bold',color=INK)
            a=fig.add_axes([.025,y,.365,.205]);im(a,source,'',use)
            a=fig.add_axes([.43,y-.005,.54,.23]);a.set_facecolor('#132635')
            lo,hi=np.percentile(q[:,:2],[.5,99.5],axis=0);pad=np.maximum((hi-lo)*.045,.2)
            order=np.argsort(q[:,2]);a.scatter(q[order,0],q[order,1],c=q[order,2],cmap='viridis',vmin=np.percentile(q[:,2],2),vmax=np.percentile(q[:,2],98),s=.28,lw=0,rasterized=True)
            a.plot(trajectory[:,0],trajectory[:,1],c='#ffe2a3',lw=.85)
            a.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);a.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);a.set_aspect('equal');a.set_xticks([]);a.set_yticks([])
            # Scale bar in unchanged map coordinates.
            length=5 if hi[0]-lo[0]<80 else 20;x=lo[0]+.04*(hi[0]-lo[0]);yy=lo[1]+.06*(hi[1]-lo[1]);a.plot([x,x+length],[yy,yy],c='white',lw=2);a.text(x,yy+.025*(hi[1]-lo[1]),f'{length} m',color='white',fontsize=9)
            fig.text(.025,y-.025,f'{duration/60:.1f} min  |  {distance:.1f} m estimated path  |  {len(p):,} map points',fontsize=10,color=GRAY)
            entry=dict(bag=name,raw_bag=inventory[name]['path'],raw_bag_exists=Path(inventory[name]['path']).is_file(),duration_s=duration,estimated_path_m=distance,source_rgb=str(source.relative_to(ROOT)),map=str(mapfile.relative_to(ROOT)),points=len(p),display_stride=stride,display_xy_quantiles=[.5,99.5],display_z_color_quantiles=[2,98],color='Height, not RGB or semantic class',algorithm='Archived LIVO + map-audit reviewed offline backend',visibility1_applied=False)
            records.append(entry)
        fig.text(.025,.025,'Camera RGB (left) / height-colored point cloud and trajectory (right). Per-recording display scales.\nCentral 99% XY extent shown; six recordings may revisit the same physical site. No GT accuracy is inferred.',fontsize=10,color=GRAY,linespacing=1.5)
        save(fig,'rosbag_routes_'+str(part+1))
    PROV.extend(records);return records

def wire(ax,box,color=BLUE):
    corners=np.array(list(product(*zip(box['min'],box['max']))))
    for i,c in enumerate(corners):
        for v in corners[i+1:]:
            if np.count_nonzero(c!=v)==1:ax.plot(*np.array([c,v]).T,color=color,lw=1.4)

def outdoor(use):
    doc=read(GEOM/'objects.json',use);frames=read(GEOM/'geometry_frames.json',use)
    read(ROOT/'docs/reports/large_semantic_mapping_2026-11-06/summary.json',use)
    read(GEOM/'summary.json',use);read(EXP/'boxes_ground_20261106/image_ground/model.json',use)
    full=cloud(Path(doc['source_map']),use);ground=cloud(GEOM/'ground_corrected.ply',use)
    confirmed=[o for o in doc['objects'] if o['confirmed']]
    examples=[next(o for o in confirmed if o['id']==i) for i in ['potted_plant_0116','door_0036']]
    traj=np.loadtxt(use(Path(doc['source_map']).parent/'trajectory_corrected.tum'))[:,1:4]
    fig=plt.figure(figsize=(12,11.5));fig.text(.025,.975,'Outdoor semantic mapping  |  rosbag2_20260415_162852',fontsize=18,weight='bold',color=INK,va='top')
    fig.text(.025,.94,'Archived LIVO + Grounding DINO tiny + depth association; 440 semantic keyframes',fontsize=11,color=GRAY)
    ax=fig.add_axes([.04,.51,.92,.395]);ax.set_facecolor('#132635');p=full[::max(1,int(np.ceil(len(full)/160000)))]
    ax.scatter(p[:,0],p[:,1],s=.22,c='#63788a',alpha=.65,rasterized=True)
    ax.scatter(ground[:,0],ground[:,1],s=.35,c='#55c690',alpha=.8,rasterized=True)
    ax.plot(traj[:,0],traj[:,1],c='#ffe2a3',lw=1)
    for o in confirmed:
        c=GREEN if o['label']=='potted_plant' else BLUE
        ax.scatter(*o['position'][:2],s=27,c=c,edgecolor='white',lw=.5,zorder=4)
    for letter,o,offset in [('A',examples[0],(12,14)),('B',examples[1],(-60,18))]:
        ax.annotate(letter+' / '+o['id'],o['position'][:2],xytext=offset,textcoords='offset points',color='white',fontsize=9,bbox=dict(facecolor='#183247',edgecolor='#8cb8cf',pad=3),arrowprops=dict(arrowstyle='->',color='white'))
    ax.set_aspect('equal');ax.tick_params(labelsize=8);ax.set_xlabel('X (m)',fontsize=9);ax.set_ylabel('Y (m)',fontsize=9);ax.set_title('Map + ground support + persistent landmark records',loc='left',fontsize=12,color=INK,weight='bold')
    for row,(letter,o) in enumerate(zip(['A','B'],examples)):
        ob=max(o['observations'],key=lambda q:(q['box'][2]-q['box'][0])*(q['box'][3]-q['box'][1]));kid=ob['keyframe_id'];name=f'keyframe_{kid:05d}';y=.275-row*.205
        a=fig.add_axes([.035,y,.30,.17]);im(a,LAND/'frames'/(name+'.jpg'),f'{letter}  RGB / KF {kid}',use)
        a=fig.add_axes([.355,y,.30,.17]);im(a,GEOM/'evidence'/(name+'.jpg'),'Detection + ground candidates',use)
        a=fig.add_axes([.68,y-.014,.29,.20],projection='3d');center=np.array(o['box3d']['center']);q=full[np.linalg.norm(full-center,axis=1)<2.5];q=q[::max(1,int(np.ceil(len(q)/16000)))];a.scatter(*q.T,s=.3,c='#a4b1bd',alpha=.45,rasterized=True)
        wire(a,o['box3d']);a.set_box_aspect([1,1,.7]);a.view_init(25,-55);a.set_axis_off();a.set_title('Local context + measured bounds',fontsize=10,color=INK,y=.96)
        PROV.append(dict(outdoor_object=o['id'],frame=kid,image=str((LAND/'frames'/(name+'.jpg')).relative_to(ROOT)),evidence=str((GEOM/'evidence'/(name+'.jpg')).relative_to(ROOT)),display_selection='Largest observed image box per selected confirmed record; post-hoc illustration',map_revision=doc['map_revision'],model=doc['model_source'],visibility1_applied=False,local_context_radius_m=2.5))
    fig.text(.035,.022,'19 repeated-observation records: 15 plant + 4 door/entrance candidates. People remain transient.\nGreen map points = geometrically checked ground; boxes bound observed surfaces, not complete objects.',fontsize=10,color=GRAY,linespacing=1.5)
    save(fig,'rosbag_outdoor_semantics')
    # Three different actual keyframes; corresponding masks and ground supports.
    fig=plt.figure(figsize=(12,10.2));fig.text(.025,.975,'Ground segmentation & transient observations',fontsize=19,color=INK,weight='bold',va='top')
    fig.text(.025,.93,'rosbag2_20260415_162852 / archived SegFormer + local geometric verification',fontsize=11,color=GRAY)
    for row,kid in enumerate([162,537,923]):
        frame=next(f for f in frames if f['keyframe_id']==kid);name=f'keyframe_{kid:05d}';y=.66-row*.28
        a=fig.add_axes([.025,y,.30,.20]);im(a,LAND/'frames'/(name+'.jpg'),f'RGB / KF {kid}',use)
        z=np.load(use(EXP/'boxes_ground_20261106/image_ground'/(name+'.npz')));assert z['ground'].shape==(360,640)
        a=fig.add_axes([.355,y,.30,.20]);im(a,GEOM/'evidence'/(name+'.jpg'),'2D ground + object detections',use)
        data=np.load(use(GEOM/'frames'/(name+'.npz')));points=data['ground_corrected'];assert len(points)>0 and len(points)==frame['verified_ground_points']
        a=fig.add_axes([.68,y-.01,.30,.23],projection='3d');center=points.mean(0);q=full[np.linalg.norm(full-center,axis=1)<8];q=q[::max(1,int(np.ceil(len(q)/15000)))];a.scatter(*q.T,s=.25,c='#bcc6ce',alpha=.4,rasterized=True);a.scatter(*points.T,s=5,c=GREEN,rasterized=True)
        a.set_box_aspect([1,1,.6]);a.view_init(40,-50);a.set_axis_off();a.set_title(f'{len(points)} verified ground points',fontsize=10,color=INK,y=.97)
        fig.text(.025,y-.03,f'KF {kid}: image candidates checked against measured LiDAR. Person observations are excluded from the static map.',fontsize=10,color=GRAY)
        PROV.append(dict(ground_frame=kid,pixel_ground_candidates=int(z['ground'].sum()),verified_ground_points=len(points),model='SegFormer B0 Cityscapes + archived local geometric verification',display='Image candidates and verified 3D ground are distinct; local context within 8 m',visibility1_applied=False))
    fig.text(.025,.02,'No new model inference. Image segmentation is not a traversability guarantee; these are qualitative engineering examples.',fontsize=10,color=GRAY)
    save(fig,'rosbag_ground_details')

def build(use):
    PROV.clear()
    OUT.mkdir(parents=True,exist_ok=True);plt.rcParams['svg.fonttype']='none'
    inventory=six_bags(use);outdoor(use)
    (OUT/'rosbag_visual_provenance.json').write_text(json.dumps(dict(recordings=inventory,panels=PROV,no_new_inference=True),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':build(lambda p:Path(p))
