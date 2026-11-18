"""Same four current robot maps as manuscript table; existing outputs only."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from omni_semantic_map.discovery_measured_input import robot_measured_visibility
R=Path(__file__).resolve().parents[2];P=R/'docs/paper/visibility_revision/latex';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=read(R/'build/benchmarks/revision_assessment_v1/robot_correspondence_audit.json');ctx=read(R/'build/benchmarks/revision_robot_v1/protocol.json')['contexts'][0];job=read(ctx['job']);f=next(v for v in job['frames'] if v['frame_id']==126);rgb=Image.open(f['rgb']);cloud=np.load(f['cloud'])['points'];full=np.load(Path(ctx['full'])/'geometry.npz')['xyz'];ii=np.linspace(0,len(full)-1,60000,dtype=int);lo=full[:,:2].min(0);hi=full[:,:2].max(0)
fig,axes=plt.subplots(4,2,figsize=(12,12),gridspec_kw=dict(width_ratios=[1.2,1]),constrained_layout=True);rows={}
for i,(m,name) in enumerate([('strong_baseline','Strong baseline'),('unconditional','Unconditional'),('history_only','History-only'),('full_authorization','Ours')]):
 z=a['methods'][m];folder=Path(z['folder']);assert sha(folder/'geometry.npz')==z['geometry_sha256'];assert sha(folder/'classification.json')==z['classification_sha256'];g=np.load(folder/'geometry.npz');infos=read(folder/'classification.json')['objects'];carids=[v['object_id'] for v in infos if v['predictions']['mean']['train_id']==13];vis,xy,_=robot_measured_visibility(g['xyz'],cloud,f);ax=axes[i,0];ax.imshow(rgb);ax.scatter(xy[:,0],xy[:,1],c=g['instance'][vis]%20,cmap='tab20',vmin=0,vmax=19,s=1,alpha=.8);ax.axis('off');ax.set_title(name+f' | frame 126 | {len(vis):,} visible points',fontsize=11,color='#008f7a' if m=='full_authorization' else '#163044');idx=np.flatnonzero(np.isin(g['instance'],carids));ax=axes[i,1];ax.scatter(full[ii,0],full[ii,1],c='#ced4d9',s=.15);ax.scatter(g['xyz'][idx,0],g['xyz'][idx,1],c=g['instance'][idx]%20,cmap='tab20',vmin=0,vmax=19,s=.8);ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal');ax.set_title(f'{len(idx):,} predicted car points | {z["old_point_id_changes"]:,} old edits',fontsize=10);rows[m]=dict(visible_points=len(vis),car_points=len(idx),map_sha256=z['geometry_sha256'])
fig.suptitle('Current four-method garage comparison / rosbag2245 / 330-frame maps\nAll-class image projections (left); vehicle-class map readout (right)',fontsize=14,fontweight='bold');out=P/'figures/current/robot_current_four.png';fig.savefig(out,dpi=180);plt.close(fig)
data=dict(frame=126,frame_selection='Same previously published frame; not newly selected for this revision',methods=rows,raw_image_sha256=sha(f['rgb']),current_cloud_sha256=sha(f['cloud']),coordinate_scope='Offline full archive; given LIVO poses; not robot semantic GT',display='Repeated tab20 instance palette. IDs across methods are not physical association evidence. Context deterministic60000point subsample; vehicle points unthinned.',figure_sha256=sha(out),source_sha256=sha(__file__))
(P/'data/review_20261117/robot_current_figure.json').write_text(json.dumps(data,indent=2));print(rows)
