import hashlib,json,shutil,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
sys.path.insert(0,'FAST_LIVO2/scripts')
from map_quality import temporal_events,trajectory_poses
from revisit_geometry import optimize_graph
from optimize_keyframe_map import export_map,corrected_trajectory
root=Path(__file__).resolve().parent
assets=Path('docs/experiments/fast_livo2_vbot_map_audit_2026-11-05');assets.mkdir(exist_ok=True)
entries=json.loads(Path('build/experiments/full_20261105/inventory.json').read_text());entries.sort(key=lambda e:e['duration_s'])
lio_run=json.loads((root/'lio_full/status.json').read_text());livo_run=json.loads(Path('build/experiments/z_20261105/si_full/status.json').read_text())
assert lio_run['executable_sha256']==livo_run['executable_sha256']
assert len(lio_run['results'])==len(livo_run['results'])==6
assert all(x['source_complete'] and not x['failures'] for x in lio_run['results'])
summary={'native_binary_sha256':lio_run['executable_sha256'],'reused_livo_status':str(Path('build/experiments/z_20261105/si_full/status.json').resolve()),'new_lio_status':str((root/'lio_full/status.json').resolve()),'source_duration_s':sum(e['duration_s'] for e in entries),'bags':[],'warnings':[]}
for entry in entries:
 name=entry['name'];folder=root/'audit_final'/name;d=json.loads((folder/'result.json').read_text());vo_src=Path(d['sources']['LIVO']);io_src=Path(d['sources']['LIO'])
 assert json.loads((io_src.parent/'batch_result.json').read_text())['counts']==json.loads((vo_src.parent/'batch_result.json').read_text())['counts']
 variants={};mode_data={}
 for mode in ['LIO','LIVO']:
  maps=root/'split_maps'/mode/name;result=json.loads((maps/'result.json').read_text());src=Path(d['sources'][mode]);rows=np.genfromtxt(src/'keyframes.csv',delimiter=',',names=True);mode_data[mode]=(maps,result,src,rows)
  for phase,file in [('before','trajectory_gravity.tum'),('after','trajectory_corrected.tum')]:variants[mode+'_'+phase]=np.loadtxt(maps/file)
 q=trajectory_poses(variants['LIVO_before'],[variants['LIVO_before'][0,0]])[0]@np.linalg.inv(trajectory_poses(variants['LIO_before'],[variants['LIO_before'][0,0]])[0])
 if d['events']:
  event=max(d['events'],key=lambda e:e['stamps'][0]-e['stamps'][1]);center=trajectory_poses(variants['LIVO_before'],[event['stamps'][1]])[0][:3,3]
 else:center=variants['LIVO_before'][0,1:4]
 fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
 for ax,(key,tum) in zip(axes.ravel(),variants.items()):
  mode,phase=key.split('_');mp=mode_data[mode][0]/('map_gravity_only.pcd' if phase=='before' else 'map_loop_corrected.pcd');cloud=o3d.io.read_point_cloud(str(mp));xyz=np.asarray(cloud.points)
  if mode=='LIO':xyz=xyz@q[:3,:3].T+q[:3,3]
  xyz=xyz-center;xyz=xyz[(np.linalg.norm(xyz[:,:2],axis=1)<15)&(xyz[:,2]>-8)&(xyz[:,2]<8)];xyz=xyz[::max(1,len(xyz)//70000)]
  ax.scatter(xyz[:,0],xyz[:,2],s=.35,c=xyz[:,2],vmin=-8,vmax=8,cmap='viridis',alpha=.45,rasterized=True);ax.set(title=key,xlabel='Common initial-frame X (m)',ylabel='Gravity-direction Z (m)',xlim=(-15,15),ylim=(-8,8));ax.grid(alpha=.15)
 fig.suptitle(name+'\nSame 15 m ROI / fixed SI frontend and split-scan backend; colors show height, not semantics')
 fig.savefig(folder/'map_xz.png',dpi=150);plt.close(fig)
 for f in folder.glob('*.png'):shutil.copy2(f,assets/(name+'_'+f.name))
 for f in ['result.json','image_evidence.json']:shutil.copy2(folder/f,assets/(name+'_'+f))
 # Rebuild only maps containing a loop event not supported by reserved scans.
 # Keep the original fixed-backend results intact as experimental controls.
 times=np.r_[mode_data['LIO'][3]['stamp'],mode_data['LIVO'][3]['stamp']];edges=[]
 for mode,(maps,result,src,rows) in mode_data.items():
  offset=0 if mode=='LIO' else len(mode_data['LIO'][3])
  for e in result['edges']:edges.append(dict(e,mode=mode,native_i=e['i'],native_j=e['j'],i=e['i']+offset,j=e['j']+offset))
 groups=temporal_events(edges,times)
 reviewed={}
 for mode,(maps,result,src,rows) in mode_data.items():
  accepted=[];quarantine=[]
  for group,event in zip(groups,d['events']):
   for e in group:
    if e['mode']!=mode:continue
    original=dict(e,i=e['native_i'],j=e['native_j'])
    if event['mode_checks'][mode]['status']=='supported_by_reserved_scans':accepted.append(original)
    else:quarantine.append(original)
  if not quarantine:
   reviewed[mode]={'quarantined_edges':0,'map_path':str((maps/'map_loop_corrected.pcd').resolve())};continue
  dest=root/'reviewed_maps'/name/mode;dest.mkdir(parents=True,exist_ok=False);base=np.load(maps/'keyframe_poses.npz')['before'];optimized,retained=optimize_graph(base,[e for e in accepted if not e['heldout_from_graph']])
  points=export_map(src,rows,optimized,dest/'map_loop_reviewed.pcd');native=np.loadtxt(src/'trajectory_final.tum');aligned,corrected=corrected_trajectory(native,rows['stamp'],base,optimized,np.array(result['gauge_transform']));np.savetxt(dest/'trajectory_reviewed.tum',corrected,fmt='%.12f')
  reviewed[mode]={'quarantined_edges':len(quarantine),'quarantined_training_edges':sum(not e['heldout_from_graph'] for e in quarantine),'map_path':str((dest/'map_loop_reviewed.pcd').resolve()),'points':points,'end_height_change_m':float(corrected[-1,3]-corrected[0,3]),'note':'Removed unverified event, not declared a proven false match. Offline diagnostic map only.'};(dest/'review.json').write_text(json.dumps(reviewed[mode],indent=2));summary['warnings'].append({'bag':name,'mode':mode,**reviewed[mode]})
 d['reviewed_maps']=reviewed;summary['bags'].append(d)
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');keys=['LIO_before','LIO_after','LIVO_before','LIVO_after'];colors=['#94a3b8','#475569','#f59e0b','#2563eb']
for ax,metric,label in [(axes[0],'median_abs_layer_gap_m','Median two-visit plane gap (cm)'),(axes[1],'median_balanced_width90_m','Balanced two-visit 90% width (cm)')]:
 for k,key in enumerate(keys):
  values=[b['surfaces']['vertical_plane']['variants'].get(key,{}).get(metric,float('nan'))*100 for b in summary['bags']];ax.barh(np.arange(6)+(k-1.5)*.18,values,height=.17,label=key,color=colors[k])
 ax.set(yticks=np.arange(6),yticklabels=[e['name'].replace('rosbag2_20260415_','0415_') for e in entries],xlabel=label);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Fixed planes and identical reserved scan points across four map variants\nNo verified revisit in roabag2336; after includes a flagged April14 LIO event (see report)');fig.savefig(assets/'surface_comparison.png',dpi=150);plt.close(fig)
summary['total_events']=sum(b['visit_events_union'] for b in summary['bags']);summary['ground_patches']=sum(b['surfaces']['ground']['patches'] for b in summary['bags']);summary['vertical_patches']=sum(b['surfaces']['vertical_plane']['patches'] for b in summary['bags'])
summary['supported_physical_events']=sum(all(c['status']=='supported_by_reserved_scans' for c in e['mode_checks'].values()) for b in summary['bags'] for e in b['events'])
summary['note']='No external truth; event support is not proof of absence of false loops. Surface metrics use fixed reference-only planes and fixed point samples. Ground evidence is sparse. Both frontends consume all scans; only loop fitting holds out whole scans.'
(assets/'summary.json').write_text(json.dumps(summary,indent=2))
for name in ['quality_tests.log','geometry_tests.log','ruff.log','build_install.log','split_maps.log','audit_final_batch.log']:shutil.copy2(root/name,assets/name)
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('FAST_LIVO2/scripts/map_quality.py'),Path('FAST_LIVO2/scripts/audit_map_quality.py'),Path('FAST_LIVO2/scripts/optimize_keyframe_map.py'),Path('FAST_LIVO2/scripts/revisit_geometry.py'),Path('FAST_LIVO2/scripts/extract_revisit_images.py')]};(assets/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
print('summary',summary['total_events'],summary['supported_physical_events'],summary['ground_patches'],summary['vertical_patches']);print(summary['warnings'])
