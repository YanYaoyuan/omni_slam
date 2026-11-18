import hashlib,json,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
out=Path('docs/experiments/fast_livo2_vbot_vertical_2026-11-05');out.mkdir(exist_ok=True)
entries=json.loads(Path('build/experiments/full_20261105/inventory.json').read_text());entries.sort(key=lambda e:e['duration_s'])
summary={'duration_s':sum(e['duration_s'] for e in entries),'bags':[], 'backend_sha256':{}, 'note':'Internal revisit consistency only; shared data and submaps, no independent ground truth. Gravity alignment is a gauge transform. Graph corrects offline map only.'}
fig,axes=plt.subplots(6,2,figsize=(12,18),layout='constrained')
for k,e in enumerate(entries):
 name=e['name'];base=root/'archive_full'/name/'livo';si=root/'si_full'/name/'livo';mapped=root/'si_maps'/name
 oldmap=root/'baseline_maps'/name
 d=json.loads((mapped/'result.json').read_text());od=json.loads((oldmap/'result.json').read_text())
 source=json.loads((si.parent/'batch_result.json').read_text());assert source['source_complete'] and not source['failures']
 before=np.loadtxt(mapped/'trajectory_gravity.tum');after=np.loadtxt(mapped/'trajectory_corrected.tum');old=np.loadtxt(oldmap/'trajectory_gravity.tum')
 raw=np.loadtxt(si/'trajectory_final.tum');legacy=np.loadtxt(base/'trajectory_final.tum')
 prior=np.loadtxt(Path('build/experiments/opt_20261105/full_lidar_timed')/name/'livo/trajectory_final.tum');assert np.array_equal(legacy,prior)
 smoke=np.loadtxt(root/'final_smoke'/name/'livo/trajectory_final.tum');assert np.array_equal(smoke[:-1],raw[:len(smoke)-1])
 m=re.search(r'raw mean norm=([0-9.]+); acceleration scale=([0-9.]+)',(root/'final_smoke'/name/'livo/node.log').read_text(errors='replace'));assert m
 norm=float(m[1]);dt=np.diff(raw[:,0]);assert np.all(dt>0) and np.isfinite(raw).all()
 speeds=np.linalg.norm(np.diff(raw[:,1:4],axis=0),axis=1)/dt
 assert np.percentile(speeds,99)<2 and speeds.max()<5
 for f in ('optimize_keyframe_map.py','revisit_geometry.py'):
  digest=hashlib.sha256((Path('FAST_LIVO2/scripts')/f).read_bytes()).hexdigest();assert d['backend_sha256'][f]==digest; summary['backend_sha256'][f]=digest
 assert d['source_index_sha256']==hashlib.sha256((si/'keyframes.csv').read_bytes()).hexdigest()
 item={'name':name,'duration_s':e['duration_s'],'counts':source['counts'],'native_poses':len(raw),'finite':True,'p99_speed_m_s':float(np.percentile(speeds,99)),'max_speed_m_s':float(speeds.max()),'peak_rss_mib':json.loads((si/'metrics.json').read_text())['peak_rss_mib'],'initial_raw_acc_norm_m_s2':norm,'legacy_acceleration_scale':9.81/norm,'explicit_acceleration_scale':float(m[2]),'archive_matches_previous_full':True,'final_build_prefix_interior_equal':True,'prefix_all_equal':bool(np.array_equal(smoke,raw[:len(smoke)])),'prefix_last_position_difference_m':float(np.linalg.norm(smoke[-1,1:4]-raw[len(smoke)-1,1:4])),'keyframes':d['keyframes'],'geometric_loops':d['geometric_loops'],'heldout_loops':d['heldout_loops'],'baseline_heldout_before_m':od['heldout_before']['vertical_rmse_m'],'si_heldout_before_m':d['heldout_before']['vertical_rmse_m'],'si_heldout_after_m':d['heldout_after']['vertical_rmse_m'],'gravity':d['gravity_alignment'],'end_height_before_m':float(before[-1,3]-before[0,3]),'end_height_after_m':float(after[-1,3]-after[0,3]),'height_range_before_m':float(np.ptp(before[:,3])),'height_range_after_m':float(np.ptp(after[:,3])),'corrected_map_points':d['corrected_map_points'],'map_path':str((mapped/'map_loop_corrected.pcd').resolve()),'map_bytes':(mapped/'map_loop_corrected.pcd').stat().st_size,'map_sha256':hashlib.sha256((mapped/'map_loop_corrected.pcd').read_bytes()).hexdigest(),'result_path':str((mapped/'result.json').resolve())}
 summary['bags'].append(item)
 for a,label,color in [(old,'Previous frontend / gravity frame','0.6'),(before,'SI frontend / gravity frame','#d97706'),(after,'SI + offline loop map','#2563eb')]:
  axes[k,0].plot(a[:,1],a[:,2],label=label,color=color,lw=1)
  axes[k,1].plot(a[:,0]-a[0,0],a[:,3]-a[0,3],label=label,color=color,lw=1)
 axes[k,0].set(title=name,xlabel='X (m)',ylabel='Y (m)');axes[k,0].axis('equal')
 axes[k,1].set(xlabel='Time (s)',ylabel='Gravity-direction height (m)')
 for ax in axes[k]:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Six full recordings: unit correction and geometry-verified offline loop maps\nEach run has its own stationary-force gravity alignment; no trajectory ground truth',fontsize=13)
fig.savefig(out/'trajectories.png',dpi=140);plt.close(fig)
summary['total_counts']={key:sum(b['counts'].get(key,0) for b in summary['bags']) for key in ('lidar','imu','video_packets','decoded_frames','image')}
summary['total_geometric_loops']=sum(b['geometric_loops'] for b in summary['bags'])
summary['total_heldout_loops']=sum(b['heldout_loops'] for b in summary['bags'])
(out/'summary.json').write_text(json.dumps(summary,indent=2))
for name in ('20260414_4','rosbag2245'):
 import shutil
 shutil.copy2(root/'si_maps'/name/'trajectory_comparison.png',out/(name+'_trajectory.png'))
for name in ('geometry_tests_final.log','tests_final.log','ruff_final.log','build_final.log'):
 (out/name).write_bytes((root/name).read_bytes())
(out/'si_overrides.yaml').write_bytes(Path('FAST_LIVO2/config/vbot_si_overrides.yaml').read_bytes())
for b in summary['bags']:print(b['name'],b['geometric_loops'],b['si_heldout_before_m'],b['si_heldout_after_m'])
