import json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parent; repo=Path.cwd();source=root/'rosbag2245_final';d=json.loads((source/'objects.json').read_text());a=repo/'docs/experiments/livo_semantic_integration_2026-11-06';a.mkdir(exist_ok=True)
confirmed=[o for o in d['objects'] if o['confirmed']];special=next(o for o in confirmed if o['id']=='car_0001')
summary={'bag':d['bag'],'processed_keyframes':len(d['frames']),'skipped_frames':sum('skipped' in f for f in d['frames']),'car_detections':sum(f.get('detections',0) for f in d['frames']),'depth_supported_observations':sum(f.get('supported_observations',0) for f in d['frames']),'object_proposals':len(d['objects']),'queryable_object_records':len(confirmed),'detector':d['detector'],'weights_sha256':d['weights_sha256'],'wall_s':d['wall_s'],'parameters':d['parameters'],'long_revisit':{'id':special['id'],'observations':len(special['observations']),'span_s':special['observations'][-1]['stamp']-special['observations'][0]['stamp'],'position':special['position'],'spread_p90_m':special['position_spread_p90_m']},'manager':json.loads((root/'managed_replay_v2/result.json').read_text()),'frontend':json.loads((root/'frontend_regression/comparison.json').read_text()),'limitations':d['limitations']}
(a/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
for name in ['query_cars.json','query_car_0001.json']:shutil.copy2(source/name,a/name)
# Self-contained evidence catalog: each row identifies the object being queried.
(a/'evidence').mkdir(exist_ok=True)
for o in confirmed:
 for ob in [o['observations'][0],o['observations'][-1],max(o['observations'],key=lambda x:x['score'])]:
  shutil.copy2(source/ob['evidence'],a/ob['evidence'])
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',21)
canvas=Image.new('RGB',(1280,445),'white');draw=ImageDraw.Draw(canvas)
for j,obs in enumerate([special['observations'][0],special['observations'][-1]]):
 canvas.paste(Image.open(source/obs['evidence']),(j*640,48))
 draw.text((j*640+8,10),f"car_0001 / KF {obs['keyframe_id']} / camera time {obs['image_stamp']:.3f}",font=font,fill='black')
draw.text((10,413),f"Same object record across {summary['long_revisit']['span_s']:.1f} s; green: detection, red: LiDAR support",font=font,fill='black');canvas.save(a/'revisit_car_0001.png')
cloud=o3d.io.read_point_cloud(d['source_map']);xyz=np.asarray(cloud.points);xyz=xyz[::max(1,len(xyz)//50000)]
tum=np.loadtxt('build/experiments/z_20261105/si_maps/rosbag2245/trajectory_corrected.tum')
fig,ax=plt.subplots(figsize=(13,7),layout='constrained');ax.scatter(xyz[:,0],xyz[:,1],s=.3,c='0.78',alpha=.4,rasterized=True);ax.plot(tum[:,1],tum[:,2],c='#64748b',lw=.8,label='LIVO corrected trajectory')
for o in confirmed:
 x,y,_=o['position'];ax.scatter(x,y,s=32,c='#dc2626' if o['id']=='car_0001' else '#2563eb');ax.annotate(o['id'].replace('car_',''),(x,y),xytext=(3,4),textcoords='offset points',fontsize=7)
ax.set(xlabel='Offline corrected-map X (m)',ylabel='Y (m)',title=f"rosbag2245: {len(confirmed)} queryable vehicle records\nRecord IDs are not a ground-truth count of unique cars");ax.set_aspect('equal');ax.legend();ax.grid(alpha=.2);fig.savefig(a/'objects_xy.png',dpi=160);plt.close(fig)
# Cropped evidence removes clutter from neighbouring labels; complete frames remain linked.
for page in range((len(confirmed)+7)//8):
 group=confirmed[page*8:page*8+8];sheet=Image.new('RGB',(1280,230*len(group)),'white');draw=ImageDraw.Draw(sheet)
 for i,o in enumerate(group):
  for j,ob in enumerate([o['observations'][0],o['observations'][-1]]):
   raw=Image.open(source/'frames'/Path(ob['evidence']).name);x1,y1,x2,y2=ob['box'];crop=raw.crop((max(0,int(x1)-5),max(0,int(y1)-5),min(640,int(x2)+5),min(360,int(y2)+5)));crop.thumbnail((620,185));sheet.paste(crop,(j*640+10,i*230+38));draw.text((j*640+10,i*230+7),f"{o['id']} / KF {ob['keyframe_id']} / {o['observation_count']} obs",font=font,fill='black')
 sheet.save(a/f'object_review_{page+1}.jpg')
for f in ['colcon_final.log','unit_tests_final.log','manager_tests.log','save_cancel_tests.log','native_tests.log','ruff.log']:shutil.copy2(root/f,a/f)
files=[repo/'FAST_LIVO2/src/LIVMapper.cpp',repo/'FAST_LIVO2/include/LIVMapper.h',repo/'FAST_LIVO2/scripts/build_semantic_map.py',repo/'FAST_LIVO2/scripts/prepare_managed_vbot.py',repo/'FAST_LIVO2/test/check_managed_livo.py',repo/'omni_slam_manager/omni_slam_manager/manager_node.py',repo/'omni_slam_manager/omni_slam_manager/livo_profile.py',repo/'omni_slam_interfaces/msg/LivoKeyframe.msg',repo/'omni_semantic_map/omni_semantic_map/objects.py',repo/'omni_semantic_map/omni_semantic_map/cli.py']
(a/'source_manifest.json').write_text(json.dumps({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
