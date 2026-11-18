#!/usr/bin/env python3
"""Self-contained interactive 3D map: measured boxes, ground, transient people."""
import argparse,base64,json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from plotly.offline import get_plotlyjs


def sampled(points,limit):
    p=np.asarray(points);return p[::max(1,int(np.ceil(len(p)/limit)))].round(3).tolist()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--geometry',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    doc=json.loads((args.geometry/'objects.json').read_text());frames=json.loads((args.geometry/'geometry_frames.json').read_text())
    full_cloud=o3d.io.read_point_cloud(doc['source_map']);full=np.asarray(full_cloud.points);tree=cKDTree(full)
    cloud=full_cloud.voxel_down_sample(.30)
    ground=o3d.io.read_point_cloud(str(args.geometry/'ground_corrected.ply'))
    objects=[];images={}
    def evidence(kid):
        key=str(kid)
        if key not in images:
            p=args.geometry/'evidence'/f'keyframe_{kid:05d}.jpg';images[key]='data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()
        return key
    for o in doc['objects']:
        if not o.get('box3d'):continue
        best=max(o['observations'],key=lambda ob:ob['score'])
        local=full[tree.query_ball_point(o['box3d']['center'],9.)]
        objects.append(dict(id=o['id'],label=o['label'],confirmed=o['confirmed'],box=o['box3d'],
            observations=o['observation_count'],evidence=evidence(best['keyframe_id']),local_context=sampled(local,6500)))
    people=[]
    for f in frames:
        boxes=[b for b in f['boxes'] if b['dynamic']]
        if boxes:people.append(dict(keyframe_id=f['keyframe_id'],elapsed=f['stamp']-frames[0]['stamp'],boxes=boxes,evidence=evidence(f['keyframe_id'])))
    data=dict(context=sampled(cloud.points,100000),ground=sampled(ground.points,130000),objects=objects,people=people,images=images,
              frame=doc['frame'],summary=json.loads((args.geometry/'summary.json').read_text()))
    # JSON stays data, never an executable string or HTML expression.
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>目标三维框与地面分割</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1521;color:#edf4ff;font:15px system-ui,"Noto Sans CJK SC",sans-serif}header{padding:18px 24px;border-bottom:1px solid #294056}h1{font-size:22px;margin:0 0 12px}button,select{background:#193148;border:1px solid #42637d;color:#fff;border-radius:6px;padding:8px;margin:0 12px 5px 0}label{margin-right:15px;white-space:nowrap}main{display:grid;grid-template-columns:minmax(0,1fr) 345px;height:calc(100vh - 125px)}#map{height:100%;min-height:520px}aside{padding:20px;border-left:1px solid #294056;overflow:auto}#photo{width:100%;margin-top:16px}#details{line-height:1.9;white-space:pre-line}.small{color:#9ab0c7;font-size:13px;line-height:1.8}.key{color:#65dd94}.tentative{color:#ffbe62}#status{font-size:13px;margin-left:8px;color:#9ab0c7}@media(max-width:900px){main{display:block;height:auto}#map{height:65vh}aside{border-left:0}}
</style><header><h1>目标三维框与地面分割</h1><button id="reset">全图</button><select id="target"><option value="">选择目标并放大</option></select><label><input id="tentative" type="checkbox">其他暂定目标</label><label><input id="ground" type="checkbox" checked>地面</label><label><input id="context" type="checkbox" checked>环境点云</label><select id="people"><option value="">行人：不显示</option></select><span id="status"></span></header>
<main><div id="map"></div><aside><div id="details">选择三维框或上方目标，查看位置。</div><img id="photo" hidden><p class="small"><span class="key">绿色点：</span>图像分割与局部几何共同支持的地面。<br>蓝框：满足重复观测门槛的记录。<br><span class="tentative">橙框：</span>暂定记录；消火栓目前仅有一次可靠深度观测。<br>紫框：所选关键帧中的行人框，不保留为永久目标。</p><p class="small">框表示已观测表面的范围，不能当作完整物体尺寸。地面分割不等同于可通行区域。<br>坐标系：离线校正地图，单位：米。拖动旋转，滚轮缩放。</p></aside></main><script>__PLOTLY__</script><script>
const D=__DATA__;
const names={door:'门 / 入口',potted_plant:'花盆',fire_hydrant:'消火栓',person:'行人'};
const edges=[];for(let i=0;i<8;i++)for(let j=i+1;j<8;j++){let n=i^j;if((n&(n-1))===0)edges.push([i,j]);}
const el=id=>document.getElementById(id);
function points(p,color,size,name){return {type:'scatter3d',mode:'markers',x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),marker:{size,color,opacity:1},name,hoverinfo:'skip'};}
function available(){return D.objects.filter(o=>o.confirmed||el('tentative').checked||o.label==='fire_hydrant');}
function wire(objects,color,name){const x=[],y=[],z=[];for(const o of objects){const c=o.box.corners;for(const [i,j] of edges){x.push(c[i][0],c[j][0],null);y.push(c[i][1],c[j][1],null);z.push(c[i][2],c[j][2],null);}}return {type:'scatter3d',mode:'lines',x,y,z,line:{color,width:5},hoverinfo:'skip',name};}
function marker(objects){return {type:'scatter3d',mode:'markers',x:objects.map(o=>o.box.center[0]),y:objects.map(o=>o.box.center[1]),z:objects.map(o=>o.box.center[2]),customdata:objects.map(o=>o.id),text:objects.map(o=>(names[o.label]||o.label)+' · '+o.id),hovertemplate:'%{text}<br>X %{x:.2f} / Y %{y:.2f} / Z %{z:.2f} m<extra></extra>',marker:{size:5,color:objects.map(o=>o.confirmed?'#69baff':'#ffbe62')},name:'目标位置'};}
function scene(){return {bgcolor:'#0b1521',aspectmode:'data',xaxis:{title:'X / m',backgroundcolor:'#0b1521',gridcolor:'#263748',color:'#9bb3ca'},yaxis:{title:'Y / m',backgroundcolor:'#0b1521',gridcolor:'#263748',color:'#9bb3ca'},zaxis:{title:'Z / m',backgroundcolor:'#0b1521',gridcolor:'#263748',color:'#9bb3ca'},camera:{eye:{x:1.3,y:-1.5,z:1.1}}};}
const layout={paper_bgcolor:'#0b1521',font:{color:'#eaf2ff'},margin:{l:0,r:0,t:0,b:0},scene:scene(),showlegend:false,uirevision:'map'};
let ready=false;
async function redraw(){const objs=available();const traces=[points(D.context,'#526577',1,'环境'),points(D.ground,'#42ce79',2,'地面'),wire(objs.filter(o=>o.confirmed),'#69baff','稳定记录'),wire(objs.filter(o=>!o.confirmed),'#ffbe62','暂定记录'),marker(objs)];traces[0].visible=el('context').checked;traces[1].visible=el('ground').checked;
const person=D.people.find(p=>String(p.keyframe_id)===el('people').value);if(person){const temp=person.boxes.map(b=>({box:{corners:b.corners_corrected}}));traces.push(wire(temp,'#f68bdf','当前行人'));}
const selected=D.objects.find(o=>o.id===el('target').value);if(selected&&el('context').checked)traces.push(points(selected.local_context,'#92a6b9',2,'局部环境'));
await Plotly.react('map',traces,layout,{responsive:true,displaylogo:false});if(!ready){el('map').on('plotly_click',ev=>{let id=ev.points[0].customdata;if(id)focus(id);});ready=true;window.__mapReady=true;}el('status').textContent=`${objs.length} 条目标记录 · ${D.ground.length.toLocaleString()} 个地面显示点`;}
function fill(){const previous=el('target').value;el('target').replaceChildren(new Option('选择目标并放大',''));for(const o of available())el('target').add(new Option(`${names[o.label]||o.label} · ${o.id}${o.confirmed?'':'（暂定）'}`,o.id));el('target').value=previous;}
function zoom(c,e=[1,1,1]){const r=Math.max(3.5,...e.map(v=>v*1.7));Plotly.relayout('map',{'scene.xaxis.range':[c[0]-r,c[0]+r],'scene.yaxis.range':[c[1]-r,c[1]+r],'scene.zaxis.range':[c[2]-r*.7,c[2]+r*.7],'scene.aspectmode':'manual','scene.aspectratio':{x:1,y:1,z:.7},'scene.camera.eye':{x:1.5,y:-1.7,z:1.0}});}
function photo(key){el('photo').src=D.images[key];el('photo').hidden=false;}
async function focus(id){const o=D.objects.find(o=>o.id===id);if(!o)return;if(!available().includes(o)){el('tentative').checked=true;fill();}el('target').value=id;el('people').value='';await redraw();zoom(o.box.center,o.box.extent);el('details').textContent=`${names[o.label]||o.label} · ${o.id}\n${o.confirmed?'重复观测记录':'暂定：深度观测不足'}\n位置 X / Y / Z\n${o.box.center.map(v=>v.toFixed(2)).join(' / ')} m\n观测框范围\n${o.box.extent.map(v=>v.toFixed(2)).join(' × ')} m\n${o.observations} 次观测`;photo(o.evidence);}
el('target').onchange=()=>focus(el('target').value);
el('tentative').onchange=()=>{fill();redraw();};el('ground').onchange=redraw;el('context').onchange=redraw;
for(const p of D.people)el('people').add(new Option(`行人 KF ${p.keyframe_id} · ${Math.floor(p.elapsed/60)}:${String(Math.floor(p.elapsed%60)).padStart(2,'0')}`,p.keyframe_id));
el('people').onchange=async()=>{el('target').value='';await redraw();const p=D.people.find(p=>String(p.keyframe_id)===el('people').value);if(p){const c=[0,1,2].map(k=>p.boxes[0].corners_corrected.reduce((s,x)=>s+x[k],0)/8);zoom(c);el('details').textContent=`行人：KF ${p.keyframe_id}\n只显示该时刻的 ${p.boxes.length} 个观测框。\n不推断当前仍在此处，不关联个人身份。`;photo(p.evidence);}};
el('reset').onclick=async()=>{el('target').value='';el('people').value='';await redraw();Plotly.relayout('map',{'scene.xaxis.autorange':true,'scene.yaxis.autorange':true,'scene.zaxis.autorange':true,'scene.aspectmode':'data','scene.camera.eye':{x:1.3,y:-1.5,z:1.1}});};
fill();redraw().then(()=>{if(location.hash)focus(decodeURIComponent(location.hash.slice(1)));});
</script></html>'''.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',encoded)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(html)
    print(args.output,args.output.stat().st_size,'objects',len(objects),'people frames',len(people),flush=True)


if __name__=='__main__':main()
