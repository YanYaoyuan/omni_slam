"""Causal ground / 3D support-box overlays for the recorded frontend map."""
from collections import deque
import json
from pathlib import Path
import cv2
import numpy as np

EDGES=[(i,j) for i in range(8) for j in range(i+1,8) if bin(i^j).count('1')==1]


class GeometryOverlay:
    def __init__(self,directory,width,height):
        self.root=Path(directory);self.frames=json.loads((self.root/'geometry_frames.json').read_text())
        self.index=0;self.ground=np.zeros((height,width),bool);self.static={};self.dynamic=[];self.scans=deque();self.focus=None

    def add_scan(self,stamp,points):
        self.scans.append((stamp,points[::3].copy()))
        while self.scans and self.scans[0][0]<stamp-5:self.scans.popleft()

    def update(self,stamp,pixels):
        while self.index<len(self.frames) and self.frames[self.index]['stamp']<=stamp+1e-6:
            f=self.frames[self.index];p=np.load(self.root/'frames'/f"keyframe_{f['keyframe_id']:05d}.npz")['ground_native']
            uv=pixels(p);h,w=self.ground.shape;valid=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            uv=uv[valid];self.ground[uv[:,1],uv[:,0]]=True
            self.dynamic=[]
            for b in f['boxes']:
                record=dict(b,stamp=f['stamp'])
                if b['dynamic']:self.dynamic.append(record)
                else:
                    old=self.static.get(b['id']);record['count']=1 if old is None else old['count']+1
                    record['first']=f['stamp'] if old is None else old['first'];self.static[b['id']]=record
            candidates=[b for b in f['boxes'] if not b['dynamic']]
            if candidates:
                candidates.sort(key=lambda b:({'fire_hydrant':0,'door':1}.get(b['label'],2),-b['support_points']))
                self.focus=self.static[candidates[0]['id']]
            self.index+=1

    @staticmethod
    def stable(b):return b.get('count',0)>=3 and b['stamp']-b.get('first',b['stamp'])>=4

    def draw(self,world,stamp,pixels,projection):
        self.update(stamp,pixels)
        # Neutral context makes green, geometrically verified ground distinct.
        gray=np.mean(world,axis=2).astype(np.uint8)
        frame=np.stack([gray*.65,gray*.77,gray*.9],axis=-1).astype(np.uint8)
        frame[self.ground]=(62,202,105)
        boxes=[b for b in self.static.values() if self.stable(b) or stamp-b['stamp']<4 or b['label']=='fire_hydrant']
        boxes += [b for b in self.dynamic if stamp-b['stamp']<1.0]
        for b in boxes:
            c=np.asarray(b['corners_native']);uv=pixels(c)
            color=(244,132,218) if b['dynamic'] else ((100,185,255) if self.stable(b) else (255,186,77))
            for i,j in EDGES:cv2.line(frame,tuple(uv[i]),tuple(uv[j]),color,2,cv2.LINE_AA)
            center=uv.mean(axis=0).astype(int);cv2.circle(frame,tuple(center),3,color,-1)
        focus=self.focus
        if focus is not None and self.scans:
            # Local inset makes measured object dimensions visible in a large map.
            corners=np.asarray(focus['corners_native']);center=corners.mean(axis=0)
            iw,ih=320,220;inset=np.zeros((ih,iw,3),np.uint8);inset[:]=(7,17,27)
            scale=24.;pc=projection(center)
            def small(p):
                uv=(projection(p)-pc)*scale
                return np.stack([uv[...,0]+iw/2,ih/2-uv[...,1]+10],axis=-1).astype(int)
            pts=np.concatenate([p for _,p in self.scans]);near=(np.linalg.norm(pts[:,:2]-center[:2],axis=1)<7)&(abs(pts[:,2]-center[2])<4)
            uv=small(pts[near]);ok=(uv[:,0]>=1)&(uv[:,0]<iw-1)&(uv[:,1]>=28)&(uv[:,1]<ih-40);uv=uv[ok];inset[uv[:,1],uv[:,0]]=(101,122,141)
            uv=small(corners);color=(100,185,255) if self.stable(focus) else (255,186,77)
            for i,j in EDGES:cv2.line(inset,tuple(uv[i]),tuple(uv[j]),color,2,cv2.LINE_AA)
            cv2.putText(inset,focus['id'],(10,20),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1,cv2.LINE_AA)
            cv2.putText(inset,f'XYZ {center[0]:.1f} {center[1]:.1f} {center[2]:.1f} m',(10,ih-24),cv2.FONT_HERSHEY_SIMPLEX,.42,(222,234,246),1,cv2.LINE_AA)
            cv2.putText(inset,f"observed {max(0,stamp-focus['stamp']):.1f}s ago",(10,ih-7),cv2.FONT_HERSHEY_SIMPLEX,.38,(145,167,188),1,cv2.LINE_AA)
            cv2.rectangle(inset,(0,0),(iw-1,ih-1),color,1)
            frame[8:8+ih,-8-iw:-8]=inset
        return frame
