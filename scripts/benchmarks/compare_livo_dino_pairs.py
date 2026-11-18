"""Fixed six-pair appearance diagnostic; no inference or threshold fitting."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from audit_evidence_retention import OUT,REPORT,dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.livo_dino_support import unit


def mutual(scores):
    if not scores.size:return np.array([],int),np.array([],int)
    j=scores.argmax(1);i=np.arange(len(j));valid=(scores.argmax(0)[j]==i)
    return i[valid],j[valid]


def main():
    root=OUT/'identity_geometry_v1';protocol=root/'dino_pair_protocol.json';spec=json.loads(protocol.read_text())
    cache=OUT/'livo_full_strong_v2/cache_supportcc';m=json.loads((cache/'manifest.json').read_text())
    dino=OUT/'dino_livo_tokens_v2';dm=json.loads((dino/'manifest.json').read_text());assert dm['complete'] and dm['frames']==330
    for r in dm['records']:assert sha256(dino/r['file'])==r['sha256']
    lookup={i:(r,i-r['observation_offset']) for r in m['records'] for i in range(r['observation_offset'],r['observation_offset']+r['masks'])}
    job=json.loads((OUT/'jobs/livo_full_v2.json').read_text());frames={f['frame_id']:f for f in job['frames']}
    report=REPORT/'identity_geometry_v1';report.mkdir(exist_ok=True)
    def observation(oid):
        r,index=lookup[oid];fid=r['frame_id'];f=dict(np.load(cache/'frames'/r['file']));d=dict(np.load(dino/f'{fid:06d}.npz'))
        lo,hi=d['offsets'][index:index+2];assert hi>lo,'empty measured DINO support'
        return dict(frame=fid,index=index,siglip=unit(f['features'][index]),dino=d['mean'][index],xyz=d['local_xyz'][lo:hi],tokens=unit(d['local_features'][lo:hi]),seg=f['segmentation'])
    rows=[]
    for group in ['positive_hypotheses','same_frame_distinct_pairs']:
        for pair in spec[group]:
            a=observation(pair['a']);b=observation(pair['b'])
            cos=a['tokens']@b['tokens'].T;ia,ib=mutual(cos);dist=cdist(a['xyz'],b['xyz']);ga,gb=mutual(-dist)
            metrics=dict(siglip_cosine=float(a['siglip']@b['siglip']),dino_pooled_cosine=float(a['dino']@b['dino']),
                mutual_appearance_pairs=len(ia),mutual_appearance_cosine=float(np.mean(cos[ia,ib])) if len(ia)>=3 else None,
                anchors_a=len(a['xyz']),anchors_b=len(b['xyz']))
            for radius in [.2,.4]:
                valid=dist[ga,gb]<radius;x,y=ga[valid],gb[valid]
                metrics[f'geometry_{radius:.1f}m']=dict(mutual_pairs=len(x),cosine=float(np.mean(cos[x,y])) if len(x)>=3 else None,
                    null_reason='fewer than three reciprocal geometry anchors' if len(x)<3 else None)
            panels=[]
            for oid,o in [(pair['a'],a),(pair['b'],b)]:
                image=cv2.imread(frames[o['frame']]['rgb']);mask=o['seg']==o['index'];image[mask]=(image[mask]*.6+np.array([0,210,255])*.4).astype(np.uint8)
                cv2.rectangle(image,(0,0),(image.shape[1],26),(18,18,18),-1)
                cv2.putText(image,f'frame {o["frame"]} | observation {oid+1}',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA);panels.append(image)
            picture=report/f'pair_{pair["a"]+1}_{pair["b"]+1}.jpg';assert cv2.imwrite(str(picture),np.concatenate(panels,axis=1))
            rows.append(dict(group=group,**pair,frame_a=a['frame'],frame_b=b['frame'],metrics=metrics,evidence_image=str(picture),image_sha256=sha256(picture)))
    result=dict(complete=True,protocol_sha256=sha256(protocol),dino_manifest_sha256=sha256(dino/'manifest.json'),rows=rows,
        scope='Six diagnostic pairs, not benchmark metrics. Positive hypotheses 495 and 671 remain uncertain; no threshold or method is selected from these scores.')
    dump(root/'dino_pairs.json',result);dump(report/'dino_pairs.json',result);dump(report/'dino_pair_protocol.json',spec)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
