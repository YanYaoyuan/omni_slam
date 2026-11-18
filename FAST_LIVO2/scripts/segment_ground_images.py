#!/usr/bin/env python3
"""Pixel-wise outdoor ground candidates, kept distinct from geometric verification."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor,SegformerForSemanticSegmentation


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    processor=AutoImageProcessor.from_pretrained(str(args.model),local_files_only=True)
    model=SegformerForSemanticSegmentation.from_pretrained(str(args.model),local_files_only=True).eval()
    paths=sorted((args.input/'frames').glob('*.jpg'))
    if args.limit:paths=[paths[i] for i in np.linspace(0,len(paths)-1,args.limit).astype(int)]
    summary=[];start=time.monotonic()
    for i,p in enumerate(paths):
        cached=args.output/(p.stem+'.npz')
        if cached.exists():continue
        image=cv2.imread(str(p));rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        inputs=processor(images=rgb,return_tensors='pt',size={'height':384,'width':640})
        with torch.inference_mode():
            output=model(**inputs).logits
            logits=torch.nn.functional.interpolate(output,size=(360,640),mode='bilinear',align_corners=False)
            probability=logits.softmax(1);confidence,labels=probability.max(1)
        labels=labels[0].numpy().astype(np.uint8);confidence=confidence[0].numpy()
        ground=np.isin(labels,[0,1,9])&(confidence>=.65)
        # Retain the full semantic mask and confidence, not only a painted image.
        np.savez_compressed(cached,labels=labels,confidence=confidence.astype(np.float16),ground=ground)
        annotated=image.copy();annotated[ground]=(annotated[ground]*.50+np.array([80,220,90])*.50).astype(np.uint8)
        cv2.imwrite(str(args.output/(p.stem+'.jpg')),annotated)
        summary.append(dict(keyframe=p.stem,ground_fraction=float(ground.mean())))
        if i%25==0:print(i,len(paths),round(time.monotonic()-start,1),float(ground.mean()),flush=True)
    (args.output/'model.json').write_text(json.dumps({'source':json.loads((args.model/'source.json').read_text()),'ground_classes':[0,1,9],'id2label':model.config.id2label,'confidence_threshold':.65,'input_size':[640,384],'output_size':[640,360],'note':'Image candidates only, not traversability or LiDAR-validated ground.'},ensure_ascii=False,indent=2))
    print('done',len(paths),round(time.monotonic()-start,1),flush=True)


if __name__=='__main__':main()
