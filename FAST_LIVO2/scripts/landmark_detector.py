"""Local open-vocabulary landmark detector used by the large-scene experiment."""
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import batched_nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

LABELS = ['person', 'door', 'potted plant', 'fire hydrant', 'fire extinguisher']
THRESHOLDS = {'person': .38, 'door': .35, 'bench': .35, 'potted plant': .35,
              'fire hydrant': .42, 'fire extinguisher': .45}


class LandmarkDetector:
    def __init__(self, model_path):
        torch.set_num_threads(4)
        self.processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True)
        self.model=AutoModelForZeroShotObjectDetection.from_pretrained(str(model_path),local_files_only=True).eval()
        self.prompt='. '.join(LABELS)+'.'

    def __call__(self, rgb):
        inputs=self.processor(images=rgb,text=self.prompt,return_tensors='pt',
                              size={'shortest_edge':360,'longest_edge':640})
        with torch.inference_mode():
            outputs=self.model(**inputs)
        result=self.processor.post_process_grounded_object_detection(
            outputs,inputs.input_ids,threshold=.30,text_threshold=.25,
            target_sizes=[rgb.shape[:2]])[0]
        detections=[]
        for label,box,score in zip(result['text_labels'],result['boxes'],result['scores']):
            label=label.strip()
            if label not in LABELS or float(score)<THRESHOLDS[label]:continue
            b=box.detach().cpu().numpy()
            b[[0,2]]=np.clip(b[[0,2]],0,rgb.shape[1]-1)
            b[[1,3]]=np.clip(b[[1,3]],0,rgb.shape[0]-1)
            if b[2]-b[0]<8 or b[3]-b[1]<12:continue
            detections.append(dict(label=label.replace(' ','_'),score=float(score),box=b.tolist()))
        if detections:
            ids=batched_nms(torch.tensor([d['box'] for d in detections]),
                torch.tensor([d['score'] for d in detections]),
                torch.tensor([LABELS.index(d['label'].replace('_',' ')) for d in detections]),.5)
            detections=[detections[i] for i in ids.tolist()]
        return detections


if __name__=='__main__':
    import argparse,json,time,cv2
    from PIL import Image,ImageDraw,ImageFont
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--images',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    detector=LandmarkDetector(args.model);records=[];pictures=[]
    for path in sorted(args.images.glob('*_[012].jpg')):
        rgb=np.asarray(Image.open(path).convert('RGB'));began=time.monotonic();detections=detector(rgb);elapsed=time.monotonic()-began
        annotated=rgb.copy()
        for d in detections:
            x1,y1,x2,y2=map(int,d['box']);cv2.rectangle(annotated,(x1,y1),(x2,y2),(50,255,120),2)
            cv2.putText(annotated,f"{d['label']} {d['score']:.2f}",(x1,max(y1-5,15)),cv2.FONT_HERSHEY_SIMPLEX,.45,(50,255,120),1,cv2.LINE_AA)
        im=Image.fromarray(annotated);im.save(args.output/path.name);pictures.append((path.name,im));records.append(dict(image=str(path),detections=detections,seconds=elapsed));print(path.name,round(elapsed,2),detections,flush=True)
    (args.output/'detections.json').write_text(json.dumps(records,indent=2))
    sheet=Image.new('RGB',(1920,420*((len(pictures)+2)//3)),'#101b28');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',21)
    for i,(name,im) in enumerate(pictures):
        x=i%3*640;y=i//3*420;sheet.paste(im,(x,y+40));draw.text((x+8,y+5),name,font=font,fill='white')
    sheet.save(args.output/'contact.jpg')
