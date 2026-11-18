"""Two preselected real-image SAM3 checks; no benchmark or method claim.

Explicit local checkpoint and native image predictor. Never downloads models.
This checks whether new masks carry measured support missing in the old frontend.
"""
import argparse
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from paper_final_common import ROOT,read,dump,sha

OUT=ROOT/'build/benchmarks/sam3_discovery_probe_v1'
MODEL=Path('/home/user/vbot/sam3')


def freeze():
    assert not OUT.exists()
    specs=[('kitti360_0000',ROOT/'build/benchmarks/kitti360_development_v1/jobs/2013_05_28_drive_0000_sync.json'),
        ('rosbag2245',ROOT/'build/benchmarks/evidence_retention_20261111/jobs/livo_full_v2.json')]
    scenes=[];files={}
    for key,path in specs:
        job=read(path);rank=len(job['frames'])//2;frame=job['frames'][rank]
        inputs=[path,Path(frame['rgb']),Path(frame.get('measured',frame.get('cloud')))]
        for f in inputs:files[str(f)]=sha(f)
        raw_cache=Path(job['output'])/'cache';manifest=read(raw_cache/'manifest.json');rec=manifest['records'][rank]
        assert rec['frame_id']==frame['frame_id'];old=raw_cache/'frames'/rec['file'];assert sha(old)==rec['sha256']
        files[str(old)]=sha(old)
        if key.startswith('kitti'):
            files[str(raw_cache/'geometry.npz')]=sha(raw_cache/'geometry.npz')
        scenes.append(dict(key=key,job=str(path),frame_rank=rank,frame=frame,camera=job['config']['cam'],old_cache=str(raw_cache),cache_record=rec))
    weights=MODEL/'checkpoints/sam3.pt';files[str(weights)]=sha(weights)
    sources={str(f):sha(f) for f in (MODEL/'sam3').rglob('*.py')}
    for f in [Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/sparse_lidar_support.py',ROOT/'omni_semantic_map/omni_semantic_map/sparse_semantic_support.py']:
        sources[str(f)]=sha(f)
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),scenes=scenes,input_hashes=files,source_hashes=sources,
        checkpoint=str(weights),prompts=['car','building'],confidence_threshold=.5,resolution=1008,autocast='bfloat16',
        selection='Middle cached frame in each exposed sequence, chosen by index before SAM3 inference. No retries, prompt search or GT.',
        scope='Runtime and measured-support probe only. Not a map, independent benchmark, tracking result or accuracy comparison. No model weight updates.',
        hardware='GPU0 only, 50% process allocator limit, require >=18GiB driver free before loading. CPU threads4. No modification of other processes.'))
    print('FROZEN',[(s['key'],s['frame']['frame_id']) for s in scenes],flush=True)


def predict():
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    p=read(OUT/'protocol.json');assert not (OUT/'predictions_complete.json').exists()
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    torch.set_num_threads(4);assert torch.cuda.is_available()
    free,total=torch.cuda.mem_get_info();assert free>=18*1024**3,(free,total)
    torch.cuda.set_per_process_memory_fraction(.5);torch.cuda.reset_peak_memory_stats();torch.manual_seed(0)
    start=time.monotonic();model=build_sam3_image_model(checkpoint_path=p['checkpoint'],load_from_HF=False,device='cuda',compile=False)
    processor=Sam3Processor(model,resolution=p['resolution'],confidence_threshold=p['confidence_threshold'])
    torch.cuda.synchronize();load_seconds=time.monotonic()-start;files={};stats=[]
    for s in p['scenes']:
        dest=OUT/s['key'];dest.mkdir(exist_ok=False);image=Image.open(s['frame']['rgb']).convert('RGB')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            torch.cuda.synchronize();start=time.monotonic();state=processor.set_image(image);torch.cuda.synchronize();encode=time.monotonic()-start
            for prompt in p['prompts']:
                processor.reset_all_prompts(state);torch.cuda.synchronize();start=time.monotonic()
                result=processor.set_text_prompt(prompt=prompt,state=state);torch.cuda.synchronize();seconds=time.monotonic()-start
                masks=result['masks'].detach().cpu().numpy().reshape(-1,image.height,image.width).astype(bool)
                scores=result['scores'].detach().float().cpu().numpy();boxes=result['boxes'].detach().float().cpu().numpy()
                assert len(masks)==len(scores)==len(boxes) and np.isfinite(scores).all()
                path=dest/(prompt+'.npz');np.savez_compressed(path,masks_packed=np.packbits(masks,axis=2),shape=np.asarray(masks.shape),scores=scores,boxes=boxes)
                files[str(path)]=sha(path);stats.append(dict(scene=s['key'],frame_id=s['frame']['frame_id'],prompt=prompt,
                    masks=len(masks),image_encode_seconds=encode,decode_seconds=seconds,scores=scores.tolist()))
                print('PREDICTED',s['key'],prompt,len(masks),'masks',seconds,'s',flush=True)
            del state
    dump(OUT/'predictions_complete.json',dict(complete=True,files=files,protocol_sha256=sha(OUT/'protocol.json'),results=stats,
        load_seconds=load_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda))


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from omni_semantic_map.sparse_lidar_support import project_measured_visibility
    from omni_semantic_map.sparse_semantic_support import sparse_support
    p=read(OUT/'protocol.json');seal=read(OUT/'predictions_complete.json')
    assert seal['protocol_sha256']==sha(OUT/'protocol.json')
    for f,h in seal['files'].items():assert sha(f)==h,f
    dest=ROOT/'output/sam3_discovery_probe_v1';dest.mkdir(parents=True,exist_ok=True);details='';data=[]
    for s in p['scenes']:
        frame=s['frame'];rgb=np.asarray(Image.open(frame['rgb']).convert('RGB'));h,w=rgb.shape[:2]
        with np.load(Path(s['old_cache'])/'frames'/s['cache_record']['file']) as f:old=f['segmentation']>=0
        if s['key'].startswith('kitti'):
            xyz=np.load(Path(s['old_cache'])/'geometry.npz')['xyz'][:s['cache_record']['map_points']]
            depth=np.load(frame['measured'])['depth'];x,y,_,visible=project_measured_visibility(xyz,np.asarray(frame['pose']),s['camera'],depth,SimpleNamespace(depth_absolute_m=.05,depth_relative=0.))
            ids=np.flatnonzero(visible);px=np.column_stack([x[ids],y[ids]])
        else:
            points=np.load(frame['cloud'])['points'];pr=sparse_support(points,frame['camera_from_tracking'],frame['intrinsic'],np.full((h,w),-1,int));ids=pr['visible_indices'];px=pr['visible_pixels']
        old_measured=old[px[:,1],px[:,0]];fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
        axes[0].imshow(rgb);axes[0].set_title('Original RGB');scene_stats=[]
        for ax,prompt in zip(axes[1:],p['prompts']):
            z=np.load(OUT/s['key']/(prompt+'.npz'));masks=np.unpackbits(z['masks_packed'],axis=2)[:,:,:w].astype(bool);union=masks.any(0)
            active=union[px[:,1],px[:,0]];record=dict(scene=s['key'],frame_id=frame['frame_id'],prompt=prompt,masks=len(masks),
                visible_measured_points=len(ids),old_any_mask_points=int(old_measured.sum()),new_prompt_mask_points=int(active.sum()),
                newly_covered_vs_old_any_mask=int((active&~old_measured).sum()))
            scene_stats.append(record);data.append(record);ax.imshow(rgb)
            color=np.zeros((h,w,4))
            for i,mask in enumerate(masks):color[mask]=plt.cm.tab20((i%20)/19);color[mask,3]=.45
            ax.imshow(color);ax.set_title(f'{prompt}: {len(masks)} masks\n{record["newly_covered_vs_old_any_mask"]} extra measured supports')
        for ax in axes:ax.axis('off')
        fig.suptitle(s['key']+f' | fixed middle frame {frame["frame_id"]}');fig.savefig(dest/(s['key']+'.png'),dpi=140);plt.close(fig)
        details+=f'<h2>{s["key"]} / 帧 {frame["frame_id"]}</h2><img src="{s["key"]}.png"><pre>'+str(scene_stats)+'</pre>'
    result=dict(complete=True,probe_only=True,no_accuracy_claim=True,measured_supports=data,prediction_seal_sha256=sha(OUT/'predictions_complete.json'),runtime=seal)
    dump(dest/'summary.json',result)
    (dest/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SAM3 新观测链路验证</title><style>body{max-width:1250px;margin:35px auto;padding:0 24px;font:16px/1.7 system-ui;color:#24384a}img{width:100%}pre{white-space:pre-wrap}</style><h1>SAM3：真实图像到新增测量支持</h1><p>两个已暴露序列的中间帧，推理前固定；相同 car/building 提示、默认 0.5 阈值。本页只验证模型及新增观测链路，不是新机制跑分，也不能把新增点数当作正确率。没有用 GT 选择帧、提示或掩码。</p>'+details+'<p>GPU 峰值分配显存 '+f'{seal["peak_allocated_bytes"]/1024**3:.2f} GiB'+'。新掩码依然可能含错误区域，完整对象关联、等预算策略对照和公共 benchmark 评分尚需执行。</p><p><a href="summary.json">推理与测量支持记录</a></p></html>')
    print(dest/'index.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','predict','report']);globals()[parser.parse_args().action]()
