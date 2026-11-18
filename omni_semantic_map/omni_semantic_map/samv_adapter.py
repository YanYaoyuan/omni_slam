"""Cached, single-sample adapter for the pinned external SAM-V model.

No learned modules or thresholds are changed. Keep image resizing and prompt
scaling consistent with the authors' square 1024 input path. Split panoramic
logits before interpolation, then invert the image resize for measured pixels.
"""
import numpy as np


def scale_prompts(pixels, labels, image_hw):
    xy = np.asarray(pixels, np.float32); lab = np.asarray(labels, np.int64)
    h, w = image_hw
    if min(h,w)<=0 or xy.ndim!=2 or xy.shape[1]!=2 or lab.shape!=(len(xy),):
        raise ValueError('Invalid prompt dimensions')
    if not len(xy) or not np.isfinite(xy).all() or np.any(xy<0) or np.any(xy >= [w,h]) or not np.isin(lab,[0,1]).all():
        raise ValueError('Invalid prompt coordinates or labels')
    return xy*np.array([1024/w,1024/h],np.float32), lab


def logits_to_original(logits, frames, image_hw):
    import torch.nn.functional as F
    if logits.ndim!=4 or logits.shape[:2]!=(1,1) or frames<1 or logits.shape[3]%frames:
        raise ValueError('Invalid panoramic logits')
    h,w=logits.shape[2],logits.shape[3]//frames
    per_frame=logits[0,0].reshape(h,frames,w).permute(1,0,2)[:,None].float()
    native=F.interpolate(per_frame,size=(1024,1024),mode='bilinear',align_corners=False)
    return F.interpolate(native,size=tuple(image_hw),mode='bilinear',align_corners=False)[:,0]


def encode_context(model, images, sam_features=None):
    """images: [1,N,3,1024,1024], RGB0..255; no future/causal claim."""
    if images.ndim!=5 or images.shape[0]!=1 or tuple(images.shape[-3:])!=(3,1024,1024):
        raise ValueError('One square-preprocessed sample required')
    if sam_features is None:
        sam_features,_=model.encode_sam_batched(model.sam.preprocess(images))
    vggt_input,_=model.preprocess_vggt_images(images)
    vggt,cameras,_=model.encode_vggt_batched(vggt_input)
    fused=model.fuse_embeddings_batched(sam_features,vggt)
    _,n,c,h,w=fused.shape
    concat=fused.permute(0,2,3,1,4).reshape(1,c,h,n*w)
    return dict(sam=sam_features,vggt=vggt,cameras=cameras,concat=concat,frames=n)


def decode_context(model, context, coordinates, labels, frame_indices):
    sparse,dense=model.sam.prompt_encoder(points=(coordinates[None],labels[None]),boxes=None,masks=None)
    fused=model.fuse_prompts(sparse,context['cameras'],context['vggt'],frame_indices[None],coordinates[None])
    n=context['frames']
    outputs=model.sam.mask_decoder(image_embeddings=context['concat'],
        image_pe=model.sam.prompt_encoder.get_dense_pe().repeat(1,1,1,n),
        sparse_prompt_embeddings=fused,dense_prompt_embeddings=dense.repeat(1,1,1,n),multimask_output=False)
    if not all(value.isfinite().all().item() for value in outputs):
        raise ValueError('External decoder produced nonfinite values')
    return outputs
