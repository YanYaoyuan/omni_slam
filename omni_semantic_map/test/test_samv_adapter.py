import numpy as np
import pytest
import torch
from omni_semantic_map.samv_adapter import scale_prompts,logits_to_original


def test_prompt_scaling_is_per_axis_and_preserves_labels():
    pixels=[[704,188],[0,0],[1407,375]]
    xy,labels=scale_prompts(pixels,[1,0,1],(376,1408))
    np.testing.assert_allclose(xy[0],[512,512])
    np.testing.assert_allclose(xy*[1408/1024,376/1024],pixels,atol=1e-4)
    np.testing.assert_array_equal(labels,[1,0,1])
    with pytest.raises(ValueError):scale_prompts([[1408,0]],[1],(376,1408))


def test_panorama_is_split_before_resize_without_cross_frame_bleed():
    panorama=torch.cat([torch.full((1,1,3,4),float(v)) for v in [-10,0,10]],dim=3)
    result=logits_to_original(panorama,3,(19,41))
    assert result.shape==(3,19,41)
    for i,v in enumerate([-10,0,10]):torch.testing.assert_close(result[i],torch.full((19,41),float(v)))
    with pytest.raises(ValueError):logits_to_original(panorama,5,(19,41))
