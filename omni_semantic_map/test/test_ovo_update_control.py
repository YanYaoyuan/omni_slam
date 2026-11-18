import numpy as np
import pytest
from omni_semantic_map.replay_fusion import FusionObservation
from omni_semantic_map.ovo_update_control import initialize,correct

@pytest.mark.parametrize('offset,expected_objects',[(.01,1),(5.,2)])
def test_official_ovo_kernel_merges_only_supported_close_instances(offset,expected_objects):
 xyz=np.array(np.meshgrid(*([np.linspace(0,.1,3)]*3))).reshape(3,-1).T
 records=[]
 for k in range(3):
  records.append([FusionObservation(k,k,1,xyz,np.ones_like(xyz)*.5,[1,0],[1,0]),FusionObservation(k+3,k,1,xyz+offset,np.ones_like(xyz)*.5,[1,0],[1,0])])
 objects=[dict(object_id=1,observation_ids=[0,1,2],image_feature=np.array([1.,0.])),dict(object_id=4,observation_ids=[3,4,5],image_feature=np.array([1.,0.]))]
 state,points,frames,labels,owners=initialize(records,objects)
 result=correct(state,points,frames,labels,owners,np.repeat(np.eye(4)[None],3,0),6)
 assert len(state.objects)==expected_objects and len(np.unique(result))==expected_objects
 assert np.all(result[:3]==1)
 assert result[3]==(1 if expected_objects==1 else 4)
