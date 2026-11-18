"""Replay the existing strong reference before new-window model inference."""
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from prepare_batch_transport_assessment import OUT,verify
from run_intrinsic_public_holdout import isolated_adapter


def main():
    p=verify();folder=OUT/'backbone_validation';assert not (folder/'golden.json').exists()
    native=isolated_adapter('run_robot_geometry_reference.py');native.OUT=folder;native.golden()
    result=read(folder/'golden.json');assert result['complete']
    import torch
    weights=torch.load(p['selected_model'],map_location='cpu',weights_only=False)
    assert weights['training']==['kitti360_0000','kitti360_0002']
    dump(folder/'validation.json',dict(complete=True,source_sha256=sha(__file__),golden_sha256=sha(folder/'golden.json'),
        geometry_backend_sha256=sha(ROOT/'scripts/benchmarks/geometry_reference_backend.py'),selected_model_sha256=sha(p['selected_model']),
        selected_model_training=weights['training'],new_reference_decoded=False,new_inference=False,
        scope='Exact replay of both old geometry_graph maps,groups,classscores and raw-query components in4oldframes. Deploymentmodel trainingmetadata checked. New frontend model stage not executed.'))
    print('ASSESSMENT BACKBONE VERIFIED',flush=True)


if __name__=='__main__':main()
