"""Exercise actual reference/evaluation code on synthetic files, not test GT."""
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import run_hypersim_benchmark as runner
from paper_final_common import ROOT,dump,sha
sys.path.insert(0,str(runner.META/'python_deps'))
import h5py


def main():
    # The actual independent reference builder must refuse GT access before seal.
    assert not (runner.OUT/'predictions_complete.json').exists()
    try:runner.reference('ai_001_010')
    except FileNotFoundError as e:assert 'predictions_complete.json' in str(e)
    else:raise AssertionError('Reference decoding gate did not stop unsealed access')
    with tempfile.TemporaryDirectory(prefix='omni_hypersim_evaluator_') as tmp:
        root=Path(tmp);runner.OUT=root/'out';runner.DATA=root/'data';scene='synthetic'
        directory=runner.DATA/'data'/scene/'images/scene_cam_00_geometry_hdf5';directory.mkdir(parents=True)
        shape=(8,8);distance=np.full(shape,2.,np.float16);semantic=np.zeros(shape,np.int16);instances=np.full(shape,-1,np.int16)
        semantic[0,0]=5;instances[0,0]=0  # Raw instance zero must not become background.
        semantic[0,4]=5;instances[0,4]=1
        semantic[4,0]=1;instances[4,0]=7  # Wall becomes background.
        files=[];frames=[]
        for frame in [0,1]:
            for name,a in [('depth_meters',distance),('semantic',semantic),('semantic_instance',instances)]:
                path=directory/f'frame.{frame:04d}.{name}.hdf5'
                with h5py.File(path,'w') as h:h.create_dataset('dataset',data=a)
                files.append(dict(path=str(path),sha256=sha(path)))
            frames.append(dict(frame_id=frame,pose=np.eye(4).tolist()))
        dump(runner.DATA/(scene+'_download.json'),dict(files=files))
        job=root/'job.json';dump(job,dict(config={'cam':dict(H=8,W=8,fx=4,fy=4,cx=3.5,cy=3.5)},frames=frames))
        protocol=dict(scenes=[scene],jobs={scene:str(job)},reference=dict(pixel_stride=4,voxel_size_m=.05,excluded_nyu40=[1,2,22]),primary='synthetic smoke')
        dump(runner.OUT/'protocol.json',protocol);dump(runner.OUT/'predictions_complete.json',dict(complete=True))
        runner.gate=lambda:(protocol,{})
        runner.reference(scene)
        ref=np.load(runner.OUT/'reference'/scene/'reference.npz')
        assert len(ref['xyz'])==3 and set(ref['instance'])=={0,1,2}
        choices={}
        for method in ['history2','simple_ratio']:
            folder=root/method;folder.mkdir();choices[method]=folder
            fg=ref['instance']>0
            labels=ref['instance'][fg]*10 if method=='simple_ratio' else np.full(int(fg.sum()),10)
            np.savez_compressed(folder/'geometry.npz',xyz=ref['xyz'][fg],instance=labels)
            dump(folder/'classification.json',dict(objects=[dict(object_id=int(o),predictions={'mean':{'retained':True}}) for o in np.unique(labels)]))
        runner.paths_for=lambda s:choices;runner.METHODS=list(choices)
        runner.evaluate();result=json.loads((runner.OUT/'summary.json').read_text())
        assert result['scene_mean_pq100']==dict(history2=0.,simple_ratio=100.)
        record=dict(complete=True,reference_gate_blocks_before_seal=True,raw_instance_zero_preserved=True,
                    structural_background_and_unknown_handled=True,duplicate_frames_voxelized_once=True,
                    perfect_prediction_pq100=100.,merged_prediction_pq100=0.,
                    strict_half_iou_not_matched=True,source_sha256=sha(runner.__file__),script_sha256=sha(__file__))
        dump(ROOT/'build/benchmarks/hypersim_independent_20261113/evaluator_check.json',record)
        print(record)


if __name__=='__main__':main()
