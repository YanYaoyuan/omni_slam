"""Full 330-frame real-robot visibility revision with correct atomic calibration.

No retraining or odometry rerun. Outputs are new; archived geometry stays frozen.
No accuracy claim is made without independently annotated reference evidence.
"""
from dataclasses import asdict
from pathlib import Path
import time
import os
import platform
import resource
import cv2
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from omni_semantic_map.measured_visibility import camera_from_record, measured_depth
from omni_semantic_map.visibility_cleanup import CleanupConfig, frame_evidence, rejection_mask, statistical_outliers

OUT = ROOT/'build/experiments/garage_visibility_20261113'
CACHE = ROOT/'build/benchmarks/evidence_retention_20261111/livo_full_strong_v2/cache_supportcc'
SOURCE = ROOT/'build/experiments/garage_current_20261112/current'
JOB = ROOT/'build/benchmarks/evidence_retention_20261111/jobs/livo_full_v2.json'


def main():
    assert not OUT.exists(), 'Preserve prior robot experiment'
    OUT.mkdir(parents=True)
    cv2.setNumThreads(1)
    cfg = CleanupConfig()
    manifest, job = read(CACHE/'manifest.json'), read(JOB)
    assert manifest['job_sha256'] == sha(JOB)
    paths = [Path(__file__), ROOT/'omni_semantic_map/omni_semantic_map/measured_visibility.py',
             ROOT/'omni_semantic_map/omni_semantic_map/visibility_cleanup.py',
             SOURCE/'geometry.npz', SOURCE/'classification.json', CACHE/'manifest.json', JOB]
    dump(OUT/'protocol.json', dict(scene='rosbag2245', config=asdict(cfg), minimum_refutations=1,
        methods=['baseline', 'sor', 'ours_visibility1'], frames=330,
        inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        camera='Each atomic packet supplies K, image dimensions, world camera pose, and camera-from-tracking transform. Template cache cam is not used.',
        depth='Nearest actual LiDAR measurement at each exact pixel; unknown pixels stay zero; no interpolation or depth completion.',
        coordinates='Single archived omni_odom frame; no alignment or corrected-map mixing.',
        reference_status='No independent ground-truth annotations available at execution. Counts and consistency are not accuracy.',
        review_frame_indices=[0,65,131,197,263,329],
        timing='One warmup and five CPU repetitions of evidence+rejection and SOR separately, after input loading. OpenCV and BLAS one thread; wall time, not guaranteed realtime.'))
    source = np.load(SOURCE/'geometry.npz'); xyz, labels = source['xyz'], source['instance']
    rows = read(SOURCE/'classification.json')['objects']
    retained = [r['object_id'] for r in rows if r['predictions']['mean']['retained']]
    selected = np.flatnonzero(np.isin(labels, retained)); points, ids = xyz[selected], labels[selected]
    owner = {int(o):r['object_id'] for r in rows for o in r['observation_ids']}
    assert len(owner) == sum(len(r['observation_ids']) for r in rows)
    poses = np.load(CACHE/'poses.npy')
    frames = []
    calibration = []
    for index, (rec, raw) in enumerate(zip(manifest['records'], job['frames'])):
        assert rec['frame_id'] == raw['frame_id']
        assert raw['world_frame'] == 'omni_odom'
        camera = camera_from_record(raw)
        T = np.asarray(raw['pose']); C = np.asarray(raw['camera_from_tracking'])
        pose = T @ np.linalg.inv(C)
        np.testing.assert_allclose(pose, raw['camera_pose'], atol=1e-8)
        np.testing.assert_allclose(pose, poses[index], atol=1e-8)
        image_age = (raw['stamp_ns']-raw['image_stamp_ns'])*1e-9
        assert -1e-3 <= image_age <= .1
        for name in ['cloud','rgb']:
            assert sha(raw[name]) == raw['sha256'][raw[name]]
        path = CACHE/'frames'/rec['file'];assert sha(path) == rec['sha256']
        with np.load(path) as z:
            masks = np.unpackbits(z['masks_packed'],axis=2)[:,:,:camera['W']].astype(bool)
        assert masks.shape[1:] == (camera['H'], camera['W'])
        cloud = np.load(raw['cloud'])['points']
        depth = measured_depth(cloud, C, camera)
        owners = np.array([owner.get(rec['observation_offset']+i,0) for i in range(rec['masks'])])
        frames.append((pose,camera,depth,masks,owners))
        calibration.append(dict(frame_id=rec['frame_id'], image_age_s=image_age,
                                measured_pixels=int((depth>0).sum()), camera=camera,
                                source_rgb=raw['rgb'], source_cloud=raw['cloud']))
    assert len(frames) == 330
    dump(OUT/'calibration_audit.json',dict(complete=True,frames=calibration,
        cache_template_dimensions=manifest['config']['cam'], actual_dimensions=frames[0][1],
        all_atomic_transforms_checked=True))

    def revision(record_events=False):
        votes = np.zeros((2,len(points)),np.uint16); visible = np.zeros(len(points),np.uint16); events=[]
        for i,(pose,cam,depth,masks,owners) in enumerate(frames):
            p,n,_,_,v = frame_evidence(points,ids,pose,cam,depth,masks,owners,cfg)
            votes += np.array([p,n],np.uint16); visible += v
            if record_events:
                events.append(dict(frame_id=manifest['records'][i]['frame_id'],positive=int(p.sum()),negative=int(n.sum()),visible=int(v.sum())))
        rejected = rejection_mask(votes[0],votes[1],cfg,single_frame=True)
        return rejected,votes,visible,events

    revision(); statistical_outliers(points,ids,cfg)
    durations={'ours_visibility1':[],'sor':[]}
    for _ in range(5):
        start=time.perf_counter(); removed,votes,visible,_=revision();durations['ours_visibility1'].append(time.perf_counter()-start)
        start=time.perf_counter();sor=statistical_outliers(points,ids,cfg);durations['sor'].append(time.perf_counter()-start)
    checked,_,_,events=revision(True);assert np.array_equal(checked,removed)
    np.savez_compressed(OUT/'evidence.npz',source_indices=selected,positive=votes[0],negative=votes[1],visible=visible)
    dump(OUT/'frame_events.json',events)
    per_object=[]
    for row in rows:
        oid=row['object_id']
        if oid not in retained:continue
        mask=ids==oid
        per_object.append(dict(object_id=oid,category=row['predictions']['mean']['name'],points=int(mask.sum()),
            observed_points=int((mask&(visible>0)).sum()),decisive_points=int((mask&(votes.sum(0)>0)).sum()),
            ours_removed=int((mask&removed).sum()),sor_removed=int((mask&sor).sum())))
    dump(OUT/'per_object.json',per_object)
    stats={}
    for name,reject in [('baseline',np.zeros(len(points),bool)),('sor',sor),('ours_visibility1',removed)]:
        folder=OUT/name;folder.mkdir();keep=np.ones(len(xyz),bool);keep[selected[reject]]=False
        np.savez_compressed(folder/'geometry.npz',xyz=xyz[keep],instance=labels[keep])
        np.save(folder/'retained_source_indices.npy',np.flatnonzero(keep))
        (folder/'classification.json').write_bytes((SOURCE/'classification.json').read_bytes())
        stats[name]=dict(removed=int(reject.sum()),kept_foreground=int((~reject).sum()),
            emptied_objects=[int(o) for o in retained if np.all(reject[ids==o])])
    cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown')
    dump(OUT/'run.json',dict(complete=True,frames=330,foreground_objects=len(retained),foreground_points=len(points),
         observed_points=int((visible>0).sum()),decisive_points=int((votes.sum(0)>0).sum()),methods=stats,
         timing={k:dict(repeats_s=v,median_s=float(np.median(v)),p95_s=float(np.quantile(v,.95))) for k,v in durations.items()},
         hardware=dict(cpu=cpu,platform=platform.platform(),cpu_affinity=sorted(os.sched_getaffinity(0)),
                       opencv_threads=cv2.getNumThreads(),openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS')),
         cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
         no_reference_accuracy_claim=True,protocol_sha256=sha(OUT/'protocol.json'),
         artifacts={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}))
    print('ROBOT COMPLETE',stats,flush=True)


if __name__=='__main__':main()
