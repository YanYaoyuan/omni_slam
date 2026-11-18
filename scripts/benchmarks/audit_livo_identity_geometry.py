"""Frozen-candidate causal identity and measured multi-visit geometry audit.

No registration, pose correction, GT position or identity labels are used.
Nearest-surface distances are internal consistency, sensitive to visibility.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree
from audit_evidence_retention import ROOT, OUT, REPORT, dump
from omni_semantic_map.flat_benchmark import sha256

DEST=OUT/'identity_geometry_v1'


def main():
    dest=DEST;dest.mkdir(exist_ok=False)
    source=OUT/'scale_ablation_v1/cache_supportcc/both'
    cache=OUT/'livo_full_strong_v2/cache_supportcc'
    job_path=OUT/'jobs/livo_full_v2.json';job=json.loads(job_path.read_text())
    manifest=json.loads((cache/'manifest.json').read_text())
    frames={r['frame_id']:r for r in job['frames']};lookup={}
    for record in manifest['records']:
        for i in range(record['masks']):lookup[record['observation_offset']+i]=(record,i)
    classification=json.loads((source/'classification.json').read_text())['objects']
    frozen=dict(scope=__doc__,candidate_run_sha256=sha256(source/'run.json'),cache_sha256=sha256(cache/'manifest.json'),
        source_sha256=sha256(Path(__file__)),selection='All retained foreground objects; split temporal visits at >30 seconds between member observations.',
        visit_gap_s=30,comparison='First versus each later visit; pool measured supports, deduplicate at original 2.5cm output voxel scale; symmetric nearest surface distances and directed overlap at 0.20m / 0.40m. No ICP fit.',
        control='Adjacent member-frame surface distance within each visit, same statistic. Visibility and partial overlap affect both; neither is GT accuracy.')
    dump(dest/'protocol.json',frozen)
    xyz=np.load(cache/'geometry.npz')['xyz'];objects=[];frame_cache={}
    def cloud(oid):
        record,index=lookup[oid];file=cache/'frames'/record['file']
        if file not in frame_cache:
            assert sha256(file)==record['sha256'];frame_cache[file]=dict(np.load(file))
        f=frame_cache[file];ids=f['point_ids'][f['offsets'][index]:f['offsets'][index+1]]
        return xyz[ids]
    def voxel(points):
        if not len(points):return points
        _,ids=np.unique(np.floor(points/.025).astype(np.int64),axis=0,return_index=True)
        return points[np.sort(ids)]
    def compare(a,b):
        if not len(a) or not len(b):return dict(valid=False,reason='empty measured support')
        dab,idx=cKDTree(b).query(a);dba,_=cKDTree(a).query(b)
        d=np.r_[dab,dba]
        return dict(valid=True,points_a=len(a),points_b=len(b),symmetric_nn_median_m=float(np.median(d)),symmetric_nn_p90_m=float(np.quantile(d,.9)),
            overlap_a_to_b_020=float(np.mean(dab<.2)),overlap_b_to_a_020=float(np.mean(dba<.2)),
            overlap_a_to_b_040=float(np.mean(dab<.4)),overlap_b_to_a_040=float(np.mean(dba<.4)),
            nearest_displacement_median_m=np.median(b[idx]-a,axis=0).tolist(),
            displacement_scope='Nearest-surface displacement, not a rigid pose estimate or GT drift.')
    for row in classification:
        if not row['predictions']['mean']['retained']:continue
        byframe=defaultdict(list)
        for oid in row['observation_ids']:byframe[lookup[oid][0]['frame_id']].append(oid)
        fids=sorted(byframe);visits=[]
        for fid in fids:
            if not visits or (frames[fid]['stamp_ns']-frames[visits[-1][-1]]['stamp_ns'])/1e9>30:visits.append([])
            visits[-1].append(fid)
        clouds={fid:voxel(np.concatenate([cloud(o) for o in byframe[fid]])) for fid in fids}
        visit_clouds=[voxel(np.concatenate([clouds[f] for f in visit])) for visit in visits]
        alias=min(row['observation_ids'])+1
        if len(visits)>1:np.savez_compressed(dest/f'object_{alias}_visits.npz',**{f'visit_{i}':p for i,p in enumerate(visit_clouds)})
        objects.append(dict(alias=alias,name=row['predictions']['mean']['name'],frames=fids,visits=visits,
            comparisons=[dict(visit_a=0,visit_b=i,**compare(visit_clouds[0],p)) for i,p in enumerate(visit_clouds) if i],
            adjacent_comparisons=[dict(frame_a=a,frame_b=b,**compare(clouds[a],clouds[b])) for visit in visits for a,b in zip(visit,visit[1:])]))
    dump(dest/'geometry.json',dict(complete=True,protocol_sha256=sha256(dest/'protocol.json'),objects=objects,scope=__doc__))
    # Instrument a separate exact replay, preserving every frozen run.
    import run_livo_scale_ablation as scale
    from omni_semantic_map.replay_fusion import ConceptGraphsBackend
    replay=dest/'replay';replay.mkdir();(replay/'protocol.json').write_bytes((OUT/'scale_ablation_v1/protocol.json').read_bytes())
    scale.ROOT_OUT=replay
    original=ConceptGraphsBackend.integrate;history=[];targets=[5,3074,494,670]
    def observed(self,observations,pose):
        result=original(self,observations,pose)
        labels=self.labels(finalize=False)
        history.append(dict(frame_id=observations[0].frame_id if observations else None,target_aliases={str(t+1):labels.get(t) for t in targets}))
        return result
    ConceptGraphsBackend.integrate=observed
    try:scale.execute('cache_supportcc','both')
    finally:ConceptGraphsBackend.integrate=original
    copied=replay/'cache_supportcc/both'
    for key in ['xyz','instance']:
        np.testing.assert_array_equal(np.load(copied/'geometry.npz')[key],np.load(source/'geometry.npz')[key])
    assert json.loads((copied/'classification.json').read_text())==json.loads((source/'classification.json').read_text())
    dump(dest/'causal_identity.json',dict(complete=True,exact_final_reproduction=True,history=history,
        scope='Native causal labels before final postprocessing, versus the unchanged final object partition. Not GT identity accuracy.'))
    report=REPORT/'identity_geometry_v1'
    for name in ['protocol.json','geometry.json','causal_identity.json']:
        dump(report/name,json.loads((dest/name).read_text()))
    print('COMPLETE identity/geometry',len(objects),'objects;',sum(len(o['visits'])>1 for o in objects),'multi-visit',flush=True)


if __name__=='__main__':main()
