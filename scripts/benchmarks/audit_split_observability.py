"""Diagnose whether existing masks can distinguish oracle partitions.

This consumes reference labels exclusively for diagnosis; it does not emit map
predictions or choose inference parameters. All candidate objects are retained.
"""
from itertools import combinations
import numpy as np
from paper_final_common import ROOT, read, dump
from run_reversible_revision import OUT, verify, build_views
from omni_semantic_map.visibility_cleanup import CleanupConfig
from omni_semantic_map.reversible_split import SplitConfig, view_gain


def main():
    assert not (OUT/'oracle_mask_separability.json').exists()
    protocol=verify(); results={}
    for scene in protocol['scenes']:
        key=scene['key']; ref=np.load(scene['reference'])
        xyz=ref['xyz']; truth=ref['instance'].copy()
        if 'valid' in ref.files:truth[~ref['valid']]=0
        labels=np.load(ROOT/'build/benchmarks/visibility_cleanup_20261112/evaluation'/key/'native_bridge_2.npz')['instance']
        rows=read(scene['classification'])['objects']
        views,_,_,_=build_views(scene,xyz,labels,rows,CleanupConfig())
        records=[]
        for oid in np.unique(labels):
            if oid<=0:continue
            gids,counts=np.unique(truth[labels==oid],return_counts=True)
            significant=gids[(gids>0)&(counts>=5)]
            for ga,gb in combinations(significant,2):
                events=[]
                for i,v in enumerate(views):
                    select=(labels[v.indices]==oid)&np.isin(truth[v.indices],[ga,gb])
                    local=v.indices[select]; child=(truth[local]==gb).astype(np.uint8)
                    gain=view_gain(v.membership[:,select],child,SplitConfig())
                    if gain is not None:events.append(dict(index=i,frame_id=v.frame_id,gain=float(gain)))
                records.append(dict(object_id=int(oid),gt_pair=[int(ga),int(gb)],events=events,
                    separating_frames=sum(e['gain']>=.1 for e in events),
                    mean_gain=float(np.mean([e['gain'] for e in events])) if events else None))
        results[key]=records
        print(key,[(r['object_id'],r['gt_pair'],r['separating_frames'],r['mean_gain'])for r in records],flush=True)
    dump(OUT/'oracle_mask_separability.json',dict(diagnostic_only=True,gt_not_used_by_mapper=True,
        scope='Masks evaluated on two GT-supported subsets of the original fixed prediction; excludes background and other GT. This is an optimistic separability diagnostic, not a map score.',results=results))


if __name__=='__main__':main()
