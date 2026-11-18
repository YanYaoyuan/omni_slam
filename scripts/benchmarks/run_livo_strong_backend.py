"""Run the unchanged frozen CG backend with observational causal diagnostics."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from audit_evidence_retention import ROOT, dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.replay_fusion import ConceptGraphsBackend
from retention_holdout_backend import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    original=ConceptGraphsBackend.integrate;history=[];previous={}
    def integrate(self,observations,pose):
        nonlocal previous
        began=time.monotonic();result=original(self,observations,pose);elapsed=time.monotonic()-began
        labels=self.labels(finalize=False)
        changed=[i for i,v in previous.items() if v>0 and labels.get(i,0)!=v]
        objects=[]
        for obj in self._objects:
            ids=sorted(set(map(int,obj['observation_ids'])));xyz=np.asarray(obj['pcd'].points)
            objects.append(dict(persistent_alias=min(ids)+1,observation_ids=ids,points=len(xyz),center=xyz.mean(0).tolist()))
        history.append(dict(frame_id=observations[0].frame_id if observations else None,
            received_observations=len(observations),association_wall_s=elapsed,
            rejected_observations=list(map(int,self.rejected)),changed_past_observations=changed,
            labels={str(k):int(v) for k,v in labels.items()},objects=objects))
        previous=labels;return result
    ConceptGraphsBackend.integrate=integrate
    try:run(a.cache,a.output)
    finally:ConceptGraphsBackend.integrate=original
    # Diagnostics are outside the frozen backend run manifest, which remains
    # verifiable and contains exactly its ordinary output files.
    dump(a.output/'causal_diagnostics.json',dict(complete=True,source_sha256=sha256(Path(__file__)),frames=history,
        scope='Native causal membership and surface-centroid changes, no identity GT. Renumbered final evaluation IDs are not persistent IDs.',
        changed_past_observation_events=sum(len(r['changed_past_observations']) for r in history)))


if __name__=='__main__':main()
