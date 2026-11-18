"""Post-v1 development ablation: isolate consensus from quality/exclusivity."""
import argparse
from pathlib import Path
import numpy as np
import public_category_evidence as base
from public_dino_protocol import read,verify
from audit_evidence_retention import ROOT,OUT,dump
from omni_semantic_map.flat_benchmark import sha256
from omni_semantic_map.category_evidence import readout as original,METHODS as ORIGINAL_METHODS

DEST=OUT/'public_category_evidence_v2'
CONTROLS=('consensus_uniform','consensus_sqrt_area','consensus_area')


def readout(features,areas,exclusive_areas,text,*,method='mean',dino=None):
    if method not in CONTROLS:return original(features,areas,exclusive_areas,text,method=method,dino=dino)
    result=original(features,areas,exclusive_areas,text,method='quality_siglip',dino=dino)
    a=np.asarray(areas,np.float64)
    weights=np.ones(len(a)) if method=='consensus_uniform' else np.sqrt(a) if method=='consensus_sqrt_area' else a.copy()
    weights*=result['selected'];weights/=weights.sum()
    descriptor=weights@np.asarray(features,np.float64);norm=np.linalg.norm(descriptor)
    if norm<=1e-12:raise ValueError('degenerate descriptor')
    result.update(scores=np.asarray(text,np.float64)@(descriptor/norm),weights=weights,
        effective_sample_size=float(1/(weights@weights)))
    return result


def freeze():
    assert not DEST.exists()
    spec=read(base.DEST/'protocol.json');verify(spec)
    spec.update(methods=list(ORIGINAL_METHODS)+list(CONTROLS),
        extension='Post-v1 development ablation after viewing all v1 results; no independent-test claim.',
        added_controls='Same SigLIP centrality-selected views; respectively uniform, sqrt(area), and area weights. Compare to quality_siglip to isolate the extra exclusivity term.',
        parent_category_protocol_sha256=sha256(base.DEST/'protocol.json'))
    source=Path(__file__).resolve();spec['source_sha256'][str(source)]=sha256(source)
    dump(DEST/'protocol.json',spec);(DEST/'sources').mkdir();(DEST/'sources'/source.name).write_bytes(source.read_bytes())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    spec=read(DEST/'protocol.json');verify(spec);base.DEST=DEST;base.METHODS=spec['methods'];base.readout=readout
    if a.run:
        for e in spec['scenes']:base.predict(e['key'],spec)
        dump(DEST/'predictions_complete.json',dict(complete=True,files={e['key']:sha256(DEST/'runs'/e['key']/'run.json') for e in spec['scenes']}))
        for e in spec['scenes']:base.evaluate(e['key'],spec)
        base.summarize(spec)
