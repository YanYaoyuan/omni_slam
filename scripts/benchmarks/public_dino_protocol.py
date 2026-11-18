"""Freeze all-scene DINO history verification and separate category evaluation."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from audit_evidence_retention import ROOT,OUT,dump
from omni_semantic_map.flat_benchmark import sha256

DEST=OUT/'public_dino_v1'
PARENT=OUT/'public_history_v1'
MODEL=ROOT/'build/benchmarks/flat_dino_20261109/model'


def read(p):return json.loads(Path(p).read_text())


def freeze():
    assert not DEST.exists()
    scenes=read(PARENT/'protocol.json')['scenes']
    for entry in scenes:
        cache=Path(entry['cache']);m=read(cache/'manifest.json');frames=[]
        if entry['dataset']=='Replica':
            rgb=sorted((cache.parents[2]/'OVO_Replica_full'/entry['scene']/'results').glob('frame*.jpg'));assert len(rgb)==2000
            images={r['frame_id']:rgb[r['frame_id']] for r in m['records']}
            candidates=[ROOT/'build/benchmarks'/root/'data'/entry['scene']/'info_semantic.json' for root in ['replica_independent_20261109','ownership_20261109/heldout','expanded_semantic_20261109/replica']]
            entry['semantic_metadata']=str(next(p for p in candidates if p.exists()))
        else:
            job=read(OUT/'jobs/v1'/(entry['scene']+'.json'))
            images={r['frame_id']:Path(r['rgb']) for r in job['frames']}
            entry['semantic_metadata']=str(OUT/'official_annotation_audit/cvpr18'/('scenenn_seg_'+entry['scene']+'.hdf5'))
        for r in m['records']:
            p=images[r['frame_id']];frames.append(dict(frame_id=r['frame_id'],rgb=str(p),sha256=sha256(p)))
        entry['images']=frames;entry['semantic_metadata_sha256']=sha256(entry['semantic_metadata'])
        entry['parent_run_sha256']=sha256(PARENT/'runs'/entry['key']/'run.json')
    sources=[ROOT/'scripts/benchmarks'/f for f in ['public_dino_protocol.py','extract_public_dino.py','run_public_dino_graph.py','evaluate_public_dino.py']]
    sources += [ROOT/'omni_semantic_map/omni_semantic_map'/f for f in ['dino_history_evidence.py','semantic_instance_diagnostic.py','livo_dino_support.py','history_graph.py']]
    sources += [ROOT/'scripts/benchmarks'/f for f in ['run_public_history_graph.py','run_livo_history_graph.py']]
    xml=OUT/'official_annotation_audit/nyu_class/nyu_color.xml'
    names={c.attrib['text'].replace(' ','-').lower():int(c.attrib['id']) for c in ET.parse(xml).getroot().iter('class')}
    aliases={'tv-screen':'television','nightstand':'night-stand','shelf':'shelves','book':'books'}
    vocab=read(Path(scenes[-1]['cache'])/'vocabulary.json')
    mapping={name:names.get(aliases.get(name,name),-1) for name in vocab['names']}
    dump(DEST/'protocol.json',dict(scenes=scenes,source_sha256={str(p):sha256(p) for p in sources},
        model_sha256={str(p):sha256(p) for p in MODEL.iterdir() if p.is_file()},
        parent_protocol_sha256=sha256(PARENT/'protocol.json'),
        feature='Frozen DINOv2-small; full RGB bicubic width448, height nearest multiple14, no crop; ImageNet normalization FP32; measured-support weighted pooling, <=24 local tokens using unchanged livo_dino_support.pool.',
        variants=['baseline','native_bridge_2','dino_pooled_2','dino_local_2'],
        candidate='Identical native ambiguous queries and pre-final nodes to public_history_v1. DINO never proposes new edges.',
        pooled='Cosine of support-pooled vectors; each historical branch uses maximum over its causal prior observations.',
        local='Mean cosine of reciprocal nearest appearance token pairs, at least3 pairs; no geometric correspondence claim or distance cutoff. Each branch uses maximum valid prior score.',
        vote='Both historical branches must exceed maximum same-frame competitor appearance score. Any missing branch score, absent competitor, or unknown competitor score retains the native vote, recorded separately.',
        graph='Same two-distinct-query-frame votes, component coframe veto, and common native proposal order. Original finalization and SigLIP mean unchanged.',
        missing_policy='Pool and local share conservative native fallback, unlike earlier LIVO strict unsupported rejection. No fallback score is fabricated.',
        no_training=True,no_sweep=True,all_scenes=True,development_only=True,
        semantic='Separate class-aware instance PQ and matched-instance class accuracy; same predicted foreground points. Unknown predicted ontology class counts as wrong, never dropped.',
        replica_gt='Exact provided instance id+1 to original class_id; missing/negative GT classes void only for category evaluation, report domain coverage.',
        scenenn_gt='Exact CVPR18 annotation coordinates to original instance mesh; dominant NYU class among unique annotated vertices per instance. Require>=.7 purity; ambiguous/absent class void only for category metrics. This cross-release diagnostic is not an official semantic benchmark.',
        nyu_names=names,nyu_prediction_mapping=mapping,nyu_metadata_sha256=sha256(xml),
        nyu_source='https://hkust-vgd.ust.hk/scenenn/contrib/nyu_class/nyu_color.xml',
        limits='Previously viewed scenes. No LIVO localization claim. Candidate GT signatures are post-hoc. Class-aware PQ pooled over instances is a diagnostic, not official per-class macro panoptic PQ.'))
    for p in sources:
        target=DEST/'sources'/p.name;target.parent.mkdir(exist_ok=True);target.write_bytes(p.read_bytes())


def verify(spec):
    for p,h in spec['source_sha256'].items():assert sha256(p)==h,p


if __name__=='__main__':freeze()
