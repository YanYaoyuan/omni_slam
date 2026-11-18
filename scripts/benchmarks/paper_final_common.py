"""Immutable paper-v1 execution paths and provenance helpers."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'omni_semantic_map'))
RUN=ROOT/'build/benchmarks/paper_final_v1_20261111'
FROZEN=ROOT/'docs/paper/frozen_v1'
STORAGE=Path('/media/user/50D2BD30D2BD1AE4/WorkSpace/omni_slam_benchmarks/PaperFinalV1_20261111')
OLD=ROOT/'build/benchmarks/evidence_retention_20261111'
PY=ROOT/'build/benchmarks/ovo_full_20261110/venv311/bin/python'
CGPY=ROOT/'build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python'
SCENES=['036','038','041','045','047','049']


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def verify_execution():
    gate=read(RUN/'execution_manifest.json')
    for name,want in gate['source_sha256'].items():assert sha(ROOT/name)==want,name
    assert gate['paper_manifest_sha256']==sha(FROZEN/'manifest.json')
    assert gate['data_amendment_sha256']==sha(RUN/'data_amendment.json')
    if 'annotation_adapter_amendment_sha256' in gate:
        assert gate['annotation_adapter_amendment_sha256']==sha(RUN/'annotation_adapter_amendment.json')
        amendment=read(RUN/'annotation_adapter_amendment.json')
        assert sha(RUN/'predictions_complete.json')==amendment['predictions_gate_sha256']
        parent=RUN/'execution_revisions/v2_before_annotation_adapter/execution_manifest.json'
        assert sha(parent)==gate['prediction_execution_manifest_sha256']
        assert read(RUN/'predictions_complete.json')['execution_manifest_sha256']==sha(parent)
        for name,want in amendment['evidence_sha256'].items():assert sha(ROOT/name)==want,name
    return gate
def run_scenes():
    amendment=read(RUN/'data_amendment.json')
    assert amendment['selected_scenes']==SCENES and not amendment['replacement_scenes']
    return amendment['executable_scenes']
def scored_scenes():
    path=RUN/'annotation_adapter_amendment.json'
    return read(path)['scored_scenes'] if path.exists() else run_scenes()
def unavailable_scenes():
    unavailable=dict(read(RUN/'data_amendment.json')['unscored_scenes'])
    path=RUN/'annotation_adapter_amendment.json'
    if path.exists():unavailable.update(read(path)['unscored_scenes'])
    return unavailable
def resource_usage():
    import resource
    return {'cpu_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
