"""Verify current document evidence; stdlib only, no mutation or GPU access."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/paper/visibility_revision'
BASE=ROOT/'build/benchmarks/visibility_independent_20261112'

def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def check(condition,message):
    if not condition:raise ValueError(message)

def verify(allow_unreviewed=False):
    manifest=read(OUT/'build_manifest.json');supp=read(OUT/'supplement_manifest.json')
    paths={**manifest['input_sha256'],**manifest['output_sha256'],**supp['inputs_sha256']}
    for name,digest in paths.items():
        p=ROOT/name;check(p.is_file(),'missing evidence: '+name)
        check(sha(p)==digest,'changed evidence: '+name)
    for name,digest in supp['generated_sha256'].items():check(sha(OUT/name)==digest,'changed supplement: '+name)
    check(sha(ROOT/'scripts/paper/build_visibility_manuscript.py')==manifest['script_sha256'],'changed manuscript builder')
    check(sha(ROOT/'scripts/paper/build_visibility_supplement.py')==supp['source_sha256'],'changed supplement builder')
    protocol=read(BASE/'protocol.json');check(protocol['primary']=='visibility1','primary method differs')
    for name,digest in protocol['source_sha256'].items():check(sha(ROOT/name)==digest,'changed frozen code: '+name)
    check(not supp['development_gate_passed'] and not supp['independent_gate_passed'],'historical failed gate was relabeled')
    text=(OUT/'manuscript_resolved.md').read_text();check('{{' not in text,'unresolved document tokens')
    metrics=[read(BASE/'evaluation'/s/'metrics.json')['methods'] for s in ['052','054','057','061']]
    ours=[m['visibility1']['score']['observed_voxel_instance_pq']*100 for m in metrics]
    expected='| **Ours** | '+' | '.join(f'{x:.3f}' for x in [*ours,statistics.mean(ours)])+' |'
    check(expected in text,'main Ours table differs from frozen scores')
    for name,value in manifest['means_pq100'].items():
        got=statistics.mean(m[name]['score']['observed_voxel_instance_pq']*100 for m in metrics)
        check(abs(value-got)<1e-10,'mean mismatch: '+name)
    for section in ['Development Results Are Separate Evidence','Relationship to Existing Point Filtering','Reproduction and Artifact Scope']:
        check(section in text,'missing required supplement: '+section)
    for name in ['manuscript_resolved.md','editorial_note_zh.md','REPRODUCE.md']:
        for target in re.findall(r'\]\(([^)]+)\)',(OUT/name).read_text()):
            if not target.startswith(('http:','https:','#')):check((OUT/target).is_file(),'broken link: '+target)
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,target in attrs:
                if key in ['src','href'] and not target.startswith(('http:','https:','#')):
                    check((OUT/target).is_file(),'broken HTML link: '+target)
    for document in ['index.html','visual_gallery.html']:
        Links().feed((OUT/document).read_text())
    for figure in ['method_architecture','dataset_scene_057','dataset_scene_061','object_comparison_real','garage_real_evidence','rosbag_routes_1','rosbag_routes_2','rosbag_outdoor_semantics','rosbag_ground_details']:
        for ext in ['png','svg']:
            check((OUT/'figures'/(figure+'.'+ext)).is_file(),'missing real-data visual: '+figure)
    check('visibility1 cleanup used in the benchmark is not applied to this garage result' in text,'garage branch scope missing')
    rosbag=read(OUT/'figures/rosbag_visual_provenance.json')
    check(len(rosbag['recordings'])==6,'six recording inventory missing')
    check(all(r['raw_bag_exists'] and not r['visibility1_applied'] for r in rosbag['recordings']),'incorrect ROSBAG scope')
    check('visibility1 is not applied to these outdoor results' in text,'outdoor branch scope missing')
    visual=False
    if not manifest['pdf_render_review_pending']:
        qa=read(OUT/'render_qa.json')
        visual=qa['complete'] and qa['all_pages_rendered_and_visually_reviewed'] and qa['pdf_sha256']==sha(ROOT/'output/pdf/omni_slam_visibility_revision_draft.pdf')
        check(visual,'PDF visual review does not match current PDF')
        check(sha(OUT/'render_qa.json')==manifest['render_qa_sha256'],'changed render QA')
    if not allow_unreviewed:check(visual,'PDF was rebuilt and needs visual review; use the documented Poppler check')
    return dict(evidence_verified=True,hashed_paths=len(paths),methods=7,primary='visibility1',
        development_and_test_separated=True,main_scores_unchanged=True,visual_review_current=visual,
        no_new_mapping_or_inference=True,scope='Existing-result paper package; not validation of broader unmeasured paper claims.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--allow-unreviewed-pdf',action='store_true');args=parser.parse_args()
    try:print(json.dumps(verify(args.allow_unreviewed_pdf),indent=2))
    except (OSError,ValueError,KeyError) as exc:raise SystemExit('VERIFICATION FAILED: '+str(exc))
