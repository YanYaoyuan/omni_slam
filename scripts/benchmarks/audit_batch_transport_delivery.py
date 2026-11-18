"""Verify delivered experiments without equating execution with a passed gate."""
from html.parser import HTMLParser
from paper_final_common import ROOT,read,dump,sha
from run_batch_transport_assessment import OUT as PUBLIC,seal,verify
from run_batch_transport_assessment_corrected import verify_amendment
from run_batch_transport_robot import OUT as ROBOT,verify as verify_robot


class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):self.refs.extend(v for k,v in attrs if k in ['href','src'])


def main():
    verify(True);verify_amendment();verify_robot();seals={}
    for root,names in [(PUBLIC,['masks_complete','features_complete','supports_complete','baseline_complete','fields_complete','predictions_complete']),(ROBOT,['baseline_complete','anchors_complete','prepared_complete','fields_complete','predictions_complete']),(PUBLIC/'semantics',['predictions_complete','aggregation_complete']),(ROBOT/'semantics',['predictions_complete','aggregation_complete'])]:
        for name in names:
            path=root/(name+'.json');d=seal(path);seals[str(path)]=dict(sha256=sha(path),verified_files=len(d['files']))
    links=[];reports={}
    for folder in [ROOT/'output/batch_transport_assessment_v1',ROOT/'output/batch_transport_robot_v1']:
        manifest=read(folder/'manifest.json');assert manifest['complete']
        for f,h in manifest['files'].items():assert sha(f)==h,f
        parser=Links();parser.feed((folder/'index.html').read_text())
        for ref in parser.refs:
            if '://' not in ref and not ref.startswith('#'):
                path=(folder/ref).resolve();assert path.exists(),str(path);links.append(str(path))
        reports[str(folder/'manifest.json')]=sha(folder/'manifest.json')
    p=read(PUBLIC/'summary.json');r=read(ROBOT/'summary.json');a=read(PUBLIC/'prediction_audit.json');domain=read(PUBLIC/'reference_domain_audit.json')
    assert p['complete'] and r['complete'] and a['all_ten_maps_checked'] and not p['independent_gate']
    assert all(v is None for v in p['mean_primary_ap'].values()) and domain['results']['kitti360_0007']['target_reference_points']==0
    assert len(p['results'])==2 and all(len(v['methods'])==5 for v in p['results'].values()) and len(r['results'])==5
    assert read(ROBOT/'semantics/predictions_complete.json')['forwards']==330 and read(PUBLIC/'semantics/predictions_complete.json')['forwards']==89
    assert len(read(ROOT/'output/batch_transport_assessment_v1/all_objects.json')['kitti360_0006'])==55
    assert all(len(v['per_object'])==61 for v in r['results'].values())
    old=read(ROOT/'output/batch_stable_transport_v1/manifest.json')
    for f,h in old['files'].items():assert sha(f)==h,f
    status=dict(new_mechanism_implemented_and_development_tested=True,new_public_predictions_and_scoring_executed=True,public_valid_target_windows=1,public_target_absent_windows=1,public_quality_gate_passed=False,same_version_robot_five_controls_complete=True,robot_semantic_accuracy_verified=False,independent_mechanism_advantage_verified=False,goal_complete=False)
    dump(PUBLIC/'delivery_audit.json',dict(complete=True,source_sha256=sha(__file__),verified_seals=seals,report_hashes=reports,checked_local_links=links,visually_reviewed_figures=['kitti360_0006.png','kitti360_0007.png','robot/maps.png','robot/projection_000085.png','robot/projection_000167.png','robot/projection_000249.png'],old_development_report_unchanged=True,requirements=status,remaining='Independent mechanism superiority remains unverified; reserved0007 had no targetGT. Do not promote failed gate or claim robot accuracy. No replacement scene/threshold sweep performed.'))
    print('DELIVERY AUDIT',status,'verified seals',len(seals),'links',len(links),flush=True)


if __name__=='__main__':main()
